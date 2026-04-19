#!/usr/bin/env python3
"""
task3_2_prosody_warping.py
===========================
Task 3.2 – Prosody Warping via Dynamic Time Warping (DTW)

Goal
────
Transfer the "teaching style" prosody (F0 contour + energy envelope) from
the professor's lecture onto the synthesised Maithili speech so that the
pedagogical rhythm and intonational patterns are preserved.

Pipeline
────────
1. Extract F0 (fundamental frequency) from source (professor) audio using
   the PYIN algorithm (probabilistic YIN, Mauch & Dixon 2014).
2. Extract energy (RMS per frame) from source audio.
3. Extract F0 and energy from the synthesised target audio.
4. Apply DTW to find the optimal warping path between source and target
   prosodic sequences.
5. Warp the source F0/energy onto the target timeline.
6. Modify the target waveform: apply PSOLA/WORLD vocoder to inject the
   warped F0, and scale the signal frame-by-frame to match warped energy.

F0 extraction (PYIN)
─────────────────────
PYIN is preferred over plain YIN because it handles voiced/unvoiced decisions
probabilistically, giving cleaner F0 tracks with fewer octave errors — which
is critical for Indian speech where pitch range is wide.

DTW cost matrix
────────────────
Given source sequence  s = [s_1 … s_M]  and target  t = [t_1 … t_N] , the
DTW distance is:

    D(i, j) = |s_i − t_j| + min( D(i−1,j), D(i,j−1), D(i−1,j−1) )

The warping path  W = [(i_1,j_1) … (i_K,j_K)]  maps target frame k to
source frame i_k.  We then resample the source F0/energy along this path.

Design choice (non-obvious)
────────────────────────────
Rather than warping in the log-F0 domain only, we normalise both sequences
to z-scores before computing DTW.  This makes the alignment invariant to
absolute pitch differences between speaker genders (professor vs student)
while preserving relative pitch shape.  After DTW we map the warped
normalised contour back to the student's absolute F0 range by:

    F0_warped(t) = exp( z_warped(t) · σ_student + μ_student )

where μ, σ are estimated from the student's voiced frames only.

Usage
─────
    python task3_2_prosody_warping.py \
        --source data/original_segment_22k.wav \
        --target data/synth_flat.wav \
        --output data/synth_prosody_warped.wav
"""

import os
import argparse
import logging
from typing import Tuple, Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# Frame parameters (22.05 kHz)
SR_DEFAULT = 22_050
HOP_MS     = 10          # 10 ms hop → 220 samples @ 22 kHz
FRAME_MS   = 25          # 25 ms window


# ═══════════════════════════════════════════════════════════════════════════════
# F0 extraction (PYIN via librosa)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_f0(audio: np.ndarray, sr: int,
               fmin: float = 50.0, fmax: float = 600.0) -> np.ndarray:
    """
    Extract F0 in Hz using PYIN.

    Returns array of shape (T,) with NaN for unvoiced frames.
    """
    try:
        import librosa
        hop = max(1, int(sr * HOP_MS / 1000))
        f0, voiced_flag, voiced_prob = librosa.pyin(
            audio,
            fmin       = fmin,
            fmax       = fmax,
            sr         = sr,
            hop_length = hop,
            frame_length = int(sr * FRAME_MS / 1000),
        )
        # PYIN returns 0.0 for unvoiced; replace with NaN
        f0 = f0.astype(np.float32)
        f0[~voiced_flag] = np.nan
        return f0
    except ImportError:
        log.warning("librosa not found, using simple autocorrelation F0.")
        return _autocorr_f0(audio, sr)


def _autocorr_f0(audio: np.ndarray, sr: int,
                 fmin: float = 50.0, fmax: float = 600.0) -> np.ndarray:
    """Naive autocorrelation-based F0 (fallback)."""
    hop        = int(sr * HOP_MS / 1000)
    win_len    = int(sr * FRAME_MS / 1000)
    n_frames   = (len(audio) - win_len) // hop + 1
    f0_out     = np.full(n_frames, np.nan, dtype=np.float32)
    min_period = int(sr / fmax)
    max_period = int(sr / fmin)

    for i in range(n_frames):
        frame = audio[i * hop : i * hop + win_len].astype(np.float64)
        if np.abs(frame).mean() < 1e-4:
            continue
        frame -= frame.mean()
        corr = np.correlate(frame, frame, mode="full")
        corr = corr[len(corr) // 2:]
        lags  = corr[min_period:max_period]
        if lags.size == 0:
            continue
        peak  = lags.argmax() + min_period
        if corr[0] > 0:
            f0_out[i] = sr / peak
    return f0_out


# ═══════════════════════════════════════════════════════════════════════════════
# Energy (RMS) extraction
# ═══════════════════════════════════════════════════════════════════════════════

def extract_energy(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Compute per-frame RMS energy.

    Returns array of shape (T,).
    """
    hop   = int(sr * HOP_MS   / 1000)
    win   = int(sr * FRAME_MS / 1000)
    n_frames = (len(audio) - win) // hop + 1
    energy   = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        frame        = audio[i * hop : i * hop + win]
        energy[i]    = np.sqrt(np.mean(frame.astype(np.float64) ** 2) + 1e-10)
    return energy


# ═══════════════════════════════════════════════════════════════════════════════
# Dynamic Time Warping
# ═══════════════════════════════════════════════════════════════════════════════

def dtw_path(
    source: np.ndarray,
    target: np.ndarray,
    dist:   str = "abs",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute DTW warping path between two 1-D sequences.

    Parameters
    ──────────
    source, target : 1-D arrays of shape (M,) and (N,)
    dist           : distance function  "abs" | "sq"

    Returns
    ───────
    path_s : source indices (length K)
    path_t : target indices (length K)
    """
    M, N = len(source), len(target)

    # Cost matrix
    if dist == "sq":
        C = (source[:, None] - target[None, :]) ** 2
    else:
        C = np.abs(source[:, None] - target[None, :])

    # Accumulated cost
    D = np.full((M, N), np.inf, dtype=np.float64)
    D[0, 0] = C[0, 0]
    for i in range(1, M):
        D[i, 0] = D[i-1, 0] + C[i, 0]
    for j in range(1, N):
        D[0, j] = D[0, j-1] + C[0, j]
    for i in range(1, M):
        for j in range(1, N):
            D[i, j] = C[i, j] + min(D[i-1, j], D[i, j-1], D[i-1, j-1])

    # Backtrack
    i, j   = M - 1, N - 1
    path_s = [i]
    path_t = [j]
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            choice = np.argmin([D[i-1, j], D[i, j-1], D[i-1, j-1]])
            if choice == 0:
                i -= 1
            elif choice == 1:
                j -= 1
            else:
                i -= 1
                j -= 1
        path_s.append(i)
        path_t.append(j)

    path_s = np.array(path_s[::-1], dtype=int)
    path_t = np.array(path_t[::-1], dtype=int)
    return path_s, path_t


def _nan_interp(x: np.ndarray) -> np.ndarray:
    """Linear interpolation over NaN values."""
    nans = np.isnan(x)
    if nans.all():
        return np.zeros_like(x)
    idx  = np.arange(len(x))
    xout = x.copy()
    xout[nans] = np.interp(idx[nans], idx[~nans], x[~nans])
    return xout


def warp_prosody(
    src_f0:     np.ndarray,   # (M,)
    src_energy: np.ndarray,   # (M,)
    tgt_f0:     np.ndarray,   # (N,)
    tgt_energy: np.ndarray,   # (N,)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply DTW to produce prosody-warped F0 and energy for the target.

    Returns warped_f0 (N,) and warped_energy (N,) arrays.
    """
    # Interpolate over NaN (unvoiced) frames for DTW alignment
    src_f0_i = _nan_interp(src_f0)
    tgt_f0_i = _nan_interp(tgt_f0)

    # ── Normalise to z-scores ────────────────────────────────────────────────
    src_mu, src_sigma = src_f0_i.mean(), src_f0_i.std() + 1e-6
    tgt_mu, tgt_sigma = tgt_f0_i.mean(), tgt_f0_i.std() + 1e-6

    src_z = (src_f0_i - src_mu) / src_sigma
    tgt_z = (tgt_f0_i - tgt_mu) / tgt_sigma

    log.info(
        f"DTW: source={len(src_z)} frames, target={len(tgt_z)} frames"
    )

    # DTW alignment on z-score F0
    path_s, path_t = dtw_path(src_z, tgt_z)

    # Build target-length warped source z-scores
    N = len(tgt_z)
    warped_src_z = np.zeros(N, dtype=np.float32)
    for t in range(N):
        # All source frames that map to target frame t
        src_frames = path_s[path_t == t]
        if len(src_frames):
            warped_src_z[t] = src_z[src_frames].mean()
        else:
            warped_src_z[t] = 0.0

    # Map warped z-scores back to target speaker's absolute F0 range
    # warped_f0 = exp(z_warped * σ_tgt + μ_tgt)  [in Hz]
    log_tgt_voiced = np.log(tgt_f0_i[tgt_f0_i > 0] + 1e-9)
    if len(log_tgt_voiced) > 5:
        log_mu  = log_tgt_voiced.mean()
        log_sig = log_tgt_voiced.std() + 1e-6
    else:
        log_mu, log_sig = np.log(150.0), 0.3

    warped_f0 = np.exp(warped_src_z * log_sig + log_mu).astype(np.float32)

    # Restore NaN for frames where target was originally unvoiced
    voiced_mask = ~np.isnan(tgt_f0)
    warped_f0[~voiced_mask] = np.nan

    # ── Energy warping (simpler: just resample source energy to target length) ─
    from scipy.interpolate import interp1d
    src_t = np.linspace(0, 1, len(src_energy))
    tgt_t = np.linspace(0, 1, N)
    energy_interp = interp1d(src_t, src_energy, kind="linear",
                              bounds_error=False, fill_value="extrapolate")
    warped_energy = energy_interp(tgt_t).astype(np.float32)
    warped_energy = np.clip(warped_energy, 0, None)

    log.info(
        f"Warped F0: voiced={np.sum(~np.isnan(warped_f0))}/{N}  "
        f"mean={np.nanmean(warped_f0):.1f} Hz"
    )
    return warped_f0, warped_energy


# ═══════════════════════════════════════════════════════════════════════════════
# WORLD-vocoder based prosody injection
# ═══════════════════════════════════════════════════════════════════════════════

def inject_prosody_world(
    audio:         np.ndarray,
    sr:            int,
    warped_f0:     np.ndarray,
    warped_energy: np.ndarray,
) -> np.ndarray:
    """
    Use WORLD vocoder to re-synthesise audio with the warped F0/energy.

    Falls back to pitch-shifting via torchaudio if pyworld is unavailable.
    """
    try:
        import pyworld as pw

        audio64 = audio.astype(np.float64)
        frame_period = HOP_MS  # ms

        # Analysis
        _f0, sp, ap = pw.wav2world(audio64, sr, frame_period=frame_period)

        # Replace F0 with warped version (trim/pad to match analysis length)
        T_world = len(_f0)
        T_warp  = len(warped_f0)

        if T_warp >= T_world:
            new_f0 = warped_f0[:T_world].astype(np.float64)
        else:
            new_f0 = np.concatenate([
                warped_f0,
                np.full(T_world - T_warp, warped_f0[-1]),
            ]).astype(np.float64)

        # Replace NaN (unvoiced) with 0 for WORLD
        new_f0 = np.where(np.isnan(new_f0), 0.0, new_f0)

        # Scale spectral magnitude to match warped energy
        T_sp = sp.shape[0]
        if len(warped_energy) >= T_sp:
            e_ref = warped_energy[:T_sp]
        else:
            e_ref = np.pad(warped_energy, (0, T_sp - len(warped_energy)),
                           mode="edge")
        e_orig = np.sqrt(np.mean(sp ** 2, axis=1) + 1e-12)
        gain   = (e_ref / (e_orig + 1e-12))[:, None]
        sp_mod = sp * gain

        # Synthesis
        synth = pw.synthesize(new_f0, sp_mod, ap, sr,
                              frame_period=frame_period)
        return synth.astype(np.float32)

    except ImportError:
        log.warning("pyworld not available – applying energy scaling only.")
        return _energy_scale_only(audio, sr, warped_energy)


def _energy_scale_only(
    audio:         np.ndarray,
    sr:            int,
    warped_energy: np.ndarray,
) -> np.ndarray:
    """
    Scale audio frame-by-frame to match warped energy (fallback).
    """
    hop = int(sr * HOP_MS / 1000)
    win = int(sr * FRAME_MS / 1000)
    out = audio.copy()
    T   = len(warped_energy)
    for i in range(T):
        start = i * hop
        end   = min(start + win, len(audio))
        chunk = audio[start:end]
        rms   = np.sqrt(np.mean(chunk ** 2) + 1e-10)
        gain  = warped_energy[i] / rms
        out[start:end] = chunk * np.clip(gain, 0.1, 10.0)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def prosody_warp(
    source_path: str,
    target_path: str,
    output_path: str,
    sr:          int = SR_DEFAULT,
) -> str:
    """
    Full prosody warping pipeline.

    source_path : professor lecture audio (prosody donor)
    target_path : flat synthesised speech (prosody recipient)
    output_path : output with professor's prosody applied

    Returns output_path.
    """
    log.info(f"Loading source  : {source_path}")
    src_audio, src_sr = sf.read(source_path, dtype="float32")
    if src_audio.ndim > 1:
        src_audio = src_audio.mean(axis=1)
    if src_sr != sr:
        src_audio = torchaudio.functional.resample(
            torch.from_numpy(src_audio), src_sr, sr
        ).numpy()

    log.info(f"Loading target  : {target_path}")
    tgt_audio, tgt_sr = sf.read(target_path, dtype="float32")
    if tgt_audio.ndim > 1:
        tgt_audio = tgt_audio.mean(axis=1)
    if tgt_sr != sr:
        tgt_audio = torchaudio.functional.resample(
            torch.from_numpy(tgt_audio), tgt_sr, sr
        ).numpy()

    # Extract prosodic features
    log.info("Extracting F0 (PYIN) from source …")
    src_f0  = extract_f0(src_audio, sr)
    log.info("Extracting F0 (PYIN) from target …")
    tgt_f0  = extract_f0(tgt_audio, sr)

    log.info("Extracting RMS energy …")
    src_energy = extract_energy(src_audio, sr)
    tgt_energy = extract_energy(tgt_audio, sr)

    # DTW prosody warp
    log.info("Running DTW …")
    warped_f0, warped_energy = warp_prosody(
        src_f0, src_energy, tgt_f0, tgt_energy
    )

    # Inject warped prosody via WORLD vocoder
    log.info("Injecting prosody via WORLD vocoder …")
    output_audio = inject_prosody_world(tgt_audio, sr, warped_f0, warped_energy)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    sf.write(output_path, output_audio, sr, subtype="PCM_16")
    log.info(f"Saved prosody-warped audio → {output_path}")

    # Save prosody traces for reporting
    traces_path = output_path.replace(".wav", "_prosody_traces.npz")
    np.savez(
        traces_path,
        src_f0     = src_f0,
        tgt_f0     = tgt_f0,
        warped_f0  = warped_f0,
        src_energy = src_energy,
        tgt_energy = tgt_energy,
        warped_energy = warped_energy,
    )
    log.info(f"Prosody traces saved → {traces_path}")
    return output_path


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",  required=True,
                        help="Source (professor) WAV for prosody extraction")
    parser.add_argument("--target",  required=True,
                        help="Target (synthesised flat) WAV to modify")
    parser.add_argument("--output",  required=True,
                        help="Output prosody-warped WAV")
    parser.add_argument("--sr",      type=int, default=SR_DEFAULT)
    args = parser.parse_args()

    prosody_warp(args.source, args.target, args.output, args.sr)
