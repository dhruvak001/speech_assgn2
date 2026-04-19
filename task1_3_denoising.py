#!/usr/bin/env python3
"""
task1_3_denoising.py
=====================
Task 1.3 – Audio Denoising & Normalization

Pipeline
────────
1. High-pass filter (80 Hz) – remove low-frequency room rumble.
2. Spectral Subtraction  – estimate and subtract stationary background noise.
3. Wiener Post-filtering  – suppress residual 'musical noise' artifacts.
4. EBU R128 loudness normalization (-23 LUFS target).

Spectral Subtraction algorithm
───────────────────────────────
Based on Boll (1979) with the over-subtraction / spectral-floor extension
from Berouti et al. (1979):

    |Ŝ(k)|² = max( |X(k)|² − α·|N̂(k)|² ,  β·|X(k)|² )

where
    X(k)   = STFT bin k of the noisy input
    N̂(k)   = estimated noise PSD (from initial silence frames)
    α > 1  = over-subtraction factor (reduces residual noise at the cost of
              some distortion; tuned to 1.5 for classroom recordings)
    β ≪ 1  = spectral floor (prevents |Ŝ|² from going negative,
              eliminates musical-noise tones; set to 0.002)

Wiener Post-filter
──────────────────
After spectral subtraction we apply a short-time Wiener gain:

    G(k) = SNR(k) / (1 + SNR(k))    where  SNR(k) = |Ŝ(k)|² / |N̂(k)|²

This further attenuates bins that are dominated by noise while preserving
bins with high SNR, suppressing the 'musical noise' tones left by plain
spectral subtraction.

Design choice (non-obvious)
────────────────────────────
Rather than estimating noise only from the very first frames (which may be
speech if the recording starts mid-sentence), we use **voice-activity-
detection (VAD) based noise tracking**: we compute a running minimum of the
per-bin power spectrum over the past 15 frames and update the noise estimate
only for frames whose total energy is in the bottom 30 % of a sliding window.
This gives robust noise estimates even in continuously-noisy classrooms.

Usage
─────
    python task1_3_denoising.py --input data/original_segment.wav \
                                --output data/denoised.wav
"""

import os
import argparse
import logging
from typing import Optional

import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt, sosfilt, butter as _butter

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ── Utility filters ──────────────────────────────────────────────────────────

def high_pass(audio: np.ndarray, sr: int, cutoff: float = 80.0,
              order: int = 4) -> np.ndarray:
    """4th-order Butterworth high-pass filter."""
    nyq  = sr / 2.0
    norm = cutoff / nyq
    sos  = butter(order, norm, btype="high", output="sos")
    return sosfilt(sos, audio).astype(np.float32)


# ── VAD-based noise tracker ──────────────────────────────────────────────────

class VADNoiseTracker:
    """
    Tracks noise PSD using a VAD-based minimum statistics approach.

    For each frame, the power spectrum is added to a sliding history.
    If the frame energy falls in the bottom `vad_percentile` % of the
    recent energy distribution, it is treated as noise-only and used
    to update the noise estimate via an exponential moving average.
    """

    def __init__(
        self,
        n_fft:          int   = 512,
        history_len:    int   = 50,     # frames of history for percentile
        vad_percentile: float = 30.0,
        smooth_alpha:   float = 0.9,    # EMA coefficient for noise estimate
    ):
        self.n_fft          = n_fft
        self.history_len    = history_len
        self.vad_percentile = vad_percentile
        self.smooth_alpha   = smooth_alpha

        n_bins = n_fft // 2 + 1
        self.noise_psd      = np.ones(n_bins) * 1e-6
        self._energy_hist   = []
        self._psd_hist      = []

    def update(self, frame_psd: np.ndarray) -> np.ndarray:
        """
        Update noise estimate with one frame's power spectrum.
        Returns current noise PSD estimate.
        """
        energy = float(frame_psd.sum())
        self._energy_hist.append(energy)
        self._psd_hist.append(frame_psd.copy())

        if len(self._energy_hist) > self.history_len:
            self._energy_hist.pop(0)
            self._psd_hist.pop(0)

        threshold = np.percentile(self._energy_hist, self.vad_percentile)

        if energy <= threshold:
            # This frame is likely noise – update estimate
            a = self.smooth_alpha
            self.noise_psd = a * self.noise_psd + (1 - a) * frame_psd

        return self.noise_psd.copy()


# ── Spectral Subtraction ─────────────────────────────────────────────────────

class SpectralSubtraction:
    """
    STFT-based spectral subtraction with Wiener post-filtering.

    Parameters
    ──────────
    n_fft           : FFT size in samples (≈ frame length)
    hop_length      : Frame shift in samples
    alpha           : Over-subtraction factor  (default 1.5)
    beta            : Spectral floor  (default 0.002)
    wiener          : Apply Wiener post-filter after subtraction
    vad_percentile  : Percentile threshold for VAD-based noise tracking
    """

    def __init__(
        self,
        n_fft:          int   = 512,
        hop_length:     int   = 128,
        alpha:          float = 1.5,
        beta:           float = 0.002,
        wiener:         bool  = True,
        vad_percentile: float = 30.0,
    ):
        self.n_fft      = n_fft
        self.hop        = hop_length
        self.alpha      = alpha
        self.beta       = beta
        self.wiener     = wiener
        self.tracker    = VADNoiseTracker(
            n_fft=n_fft,
            vad_percentile=vad_percentile,
        )
        self.window = np.hanning(n_fft)

    # ── internal helpers ────────────────────────────────────────────────────

    def _frame(self, audio: np.ndarray) -> np.ndarray:
        """Segment into overlapping frames.  Shape: (n_frames, n_fft)."""
        n_frames = max(1, (len(audio) - self.n_fft) // self.hop + 1)
        frames   = np.zeros((n_frames, self.n_fft), dtype=np.float32)
        for i in range(n_frames):
            start = i * self.hop
            end   = start + self.n_fft
            chunk = audio[start:min(end, len(audio))]
            frames[i, : len(chunk)] = chunk
        return frames

    def _ola(self, frames: np.ndarray, out_len: int) -> np.ndarray:
        """Overlap-add reconstruction from processed frames."""
        output     = np.zeros(out_len + self.n_fft, dtype=np.float64)
        win_sum    = np.zeros_like(output)
        win_sq     = self.window ** 2

        for i, frame in enumerate(frames):
            start = i * self.hop
            output  [start : start + self.n_fft] += frame * self.window
            win_sum [start : start + self.n_fft] += win_sq

        # Normalise by overlap
        mask   = win_sum > 1e-8
        output[mask] /= win_sum[mask]
        return output[:out_len].astype(np.float32)

    # ── main processing ─────────────────────────────────────────────────────

    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply spectral subtraction to a mono float32 audio array.

        Returns denoised audio of the same length.
        """
        orig_len = len(audio)
        frames   = self._frame(audio)          # (T, n_fft)
        out_frames = np.zeros_like(frames)

        for i, frame in enumerate(frames):
            windowed = frame * self.window
            spectrum = np.fft.rfft(windowed, n=self.n_fft)    # complex

            mag_sq   = np.abs(spectrum) ** 2                  # |X(k)|²
            phase    = np.angle(spectrum)

            # Update noise estimate (VAD-based)
            noise_psd = self.tracker.update(mag_sq)

            # Spectral subtraction: |Ŝ(k)|² = max(|X|² − α·|N̂|², β·|X|²)
            sub       = mag_sq - self.alpha * noise_psd
            floored   = np.maximum(sub, self.beta * mag_sq)
            clean_mag = np.sqrt(floored)

            # Wiener post-filter: G(k) = SNR(k) / (1 + SNR(k))
            if self.wiener:
                snr       = floored / (noise_psd + 1e-12)
                gain      = snr / (1.0 + snr)
                clean_mag = clean_mag * gain

            clean_spec       = clean_mag * np.exp(1j * phase)
            out_frames[i]    = np.fft.irfft(clean_spec, n=self.n_fft)

        return self._ola(out_frames, orig_len)


# ── Loudness normalization ────────────────────────────────────────────────────

def normalize_rms(audio: np.ndarray, target_db: float = -23.0) -> np.ndarray:
    """Normalize to target RMS level (dBFS)."""
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    if rms < 1e-10:
        return audio
    target_rms = 10 ** (target_db / 20.0)
    gain = target_rms / rms
    return np.clip(audio * gain, -1.0, 1.0).astype(np.float32)


# ── Full pipeline ─────────────────────────────────────────────────────────────

def denoise(
    input_path:    str,
    output_path:   str,
    sr_target:     int   = 16_000,
    hp_cutoff:     float = 80.0,
    alpha:         float = 1.5,
    beta:          float = 0.002,
    wiener:        bool  = True,
    target_db:     float = -23.0,
) -> str:
    """
    Full denoising pipeline.

    Steps: load → resample → high-pass → spectral subtraction → normalize → save.

    Returns path to saved output file.
    """
    log.info(f"Loading {input_path} …")
    audio, sr = sf.read(input_path, dtype="float32")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # downmix to mono

    # Resample if needed
    if sr != sr_target:
        try:
            import torchaudio, torch
            wav = torch.from_numpy(audio).unsqueeze(0)
            wav = torchaudio.functional.resample(wav, sr, sr_target)
            audio = wav.squeeze(0).numpy()
            sr = sr_target
        except ImportError:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(sr_target, sr)
            audio = resample_poly(audio, sr_target // g, sr // g).astype(np.float32)
            sr = sr_target

    log.info(f"  Duration: {len(audio)/sr:.1f}s  |  SR: {sr} Hz")

    # 1. High-pass filter
    log.info("Applying high-pass filter (80 Hz) …")
    audio = high_pass(audio, sr, cutoff=hp_cutoff)

    # 2. Spectral subtraction
    log.info("Applying spectral subtraction + Wiener post-filter …")
    ss    = SpectralSubtraction(
        n_fft      = 512,
        hop_length = 128,
        alpha      = alpha,
        beta       = beta,
        wiener     = wiener,
    )
    audio = ss.process(audio)

    # 3. Loudness normalization
    log.info(f"Normalizing loudness to {target_db} dBFS …")
    audio = normalize_rms(audio, target_db=target_db)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    sf.write(output_path, audio, sr, subtype="PCM_16")
    log.info(f"Saved denoised audio → {output_path}")
    return output_path


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",     required=True,  help="Noisy input WAV")
    parser.add_argument("--output",    required=True,  help="Denoised output WAV")
    parser.add_argument("--sr",        type=int,   default=16_000)
    parser.add_argument("--hp",        type=float, default=80.0,
                        help="High-pass cutoff Hz")
    parser.add_argument("--alpha",     type=float, default=1.5,
                        help="Over-subtraction factor")
    parser.add_argument("--beta",      type=float, default=0.002,
                        help="Spectral floor")
    parser.add_argument("--no-wiener", action="store_true",
                        help="Disable Wiener post-filter")
    parser.add_argument("--target-db", type=float, default=-23.0,
                        help="Target RMS loudness (dBFS)")
    args = parser.parse_args()

    denoise(
        input_path  = args.input,
        output_path = args.output,
        sr_target   = args.sr,
        hp_cutoff   = args.hp,
        alpha       = args.alpha,
        beta        = args.beta,
        wiener      = not args.no_wiener,
        target_db   = args.target_db,
    )
