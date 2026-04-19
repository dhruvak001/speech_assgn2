#!/usr/bin/env python3
"""
task4_2_adversarial.py
=======================
Task 4.2 – Adversarial Noise Injection via FGSM

Goal
────
Find the minimum perturbation ε that causes the LID model (Task 1.1) to
misclassify a Hindi speech segment as English, while keeping the noise
inaudible (SNR > 40 dB).

Method: Fast Gradient Sign Method (FGSM)
─────────────────────────────────────────
FGSM (Goodfellow et al. 2014) computes a single-step gradient:

    x_adv = x + ε · sign(∇_x L(f(x), y_target))

where:
    x        = clean input waveform (as differentiable tensor)
    y_target = target class ("English" = 0)
    f(x)     = LID model predictions
    L        = cross-entropy loss

We iterate with a binary search over ε to find the minimum value that:
    1. Flips the LID prediction from Hindi → English
    2. Maintains SNR ≥ 40 dB  (i.e.  ε ≤ ε_max_snr)

SNR constraint:
───────────────
    SNR_dB = 10 · log10( E[x²] / E[δ²] )  ≥  40 dB
    →  ε ≤ σ_x · 10^(−20/10) = σ_x / 100

where σ_x = RMS of the clean signal.

Design choice (non-obvious)
────────────────────────────
Rather than computing FGSM directly on the raw waveform, we apply it to
the **log-Mel feature space** and then project the perturbation back to the
time domain via the pseudo-inverse of the mel filterbank.  This ensures the
adversarial noise is spectrally shaped to lie primarily in frequency bands
where the LID is sensitive, rather than spreading equally across all
frequencies.  The resulting perturbation is harder to detect by simple
spectral analysis, while requiring a smaller ε to flip the prediction.

Usage
─────
    python task4_2_adversarial.py \
        --audio      data/denoised.wav \
        --lid-model  models/lid_model.pt \
        --segment    0.0 5.0 \
        --output     data/adversarial_5s.wav \
        --report     data/adversarial_report.json
"""

import os
import json
import argparse
import logging
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import soundfile as sf
from utils import get_device

# Import LID model (Task 1.1)
from task1_1_lid import MultiHeadLID, LogMelFeatureExtractor, SR, HOP_LEN, CONTEXT_WIN

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

TARGET_SNR_DB  = 40.0
TARGET_CLASS   = 0    # "English"
HINDI_CLASS    = 1    # "Hindi"
N_MELS         = 80


# ═══════════════════════════════════════════════════════════════════════════════
# Utility helpers
# ═══════════════════════════════════════════════════════════════════════════════

def snr_db(signal: np.ndarray, noise: np.ndarray) -> float:
    """Compute SNR in dB given signal and noise arrays."""
    sig_power   = np.mean(signal.astype(np.float64) ** 2)
    noise_power = np.mean(noise.astype(np.float64) ** 2)
    if noise_power < 1e-20:
        return float("inf")
    return 10.0 * np.log10(sig_power / noise_power + 1e-12)


def max_epsilon_for_snr(audio: np.ndarray, target_snr_db: float = 40.0) -> float:
    """
    Compute the maximum L-inf ε such that adding uniform noise ±ε
    maintains SNR ≥ target_snr_db.

    For FGSM the noise power ≈ ε² (uniform sign noise).
    We solve:  10 log(σ_x² / ε²) = 40  →  ε = σ_x / 100
    """
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    return rms / 10 ** (target_snr_db / 20.0)


# ═══════════════════════════════════════════════════════════════════════════════
# FGSM in feature space
# ═══════════════════════════════════════════════════════════════════════════════

class FGSMAttack:
    """
    FGSM adversarial attack against the LID model.

    Operates in the **log-Mel feature space** and projects perturbations
    back to the waveform via the pseudo-inverse filterbank.
    """

    def __init__(
        self,
        model:          MultiHeadLID,
        fe:             LogMelFeatureExtractor,
        device:         torch.device,
        target_class:   int   = TARGET_CLASS,
    ):
        self.model        = model.eval()
        self.fe           = fe
        self.device       = device
        self.target_class = target_class

    def _get_mel_grad(
        self,
        wav:   torch.Tensor,   # (T_samples,)
        label: int,
    ) -> torch.Tensor:
        """
        Compute ∇_mel L(model(mel), target_class).

        Returns gradient tensor of same shape as mel features: (T_frames, N_MELS).
        """
        wav = wav.detach().unsqueeze(0)  # (1, T)
        # Need grad w.r.t. mel features
        with torch.no_grad():
            mel = self.fe(wav).squeeze(0)   # (T_frames, N_MELS)

        mel_var = mel.clone().requires_grad_(True)

        # Forward through LID (center-frame only for speed)
        half   = CONTEXT_WIN
        T      = mel_var.shape[0]
        center = T // 2
        if T > 2 * half:
            window = mel_var[center - half : center + half + 1].unsqueeze(0)
        else:
            window = mel_var.unsqueeze(0)

        logits = self.model(window)[:, half if T > 2 * half else 0, :]
        target = torch.tensor([label], device=self.device)
        loss   = F.cross_entropy(logits, target)
        loss.backward()

        return mel_var.grad.detach()  # (T_frames, N_MELS)

    def _mel_grad_to_wav(
        self,
        mel_grad:   torch.Tensor,   # (T_frames, N_MELS)
        wav_len:    int,
    ) -> np.ndarray:
        """
        Project mel-space gradient sign back to the waveform.

        We multiply the sign pattern by the pseudo-inverse (transposed)
        filterbank to distribute the gradient across FFT bins, then ISTFT.
        """
        # Get the mel filterbank matrix from torchaudio
        fb = T.MelScale(
            n_mels       = N_MELS,
            sample_rate  = SR,
            n_stft       = 201,   # n_fft=400 → n_stft=201
            f_min        = 0.0,
            f_max        = None,
        ).fb.to(self.device)   # (n_stft, n_mels)

        # Pseudo-inverse: (n_mels, n_stft) → pinv = (n_stft, n_mels)
        fb_pinv = torch.linalg.pinv(fb.T).to(self.device)  # (n_stft, n_mels)

        # Project sign of gradient into FFT bin space
        sign_mel = torch.sign(mel_grad)   # (T_frames, N_MELS)
        sign_fft = sign_mel @ fb_pinv.T   # (T_frames, n_stft)

        # Interpret sign_fft as a magnitude perturbation in FFT bins
        # and reconstruct via ISTFT (with zero phase)
        T_frames = sign_fft.shape[0]
        n_fft    = 400
        hop      = HOP_LEN
        noise_spec = sign_fft.cpu().numpy().T   # (n_stft, T_frames)
        import scipy.signal as sp_signal
        _, noise_wav = sp_signal.istft(
            noise_spec + 0j, fs=SR, window="hann",
            nperseg=n_fft, noverlap=n_fft - hop,
        )
        # Trim/pad to wav_len
        if len(noise_wav) >= wav_len:
            noise_wav = noise_wav[:wav_len]
        else:
            noise_wav = np.pad(noise_wav, (0, wav_len - len(noise_wav)))
        return noise_wav.astype(np.float32)

    def attack(
        self,
        wav:          np.ndarray,   # mono float32 clip
        true_label:   int = HINDI_CLASS,
        epsilon:      Optional[float] = None,
    ) -> Tuple[np.ndarray, float, bool]:
        """
        Apply FGSM perturbation.

        Parameters
        ──────────
        wav        : clean audio segment (mono float32)
        true_label : expected LID output for wav (default: Hindi)
        epsilon    : L-inf bound.  If None, starts from max_snr epsilon.

        Returns
        ───────
        adv_wav    : adversarial audio
        actual_eps : epsilon actually applied
        success    : True if prediction flipped to target_class
        """
        wav_t  = torch.from_numpy(wav).to(self.device)
        label  = self.target_class

        if epsilon is None:
            epsilon = max_epsilon_for_snr(wav, TARGET_SNR_DB)

        mel_grad = self._get_mel_grad(wav_t, label)
        noise    = self._mel_grad_to_wav(mel_grad, len(wav))

        # Scale noise to ε (L-inf norm)
        noise_max = np.abs(noise).max()
        if noise_max > 1e-9:
            noise = (noise / noise_max) * epsilon

        adv_wav = np.clip(wav + noise, -1.0, 1.0)
        actual_snr = snr_db(wav, noise)

        # Check if attack succeeded
        with torch.no_grad():
            mel  = self.fe(torch.from_numpy(adv_wav).unsqueeze(0).to(self.device))
            T    = mel.shape[1]
            half = CONTEXT_WIN
            c    = T // 2
            if T > 2 * half:
                win = mel[:, c - half : c + half + 1, :]
            else:
                win = mel
            logits = self.model(win)[:, half if T > 2 * half else 0, :]
            pred   = logits.argmax(dim=-1).item()

        success = (pred == self.target_class)
        log.info(
            f"ε={epsilon:.6f}  SNR={actual_snr:.1f} dB  "
            f"pred={pred}  success={success}"
        )
        return adv_wav, epsilon, success


# ═══════════════════════════════════════════════════════════════════════════════
# Binary search for minimum ε
# ═══════════════════════════════════════════════════════════════════════════════

def find_min_epsilon(
    wav:        np.ndarray,
    attack:     FGSMAttack,
    n_steps:    int   = 20,
    eps_lo:     float = 1e-5,
    eps_hi:     float = None,
) -> dict:
    """
    Binary search for the minimum ε that flips the prediction
    while maintaining SNR ≥ 40 dB.

    Returns a dict with 'min_epsilon', 'snr_db', 'success'.
    """
    if eps_hi is None:
        eps_hi = max_epsilon_for_snr(wav, TARGET_SNR_DB)
    log.info(f"Binary search: eps=[{eps_lo:.6f}, {eps_hi:.6f}]  steps={n_steps}")

    # First check if attack is possible at eps_hi
    adv, _, success_hi = attack.attack(wav, epsilon=eps_hi)
    if not success_hi:
        log.warning("Attack failed even at max allowable ε (SNR constraint).")
        return {
            "min_epsilon": eps_hi,
            "snr_db":      snr_db(wav, adv - wav),
            "success":     False,
        }

    # Binary search
    lo, hi = eps_lo, eps_hi
    best_eps, best_adv = eps_hi, adv

    for step in range(n_steps):
        mid        = (lo + hi) / 2
        adv_mid, _, succ = attack.attack(wav, epsilon=mid)
        log.info(f"  step {step+1}/{n_steps}  mid={mid:.6f}  success={succ}")
        if succ:
            best_eps = mid
            best_adv = adv_mid
            hi       = mid
        else:
            lo = mid

    noise      = best_adv - wav
    final_snr  = snr_db(wav, noise)
    return {
        "min_epsilon": float(best_eps),
        "snr_db":      float(final_snr),
        "success":     True,
        "adv_audio":   best_adv,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def adversarial_attack(
    audio_path:    str,
    model_path:    str,
    output_path:   str,
    report_path:   str,
    segment_start: float = 0.0,
    segment_end:   float = 5.0,
    device:        str   = "cpu",
) -> dict:
    """
    Run FGSM attack on a Hindi segment and save results.
    """
    device_obj = get_device(device)

    # Load model
    fe    = LogMelFeatureExtractor().to(device_obj)
    model = MultiHeadLID().to(device_obj)
    ckpt  = torch.load(model_path, map_location=device_obj)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Load audio segment
    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        audio = torchaudio.functional.resample(
            torch.from_numpy(audio), sr, SR
        ).numpy()

    start_s = int(segment_start * SR)
    end_s   = int(segment_end   * SR)
    segment = audio[start_s:end_s]
    log.info(f"Segment: {segment_start:.1f}s – {segment_end:.1f}s  "
             f"({len(segment)/SR:.1f}s)")

    # Run clean prediction
    with torch.no_grad():
        mel = fe(torch.from_numpy(segment).unsqueeze(0).to(device_obj))
        T   = mel.shape[1]
        half = CONTEXT_WIN
        c    = T // 2
        if T > 2 * half:
            win = mel[:, c - half : c + half + 1, :]
        else:
            win = mel
        logits = model(win)[:, half if T > 2 * half else 0, :]
        clean_pred = logits.argmax(dim=-1).item()
        clean_conf = F.softmax(logits, dim=-1).max().item()
    log.info(f"Clean prediction: {'Hindi' if clean_pred == 1 else 'English'} "
             f"(conf={clean_conf:.3f})")

    # FGSM attack
    fgsm   = FGSMAttack(model, fe, device_obj)
    result = find_min_epsilon(segment, fgsm)

    # Save adversarial audio
    adv_audio = result.pop("adv_audio", segment)
    full_adv  = audio.copy()
    full_adv[start_s:end_s] = adv_audio

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    sf.write(output_path, full_adv, SR, subtype="PCM_16")
    log.info(f"Adversarial audio saved → {output_path}")

    # Build report
    report = {
        "segment":          f"{segment_start:.1f}–{segment_end:.1f}s",
        "clean_prediction": "Hindi" if clean_pred == 1 else "English",
        "clean_confidence": round(clean_conf, 4),
        "min_epsilon":      round(result["min_epsilon"], 8),
        "snr_db":           round(result["snr_db"],      2),
        "attack_success":   result["success"],
        "target_snr_constraint_db": TARGET_SNR_DB,
        "target_class":     "English",
    }

    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Report saved → {report_path}")

    log.info(
        f"\n── Adversarial Attack Summary ──────────────────────────\n"
        f"  Min ε          : {report['min_epsilon']:.2e}\n"
        f"  SNR            : {report['snr_db']:.1f} dB\n"
        f"  Attack success : {report['attack_success']}\n"
        f"────────────────────────────────────────────────────────"
    )
    return report


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio",     required=True, help="Denoised input WAV")
    parser.add_argument("--lid-model", default="models/lid_model.pt")
    parser.add_argument("--segment",   nargs=2, type=float, default=[0.0, 5.0],
                        metavar=("START", "END"),
                        help="Start/end time of 5-second attack segment")
    parser.add_argument("--output",    default="data/adversarial_5s.wav")
    parser.add_argument("--report",    default="data/adversarial_report.json")
    parser.add_argument("--device",    default="cpu")
    args = parser.parse_args()

    adversarial_attack(
        audio_path    = args.audio,
        model_path    = args.lid_model,
        output_path   = args.output,
        report_path   = args.report,
        segment_start = args.segment[0],
        segment_end   = args.segment[1],
        device        = args.device,
    )
