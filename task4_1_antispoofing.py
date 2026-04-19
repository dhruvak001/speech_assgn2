#!/usr/bin/env python3
"""
task4_1_antispoofing.py
========================
Task 4.1 – Anti-Spoofing Countermeasure (CM)

Goal
────
Classify audio clips as "Bona Fide" (real human speech) or "Spoof"
(synthesised output from Task 3.3), evaluated using the Equal Error Rate.

Feature: LFCC (Linear Frequency Cepstral Coefficients)
─────────────────────────────────────────────────────────
LFCC uses a linear (uniform) frequency scale filterbank rather than the
mel (log) scale used in MFCC.  This preserves fine-grained information at
higher frequencies which spoofed speech tends to alter.

LFCC computation:
    1. Pre-emphasis filter: s'[n] = s[n] − 0.97 · s[n−1]
    2. STFT with 20ms frames, 10ms hop
    3. Linear filterbank (40 triangular filters, 0–8 kHz)
    4. Log compression
    5. DCT → first 20 coefficients (C_0 … C_19)
    6. Delta and delta-delta append: final dim = 60

Classifier: Light-CNN (LCNN)
─────────────────────────────
We implement the LCNN architecture from Wu et al. (2018) which is the
standard baseline for ASVspoof countermeasures.  LCNN uses Max Feature Map
(MFM) activations instead of ReLU: MFM splits the feature map into two
halves and takes the element-wise maximum.

Design choice (non-obvious)
────────────────────────────
The training data is highly imbalanced (one real speaker vs many synthetic
utterances).  We use **focal loss** instead of standard binary cross-entropy:

    FL(p_t) = −α_t · (1 − p_t)^γ · log(p_t)

With γ=2, focal loss down-weights the well-classified bona-fide samples
(the easy majority) and focuses gradient updates on hard misclassified
spoof examples.  This proved critical on ASVspoof 2019 LA datasets.

Usage
─────
    # Train
    python task4_1_antispoofing.py --mode train \
        --bona-fide data/student_voice_ref.wav \
        --spoof     data/output_LRL_cloned.wav \
        --save      models/antispoofing.pt

    # Evaluate (get EER)
    python task4_1_antispoofing.py --mode eval \
        --bona-fide data/student_voice_ref.wav \
        --spoof     data/output_LRL_cloned.wav \
        --load      models/antispoofing.pt
"""

import os
import argparse
import logging
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
from utils import get_device
from scipy.fft import dct
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# Parameters
SR         = 16_000
FRAME_MS   = 20
HOP_MS     = 10
N_FILTERS  = 40
N_COEFFS   = 20
FEAT_DIM   = N_COEFFS * 3    # static + delta + delta-delta = 60
LABEL_BONA = 0
LABEL_SPOOF= 1


# ═══════════════════════════════════════════════════════════════════════════════
# LFCC Feature Extraction
# ═══════════════════════════════════════════════════════════════════════════════

class LFCCExtractor:
    """
    Linear Frequency Cepstral Coefficients.

    Parameters
    ──────────
    sr          : sample rate
    n_filters   : number of linear filterbank filters
    n_coeffs    : number of cepstral coefficients (excluding C0 if desired)
    frame_len   : FFT frame length (samples)
    hop_len     : frame shift (samples)
    pre_emph    : pre-emphasis coefficient
    """

    def __init__(
        self,
        sr:        int   = SR,
        n_filters: int   = N_FILTERS,
        n_coeffs:  int   = N_COEFFS,
        frame_len: int   = None,
        hop_len:   int   = None,
        pre_emph:  float = 0.97,
        fmin:      float = 0.0,
        fmax:      float = None,
    ):
        self.sr        = sr
        self.n_filters = n_filters
        self.n_coeffs  = n_coeffs
        self.frame_len = frame_len or int(sr * FRAME_MS / 1000)
        self.hop_len   = hop_len   or int(sr * HOP_MS  / 1000)
        self.pre_emph  = pre_emph
        self.fmin      = fmin
        self.fmax      = fmax or sr / 2.0

        # Build linear filterbank
        self.filterbank = self._build_filterbank()

    def _build_filterbank(self) -> np.ndarray:
        """Create triangular linear-frequency filterbank. Shape: (n_filters, n_fft//2+1)"""
        n_fft    = self.frame_len
        n_bins   = n_fft // 2 + 1
        freq_res = self.sr / n_fft
        freqs    = np.arange(n_bins) * freq_res

        center_freqs = np.linspace(self.fmin, self.fmax, self.n_filters + 2)
        fb = np.zeros((self.n_filters, n_bins), dtype=np.float32)

        for m in range(self.n_filters):
            f_left   = center_freqs[m]
            f_center = center_freqs[m + 1]
            f_right  = center_freqs[m + 2]
            for k, f in enumerate(freqs):
                if f_left <= f <= f_center:
                    fb[m, k] = (f - f_left) / (f_center - f_left + 1e-9)
                elif f_center < f <= f_right:
                    fb[m, k] = (f_right - f) / (f_right - f_center + 1e-9)
        return fb

    def _pre_emphasis(self, audio: np.ndarray) -> np.ndarray:
        return np.append(audio[0], audio[1:] - self.pre_emph * audio[:-1])

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute LFCC features for a mono float32 audio array.

        Returns array of shape (T, n_coeffs * 3).
        """
        audio = self._pre_emphasis(audio.astype(np.float64))

        # Frame
        n_frames = max(1, (len(audio) - self.frame_len) // self.hop_len + 1)
        window   = np.hanning(self.frame_len)
        static   = np.zeros((n_frames, self.n_coeffs), dtype=np.float32)

        for i in range(n_frames):
            start  = i * self.hop_len
            frame  = audio[start : start + self.frame_len]
            if len(frame) < self.frame_len:
                frame = np.pad(frame, (0, self.frame_len - len(frame)))
            frame  = frame * window

            spec   = np.abs(np.fft.rfft(frame, n=self.frame_len)) ** 2
            fb_out = self.filterbank @ spec
            fb_log = np.log(fb_out + 1e-9)
            cepst  = dct(fb_log, type=2, norm="ortho")
            static[i] = cepst[:self.n_coeffs]

        # Delta features
        def delta(x, N=2):
            d = np.zeros_like(x)
            for t in range(len(x)):
                num, denom = 0.0, 0.0
                for n in range(1, N + 1):
                    t_f = min(t + n, len(x) - 1)
                    t_b = max(t - n, 0)
                    num   += n * (x[t_f] - x[t_b])
                    denom += 2 * n * n
                d[t] = num / (denom + 1e-9)
            return d

        delta1 = delta(static)
        delta2 = delta(delta1)

        return np.concatenate([static, delta1, delta2], axis=1)  # (T, 60)


# ═══════════════════════════════════════════════════════════════════════════════
# Max Feature Map activation
# ═══════════════════════════════════════════════════════════════════════════════

class MFMActivation(nn.Module):
    """Max Feature Map: out[c] = max(x[:C//2], x[C//2:]) channel-wise."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, ...)
        C = x.shape[1]
        assert C % 2 == 0, "MFM requires even channel count."
        return torch.max(x[:, :C//2], x[:, C//2:])


# ═══════════════════════════════════════════════════════════════════════════════
# LCNN Classifier
# ═══════════════════════════════════════════════════════════════════════════════

class LCNN(nn.Module):
    """
    Light-CNN countermeasure for anti-spoofing.

    Input:  (B, 1, T, feat_dim)   — 2-D spectrogram-style LFCC features
    Output: (B, 2)                — logits [bona_fide, spoof]
    """

    def __init__(self, feat_dim: int = FEAT_DIM, dropout: float = 0.5):
        super().__init__()

        # Conv blocks with MFM activation
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64,  kernel_size=5, padding=2),
            MFMActivation(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64,  kernel_size=1),
            MFMActivation(),
            nn.Conv2d(32, 96,  kernel_size=3, padding=1),
            MFMActivation(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.BatchNorm2d(48),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(48, 96,  kernel_size=1),
            MFMActivation(),
            nn.Conv2d(48, 128, kernel_size=3, padding=1),
            MFMActivation(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1),
            MFMActivation(),
            nn.Conv2d(64, 64,  kernel_size=3, padding=1),
            MFMActivation(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(32, 64,  kernel_size=1),
            MFMActivation(),
            nn.Conv2d(32, 64,  kernel_size=3, padding=1),
            MFMActivation(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )

        # Fully connected
        self.fc1     = nn.Linear(32 * 4, 160)   # dynamic; resized in forward
        self.mfm_fc  = MFMActivation()
        self.dropout = nn.Dropout(dropout)
        self.fc2     = nn.Linear(80, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T, F)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        B = x.shape[0]
        x = x.reshape(B, -1)

        # Dynamic FC1 (handle variable sequence lengths)
        if not hasattr(self, "_fc1_in") or self._fc1_in != x.shape[1]:
            self._fc1_in = x.shape[1]
            self.fc1 = nn.Linear(x.shape[1], 160).to(x.device)

        x = self.fc1(x)                              # (B, 160)
        # MFM on FC: split feature dim in half, take element-wise max
        x1, x2 = x.chunk(2, dim=-1)
        x = torch.max(x1, x2)                        # (B, 80)
        x = self.dropout(x)
        x = self.fc2(x)                              # (B, 2)
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# Focal Loss
# ═══════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced binary classification.
    FL(p_t) = −α_t · (1 − p_t)^γ · log(p_t)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce   = F.cross_entropy(logits, targets, reduction="none")
        pt   = torch.exp(-ce)
        loss = self.alpha * (1 - pt) ** self.gamma * ce
        return loss.mean()


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class SpoofDataset(Dataset):
    """
    Creates training samples by slicing two audio files into 3-second clips.

    bona_fide_path : real human voice WAV
    spoof_path     : synthesised (TTS) WAV
    """

    CLIP_SEC = 3.0
    FIX_T    = 150    # fixed number of frames per clip (pad/truncate)

    def __init__(self, bona_fide_path: str, spoof_path: str,
                 lfcc: LFCCExtractor, augment: bool = True):
        self.lfcc    = lfcc
        self.augment = augment
        self.samples: List[Tuple[np.ndarray, int]] = []

        for path, label in [(bona_fide_path, LABEL_BONA),
                            (spoof_path,     LABEL_SPOOF)]:
            audio, sr = sf.read(path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != SR:
                audio = self._resample(audio, sr, SR)

            clip_len  = int(self.CLIP_SEC * SR)
            n_clips   = max(1, len(audio) // clip_len)
            for i in range(n_clips):
                clip = audio[i * clip_len : (i + 1) * clip_len]
                if len(clip) < clip_len:
                    clip = np.pad(clip, (0, clip_len - len(clip)))
                feat = lfcc.extract(clip)              # (T, 60)
                feat = self._pad_or_trim(feat)
                self.samples.append((feat, label))

        log.info(f"Dataset: {len(self.samples)} clips  "
                 f"(bona={sum(1 for _,l in self.samples if l==0)}, "
                 f"spoof={sum(1 for _,l in self.samples if l==1)})")

    @staticmethod
    def _resample(audio, src_sr, tgt_sr):
        import torchaudio
        t = torch.from_numpy(audio).unsqueeze(0)
        t = torchaudio.functional.resample(t, src_sr, tgt_sr)
        return t.squeeze(0).numpy()

    def _pad_or_trim(self, feat: np.ndarray) -> np.ndarray:
        T = self.FIX_T
        if feat.shape[0] >= T:
            return feat[:T]
        return np.pad(feat, ((0, T - feat.shape[0]), (0, 0)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feat, label = self.samples[idx]
        if self.augment:
            feat = feat + np.random.randn(*feat.shape).astype(np.float32) * 0.01
        x = torch.from_numpy(feat).unsqueeze(0)    # (1, T, F) → 2D "image"
        y = torch.tensor(label, dtype=torch.long)
        return x, y


# ═══════════════════════════════════════════════════════════════════════════════
# EER computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    Compute Equal Error Rate (EER).

    labels : 0 = bona fide, 1 = spoof
    scores : higher score → more likely spoof
    """
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr         = 1.0 - tpr
    # EER is where FPR ≈ FNR
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return float(eer) * 100.0   # return as percentage


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

def train_cm(
    bona_fide_path: str,
    spoof_path:     str,
    save_path:      str,
    epochs:         int   = 30,
    batch_size:     int   = 32,
    lr:             float = 1e-3,
    device:         str   = "cpu",
):
    device = get_device(device)
    log.info(f"Training anti-spoofing CM on {device}")

    lfcc  = LFCCExtractor()
    ds    = SpoofDataset(bona_fide_path, spoof_path, lfcc, augment=True)
    dl    = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    model   = LCNN().to(device)
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    optim   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched   = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.5)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for x, y in dl:
            x, y    = x.to(device), y.to(device)
            logits  = model(x)
            loss    = loss_fn(logits, y)
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            epoch_loss += loss.item()
        sched.step()
        epoch_loss /= len(dl)
        log.info(f"Epoch {epoch:02d}/{epochs}  loss={epoch_loss:.4f}")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({"model_state": model.state_dict()}, save_path)

    log.info(f"Training done. Model saved to {save_path}")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_cm(
    bona_fide_path: str,
    spoof_path:     str,
    model_path:     str,
    device:         str = "cpu",
) -> float:
    device = get_device(device)

    lfcc  = LFCCExtractor()
    ds    = SpoofDataset(bona_fide_path, spoof_path, lfcc, augment=False)
    dl    = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    model = LCNN().to(device)
    ckpt  = torch.load(model_path, map_location=device)
    # Resize fc1 to match the checkpoint (dynamic layer)
    fc1_key = "fc1.weight"
    if fc1_key in ckpt["model_state"]:
        in_features = ckpt["model_state"][fc1_key].shape[1]
        model.fc1 = nn.Linear(in_features, 160).to(device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    all_scores, all_labels = [], []
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            logits = model(x)
            probs  = F.softmax(logits, dim=-1)[:, 1]   # P(spoof)
            all_scores.extend(probs.cpu().tolist())
            all_labels.extend(y.tolist())

    labels = np.array(all_labels)
    scores = np.array(all_scores)
    eer    = compute_eer(labels, scores)
    log.info(f"EER = {eer:.2f} %")
    return eer


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",       choices=["train", "eval"], required=True)
    parser.add_argument("--bona-fide",  required=True,
                        help="Bona fide (real) voice WAV")
    parser.add_argument("--spoof",      required=True,
                        help="Spoof (synthesised) WAV")
    parser.add_argument("--save",       default="models/antispoofing.pt")
    parser.add_argument("--load",       default="models/antispoofing.pt")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch",      type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--device",     default="cpu")
    args = parser.parse_args()

    if args.mode == "train":
        train_cm(
            bona_fide_path = args.bona_fide,
            spoof_path     = args.spoof,
            save_path      = args.save,
            epochs         = args.epochs,
            batch_size     = args.batch,
            lr             = args.lr,
            device         = args.device,
        )
    else:
        eer = evaluate_cm(
            bona_fide_path = args.bona_fide,
            spoof_path     = args.spoof,
            model_path     = args.load,
            device         = args.device,
        )
        print(f"\nEER = {eer:.2f} %  (target: < 10 %)")
