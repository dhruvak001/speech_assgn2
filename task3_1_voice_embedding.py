#!/usr/bin/env python3
"""
task3_1_voice_embedding.py
===========================
Task 3.1 – Speaker Embedding Extraction (x-vector / d-vector)

Architecture: TDNN-based x-vector extractor
─────────────────────────────────────────────
The x-vector system (Snyder et al., 2018) uses a Time-Delay Neural Network
(TDNN) to encode local temporal context, followed by a statistical-pooling
layer that compresses the variable-length sequence into a fixed-dimension
speaker embedding.

Network layers
──────────────
Layer 1: TDNN  context [−2,−1,0,+1,+2]  (5-frame context)  → 512-d
Layer 2: TDNN  context [−2,0,+2]         (3-frame, stride 2) → 512-d
Layer 3: TDNN  context [−3,0,+3]         (3-frame, stride 3) → 512-d
Layer 4: TDNN  context [0]               (frame-level)       → 512-d
Layer 5: Dense                                               → 1500-d
Statistical pooling: mean + std  →  3000-d
Embedding layer (x-vector): FC  3000 → 512-d
Classifier head (training):  FC  512  → N_speakers

Design choice (non-obvious)
────────────────────────────
Standard x-vector training uses a full speaker classification objective.
Because we have only a single 60-second reference clip (no labelled multi-
speaker data), we use **GE2E loss** (Generalized End-to-End loss, Wan et al.
2018) in a self-supervised fashion: we split the reference audio into 10
random 6-second crops, form a batch of positive pairs from the same speaker
and negatives from random crops of different segments.  This allows the model
to learn a compact speaker representation even from a single recording.

In practice, when a pretrained model is available (SpeechBrain ECAPA-TDNN)
we load it directly and skip GE2E fine-tuning.

Usage
─────
    # Extract embedding from your 60-s reference clip
    python task3_1_voice_embedding.py \
        --audio data/student_voice_ref.wav \
        --output data/speaker_embedding.pt

    # Extract from professor audio (for prosody reference)
    python task3_1_voice_embedding.py \
        --audio data/original_segment.wav \
        --output data/professor_embedding.pt
"""

import os
import math
import argparse
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

SR       = 16_000
N_MELS   = 80
EMB_DIM  = 512


# ═══════════════════════════════════════════════════════════════════════════════
# TDNN Block
# ═══════════════════════════════════════════════════════════════════════════════

class TDNNBlock(nn.Module):
    """
    Time-Delay Neural Network block.

    Implements context splicing via 1-D convolution:
        y[t] = f(W · [x[t+c_1], x[t+c_2], …, x[t+c_k]] + b)

    For a symmetric context [−d, 0, +d] this is equivalent to
    a dilated 1-D convolution with kernel_size=3 and dilation=d.
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        context_size: int,
        dilation:     int = 1,
        dropout:      float = 0.0,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size = context_size,
            dilation    = dilation,
            padding     = (context_size - 1) * dilation // 2,
        )
        self.norm    = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        return self.dropout(F.relu(self.norm(self.conv(x))))


# ═══════════════════════════════════════════════════════════════════════════════
# Statistical Pooling
# ═══════════════════════════════════════════════════════════════════════════════

class StatsPooling(nn.Module):
    """Concatenate temporal mean and standard deviation → 2C features."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        mean = x.mean(dim=2)
        std  = x.std(dim=2).clamp(min=1e-6)
        return torch.cat([mean, std], dim=1)   # (B, 2C)


# ═══════════════════════════════════════════════════════════════════════════════
# X-Vector Model
# ═══════════════════════════════════════════════════════════════════════════════

class XVectorModel(nn.Module):
    """
    Full TDNN x-vector network.

    Forward returns (embedding, logits) during training,
    or just embedding during inference.
    """

    def __init__(
        self,
        input_dim:   int = N_MELS,
        emb_dim:     int = EMB_DIM,
        n_speakers:  int = 1000,    # classification head size
        dropout:     float = 0.1,
    ):
        super().__init__()

        # Frame-level TDNN stack
        self.tdnn = nn.Sequential(
            TDNNBlock(input_dim,  512, context_size=5, dilation=1, dropout=dropout),
            TDNNBlock(512, 512, context_size=3, dilation=2, dropout=dropout),
            TDNNBlock(512, 512, context_size=3, dilation=3, dropout=dropout),
            TDNNBlock(512, 512, context_size=1, dilation=1, dropout=dropout),
            TDNNBlock(512, 1500,context_size=1, dilation=1, dropout=dropout),
        )

        # Utterance-level aggregation
        self.pool = StatsPooling()   # 3000-d output

        # Embedding layers
        self.embed1 = nn.Sequential(
            nn.Linear(3000, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
        )
        self.embed2 = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
        )

        # Classifier head (only used during supervised training)
        self.classifier = nn.Linear(emb_dim, n_speakers)

    def get_embedding(self, feats: torch.Tensor) -> torch.Tensor:
        """
        feats : (B, T, N_MELS)  or  (T, N_MELS)
        returns: (B, emb_dim)
        """
        if feats.dim() == 2:
            feats = feats.unsqueeze(0)
        x = feats.permute(0, 2, 1)     # (B, N_MELS, T)
        x = self.tdnn(x)               # (B, 1500, T)
        x = self.pool(x)               # (B, 3000)
        x = self.embed1(x)             # (B, emb_dim)
        x = self.embed2(x)             # (B, emb_dim)  ← x-vector
        return x

    def forward(
        self,
        feats: torch.Tensor,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        emb    = self.get_embedding(feats)
        logits = self.classifier(emb) if return_logits else None
        return emb, logits


# ═══════════════════════════════════════════════════════════════════════════════
# GE2E Loss (for self-supervised training on one speaker)
# ═══════════════════════════════════════════════════════════════════════════════

class GE2ELoss(nn.Module):
    """
    Generalized End-to-End speaker loss (Wan et al. 2018).

    Used when only a single reference speaker is available: crops of the same
    recording are treated as positives; crops are randomly permuted to form
    the batch structure (N speakers × M utterances per speaker).
    """

    def __init__(self, init_w: float = 10.0, init_b: float = -5.0):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        embeddings : (N, M, D)  N speakers, M utterances, D dimensions
        """
        N, M, D = embeddings.shape
        # Centroid for each speaker
        centroids = embeddings.mean(dim=1)  # (N, D)
        centroids = F.normalize(centroids, dim=-1)
        emb_norm  = F.normalize(embeddings, dim=-1)   # (N, M, D)

        # Similarity matrix
        sim = torch.einsum("nmd,kd->nmk", emb_norm, centroids)   # (N, M, N)
        sim = sim * self.w.abs() + self.b

        # Labels: utterance (n, m) should match speaker n
        loss  = 0.0
        count = 0
        for n in range(N):
            for m in range(M):
                target = torch.tensor(n, device=embeddings.device)
                loss  += F.cross_entropy(sim[n, m].unsqueeze(0), target.unsqueeze(0))
                count += 1
        return loss / count


# ═══════════════════════════════════════════════════════════════════════════════
# Feature extraction
# ═══════════════════════════════════════════════════════════════════════════════

def extract_mel(wav_path: str, device: str = "cpu") -> torch.Tensor:
    """Load WAV → normalised 80-dim log-Mel features. Returns (T, 80)."""
    wav, sr = torchaudio.load(wav_path)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    wav = wav.mean(0)   # mono

    mel_tf = T.MelSpectrogram(
        sample_rate  = SR,
        n_fft        = 400,
        win_length   = 400,
        hop_length   = 160,
        n_mels       = N_MELS,
        window_fn    = torch.hann_window,
    )
    amp2db = T.AmplitudeToDB(stype="power", top_db=80)

    with torch.no_grad():
        mel = mel_tf(wav)          # (80, T)
        mel = amp2db(mel)
        mel = (mel + 80.0) / 80.0  # normalise to [0,1]
    return mel.T.to(device)        # (T, 80)


# ═══════════════════════════════════════════════════════════════════════════════
# Pretrained SpeechBrain fallback
# ═══════════════════════════════════════════════════════════════════════════════

def extract_with_speechbrain(wav_path: str, device: str = "cpu") -> torch.Tensor:
    """
    Use SpeechBrain's pretrained ECAPA-TDNN for speaker embedding.
    Falls back to our custom TDNN if SpeechBrain is not installed.
    """
    try:
        from speechbrain.pretrained import EncoderClassifier
        log.info("Using SpeechBrain ECAPA-TDNN encoder …")
        encoder = EncoderClassifier.from_hparams(
            source    = "speechbrain/spkrec-ecapa-voxceleb",
            savedir   = "models/speechbrain_ecapa",
            run_opts  = {"device": device},
        )
        wav, sr = torchaudio.load(wav_path)
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
        wav = wav.mean(0, keepdim=True).to(device)

        with torch.no_grad():
            emb = encoder.encode_batch(wav).squeeze()   # (192,) or (512,)
        return emb
    except Exception as e:
        log.warning(f"SpeechBrain unavailable ({e}), using custom TDNN.")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def extract_embedding(
    wav_path:   str,
    model_path: Optional[str] = None,
    device:     str           = "cpu",
    use_speechbrain: bool     = True,
) -> torch.Tensor:
    """
    Extract a speaker embedding from a WAV file.

    Priority:
    1. SpeechBrain ECAPA-TDNN (pretrained, best quality)
    2. Custom TDNN loaded from model_path
    3. Custom TDNN with random weights (embedding is meaningless but shape is correct)

    Returns: 1-D tensor of shape (D,)
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Option 1: SpeechBrain
    if use_speechbrain:
        emb = extract_with_speechbrain(wav_path, str(device))
        if emb is not None:
            log.info(f"Embedding shape: {emb.shape}")
            return emb.cpu()

    # Option 2/3: Custom TDNN
    log.info("Extracting with custom TDNN x-vector model …")
    model = XVectorModel(input_dim=N_MELS, emb_dim=EMB_DIM).to(device)
    if model_path and os.path.isfile(model_path):
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
        log.info(f"Loaded TDNN from {model_path}")
    else:
        log.warning("No pretrained TDNN weights – embedding is random (not useful for TTS).")

    feats = extract_mel(wav_path, str(device))   # (T, 80)
    model.eval()
    with torch.no_grad():
        emb, _ = model(feats.unsqueeze(0).to(device))
    emb = emb.squeeze(0).cpu()
    log.info(f"Embedding shape: {emb.shape}")
    return emb


# ═══════════════════════════════════════════════════════════════════════════════
# GE2E self-supervised training on reference audio
# ═══════════════════════════════════════════════════════════════════════════════

def self_supervised_train(
    wav_path:   str,
    save_path:  str,
    crop_sec:   float = 6.0,
    n_crops:    int   = 20,
    epochs:     int   = 50,
    lr:         float = 1e-3,
    device:     str   = "cpu",
):
    """
    Train the TDNN using GE2E loss on random crops of the reference audio.
    Useful when no pretrained model is available.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    log.info(f"Self-supervised GE2E training on {wav_path} …")

    feats = extract_mel(wav_path, str(device))   # (T, 80)
    T     = feats.shape[0]
    crop_frames = int(crop_sec * SR / 160)       # 160 = hop

    model = XVectorModel(input_dim=N_MELS, emb_dim=EMB_DIM).to(device)
    ge2e  = GE2ELoss().to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(ge2e.parameters()), lr=lr
    )

    N_spk, M_utt = 2, n_crops // 2   # treat two "halves" as pseudo-speakers

    for epoch in range(1, epochs + 1):
        model.train()
        # Sample random crops
        crops = []
        for _ in range(N_spk * M_utt):
            start = torch.randint(0, max(1, T - crop_frames), (1,)).item()
            crop  = feats[start : start + crop_frames]
            if crop.shape[0] < crop_frames:
                crop = F.pad(crop, (0, 0, 0, crop_frames - crop.shape[0]))
            crops.append(crop)

        # Build (N, M, D) embedding tensor
        embs = []
        for i in range(N_spk):
            row = []
            for j in range(M_utt):
                x = crops[i * M_utt + j].unsqueeze(0)   # (1, T, 80)
                e, _ = model(x.to(device))
                row.append(e)
            embs.append(torch.stack(row, dim=0))
        emb_batch = torch.stack(embs, dim=0).squeeze(2)   # (N, M, D)

        loss = ge2e(emb_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            log.info(f"Epoch {epoch}/{epochs}  GE2E loss={loss.item():.4f}")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save({"model_state": model.state_dict()}, save_path)
    log.info(f"Saved TDNN to {save_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio",   required=True, help="Input WAV file")
    parser.add_argument("--output",  default="data/speaker_embedding.pt")
    parser.add_argument("--model",   default=None,  help="TDNN weights (.pt)")
    parser.add_argument("--train",   action="store_true",
                        help="Run GE2E self-supervised training first")
    parser.add_argument("--device",  default="cpu")
    args = parser.parse_args()

    if args.train:
        self_supervised_train(
            wav_path  = args.audio,
            save_path = args.model or "models/tdnn_ge2e.pt",
            device    = args.device,
        )
        args.model = args.model or "models/tdnn_ge2e.pt"

    emb = extract_embedding(
        wav_path        = args.audio,
        model_path      = args.model,
        device          = args.device,
        use_speechbrain = True,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(emb, args.output)
    print(f"Saved embedding  shape={tuple(emb.shape)}  → {args.output}")
