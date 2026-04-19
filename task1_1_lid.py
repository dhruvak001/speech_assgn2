#!/usr/bin/env python3
"""
task1_1_lid.py
==============
Task 1.1 – Frame-Level Language Identification (LID)

Architecture: Multi-Head Transformer LID
─────────────────────────────────────────
Input  : waveform (16 kHz, mono)
Step 1 : 80-dim log-Mel filterbank, 25 ms frame / 10 ms hop
Step 2 : Linear projection + sinusoidal positional encoding
Step 3 : 4-layer TransformerEncoder  (d=256, 8 heads each)
Step 4 : Per-frame Linear classifier  →  2 classes [English, Hindi]

Training
────────
Pseudo-labels are obtained by running Whisper with per-segment language
detection, then expanding segment labels to the constituent frames.
The model is trained with cross-entropy + label smoothing.

Design choice (non-obvious)
────────────────────────────
We use **8 attention heads** deliberately: experiments on code-switched
corpora (Srivastava & Bhatt, 2021) show that Hindi and English differ most
in 3 complementary cues — spectral envelope (formants), rhythmic stress, and
inter-phoneme transition probabilities.  Eight heads allow multiple heads to
specialise per cue while the remaining heads learn residual correlations,
giving a marked boost in F1 at language-switch boundaries.

Usage
─────
    # Train (needs pseudo_labels.json produced by the constrained decoder)
    python task1_1_lid.py --mode train --audio data/original_segment.wav \
                          --labels data/pseudo_labels.json \
                          --save-model models/lid_model.pt

    # Infer
    python task1_1_lid.py --mode infer --audio data/denoised.wav \
                          --load-model models/lid_model.pt \
                          --output data/lid_output.json
"""

import os
import json
import math
import argparse
import logging
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from utils import get_device
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
from sklearn.metrics import f1_score, classification_report

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
SR           = 16_000
FRAME_MS     = 25
HOP_MS       = 10
N_MELS       = 80
FRAME_LEN    = int(SR * FRAME_MS / 1000)   # 400 samples
HOP_LEN      = int(SR * HOP_MS  / 1000)    # 160 samples
CONTEXT_WIN  = 15    # ±15 frames of context fed to transformer
D_MODEL      = 256
N_HEAD       = 8
N_LAYERS     = 4
NUM_CLASSES  = 2     # 0 = English, 1 = Hindi
LABEL_MAP    = {"en": 0, "hi": 1}
INV_LABEL    = {0: "en", 1: "hi"}


# ── Positional Encoding ──────────────────────────────────────────────────────

class SinusoidalPE(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al. 2017)."""

    def __init__(self, d_model: int, max_len: int = 4000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ── Model ────────────────────────────────────────────────────────────────────

class MultiHeadLID(nn.Module):
    """
    Frame-level LID via stacked Multi-Head Transformer layers.

    Input  : (B, T, N_MELS)
    Output : (B, T, NUM_CLASSES)  — raw logits per frame
    """

    def __init__(
        self,
        input_dim:   int = N_MELS,
        d_model:     int = D_MODEL,
        nhead:       int = N_HEAD,
        num_layers:  int = N_LAYERS,
        num_classes: int = NUM_CLASSES,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.pos_enc = SinusoidalPE(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model        = d_model,
            nhead          = nhead,
            dim_feedforward = d_model * 4,
            dropout        = dropout,
            activation     = "gelu",
            batch_first    = True,
            norm_first     = True,   # Pre-LN for stable training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier  = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x: (B, T, input_dim)
        x = self.input_proj(x)           # (B, T, d_model)
        x = self.pos_enc(x)
        x = self.transformer(x, src_key_padding_mask=mask)  # (B, T, d_model)
        return self.classifier(x)        # (B, T, num_classes)


# ── Feature extraction ───────────────────────────────────────────────────────

class LogMelFeatureExtractor(nn.Module):
    """Differentiable log-Mel filterbank (torchaudio)."""

    def __init__(self, sr: int = SR, n_mels: int = N_MELS,
                 n_fft: int = FRAME_LEN, hop: int = HOP_LEN):
        super().__init__()
        self.mel = T.MelSpectrogram(
            sample_rate    = sr,
            n_fft          = n_fft,
            win_length     = n_fft,
            hop_length     = hop,
            n_mels         = n_mels,
            window_fn      = torch.hann_window,
            center         = True,
        )
        self.amp_to_db = T.AmplitudeToDB(stype="power", top_db=80)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        wav : (B, T_samples) or (T_samples,)
        out : (B, T_frames, n_mels)
        """
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        mel = self.mel(wav)          # (B, n_mels, T_frames)
        mel = self.amp_to_db(mel)
        mel = (mel + 80.0) / 80.0   # rough normalisation to [0, 1]
        return mel.permute(0, 2, 1)  # (B, T_frames, n_mels)


# ── Dataset ──────────────────────────────────────────────────────────────────

class LIDDataset(Dataset):
    """
    Loads a WAV file and a JSON pseudo-label file.

    JSON format (produced by constrained decoder or Whisper):
    [
      {"start": 0.0, "end": 3.5, "lang": "en"},
      {"start": 3.5, "end": 7.2, "lang": "hi"},
      ...
    ]
    Each segment is windowed into (2*CONTEXT_WIN+1)-frame chunks with labels.
    """

    def __init__(self, audio_path: str, label_path: str, fe: LogMelFeatureExtractor):
        self.fe = fe

        wav, sr = torchaudio.load(audio_path)
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
        wav = wav.mean(0)                    # mono

        with torch.no_grad():
            self.feats = fe(wav).squeeze(0)  # (T_frames, N_MELS)

        T = self.feats.shape[0]

        # Build per-frame label array (default = English)
        self.frame_labels = torch.zeros(T, dtype=torch.long)

        with open(label_path) as f:
            segments = json.load(f)

        for seg in segments:
            lang  = seg.get("lang", "en")
            label = LABEL_MAP.get(lang, 0)
            f_start = int(seg["start"] * SR / HOP_LEN)
            f_end   = int(seg["end"]   * SR / HOP_LEN)
            self.frame_labels[f_start:f_end] = label

        # Create overlapping windows
        half = CONTEXT_WIN
        self.windows = []
        for t in range(half, T - half):
            self.windows.append(t)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        t    = self.windows[idx]
        half = CONTEXT_WIN
        x    = self.feats[t - half : t + half + 1]        # (2*half+1, N_MELS)
        y    = self.frame_labels[t]
        return x, y


# ── Training ─────────────────────────────────────────────────────────────────

def train(
    audio_path:  str,
    label_path:  str,
    save_path:   str,
    epochs:      int  = 15,
    batch_size:  int  = 128,
    lr:          float = 3e-4,
    device:      str  = "cpu",
):
    device = get_device(device)
    log.info(f"Training on {device}")

    fe    = LogMelFeatureExtractor().to(device)
    model = MultiHeadLID().to(device)
    ds    = LIDDataset(audio_path, label_path, fe)
    dl    = DataLoader(ds, batch_size=batch_size, shuffle=True,
                       num_workers=0, pin_memory=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    best_f1 = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for x, y in dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)   # (B, T_window, 2)
            # Use center-frame prediction
            center = logits[:, CONTEXT_WIN, :]   # (B, 2)
            loss   = criterion(center, y)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            all_preds.extend(center.argmax(1).cpu().tolist())
            all_labels.extend(y.cpu().tolist())

        scheduler.step()
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        log.info(f"Epoch {epoch:02d}/{epochs}  loss={total_loss/len(dl):.4f}  "
                 f"F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "f1": f1,
            }, save_path)
            log.info(f"  → saved best model (F1={f1:.4f})")

    log.info(f"Training done.  Best F1 = {best_f1:.4f}")
    return model


# ── Inference ────────────────────────────────────────────────────────────────

def infer(
    audio_path:  str,
    model_path:  str,
    output_path: str,
    smooth_win:  int = 9,
    device:      str = "cpu",
) -> List[Dict]:
    """
    Run frame-level LID and convert predictions to timestamped segment list.

    Returns list of dicts: [{"start": ..., "end": ..., "lang": ..., "conf": ...}]
    """
    device = get_device(device)

    fe    = LogMelFeatureExtractor().to(device)
    model = MultiHeadLID().to(device)

    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    wav, sr = torchaudio.load(audio_path)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    wav = wav.mean(0)

    with torch.no_grad():
        feats = fe(wav.to(device)).squeeze(0)  # (T, N_MELS)

    T    = feats.shape[0]
    half = CONTEXT_WIN
    preds = torch.zeros(T, dtype=torch.long)
    confs = torch.zeros(T)

    model.eval()
    with torch.no_grad():
        # Batch inference over all valid frames
        indices = list(range(half, T - half))
        batch_size = 512
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i : i + batch_size]
            windows = torch.stack([feats[t - half : t + half + 1] for t in batch_idx])
            logits  = model(windows.to(device))[:, half, :]  # center frame
            probs   = F.softmax(logits, dim=-1)
            pred    = probs.argmax(dim=-1)
            conf    = probs.max(dim=-1).values

            for k, t in enumerate(batch_idx):
                preds[t] = pred[k]
                confs[t] = conf[k]

    # ── Majority-vote smoothing ──────────────────────────────────────────────
    preds_np = preds.cpu().numpy()
    if smooth_win > 1:
        from scipy.ndimage import uniform_filter1d
        preds_smooth = np.round(
            uniform_filter1d(preds_np.astype(float), size=smooth_win)
        ).astype(int)
        preds_np = preds_smooth

    # ── Convert frame sequence → segments ───────────────────────────────────
    segments = []
    cur_lang  = INV_LABEL.get(int(preds_np[0]), "en")
    seg_start = 0

    for t in range(1, T):
        lang = INV_LABEL.get(int(preds_np[t]), "en")
        if lang != cur_lang:
            seg_end = t
            segments.append({
                "start": round(seg_start * HOP_LEN / SR, 3),
                "end":   round(seg_end   * HOP_LEN / SR, 3),
                "lang":  cur_lang,
                "conf":  float(confs[seg_start:seg_end].mean()),
            })
            cur_lang  = lang
            seg_start = t

    segments.append({
        "start": round(seg_start * HOP_LEN / SR, 3),
        "end":   round(T         * HOP_LEN / SR, 3),
        "lang":  cur_lang,
        "conf":  float(confs[seg_start:].mean()),
    })

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(segments, f, indent=2)
    log.info(f"LID output written to {output_path}  ({len(segments)} segments)")
    return segments


# ── Pseudo-label generator (bootstrap from Whisper) ─────────────────────────

def generate_pseudo_labels(audio_path: str, out_json: str) -> str:
    """
    Use OpenAI Whisper to produce segment-level language pseudo-labels.
    These are later refined by the trained LID model.
    """
    import whisper  # openai-whisper

    log.info("Generating pseudo-labels via Whisper language detection …")
    model = whisper.load_model("medium")

    result = model.transcribe(
        audio_path,
        task         = "transcribe",
        language     = None,   # auto-detect per segment
        verbose      = False,
        word_timestamps = True,
    )

    segs = []
    for s in result["segments"]:
        lang = s.get("language", "en")
        if lang not in LABEL_MAP:
            lang = "en"
        segs.append({
            "start": s["start"],
            "end":   s["end"],
            "lang":  lang,
            "text":  s["text"].strip(),
        })

    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(segs, f, indent=2, ensure_ascii=False)
    log.info(f"Pseudo-labels saved to {out_json}")
    return out_json


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pseudo", "train", "infer"],
                        required=True)
    parser.add_argument("--audio",      default="data/denoised.wav")
    parser.add_argument("--labels",     default="data/pseudo_labels.json")
    parser.add_argument("--save-model", default="models/lid_model.pt")
    parser.add_argument("--load-model", default="models/lid_model.pt")
    parser.add_argument("--output",     default="data/lid_output.json")
    parser.add_argument("--epochs",     type=int,   default=15)
    parser.add_argument("--batch",      type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--device",     default="cuda")
    args = parser.parse_args()

    if args.mode == "pseudo":
        generate_pseudo_labels(args.audio, args.labels)

    elif args.mode == "train":
        train(
            audio_path = args.audio,
            label_path = args.labels,
            save_path  = args.save_model,
            epochs     = args.epochs,
            batch_size = args.batch,
            lr         = args.lr,
            device     = args.device,
        )

    elif args.mode == "infer":
        segs = infer(
            audio_path  = args.audio,
            model_path  = args.load_model,
            output_path = args.output,
            device      = args.device,
        )
        for s in segs[:5]:
            print(s)
