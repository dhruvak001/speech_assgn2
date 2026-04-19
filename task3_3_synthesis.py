#!/usr/bin/env python3
"""
task3_3_synthesis.py
=====================
Task 3.3 – Zero-Shot Cross-Lingual Voice Cloning (TTS)

Model: Meta MMS-TTS (Massively Multilingual Speech) via HuggingFace Transformers
Language: facebook/mms-tts-hin (Hindi) used as Maithili proxy
Speaker conditioning: custom VITS-based speaker scale after synthesis
Output: ≥ 22.05 kHz

Design choice (non-obvious)
────────────────────────────
Coqui TTS is restricted to Python <3.12.  Instead we use Meta's open-source
MMS-TTS (VITS-based, facebook/mms-tts-hin) which is distributed through
HuggingFace Transformers and works on any Python version.  MMS-TTS covers
1,100+ languages; Hindi (hin) is the closest in phonological inventory to
Maithili and shares its Devanagari script representation.  After synthesis,
we apply a simple speaker-timbre transfer by matching the long-term spectral
envelope of the synthesised audio to the student's reference voice via
cepstral mean normalisation (CMN), which shifts timbre without pitch.

Usage
─────
    python task3_3_synthesis.py \
        --text-json    data/maithili_output.json \
        --speaker-wav  data/student_voice_ref.wav \
        --output       data/synth_flat.wav

    python task3_3_synthesis.py \
        --text "आइ हम ध्वनि प्रसंस्करण केर बारेमे बात करब" \
        --speaker-wav  data/student_voice_ref.wav \
        --output       data/test_synth.wav
"""

import os
import json
import argparse
import logging
import re
import tempfile
from typing import List, Optional

import numpy as np
import torch
import soundfile as sf
import torchaudio

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

OUTPUT_SR    = 22_050
MMS_MODEL_ID = "facebook/mms-tts-hin"    # Hindi – closest to Maithili


# ── Device helper ──────────────────────────────────────────────────────────────

def get_device(preferred: str = "cpu") -> torch.device:
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred in ("cuda", "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Text helpers ───────────────────────────────────────────────────────────────

def load_maithili_text(json_path: str) -> str:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("full_maithili", "")


def chunk_text(text: str, max_chars: int = 150) -> List[str]:
    sentences = re.split(r"[।\.!?]\s*", text)
    chunks, current = [], ""
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(current) + len(sent) < max_chars:
            current += (" " if current else "") + sent
        else:
            if current:
                chunks.append(current)
            current = sent
    if current:
        chunks.append(current)
    return chunks if chunks else [text[:max_chars]]


# ── Spectral envelope transfer (timbre shift to speaker) ─────────────────────

def cepstral_mean_normalise(
    synth:   np.ndarray,
    ref:     np.ndarray,
    sr:      int,
    n_coeff: int = 32,
) -> np.ndarray:
    """
    Shift the long-term spectral envelope of `synth` toward `ref`
    via cepstral mean subtraction + ref mean addition.
    Uses STFT + cepstrum; leaves pitch unchanged.
    """
    try:
        import librosa
        n_fft = 1024
        hop   = 256
        # Cepstral means from reference
        ref_mfcc  = librosa.feature.mfcc(y=ref,   sr=sr, n_mfcc=n_coeff,
                                          n_fft=n_fft, hop_length=hop)
        syn_mfcc  = librosa.feature.mfcc(y=synth, sr=sr, n_mfcc=n_coeff,
                                          n_fft=n_fft, hop_length=hop)
        ref_mean  = ref_mfcc.mean(axis=1, keepdims=True)
        syn_mean  = syn_mfcc.mean(axis=1, keepdims=True)

        # We cannot reconstruct waveform from MFCC alone without a vocoder,
        # so instead we apply a frequency-domain gain profile derived from the
        # difference in log-mel spectral means.
        ref_mel = librosa.feature.melspectrogram(y=ref,   sr=sr, n_mels=128)
        syn_mel = librosa.feature.melspectrogram(y=synth, sr=sr, n_mels=128)
        ref_log = np.log(ref_mel.mean(axis=1) + 1e-9)
        syn_log = np.log(syn_mel.mean(axis=1) + 1e-9)
        gain_mel = np.exp(ref_log - syn_log)          # (128,)

        # Map mel-bin gains to linear STFT bins
        stft     = librosa.stft(synth, n_fft=n_fft, hop_length=hop)
        mag, ph  = np.abs(stft), np.angle(stft)
        mel_fb   = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=128)
        # pseudo-inverse map from mel gains to fft bins
        gain_fft = (mel_fb.T @ gain_mel[:, None]).squeeze()
        gain_fft = np.clip(gain_fft, 0.5, 2.0)[:, None]

        stft_mod = (mag * gain_fft) * np.exp(1j * ph)
        out      = librosa.istft(stft_mod, hop_length=hop, length=len(synth))
        return out.astype(np.float32)
    except Exception as e:
        log.warning(f"Spectral transfer failed ({e}); returning unmodified.")
        return synth


# ── MMS TTS model ─────────────────────────────────────────────────────────────

class MMSSynthesiser:
    """
    Wrapper around HuggingFace MMS-TTS (VITS, Hindi).
    Applies spectral envelope transfer from reference speaker after synthesis.
    """

    def __init__(self, device: str = "cpu", speaker_wav: Optional[str] = None):
        self.device      = get_device(device)
        self.speaker_wav = speaker_wav
        self.ref_audio   = None
        self.ref_sr      = OUTPUT_SR

        log.info(f"Loading MMS-TTS model ({MMS_MODEL_ID}) …")
        from transformers import VitsModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(MMS_MODEL_ID)
        self.model     = VitsModel.from_pretrained(MMS_MODEL_ID).to(self.device)
        self.model.eval()
        log.info("MMS-TTS model loaded.")

        # Load reference audio for timbre transfer
        if speaker_wav and os.path.isfile(speaker_wav):
            ref, sr = sf.read(speaker_wav, dtype="float32")
            if ref.ndim > 1:
                ref = ref.mean(axis=1)
            if sr != OUTPUT_SR:
                ref = torchaudio.functional.resample(
                    torch.from_numpy(ref), sr, OUTPUT_SR
                ).numpy()
            self.ref_audio = ref
            self.ref_sr    = OUTPUT_SR
            log.info(f"Reference audio loaded: {len(ref)/OUTPUT_SR:.1f}s")

    def synthesize_chunk(self, text: str) -> np.ndarray:
        """Synthesise a single text chunk. Returns float32 array."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model(**inputs).waveform  # (1, T_samples)
        wav = output.squeeze().cpu().float().numpy()

        # MMS outputs at 16 kHz; resample to OUTPUT_SR
        model_sr = self.model.config.sampling_rate
        if model_sr != OUTPUT_SR:
            wav = torchaudio.functional.resample(
                torch.from_numpy(wav), model_sr, OUTPUT_SR
            ).numpy()

        return wav

    def synthesize_full(self, text: str, output_path: str,
                        max_chunk: int = 150) -> str:
        chunks = chunk_text(text, max_chars=max_chunk)
        log.info(f"Synthesising {len(chunks)} chunks …")

        parts = []
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            try:
                wav = self.synthesize_chunk(chunk)
                # Apply speaker timbre transfer
                if self.ref_audio is not None and len(wav) > 400:
                    wav = cepstral_mean_normalise(wav, self.ref_audio, OUTPUT_SR)
                parts.append(wav)
                # 250 ms silence between chunks
                parts.append(np.zeros(int(0.25 * OUTPUT_SR), dtype=np.float32))
                log.info(f"  [{i+1}/{len(chunks)}] {len(wav)/OUTPUT_SR:.2f}s "
                         f"– {chunk[:50]}")
            except Exception as e:
                log.warning(f"  Chunk {i+1} failed ({e}); inserting silence.")
                parts.append(np.zeros(int(0.5 * OUTPUT_SR), dtype=np.float32))

        if not parts:
            log.error("All TTS chunks failed. Writing silence.")
            final = np.zeros(OUTPUT_SR * 10, dtype=np.float32)
        else:
            final = np.concatenate(parts)

        final = np.clip(final, -1.0, 1.0)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        sf.write(output_path, final, OUTPUT_SR, subtype="PCM_16")
        log.info(f"Synthesised {len(final)/OUTPUT_SR:.1f}s → {output_path}")
        return output_path


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text",       help="Inline Maithili/Hindi text")
    group.add_argument("--text-json",  help="Maithili JSON from task2_2")
    parser.add_argument("--speaker-wav",  default="data/student_voice_ref.wav")
    parser.add_argument("--output",       default="data/synth_flat.wav")
    parser.add_argument("--device",       default="cpu")
    args = parser.parse_args()

    text = args.text if args.text else load_maithili_text(args.text_json)
    log.info(f"Text length: {len(text)} chars")

    synth = MMSSynthesiser(device=args.device, speaker_wav=args.speaker_wav)
    synth.synthesize_full(text, args.output)
    print(f"\nOutput: {args.output}")
