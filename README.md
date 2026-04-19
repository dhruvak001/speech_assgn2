# Speech Understanding – Programming Assignment 2

**Author:** Dhruva Kumar Kaushal B22AI017
**Institute:** IIT Jodhpur 
**Date:** April 2026

## End-to-End Multi-Modal Speech Processing Pipeline

---

## Overview

This repository implements a complete, ten-stage speech processing pipeline for PA-2. Starting from a single 10-minute raw English lecture audio at 16 kHz, the system chains signal processing, automatic speech recognition, language identification, phoneme conversion, low-resource machine translation, voice cloning, prosody warping, anti-spoofing, and adversarial robustness evaluation.

| Stage | Module | Description | Key Result |
|-------|--------|-------------|-----------|
| 1 | `task1_3_denoising.py` | Wiener-filter + spectral subtraction denoising | +11 dB avg SNR |
| 2 | `task1_2_constrained_decoding.py` | Whisper ASR with N-gram LM logit bias | Full transcript |
| 3 | `task1_1_lid.py` | Transformer-based frame-level Language ID | 0.944 confidence |
| 4 | `task2_1_ipa.py` | Hinglish G2P / IPA phoneme conversion | Full coverage |
| 5 | `task2_2_translation.py` | English → Maithili (530+ word corpus) | 26.3% lexical coverage |
| 6 | `task3_1_voice_embedding.py` | TDNN x-vector speaker embedding | 512-d embedding |
| 7 | `task3_3_synthesis.py` | YourTTS zero-shot voice cloning | Maithili TTS |
| 8 | `task3_2_prosody_warping.py` | PSOLA + DTW prosody transfer | F0 + energy warped |
| 9 | `task4_1_antispoofing.py` | LFCC + LCNN anti-spoofing countermeasure | EER = 45% (domain mismatch) |
| 10 | `task4_2_adversarial.py` | FGSM adversarial attack on LID | SNR = 100.3 dB at ε=1e-5 |

📄 **Full journal-style report with all theory, equations, and results:** [`PA2_Journal_Report.pdf`](./PA2_Journal_Report.pdf)

**Source video:** https://youtu.be/ZPUtA3W-7_I (segment: 2:20:00 – 2:30:00)

---

## Repository Structure

```
b22ai017_pa2/
├── PA2_Journal_Report.pdf            # ← MAIN SUBMISSION REPORT (7 pages)
├── pipeline.py                       # End-to-end orchestration
├── data_collection.py                # Stage 0: YouTube → 16 kHz WAV
├── task1_1_lid.py                    # Task 1.1: Transformer LID
├── task1_2_constrained_decoding.py   # Task 1.2: Whisper + trigram LM bias (α=0.4)
├── task1_3_denoising.py              # Task 1.3: Spectral subtraction + Wiener filter
├── task2_1_ipa.py                    # Task 2.1: Hinglish G2P / IPA phoneme mapper
├── task2_2_translation.py            # Task 2.2: Maithili translation (530+ entries)
├── task3_1_voice_embedding.py        # Task 3.1: TDNN x-vector speaker embedding
├── task3_2_prosody_warping.py        # Task 3.2: PYIN F0 + DTW + WORLD prosody warp
├── task3_3_synthesis.py              # Task 3.3: YourTTS / VITS zero-shot TTS
├── task4_1_antispoofing.py           # Task 4.1: LFCC (60-d) + LCNN + EER
├── task4_2_adversarial.py            # Task 4.2: FGSM in log-Mel space
├── evaluate.py                       # WER / MCD / EER / LID accuracy metrics
├── utils.py                          # Shared helpers
├── generate_journal_report.py        # Generates all figures → report
├── requirements.txt                  # Python dependencies
├── figures/                          # All pipeline visualisations (embedded in PDF)
└── data/                             # Generated after running pipeline
    ├── original_segment.wav          # Source audio (16 kHz mono)
    ├── original_segment_22k.wav      # Source audio (22.05 kHz for prosody)
    ├── denoised.wav                  # Post-denoising output
    ├── student_voice_ref.wav         # 60-s speaker reference recording
    ├── transcript.json               # Whisper ASR output
    ├── lid_output.json               # Per-frame language labels
    ├── ipa_output.json               # IPA phoneme sequences
    ├── maithili_output.json          # Maithili translated text
    ├── speaker_embedding.pt          # 512-d x-vector
    ├── synth_flat.wav                # Raw TTS (pre-prosody)
    ├── output_LRL_cloned.wav         # Final Maithili voice clone
    ├── adversarial_5s.wav            # FGSM adversarial example
    └── evaluation_results.json       # All computed metrics
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
brew install ffmpeg          # macOS
# sudo apt install ffmpeg    # Ubuntu/Debian
pip install yt-dlp
```

### 2. (Optional) Provide your voice reference

Record 60 seconds of your own voice and save as `data/student_voice_ref.wav` (mono, 16 kHz).

### 3. Run the full pipeline

```bash
python pipeline.py --device cuda
```

Skip YouTube download if `data/original_segment.wav` already exists:
```bash
python pipeline.py --skip-download --device cuda
```

Start from a specific stage (e.g., IPA conversion = stage 6):
```bash
python pipeline.py --start-stage 6 --device cuda
```

CPU-only mode:
```bash
python pipeline.py --device cpu
```

---

## Running Individual Modules

```bash
# Stage 0 — Download lecture segment
python data_collection.py --start 8400 --duration 600 --out-asr data/original_segment.wav

# Stage 1 — Denoise
python task1_3_denoising.py --input data/original_segment.wav --output data/denoised.wav

# Stage 2 — Transcribe with constrained decoding
python task1_2_constrained_decoding.py --audio data/denoised.wav --output data/transcript.json

# Stage 3 — LID pseudo-labels → train → infer
python task1_1_lid.py --mode pseudo --audio data/denoised.wav --labels data/pseudo_labels.json
python task1_1_lid.py --mode train  --audio data/denoised.wav --labels data/pseudo_labels.json \
    --save-model models/lid_model.pt --device cuda
python task1_1_lid.py --mode infer  --audio data/denoised.wav \
    --load-model models/lid_model.pt --output data/lid_output.json

# Stage 4 — IPA conversion
python task2_1_ipa.py --transcript data/transcript.json --output data/ipa_output.json

# Stage 5 — Maithili translation
python task2_2_translation.py --ipa-json data/ipa_output.json --output data/maithili_output.json

# Stage 6 — Speaker embedding
python task3_1_voice_embedding.py --audio data/student_voice_ref.wav \
    --output data/speaker_embedding.pt

# Stage 7 — Synthesise Maithili TTS
python task3_3_synthesis.py --text-json data/maithili_output.json \
    --speaker-wav data/student_voice_ref.wav --output data/synth_flat.wav

# Stage 8 — Prosody warping
python task3_2_prosody_warping.py \
    --source data/original_segment_22k.wav \
    --target data/synth_flat.wav \
    --output data/output_LRL_cloned.wav

# Stage 9 — Anti-spoofing (train + evaluate)
python task4_1_antispoofing.py --mode train \
    --bona-fide data/student_voice_ref.wav \
    --spoof data/output_LRL_cloned.wav \
    --save models/antispoofing.pt

# Stage 10 — Adversarial attack
python task4_2_adversarial.py --audio data/denoised.wav \
    --lid-model models/lid_model.pt --output data/adversarial_5s.wav

# Stage 11 — Full evaluation
python evaluate.py --transcript data/transcript.json \
    --synth data/output_LRL_cloned.wav \
    --reference-voice data/student_voice_ref.wav \
    --lid-output data/lid_output.json
```

---

## Architecture Summaries

### Task 1.1 – Multi-Head Language Identification
- **Input:** 80-dim log-Mel features, 25 ms frames / 10 ms hop
- **Model:** 4-layer TransformerEncoder, d_model=256, 8 attention heads, Pre-LN
- **Training:** Pseudo-labels from Whisper, AdamW + CosineAnnealing, label smoothing 0.1
- **Output:** Per-frame binary label (English=0 / Hindi=1)

### Task 1.2 – Constrained Decoding
- **Base:** OpenAI Whisper large-v3
- **LM:** Smoothed trigram LM (Lidstone k=0.05) trained on course syllabus
- **Bias:** `logits'[w] = logits[w] + α·log P_LM(w | context)`, α=0.4

### Task 1.3 – Denoising
- High-pass filter (80 Hz, 4th-order Butterworth)
- Spectral subtraction: `|Ŝ(k)|² = max(|X(k)|² − 1.5|N̂(k)|², 0.002|X(k)|²)`
- Wiener post-filter: `G(k) = SNR(k) / (1 + SNR(k))`
- EBU R128-style RMS normalisation (−23 dBFS)

### Task 2.1 – IPA Phoneme Mapping
- Word-level language routing (Devanagari / Roman Hindi / English / digits)
- Devanagari: Unicode → IPA via consonant + vowel/matra table
- English: CMU pronouncing dictionary + regex fallback

### Task 2.2 – Maithili Translation
- 530+ entry parallel corpus (general vocabulary + speech/ML terminology)
- N-gram phrase matching (up to 4-gram) with morpheme-stripped fallback
- Unknown technical terms borrowed as-is (per Maithili Language Academy guidelines)

### Task 3.1 – Speaker Embedding
- TDNN x-vector (5 dilated 1-D conv layers + statistical pooling + 512-d embedding)
- Falls back to SpeechBrain ECAPA-TDNN when available

### Task 3.2 – Prosody Warping
- F0 extraction: PYIN algorithm (50–600 Hz)
- DTW alignment on z-normalised log-F0 contours
- Prosody injection via WORLD vocoder (pyworld) or energy-scaling fallback

### Task 3.3 – Voice Synthesis
- **Model:** YourTTS (multilingual VITS, Coqui TTS) with Hindi language code
- **Speaker conditioning:** 60-s reference clip or pre-extracted embedding
- Chunked synthesis (200 char/chunk) with 300 ms inter-sentence pauses

### Task 4.1 – Anti-Spoofing
- **Features:** LFCC (40 linear filters, 20 cepstral coeffs + Δ + ΔΔ = 60-d)
- **Model:** LCNN with Max Feature Map activations
- **Loss:** Focal loss (α=0.25, γ=2)
- **Evaluation:** Equal Error Rate (EER) via ROC interpolation

### Task 4.2 – Adversarial Attack
- **Method:** FGSM in log-Mel feature space, projected back via pseudo-inverse filterbank
- **Constraint:** SNR ≥ 40 dB → ε ≤ σ_x / 100
- **Search:** Binary search (20 steps) for minimum ε that flips Hindi → English

---

## Results Summary

| Metric | Value |
|--------|-------|
| Average SNR improvement | +11 dB |
| LID confidence | 0.944 |
| Maithili translation coverage | 26.3% |
| Anti-spoofing EER | 45% (domain mismatch) |
| Adversarial attack SNR | 100.3 dB at ε=1×10⁻⁵ |

---

## References

1. Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision" (Whisper), 2023  
2. Kim et al., "Conditional VAE with Adversarial Learning for End-to-End TTS" (VITS), 2021  
3. Casanova et al., "YourTTS: Zero-Shot Multi-Speaker TTS", 2022  
4. Snyder et al., "X-vectors: Robust DNN Embeddings for Speaker Recognition", 2018  
5. Wan et al., "Generalized End-to-End Loss for Speaker Verification" (GE2E), 2018  
6. Sakoe & Chiba, "Dynamic Programming Algorithm for Spoken Word Recognition" (DTW), 1978  
7. Boll, "Suppression of Acoustic Noise in Speech Using Spectral Subtraction", 1979  
8. Mauch & Dixon, "pYIN: A Fundamental Frequency Estimator", 2014  
9. Wu et al., "Light CNN for Deep Face Representation" (LCNN), 2018  
10. Goodfellow et al., "Explaining and Harnessing Adversarial Examples" (FGSM), 2014  
11. Lin et al., "Focal Loss for Dense Object Detection", 2017  
12. Morise et al., "WORLD: A Vocoder-Based High-Quality Speech Synthesis System", 2016  
