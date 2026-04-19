#!/usr/bin/env python3
"""
pipeline.py
===========
End-to-end orchestration of the Speech Understanding PA-2 pipeline.

Stages
──────
Stage 0: Data collection   – download lecture segment from YouTube
Stage 1: Preprocessing     – denoise audio
Stage 2: Transcription     – Whisper with N-gram logit biasing
Stage 3: LID bootstrap     – generate pseudo-labels via Whisper
Stage 4: LID training      – train frame-level Multi-Head LID
Stage 5: LID inference     – produce per-segment language tags
Stage 6: IPA conversion    – Hinglish G2P → unified IPA string
Stage 7: Translation       – translate to Maithili (LRL)
Stage 8: Speaker embedding – extract x-vector from reference voice
Stage 9: TTS (flat)        – synthesise Maithili text (YourTTS/VITS)
Stage 10: Prosody warping  – DTW-warp professor prosody onto synthesis
Stage 11: Anti-spoofing CM – train LFCC + LCNN classifier
Stage 12: FGSM attack      – adversarial perturbation on Hindi segment
Stage 13: Evaluation       – WER / MCD / EER / LID switching accuracy

Usage
─────
    # Full pipeline (stages 0–13)
    python pipeline.py --device cuda

    # Skip data collection if audio already downloaded
    python pipeline.py --skip-download --device cuda

    # Start from a specific stage
    python pipeline.py --start-stage 6 --device cuda

    # Student voice reference must be recorded separately and placed at:
    #   data/student_voice_ref.wav
"""

import os
import json
import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)s | %(message)s",
    datefmt = "%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Default paths ─────────────────────────────────────────────────────────────
class Paths:
    DATA       = Path("data")
    MODELS     = Path("models")

    # Stage 0
    ORIG_ASR   = DATA / "original_segment.wav"       # 16 kHz
    ORIG_TTS   = DATA / "original_segment_22k.wav"   # 22.05 kHz

    # Stage 1
    DENOISED   = DATA / "denoised.wav"

    # Stage 2
    TRANSCRIPT = DATA / "transcript.json"

    # Stage 3
    PSEUDO_LBL = DATA / "pseudo_labels.json"

    # Stage 4
    LID_MODEL  = MODELS / "lid_model.pt"

    # Stage 5
    LID_OUT    = DATA / "lid_output.json"

    # Stage 6
    IPA_OUT    = DATA / "ipa_output.json"

    # Stage 7
    MAI_OUT    = DATA / "maithili_output.json"

    # Stage 8
    VOICE_REF  = DATA / "student_voice_ref.wav"   # user must provide
    SPK_EMB    = DATA / "speaker_embedding.pt"

    # Stage 9
    SYNTH_FLAT = DATA / "synth_flat.wav"

    # Stage 10
    SYNTH_PROS = DATA / "output_LRL_cloned.wav"   # final output

    # Stage 11
    CM_MODEL   = MODELS / "antispoofing.pt"

    # Stage 12
    ADV_WAV    = DATA / "adversarial_5s.wav"
    ADV_RPT    = DATA / "adversarial_report.json"

    # Stage 13
    EVAL_OUT   = DATA / "evaluation_results.json"


# ── Utilities ─────────────────────────────────────────────────────────────────

def banner(stage: int, title: str):
    log.info("")
    log.info("=" * 60)
    log.info(f"  STAGE {stage:02d} | {title}")
    log.info("=" * 60)


def check_file(path, label: str, abort: bool = False):
    p = Path(path)
    if p.exists():
        log.info(f"  ✓ {label}: {p}")
        return True
    msg = f"  ✗ {label} not found: {p}"
    if abort:
        log.error(msg)
        sys.exit(1)
    else:
        log.warning(msg)
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Stage implementations (thin wrappers around task modules)
# ═══════════════════════════════════════════════════════════════════════════════

def stage0_download(args):
    banner(0, "Data Collection")
    from data_collection import download_segment
    paths = download_segment(
        url          = args.url,
        start_sec    = args.start_sec,
        duration_sec = args.duration,
        output_asr   = str(Paths.ORIG_ASR),
        output_tts   = str(Paths.ORIG_TTS),
    )
    check_file(Paths.ORIG_ASR, "ASR audio (16 kHz)")
    check_file(Paths.ORIG_TTS, "TTS audio (22 kHz)")


def stage1_denoise(args):
    banner(1, "Denoising & Normalisation (Spectral Subtraction)")
    check_file(Paths.ORIG_ASR, "Input audio", abort=True)
    from task1_3_denoising import denoise
    denoise(str(Paths.ORIG_ASR), str(Paths.DENOISED))
    check_file(Paths.DENOISED, "Denoised audio")


def stage2_transcribe(args):
    banner(2, "Constrained Beam-Search Transcription (Whisper + N-gram LM)")
    check_file(Paths.DENOISED, "Denoised audio", abort=True)
    from task1_2_constrained_decoding import transcribe_constrained
    transcribe_constrained(
        audio_path  = str(Paths.DENOISED),
        output_path = str(Paths.TRANSCRIPT),
        model_size  = args.whisper_model,
        alpha       = args.ngram_alpha,
        beam_size   = args.beam,
    )
    check_file(Paths.TRANSCRIPT, "Transcript JSON")


def stage3_pseudo_labels(args):
    banner(3, "LID Bootstrap — Whisper Pseudo-Labels")
    check_file(Paths.DENOISED, "Denoised audio", abort=True)
    from task1_1_lid import generate_pseudo_labels
    generate_pseudo_labels(str(Paths.DENOISED), str(Paths.PSEUDO_LBL))
    check_file(Paths.PSEUDO_LBL, "Pseudo-labels JSON")


def stage4_lid_train(args):
    banner(4, "LID Training (Multi-Head Transformer)")
    check_file(Paths.DENOISED,   "Denoised audio",   abort=True)
    check_file(Paths.PSEUDO_LBL, "Pseudo-labels",    abort=True)
    from task1_1_lid import train
    Paths.MODELS.mkdir(exist_ok=True)
    train(
        audio_path = str(Paths.DENOISED),
        label_path = str(Paths.PSEUDO_LBL),
        save_path  = str(Paths.LID_MODEL),
        epochs     = args.lid_epochs,
        batch_size = args.lid_batch,
        lr         = args.lid_lr,
        device     = args.device,
    )
    check_file(Paths.LID_MODEL, "LID model weights")


def stage5_lid_infer(args):
    banner(5, "LID Inference — Frame-Level Language Tags")
    check_file(Paths.DENOISED,  "Denoised audio",  abort=True)
    check_file(Paths.LID_MODEL, "LID model",       abort=True)
    from task1_1_lid import infer
    infer(
        audio_path  = str(Paths.DENOISED),
        model_path  = str(Paths.LID_MODEL),
        output_path = str(Paths.LID_OUT),
        device      = args.device,
    )
    check_file(Paths.LID_OUT, "LID output JSON")


def stage6_ipa(args):
    banner(6, "IPA Conversion — Hinglish G2P")
    check_file(Paths.TRANSCRIPT, "Transcript JSON", abort=True)
    from task2_1_ipa import process_transcript
    process_transcript(str(Paths.TRANSCRIPT), str(Paths.IPA_OUT))
    check_file(Paths.IPA_OUT, "IPA output JSON")


def stage7_translate(args):
    banner(7, "Semantic Translation → Maithili")
    check_file(Paths.IPA_OUT, "IPA JSON", abort=True)
    from task2_2_translation import translate_ipa_json
    translate_ipa_json(str(Paths.IPA_OUT), str(Paths.MAI_OUT))
    check_file(Paths.MAI_OUT, "Maithili output JSON")


def stage8_speaker_emb(args):
    banner(8, "Speaker Embedding Extraction (TDNN x-vector)")
    if not check_file(Paths.VOICE_REF, "Student voice reference"):
        log.warning(
            "Place a 60-second recording of your voice at:\n"
            f"  {Paths.VOICE_REF}\n"
            "Then re-run from stage 8.  Continuing with fallback silence …"
        )
        import numpy as np, soundfile as sf
        Paths.DATA.mkdir(exist_ok=True)
        sf.write(str(Paths.VOICE_REF),
                 np.zeros(16_000 * 60, dtype=np.float32), 16_000)
    from task3_1_voice_embedding import extract_embedding
    emb = extract_embedding(
        wav_path        = str(Paths.VOICE_REF),
        device          = args.device,
        use_speechbrain = True,
    )
    import torch
    torch.save(emb, str(Paths.SPK_EMB))
    check_file(Paths.SPK_EMB, "Speaker embedding")


def stage9_synthesise(args):
    banner(9, "Zero-Shot TTS Synthesis (Meta MMS-TTS VITS)")
    check_file(Paths.MAI_OUT, "Maithili JSON", abort=True)
    from task3_3_synthesis import MMSSynthesiser, load_maithili_text
    text  = load_maithili_text(str(Paths.MAI_OUT))
    synth = MMSSynthesiser(
        device      = args.device,
        speaker_wav = str(Paths.VOICE_REF) if Paths.VOICE_REF.exists() else None,
    )
    synth.synthesize_full(text, str(Paths.SYNTH_FLAT))
    check_file(Paths.SYNTH_FLAT, "Flat synthesis")


def stage10_prosody(args):
    banner(10, "Prosody Warping (DTW F0 + Energy Transfer)")
    check_file(Paths.ORIG_TTS,  "Professor audio (22 kHz)", abort=True)
    check_file(Paths.SYNTH_FLAT, "Flat synthesis",          abort=True)
    from task3_2_prosody_warping import prosody_warp
    prosody_warp(
        source_path = str(Paths.ORIG_TTS),
        target_path = str(Paths.SYNTH_FLAT),
        output_path = str(Paths.SYNTH_PROS),
    )
    check_file(Paths.SYNTH_PROS, "Prosody-warped output")


def stage11_antispoofing(args):
    banner(11, "Anti-Spoofing CM Training (LFCC + LCNN + Focal Loss)")
    check_file(Paths.VOICE_REF,  "Bona fide audio", abort=True)
    check_file(Paths.SYNTH_PROS, "Spoof audio",     abort=True)
    from task4_1_antispoofing import train_cm
    Paths.MODELS.mkdir(exist_ok=True)
    train_cm(
        bona_fide_path = str(Paths.VOICE_REF),
        spoof_path     = str(Paths.SYNTH_PROS),
        save_path      = str(Paths.CM_MODEL),
        epochs         = args.cm_epochs,
        device         = args.device,
    )
    check_file(Paths.CM_MODEL, "Anti-spoofing model")


def stage12_adversarial(args):
    banner(12, "Adversarial Noise Injection (FGSM)")
    check_file(Paths.DENOISED,  "Denoised audio", abort=True)
    check_file(Paths.LID_MODEL, "LID model",      abort=True)
    from task4_2_adversarial import adversarial_attack
    adversarial_attack(
        audio_path    = str(Paths.DENOISED),
        model_path    = str(Paths.LID_MODEL),
        output_path   = str(Paths.ADV_WAV),
        report_path   = str(Paths.ADV_RPT),
        segment_start = 0.0,
        segment_end   = 5.0,
        device        = args.device,
    )
    check_file(Paths.ADV_WAV, "Adversarial audio")
    check_file(Paths.ADV_RPT, "Adversarial report")


def stage13_evaluate(args):
    banner(13, "Evaluation — WER / MCD / EER / LID Switching Accuracy")
    from evaluate import run_all
    run_all(
        transcript_json     = str(Paths.TRANSCRIPT),
        synth_path          = str(Paths.SYNTH_PROS),
        ref_voice_path      = str(Paths.VOICE_REF),
        lid_output_json     = str(Paths.LID_OUT),
        antispoofing_model  = str(Paths.CM_MODEL) if Paths.CM_MODEL.exists() else None,
        bona_fide_path      = str(Paths.VOICE_REF),
        spoof_path          = str(Paths.SYNTH_PROS),
        device              = args.device,
        output_json         = str(Paths.EVAL_OUT),
    )
    check_file(Paths.EVAL_OUT, "Evaluation results")


# ═══════════════════════════════════════════════════════════════════════════════
# Main dispatcher
# ═══════════════════════════════════════════════════════════════════════════════

STAGES = [
    (0,  "download",      stage0_download),
    (1,  "denoise",       stage1_denoise),
    (2,  "transcribe",    stage2_transcribe),
    (3,  "pseudo-labels", stage3_pseudo_labels),
    (4,  "lid-train",     stage4_lid_train),
    (5,  "lid-infer",     stage5_lid_infer),
    (6,  "ipa",           stage6_ipa),
    (7,  "translate",     stage7_translate),
    (8,  "speaker-emb",   stage8_speaker_emb),
    (9,  "synthesise",    stage9_synthesise),
    (10, "prosody",       stage10_prosody),
    (11, "antispoofing",  stage11_antispoofing),
    (12, "adversarial",   stage12_adversarial),
    (13, "evaluate",      stage13_evaluate),
]


def run_pipeline(args):
    # Create output directories
    Paths.DATA.mkdir(exist_ok=True)
    Paths.MODELS.mkdir(exist_ok=True)

    start = args.start_stage
    end   = args.end_stage

    log.info(f"\n{'='*60}")
    log.info(f"  Speech Understanding PA-2 Pipeline")
    log.info(f"  Running stages {start} – {end}  |  device={args.device}")
    log.info(f"{'='*60}\n")

    for num, name, fn in STAGES:
        if num < start or num > end:
            continue
        if args.skip_download and num == 0:
            log.info("Skipping stage 0 (--skip-download)")
            continue
        try:
            fn(args)
        except Exception as e:
            log.error(f"Stage {num} ({name}) failed: {e}")
            if not args.continue_on_error:
                raise
            log.warning("Continuing to next stage …")

    log.info("\n" + "=" * 60)
    log.info("  Pipeline complete.")
    log.info("  Key output files:")
    log.info(f"    Transcript   : {Paths.TRANSCRIPT}")
    log.info(f"    Maithili TTS : {Paths.SYNTH_PROS}")
    log.info(f"    LID output   : {Paths.LID_OUT}")
    log.info(f"    Eval results : {Paths.EVAL_OUT}")
    log.info("=" * 60 + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Speech Understanding PA-2 – Full Pipeline"
    )

    # Stage control
    parser.add_argument("--start-stage",      type=int, default=0)
    parser.add_argument("--end-stage",        type=int, default=13)
    parser.add_argument("--skip-download",    action="store_true")
    parser.add_argument("--continue-on-error",action="store_true")

    # Data collection
    parser.add_argument("--url",        default="https://youtu.be/ZPUtA3W-7_I")
    parser.add_argument("--start-sec",  type=int, default=8400,
                        help="Video start offset (default: 2h20m = 8400s)")
    parser.add_argument("--duration",   type=int, default=600,
                        help="Clip duration in seconds (default: 600 = 10 min)")

    # Transcription
    parser.add_argument("--whisper-model", default="large-v3",
                        choices=["tiny","base","small","medium","large-v3"])
    parser.add_argument("--ngram-alpha",   type=float, default=0.4)
    parser.add_argument("--beam",          type=int,   default=5)

    # LID training
    parser.add_argument("--lid-epochs",  type=int,   default=15)
    parser.add_argument("--lid-batch",   type=int,   default=128)
    parser.add_argument("--lid-lr",      type=float, default=3e-4)

    # Anti-spoofing
    parser.add_argument("--cm-epochs",   type=int,   default=30)

    # Hardware
    parser.add_argument("--device",      default="mps",
                        help="PyTorch device (cuda / mps / cpu)")

    args = parser.parse_args()
    run_pipeline(args)
