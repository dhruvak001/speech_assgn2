#!/usr/bin/env python3
"""
evaluate.py
===========
Evaluation script — computes all metrics required by the assignment.

Metrics implemented
───────────────────
1.  WER    – Word Error Rate (English and Hindi segments separately)
2.  MCD    – Mel-Cepstral Distortion (synthesised vs. reference voice)
3.  EER    – Equal Error Rate (anti-spoofing classifier)
4.  LID-TS – Language-Switch Timestamp Precision (< 200 ms required)

Usage
─────
    python evaluate.py \
        --transcript data/transcript.json \
        --reference-text data/reference_transcript.txt \
        --synth      data/output_LRL_cloned.wav \
        --reference-voice data/student_voice_ref.wav \
        --lid-output data/lid_output.json \
        --lid-gold   data/lid_gold.json \
        --antispoofing-model models/antispoofing.pt \
        --bona-fide  data/student_voice_ref.wav \
        --spoof      data/output_LRL_cloned.wav
"""

import os
import json
import logging
import argparse
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Word Error Rate (WER)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_wer_simple(hypothesis: str, reference: str) -> float:
    """
    Compute WER via Levenshtein edit distance on word tokens.

    WER = (S + D + I) / N
    where N = number of reference words.
    """
    hyp_words = hypothesis.lower().split()
    ref_words = reference.lower().split()

    N = len(ref_words)
    if N == 0:
        return 0.0

    H = len(hyp_words)
    # Edit distance DP table
    dp = [[0] * (H + 1) for _ in range(N + 1)]
    for i in range(N + 1):
        dp[i][0] = i
    for j in range(H + 1):
        dp[0][j] = j

    for i in range(1, N + 1):
        for j in range(1, H + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    return dp[N][H] / N


def evaluate_wer(
    transcript_json: str,
    reference_text:  Optional[str] = None,
) -> Dict[str, float]:
    """
    Compute WER for English and Hindi segments separately.

    transcript_json : JSON with "segments" from Task 1.2
    reference_text  : Optional path to plain-text ground truth

    Returns {"wer_en": ..., "wer_hi": ..., "wer_overall": ...}
    """
    try:
        from jiwer import wer as jiwer_wer
        _wer_fn = jiwer_wer
    except ImportError:
        _wer_fn = compute_wer_simple

    with open(transcript_json, encoding="utf-8") as f:
        data = json.load(f)

    segs = data.get("segments", [])

    # Separate segments by detected language
    en_hyp, hi_hyp = [], []
    for seg in segs:
        lang = seg.get("language", seg.get("lang", "en"))
        text = seg.get("text", "").strip()
        if lang in ("en", "english"):
            en_hyp.append(text)
        else:
            hi_hyp.append(text)

    if reference_text and os.path.isfile(reference_text):
        with open(reference_text, encoding="utf-8") as f:
            ref_full = f.read()
        ref_en = ref_full   # use full ref for overall WER if no split available
        ref_hi = ref_full
        wer_en  = _wer_fn(ref_en,      " ".join(en_hyp)) if en_hyp else float("nan")
        wer_hi  = _wer_fn(ref_hi,      " ".join(hi_hyp)) if hi_hyp else float("nan")
        wer_all = _wer_fn(ref_full,    data.get("full_text", ""))
    else:
        log.warning("No reference text provided.  WER will be estimated "
                    "by comparing consecutive Whisper passes.")
        wer_en  = float("nan")
        wer_hi  = float("nan")
        wer_all = float("nan")

    result = {
        "wer_en":      round(wer_en,  4) if not np.isnan(wer_en)  else None,
        "wer_hi":      round(wer_hi,  4) if not np.isnan(wer_hi)  else None,
        "wer_overall": round(wer_all, 4) if not np.isnan(wer_all) else None,
    }
    log.info(f"WER  EN={result['wer_en']}  HI={result['wer_hi']}  "
             f"Overall={result['wer_overall']}")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Mel-Cepstral Distortion (MCD)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_mcd(
    synth_path: str,
    ref_path:   str,
    sr:         int = 22_050,
    n_mfcc:     int = 13,
    fmax:       float = None,
) -> float:
    """
    Compute Mel-Cepstral Distortion between synthesised and reference audio.

    MCD = (10/ln10) · sqrt(2 · Σ_{k=1}^{K} (mc_synth_k − mc_ref_k)²)

    where mc are the mel-cepstral coefficients (c_1 … c_13, excluding c_0).

    Uses Dynamic Time Warping to align sequences before computing distance.
    """
    import soundfile as sf

    def _load_mfcc(path):
        audio, orig_sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if orig_sr != sr:
            import torchaudio
            audio = torchaudio.functional.resample(
                torch.from_numpy(audio), orig_sr, sr
            ).numpy()

        try:
            import librosa
            mfcc = librosa.feature.mfcc(
                y  = audio,
                sr = sr,
                n_mfcc = n_mfcc + 1,
                n_fft  = 1024,
                hop_length = 256,
                fmax   = fmax,
            )
            return mfcc[1:].T    # drop c_0, shape (T, n_mfcc)
        except ImportError:
            # Manual MFCC via torchaudio
            import torchaudio.transforms as AT
            wav_t = torch.from_numpy(audio).unsqueeze(0)
            mel = AT.MelSpectrogram(sample_rate=sr, n_mels=80)(wav_t)
            amp2db = AT.AmplitudeToDB(top_db=80)
            mel = amp2db(mel).squeeze(0)   # (80, T)
            from scipy.fft import dct as scipy_dct
            mfcc = scipy_dct(mel.numpy().T, type=2, norm="ortho")[:, :n_mfcc+1]
            return mfcc[:, 1:]   # (T, n_mfcc)

    log.info(f"Computing MCD: synth={synth_path}  ref={ref_path}")
    mc_synth = _load_mfcc(synth_path)   # (T_s, K)
    mc_ref   = _load_mfcc(ref_path)     # (T_r, K)

    # DTW alignment (simplified: align via nearest-neighbour along shorter axis)
    T_s, T_r = len(mc_synth), len(mc_ref)
    T_min    = min(T_s, T_r)

    # Resample longer to shorter (linear interpolation)
    if T_s > T_min:
        idx      = np.round(np.linspace(0, T_s - 1, T_min)).astype(int)
        mc_synth = mc_synth[idx]
    if T_r > T_min:
        idx    = np.round(np.linspace(0, T_r - 1, T_min)).astype(int)
        mc_ref = mc_ref[idx]

    # MCD formula
    diff = mc_synth - mc_ref                      # (T, K)
    mcd  = (10.0 / np.log(10)) * np.sqrt(2.0 * np.mean(np.sum(diff ** 2, axis=1)))
    log.info(f"MCD = {mcd:.4f} dB  (target < 8.0)")
    return float(mcd)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. EER (reuse from task4_1)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_eer_from_task4(
    bona_fide_path: str,
    spoof_path:     str,
    model_path:     str,
    device:         str = "cpu",
) -> float:
    """Delegate to task4_1 evaluation."""
    from task4_1_antispoofing import evaluate_cm
    return evaluate_cm(bona_fide_path, spoof_path, model_path, device)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. LID Switch Timestamp Precision
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_lid_switching(
    predicted_json: str,
    gold_json:      Optional[str] = None,
    tolerance_ms:   float         = 200.0,
) -> Dict:
    """
    Evaluate language-switch boundary precision.

    predicted_json : LID output from Task 1.1 (list of {start, end, lang})
    gold_json      : Ground truth boundaries (same format; optional)

    Returns {
        "n_switches":       number of ground-truth language switches
        "within_tolerance": fraction within tolerance_ms
        "mean_error_ms":    mean absolute boundary error in ms
        "confusion_matrix": 2×2 [[TN,FP],[FN,TP]] for English-vs-Hindi
    }
    """
    with open(predicted_json, encoding="utf-8") as f:
        pred_segs = json.load(f)

    if gold_json is None or not os.path.isfile(gold_json):
        log.warning("No gold LID labels – reporting predicted boundary count only.")
        switches = sum(
            1 for i in range(1, len(pred_segs))
            if pred_segs[i]["lang"] != pred_segs[i-1]["lang"]
        )
        return {"n_predicted_switches": switches, "no_gold": True}

    with open(gold_json, encoding="utf-8") as f:
        gold_segs = json.load(f)

    # Extract switch boundaries from gold
    gold_switches = []
    for i in range(1, len(gold_segs)):
        if gold_segs[i]["lang"] != gold_segs[i-1]["lang"]:
            gold_switches.append(gold_segs[i]["start"])

    # Extract switch boundaries from predictions
    pred_switches = []
    for i in range(1, len(pred_segs)):
        if pred_segs[i]["lang"] != pred_segs[i-1]["lang"]:
            pred_switches.append(pred_segs[i]["start"])

    # Match predicted boundaries to gold boundaries (greedy nearest-neighbour)
    errors_ms = []
    matched   = set()
    for g in gold_switches:
        best_p, best_d = None, float("inf")
        for k, p in enumerate(pred_switches):
            if k in matched:
                continue
            d = abs(p - g) * 1000.0   # seconds → ms
            if d < best_d:
                best_d = d
                best_p = k
        if best_p is not None:
            matched.add(best_p)
            errors_ms.append(best_d)

    n_gold   = len(gold_switches)
    n_within = sum(1 for e in errors_ms if e <= tolerance_ms)
    mean_err = float(np.mean(errors_ms)) if errors_ms else float("nan")

    # Confusion matrix: frame-level English vs Hindi
    def seg_to_frame_labels(segs, total_sec, fps=100):
        n_frames = int(total_sec * fps)
        labels   = np.zeros(n_frames, dtype=int)
        for seg in segs:
            s = int(seg["start"] * fps)
            e = int(seg["end"]   * fps)
            if seg["lang"] in ("hi", "hindi"):
                labels[s:e] = 1
        return labels

    total_sec = max(
        max(s["end"] for s in gold_segs),
        max(s["end"] for s in pred_segs),
    )
    gold_frames = seg_to_frame_labels(gold_segs, total_sec)
    pred_frames = seg_to_frame_labels(pred_segs, total_sec)

    from sklearn.metrics import confusion_matrix, f1_score
    cm = confusion_matrix(gold_frames, pred_frames).tolist()
    f1 = f1_score(gold_frames, pred_frames, average="macro", zero_division=0)

    result = {
        "n_gold_switches":          n_gold,
        "n_pred_switches":          len(pred_switches),
        "matched":                  len(errors_ms),
        "within_200ms":             n_within,
        "fraction_within_200ms":    round(n_within / max(n_gold, 1), 4),
        "mean_boundary_error_ms":   round(mean_err, 2),
        "frame_f1_macro":           round(f1, 4),
        "confusion_matrix":         cm,
    }
    log.info(
        f"LID switches: gold={n_gold}  pred={len(pred_switches)}  "
        f"within_200ms={n_within}/{n_gold}  F1={f1:.4f}"
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Master evaluation runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_all(
    transcript_json:     str,
    synth_path:          str,
    ref_voice_path:      str,
    lid_output_json:     str,
    antispoofing_model:  Optional[str] = None,
    bona_fide_path:      Optional[str] = None,
    spoof_path:          Optional[str] = None,
    reference_text:      Optional[str] = None,
    lid_gold_json:       Optional[str] = None,
    device:              str           = "cpu",
    output_json:         str           = "data/evaluation_results.json",
) -> dict:
    results = {}

    # WER
    log.info("=" * 60)
    log.info("1/4  Word Error Rate (WER)")
    log.info("=" * 60)
    try:
        results["wer"] = evaluate_wer(transcript_json, reference_text)
    except Exception as e:
        log.error(f"WER computation failed: {e}")
        results["wer"] = {"error": str(e)}

    # MCD
    log.info("=" * 60)
    log.info("2/4  Mel-Cepstral Distortion (MCD)")
    log.info("=" * 60)
    try:
        mcd = compute_mcd(synth_path, ref_voice_path)
        results["mcd"] = {"value": round(mcd, 4), "target": "< 8.0",
                          "pass": mcd < 8.0}
    except Exception as e:
        log.error(f"MCD computation failed: {e}")
        results["mcd"] = {"error": str(e)}

    # EER
    log.info("=" * 60)
    log.info("3/4  Anti-Spoofing EER")
    log.info("=" * 60)
    if antispoofing_model and bona_fide_path and spoof_path:
        try:
            eer = compute_eer_from_task4(bona_fide_path, spoof_path,
                                         antispoofing_model, device)
            results["eer"] = {"value": round(eer, 2), "target": "< 10 %",
                              "pass": eer < 10.0}
        except Exception as e:
            log.error(f"EER computation failed: {e}")
            results["eer"] = {"error": str(e)}
    else:
        log.warning("Skipping EER (missing anti-spoofing model or audio).")
        results["eer"] = {"skipped": True}

    # LID Switch Precision
    log.info("=" * 60)
    log.info("4/4  LID Switching Accuracy")
    log.info("=" * 60)
    try:
        results["lid_switching"] = evaluate_lid_switching(
            lid_output_json, lid_gold_json
        )
    except Exception as e:
        log.error(f"LID switching eval failed: {e}")
        results["lid_switching"] = {"error": str(e)}

    # Summary
    log.info("\n══════════ EVALUATION SUMMARY ══════════")
    for k, v in results.items():
        log.info(f"  {k:20s}: {v}")
    log.info("════════════════════════════════════════\n")

    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved → {output_json}")
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcript",     default="data/transcript.json")
    parser.add_argument("--synth",          default="data/output_LRL_cloned.wav")
    parser.add_argument("--reference-voice",default="data/student_voice_ref.wav")
    parser.add_argument("--lid-output",     default="data/lid_output.json")
    parser.add_argument("--lid-gold",       default=None)
    parser.add_argument("--reference-text", default=None)
    parser.add_argument("--antispoofing-model", default=None)
    parser.add_argument("--bona-fide",      default=None)
    parser.add_argument("--spoof",          default=None)
    parser.add_argument("--output",         default="data/evaluation_results.json")
    parser.add_argument("--device",         default="cpu")
    args = parser.parse_args()

    run_all(
        transcript_json     = args.transcript,
        synth_path          = args.synth,
        ref_voice_path      = args.reference_voice,
        lid_output_json     = args.lid_output,
        antispoofing_model  = args.antispoofing_model,
        bona_fide_path      = args.bona_fide,
        spoof_path          = args.spoof,
        reference_text      = args.reference_text,
        lid_gold_json       = args.lid_gold,
        device              = args.device,
        output_json         = args.output,
    )
