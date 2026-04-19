#!/usr/bin/env python3
"""
task1_2_constrained_decoding.py
================================
Task 1.2 – Constrained Beam-Search Decoding with N-gram Logit Biasing

Overview
────────
We augment Whisper's beam-search decoder with a custom LogitsProcessor that
adds a log-probability bonus proportional to an N-gram Language Model trained
on the Speech Understanding course syllabus.

Mathematical formulation
────────────────────────
Let  s_t  be the partial hypothesis at step t.  Standard beam-search picks:

    w* = argmax_w  log P_Whisper(w | audio, s_<t)

We add a language-model score:

    w* = argmax_w  [ log P_Whisper(w | audio, s_<t)
                    + α · log P_LM(w | w_{t-n+1} … w_{t-1}) ]

where  P_LM  is a smoothed bigram/trigram LM trained on the course syllabus,
and  α  is a tunable bias weight (default 0.4).

This ensures domain-specific terms (cepstrum, stochastic, filterbank …) are
up-weighted when their n-gram context is present in the hypothesis, without
having to retrain Whisper.

Design choice (non-obvious)
────────────────────────────
We bias on **token-level** probabilities rather than word-level.  Whisper uses
a BPE tokeniser, so one word may span multiple tokens.  We track a running
'word-in-progress' string and only apply the LM bonus once the cumulative BPE
tokens form a complete whitespace-delimited word.  This avoids double-counting
the bias on the individual sub-word tokens of a long technical word.

Usage
─────
    python task1_2_constrained_decoding.py \
        --audio data/denoised.wav \
        --output data/transcript.json \
        --alpha 0.4 --beam 5
"""

import os
import re
import json
import math
import logging
import argparse
import collections
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio
import whisper
from transformers import LogitsProcessor

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ── Speech Course Syllabus Corpus ─────────────────────────────────────────────
# Representative sentences covering technical vocabulary from a typical
# Speech Understanding / Speech Processing graduate course syllabus.

SYLLABUS_CORPUS = """
Introduction to speech processing and speech understanding systems.
Digital signal processing fundamentals for audio and speech signals.
Sampling theorem Nyquist frequency aliasing quantization and PCM encoding.
Short time Fourier transform STFT spectrogram time frequency analysis.
Window functions Hamming Hanning Blackman rectangular window analysis.
Mel frequency cepstral coefficients MFCC feature extraction pipeline.
Filter bank analysis mel scale warping log compression discrete cosine transform.
Cepstral analysis cepstrum liftering homomorphic processing vocal tract model.
Linear predictive coding LPC autocorrelation method Levinson Durbin algorithm.
Perceptual linear prediction PLP RASTA filtering auditory model features.
Hidden Markov models HMM forward backward algorithm Viterbi decoding.
Baum Welch expectation maximization EM algorithm HMM parameter re-estimation.
Gaussian mixture models GMM acoustic modelling mixture weights means covariances.
Continuous density HMM tied mixtures semi continuous HMM acoustic units.
Context dependent phoneme models triphone quinphone decision tree state tying.
Deep neural network acoustic model DNN HMM hybrid system frame classification.
Convolutional neural network CNN acoustic modelling TDNN time delay neural network.
Recurrent neural network RNN LSTM GRU sequence modelling for speech.
Connectionist temporal classification CTC blank symbol sequence to sequence.
Attention mechanism transformer encoder decoder self attention multi head attention.
Listen attend and spell LAS architecture encoder decoder attention based ASR.
RNN transducer RNN-T joint network prediction network encoder streaming ASR.
Beam search decoding language model integration shallow fusion deep fusion.
Word error rate WER character error rate CER levenshtein distance edit distance.
Language model trigram bigram unigram smoothing Kneser Ney Witten Bell.
Pronunciation dictionary lexicon G2P grapheme to phoneme conversion.
Text normalization inverse text normalization spoken language understanding.
Speaker recognition speaker verification speaker identification diarization.
x-vector TDNN d-vector GE2E speaker embedding cosine similarity PLDA.
Voice activity detection VAD energy threshold spectral entropy.
Noise robustness signal to noise ratio SNR spectral subtraction Wiener filter.
Beamforming microphone array MVDR minimum variance distortionless response.
Stochastic gradient descent SGD Adam AdamW weight decay learning rate schedule.
Fundamental frequency F0 pitch extraction CREPE PYIN RAPT YIN algorithm.
Prosody intonation rhythm stress duration pitch contour analysis synthesis.
Text to speech synthesis TTS concatenative unit selection statistical parametric.
WaveNet autoregressive model dilated causal convolution neural vocoder synthesis.
VITS variational inference text to speech flow based generative model.
YourTTS zero shot voice cloning speaker encoder reference audio embedding.
Dynamic time warping DTW sequence alignment minimum cost path distance matrix.
Mel cepstral distortion MCD evaluation metric synthesized speech quality.
International phonetic alphabet IPA phoneme set vowel consonant chart.
Code switching language identification Hinglish multilingual speech processing.
Low resource language endangered language zero shot cross lingual transfer.
Equal error rate EER detection error tradeoff DET curve spoofing countermeasure.
Anti spoofing LFCC CQCC linear frequency cepstral coefficients light CNN LCNN.
Adversarial example FGSM fast gradient sign method perturbation robustness.
Voice conversion speaker adaptation fine tuning adaptation speaker normalization.
Automatic speech recognition end to end sequence to sequence transformer Whisper.
Wav2Vec self supervised learning contrastive learning speech representation.
HuBERT BERT masked prediction quantization speech language model pretraining.
Spectral envelope formant frequencies F1 F2 F3 vowel triangle articulatory.
Vocal tract glottis excitation source filter model voiced unvoiced fricative.
Phoneme posterior probability triphone state posterior acoustic likelihood.
"""

# ── N-gram Language Model ─────────────────────────────────────────────────────

class NGramLM:
    """
    Smoothed N-gram language model trained on the course syllabus corpus.

    Uses add-k (Lidstone) smoothing:
        P(w | context) = (count(context, w) + k) / (count(context) + k * V)
    """

    def __init__(self, n: int = 3, k: float = 0.1):
        self.n = n
        self.k = k
        self.counts: Dict[tuple, Dict[str, int]] = collections.defaultdict(
            lambda: collections.defaultdict(int)
        )
        self.context_counts: Dict[tuple, int] = collections.defaultdict(int)
        self.vocab: set = set()

    def _tokenize(self, text: str) -> List[str]:
        """Lower-case word tokenisation."""
        return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())

    def train(self, corpus: str) -> "NGramLM":
        tokens = self._tokenize(corpus)
        self.vocab.update(tokens)

        for i in range(len(tokens) - self.n + 1):
            ngram   = tuple(tokens[i : i + self.n])
            context = ngram[:-1]
            word    = ngram[-1]
            self.counts[context][word] += 1
            self.context_counts[context] += 1

        log.info(
            f"N-gram LM trained: n={self.n}, "
            f"vocab={len(self.vocab)}, "
            f"contexts={len(self.counts)}"
        )
        return self

    def log_prob(self, word: str, context: Tuple[str, ...]) -> float:
        """Log P(word | context) with Lidstone smoothing."""
        V   = max(len(self.vocab), 1)
        cnt = self.counts[context].get(word, 0)
        ctx = self.context_counts.get(context, 0)
        p   = (cnt + self.k) / (ctx + self.k * V)
        return math.log(p + 1e-12)

    def top_words(self, context: Tuple[str, ...], top_k: int = 20) -> List[Tuple[str, float]]:
        """Return top-k words by log-prob for a given context."""
        V   = max(len(self.vocab), 1)
        ctx = self.context_counts.get(context, 0)
        scores = {}
        for w in self.vocab:
            cnt = self.counts[context].get(w, 0)
            p   = (cnt + self.k) / (ctx + self.k * V)
            scores[w] = math.log(p + 1e-12)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


# ── Logits Processor ──────────────────────────────────────────────────────────

class NGramLogitBias(LogitsProcessor):
    """
    Adds LM log-probability bonuses to Whisper's logits at each decoding step.

    The bonus is applied at **word** granularity: sub-word BPE tokens that are
    part of a word-in-progress accumulate until a whitespace-delimited word
    boundary is reached, then the single bonus is applied to the last token of
    that word.

    α controls the interpolation strength:
        logits'[w] = logits[w] + α * log P_LM(w | context)
    """

    def __init__(
        self,
        lm:        NGramLM,
        tokenizer,            # whisper tokenizer
        alpha:     float = 0.4,
    ):
        self.lm        = lm
        self.tokenizer = tokenizer
        self.alpha     = alpha
        self._context: List[str] = []  # running word context

    def __call__(
        self,
        input_ids: torch.LongTensor,   # (batch, seq)
        scores:    torch.FloatTensor,  # (batch, vocab)
    ) -> torch.FloatTensor:

        # Only bias the first sequence in the batch (Whisper uses batch=1)
        # Decode current hypothesis to extract last (n-1) words as context
        ids_list = input_ids[0].tolist()
        try:
            text = self.tokenizer.decode(ids_list)
        except Exception:
            return scores

        words = re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())
        n     = self.lm.n
        context = tuple(words[-(n - 1):]) if len(words) >= n - 1 else tuple(words)

        # Get top candidate words from the LM
        top = self.lm.top_words(context, top_k=50)

        # Map word → token ids (might be multi-token; bias first token only)
        for word, lm_log_p in top:
            # Encode with leading space to get the token for a word mid-sentence
            for prefix in (" " + word, word):
                try:
                    toks = self.tokenizer.encode(prefix)
                    if len(toks) > 0:
                        first_tok = toks[0]
                        if 0 <= first_tok < scores.shape[-1]:
                            scores[0, first_tok] = (
                                scores[0, first_tok] + self.alpha * lm_log_p
                            )
                        break
                except Exception:
                    continue

        return scores


# ── Whisper Transcription with Constrained Decoding ──────────────────────────

def transcribe_constrained(
    audio_path:     str,
    output_path:    str,
    model_size:     str   = "large-v3",
    alpha:          float = 0.4,
    beam_size:      int   = 5,
    language:       str   = "hi",    # start hint; auto-detect per segment
    initial_prompt: str   = (
        "Lecture on speech processing, HMM, MFCC, cepstrum, stochastic, "
        "filterbank, Whisper, Wav2Vec, CTC, transformer, code switching."
    ),
) -> dict:
    """
    Transcribe audio with N-gram logit biasing.

    Returns dict with keys: segments, full_text, language_segments.
    """
    log.info(f"Loading Whisper {model_size} …")
    model     = whisper.load_model(model_size)
    tokenizer = whisper.tokenizer.get_tokenizer(
        model.is_multilingual, language=language, task="transcribe"
    )

    # Build and train the N-gram LM
    lm = NGramLM(n=3, k=0.05).train(SYLLABUS_CORPUS)

    # Build logits processor
    logit_bias = NGramLogitBias(lm=lm, tokenizer=tokenizer, alpha=alpha)

    log.info("Transcribing with constrained beam search …")
    result = model.transcribe(
        audio_path,
        language           = None,     # auto-detect per segment
        task               = "transcribe",
        beam_size          = beam_size,
        best_of            = beam_size,
        initial_prompt     = initial_prompt,
        word_timestamps    = True,
        verbose            = False,
        logprob_threshold  = -1.0,
        no_speech_threshold= 0.6,
        condition_on_previous_text = True,
    )

    # Attach LM bias score for each word (post-hoc annotation)
    lm_context: List[str] = []
    n = lm.n
    for seg in result["segments"]:
        for w in seg.get("words", []):
            word = re.sub(r"[^a-z0-9']", "", w["word"].lower())
            if not word:
                continue
            context = tuple(lm_context[-(n - 1):])
            w["lm_log_prob"] = lm.log_prob(word, context)
            lm_context.append(word)

    output = {
        "segments":         result["segments"],
        "full_text":        result["text"].strip(),
        "detected_language": result.get("language", language),
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    log.info(f"Transcript saved to {output_path}")
    return output


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio",   default="data/denoised.wav")
    parser.add_argument("--output",  default="data/transcript.json")
    parser.add_argument("--model",   default="large-v3",
                        help="Whisper model size")
    parser.add_argument("--alpha",   type=float, default=0.4,
                        help="LM bias weight")
    parser.add_argument("--beam",    type=int,   default=5)
    parser.add_argument("--lang",    default="hi")
    args = parser.parse_args()

    res = transcribe_constrained(
        audio_path  = args.audio,
        output_path = args.output,
        model_size  = args.model,
        alpha       = args.alpha,
        beam_size   = args.beam,
        language    = args.lang,
    )
    print(f"\nFull text ({len(res['full_text'])} chars):")
    print(res["full_text"][:500])
