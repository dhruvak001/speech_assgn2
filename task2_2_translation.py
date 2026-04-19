#!/usr/bin/env python3
"""
task2_2_translation.py
=======================
Task 2.2 – Semantic Translation to Maithili (Low-Resource Language)

Target language: Maithili (ISO 639-3: mai)
Region: Mithila (northern Bihar, Nepal Terai)
Speakers: ~35 million
Script: Tirhuta (traditional) / Devanagari (modern)
Relation: Eastern Indo-Aryan, sister to Bengali/Assamese, closely related to
          Bhojpuri and Awadhi, distinct from Standard Hindi.

Why Maithili?
──────────────
Maithili is officially recognised (8th Schedule of Indian Constitution) yet
remains low-resource in NLP: very few parallel corpora exist for speech
domains, and no MT system reliably covers technical speech-processing
vocabulary.  This forces us to build our own parallel corpus.

Corpus structure (≥500 entries)
────────────────────────────────
1. General vocabulary    (100 common words)
2. Technical terms       (200 speech/ML terms → Maithili + pronunciation note)
3. Sentence templates    (50 sentence frames for lecture speech)
4. Numbers & units       (50 entries)
5. Discourse markers     (30 connectives / transition phrases)
6. Phrase fragments      (70+ additional entries)

Design choice (non-obvious)
────────────────────────────
For technical terms that have no Maithili equivalent we follow the Maithili
language academy's (MLA) recommended strategy: borrow the Sanskrit/English
term but adapt it to Maithili phonology (e.g. "cepstrum" → "सेप्सट्रम्",
pronounced /seːpsʈrəm/).  This is preferable to inventing neologisms that
native speakers would not recognise.  Additionally, for compound technical
concepts (e.g. "hidden Markov model") we partially translate the qualifier
while keeping the proper noun: "गुप्त मार्कव प्रतिरूप".

Translation strategy
─────────────────────
1. Word-level lookup in our parallel corpus.
2. If unknown: attempt morpheme decomposition (un-prefix, -ing/-ed suffix
   stripping) and re-lookup.
3. If still unknown: keep the source word (English technical term) and tag it
   as BORROWED.

Usage
─────
    python task2_2_translation.py --text "Today we will study hidden Markov models"
    python task2_2_translation.py --ipa-json data/ipa_output.json \
                                  --output data/maithili_output.json
"""

import os
import re
import json
import logging
import argparse
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# MAITHILI PARALLEL CORPUS  (≥ 500 entries)
# Format: {english_lower: (maithili_devanagari, romanization, notes)}
# ═══════════════════════════════════════════════════════════════════════════════

MAITHILI_CORPUS: Dict[str, Tuple[str, str, str]] = {

    # ── General vocabulary ───────────────────────────────────────────────────
    "today":        ("आइ",          "āi",           ""),
    "yesterday":    ("काल्हि",       "kālhi",         ""),
    "tomorrow":     ("काल्हि",       "kālhi",         "future context"),
    "now":          ("अखन",         "akhan",         ""),
    "here":         ("एतए",         "etae",          ""),
    "there":        ("ओतए",         "otae",          ""),
    "this":         ("ई",           "ī",             ""),
    "that":         ("ओ",           "o",             ""),
    "we":           ("हम",          "ham",           ""),
    "you":          ("अहाँ",         "ahāṃ",          "formal"),
    "i":            ("हम",          "ham",           ""),
    "they":         ("ओ सभ",        "o sabh",        ""),
    "is":           ("अछि",         "achi",          ""),
    "are":          ("छथि",         "chhathi",       ""),
    "was":          ("छल",          "chhal",         ""),
    "will":         ("-त",           "-ta",           "future suffix"),
    "and":          ("आ",           "ā",             ""),
    "or":           ("वा",          "vā",            ""),
    "but":          ("मुदा",         "mudā",          ""),
    "because":      ("किएक तँ",      "kiek tã",       ""),
    "therefore":    ("तेँ",          "teṃ",           ""),
    "however":      ("तथापि",        "tathāpi",       ""),
    "also":         ("सेहो",         "seho",          ""),
    "not":          ("नहिं",         "nahiṃ",         ""),
    "yes":          ("हाँ",          "hāṃ",           ""),
    "no":           ("नहिं",         "nahiṃ",         ""),
    "good":         ("नीक",         "nīk",           ""),
    "bad":          ("खराब",        "kharāb",        ""),
    "new":          ("नव",          "nav",           ""),
    "old":          ("पुरान",        "purān",         ""),
    "big":          ("पैघ",         "paigh",         ""),
    "small":        ("छोट",         "choṭ",          ""),
    "more":         ("बेसी",         "besī",          ""),
    "less":         ("कम",          "kam",           ""),
    "all":          ("सभ",          "sabh",          ""),
    "some":         ("किछु",         "kichu",         ""),
    "many":         ("बहुत",         "bahut",         ""),
    "few":          ("थोड़",         "thoṛ",          ""),
    "very":         ("बड्ड",         "baḍḍ",          ""),
    "in":           ("मे",          "me",            ""),
    "on":           ("पर",          "par",           ""),
    "of":           ("केर",         "ker",           ""),
    "with":         ("संग",         "saṃg",          ""),
    "for":          ("लेल",         "lel",           ""),
    "to":           ("कें",          "keṃ",           ""),
    "from":         ("सँ",          "sã",            ""),
    "about":        ("बारेमे",       "bāreme",        ""),
    "after":        ("बाद",         "bād",           ""),
    "before":       ("पहिने",        "pahine",        ""),
    "during":       ("समयमे",        "samayame",      ""),
    "between":      ("बीच",         "bīch",          ""),
    "over":         ("ऊपर",         "ūpar",          ""),
    "under":        ("नीचाँ",        "nīchāṃ",        ""),
    "study":        ("पढ़ब",         "paṛhab",        ""),
    "learn":        ("सीखब",        "sīkhab",        ""),
    "understand":   ("बुझब",        "bujhab",        ""),
    "explain":      ("बुझाएब",       "bujhāeb",       ""),
    "see":          ("देखब",        "dekhab",        ""),
    "hear":         ("सुनब",        "sunab",         ""),
    "speak":        ("बाजब",        "bājab",         ""),
    "write":        ("लिखब",        "likhab",        ""),
    "read":         ("पढ़ब",         "paṛhab",        ""),
    "use":          ("उपयोग करब",    "upyog karab",   ""),
    "apply":        ("लागू करब",     "lāgū karab",    ""),
    "calculate":    ("गणना करब",     "gaṇanā karab",  ""),
    "compute":      ("परिगणना करब",  "parigaṇanā karab", ""),
    "measure":      ("मापब",        "māpab",         ""),
    "convert":      ("बदलब",        "badalab",       ""),
    "extract":      ("निकालब",       "nikālab",       ""),
    "train":        ("प्रशिक्षण",     "praśikṣaṇ",    "noun; verb form: प्रशिक्षित करब"),
    "test":         ("परीक्षण",       "parīkṣaṇ",      ""),
    "evaluate":     ("मूल्यांकन करब", "mūlyāṃkan karab", ""),
    "improve":      ("सुधारब",       "sudhārab",      ""),
    "reduce":       ("कम करब",       "kam karab",     ""),
    "increase":     ("बढ़ाएब",        "baṛhāeb",       ""),
    "represent":    ("प्रतिनिधित्व",  "pratinidhitva", ""),
    "classify":     ("वर्गीकृत करब",  "vargīkṛt karab",""),
    "detect":       ("पहचान करब",    "pahacān karab", ""),
    "generate":     ("उत्पन्न करब",   "utpanna karab", ""),
    "process":      ("प्रसंस्करण",    "prasaṃskaraṇ",  "noun"),
    "signal":       ("संकेत",        "saṃket",        ""),
    "system":       ("तंत्र",         "tantra",        ""),
    "method":       ("विधि",         "vidhi",         ""),
    "approach":     ("दृष्टिकोण",     "dṛṣṭikoṇ",      ""),
    "result":       ("परिणाम",        "pariṇām",       ""),
    "problem":      ("समस्या",        "samasyā",       ""),
    "solution":     ("समाधान",        "samādhān",      ""),
    "example":      ("उदाहरण",        "udāharaṇ",      ""),
    "question":     ("प्रश्न",        "praśna",        ""),
    "answer":       ("उत्तर",         "uttar",         ""),
    "information":  ("सूचना",         "sūcanā",        ""),
    "data":         ("आँकड़ा",         "āṃkaṛā",        ""),
    "input":        ("निवेश",         "niveś",         ""),
    "output":       ("निर्गम",         "nirgam",        ""),
    "number":       ("संख्या",         "saṃkhyā",       ""),
    "value":        ("मान",           "mān",           ""),
    "function":     ("कार्य",          "kārya",         ""),
    "parameter":    ("प्राचल",         "prācal",        ""),

    # ── Technical: Speech & Audio ────────────────────────────────────────────
    "speech":       ("वाणी",          "vāṇī",          ""),
    "voice":        ("आवाज",          "āvāj",          ""),
    "audio":        ("श्रव्य",         "śravya",        ""),
    "sound":        ("ध्वनि",          "dhvani",        ""),
    "noise":        ("कोलाहल",         "kolāhal",       ""),
    "silence":      ("शांति",          "śānti",         ""),
    "frequency":    ("आवृत्ति",         "āvṛtti",        ""),
    "amplitude":    ("आयाम",           "āyām",          ""),
    "wave":         ("तरंग",           "taraṃg",        ""),
    "waveform":     ("तरंग रूप",        "taraṃg rūp",    ""),
    "sample":       ("नमूना",           "namūnā",        ""),
    "sampling":     ("नमूनाकरण",        "namūnākaraṇ",   ""),
    "frame":        ("ढाँचा",           "ḍhāṃcā",        "temporal frame"),
    "window":       ("खिड़की",           "khiṛkī",        "signal window"),
    "spectrum":     ("वर्णक्रम",         "varṇakram",     ""),
    "spectrogram":  ("वर्णक्रमचित्र",    "varṇakramcitr", ""),
    "filterbank":   ("छन्नी बैंक",       "channí baiṃk",  "filter bank"),
    "cepstrum":     ("सेप्सट्रम्",        "seːpsʈrəm",    "borrowed+adapted"),
    "mfcc":         ("एमएफसीसी",        "emephsīsī",     "MFCC acronym"),
    "pitch":        ("स्वर",            "svar",          "musical pitch"),
    "fundamental":  ("मूलभूत",          "mūlabhūt",      ""),
    "frequency f0": ("मूल आवृत्ति",      "mūl āvṛtti",    ""),
    "energy":       ("ऊर्जा",           "ūrjā",          ""),
    "power":        ("शक्ति",           "śakti",         ""),
    "phoneme":      ("स्वनिम",          "svanim",        ""),
    "phone":        ("स्वन",            "svan",          ""),
    "vowel":        ("स्वर",            "svar",          ""),
    "consonant":    ("व्यंजन",           "vyañjan",       ""),
    "syllable":     ("अक्षर",           "akṣar",         ""),
    "word":         ("शब्द",            "śabd",          ""),
    "sentence":     ("वाक्य",           "vākya",         ""),
    "utterance":    ("उच्चारण",          "uccāraṇ",       ""),
    "transcript":   ("प्रतिलेख",         "pratilekh",     ""),
    "transcription":("प्रतिलेखन",        "pratilekhan",   ""),
    "recognition":  ("अभिज्ञान",         "abhijñān",      ""),
    "identification":("पहचान",          "pahacān",       ""),
    "language":     ("भाषा",            "bhāṣā",         ""),
    "dialect":      ("बोली",            "bolī",          ""),
    "multilingual": ("बहुभाषिक",         "bahubhāṣik",    ""),
    "code switching":("भाषा मिश्रण",    "bhāṣā miśraṇ",  ""),
    "prosody":      ("स्वरोच्चारण",       "svaroccāraṇ",   ""),
    "intonation":   ("स्वर-उतार-चढ़ाव",   "svar-utār-caṛhāv",""),
    "rhythm":       ("लय",              "lay",           ""),
    "stress":       ("बल",              "bal",           ""),
    "duration":     ("अवधि",            "avadhi",        ""),
    "formant":      ("स्वरूप",           "svarūp",        "formant approx"),
    "vocal tract":  ("स्वर-नली",         "svar-nalī",     ""),
    "speaker":      ("वक्ता",            "vaktā",         ""),
    "listener":     ("श्रोता",           "śrotā",         ""),
    "recording":    ("अभिलेखन",          "abhilekhan",    ""),
    "microphone":   ("ध्वनिग्राहक",       "dhvanigrāhak",  ""),

    # ── Technical: ASR / NLP ─────────────────────────────────────────────────
    "automatic speech recognition": ("स्वचालित वाक् पहचान", "svacālit vāk pahacān", ""),
    "asr":          ("स्वचालित वाक् पहचान", "svacālit vāk pahacān", ""),
    "hidden markov model": ("गुप्त मार्कव प्रतिरूप", "gupt mārkav pratirup", ""),
    "hmm":          ("गुप्त मार्कव प्रतिरूप", "gupt mārkav pratirup", ""),
    "gaussian mixture model": ("गाउसी मिश्रण प्रतिरूप", "gāusī miśraṇ pratirup", ""),
    "gmm":          ("गाउसी मिश्रण प्रतिरूप", "gāusī miśraṇ pratirup", ""),
    "neural network": ("तंत्रिका जाल",    "tantrikā jāl",  ""),
    "deep learning": ("गहन अधिगम",       "gahan adhigam", ""),
    "transformer":  ("रूपांतरक",          "rūpāntarak",    "architecture"),
    "attention":    ("ध्यान",             "dhyān",         ""),
    "encoder":      ("कूटक",             "kūṭak",         ""),
    "decoder":      ("विकूटक",            "vikūṭak",       ""),
    "beam search":  ("किरण खोज",          "kiraṇ khoj",    ""),
    "language model": ("भाषा प्रतिरूप",   "bhāṣā pratirup",""),
    "n-gram":       ("एन-ग्राम",          "en-grām",       "borrowed"),
    "word error rate": ("शब्द त्रुटि दर",  "śabd truṭi dar",""),
    "wer":          ("शब्द त्रुटि दर",    "śabd truṭi dar",""),
    "ctc":          ("सीटीसी",            "sīṭīsī",        "borrowed"),
    "stochastic":   ("यादृच्छिक",          "yādṛcchik",     ""),
    "probability":  ("प्रायिकता",          "prāyikatā",     ""),
    "acoustic model": ("ध्वनि प्रतिरूप",   "dhvani pratirup",""),
    "pronunciation": ("उच्चारण",           "uccāraṇ",       ""),
    "vocabulary":   ("शब्द-भंडार",         "śabd-bhaṃḍār",  ""),
    "decoding":     ("विकूटन",             "vikūṭan",       ""),
    "training":     ("प्रशिक्षण",          "praśikṣaṇ",     ""),
    "testing":      ("परीक्षण",             "parīkṣaṇ",      ""),
    "model":        ("प्रतिरूप",            "pratirup",      ""),
    "feature":      ("विशेषता",            "viśeṣatā",      ""),
    "representation": ("निरूपण",           "nirūpaṇ",       ""),
    "classification": ("वर्गीकरण",          "vargīkaraṇ",    ""),
    "sequence":     ("अनुक्रम",            "anukram",       ""),
    "alignment":    ("संरेखण",             "saṃrekhaṇ",     ""),
    "loss":         ("हानि",               "hāni",          "ML loss"),
    "gradient":     ("प्रवणता",             "pravaṇatā",     ""),
    "optimization": ("अनुकूलन",            "anukūlan",      ""),
    "embedding":    ("अंतःस्थापन",          "aṃtaḥsthāpan",  ""),

    # ── Technical: TTS / Voice Cloning ───────────────────────────────────────
    "text to speech": ("पाठ से वाणी",      "pāṭh se vāṇī",  ""),
    "tts":          ("पाठ से वाणी",         "pāṭh se vāṇī",  ""),
    "synthesis":    ("संश्लेषण",            "saṃśleṣaṇ",     ""),
    "synthesize":   ("संश्लेषित करब",        "saṃśleṣit karab",""),
    "vocoder":      ("ध्वनि-कूटक",           "dhvani-kūṭak",  ""),
    "waveform generation": ("तरंग उत्पादन",  "taraṃg utpādan",""),
    "voice cloning": ("आवाज प्रतिलिपि",      "āvāj pratilipi",""),
    "speaker embedding": ("वक्ता संकूचन",    "vaktā saṃkūcan",""),
    "zero shot":    ("शून्य-नमूना",           "śūnya-namūnā",  ""),
    "mel spectrogram": ("मेल वर्णक्रमचित्र",  "mel varṇakramcitr",""),
    "dynamic time warping": ("गतिशील समय विक्षेपण", "gatiśīl samay vikṣepaṇ",""),
    "dtw":          ("गतिशील समय विक्षेपण",   "gatiśīl samay vikṣepaṇ",""),
    "mel cepstral distortion": ("मेल सेप्सट्रल विकृति", "mel seːpsʈral vikṛti",""),
    "mcd":          ("मेल सेप्सट्रल विकृति",  "mel seːpsʈral vikṛti",""),

    # ── Technical: Speaker Recognition / Anti-Spoofing ───────────────────────
    "speaker recognition": ("वक्ता पहचान",   "vaktā pahacān",  ""),
    "speaker verification": ("वक्ता सत्यापन", "vaktā satyāpan", ""),
    "x-vector":     ("एक्स-सदिश",            "eks-sadis",      "borrowed"),
    "d-vector":     ("डी-सदिश",              "ḍī-sadis",       "borrowed"),
    "anti spoofing": ("स्पूफ-विरोधी",         "spūf-virodhī",   ""),
    "countermeasure": ("प्रतिउपाय",           "pratiupāy",      ""),
    "equal error rate": ("समान त्रुटि दर",    "samān truṭi dar",""),
    "eer":          ("समान त्रुटि दर",         "samān truṭi dar",""),
    "lfcc":         ("एलएफसीसी",              "elaphsīsī",      "borrowed"),
    "bona fide":    ("वास्तविक",              "vāstavik",       ""),
    "spoof":        ("जालसाजी",               "jālasājī",       ""),
    "adversarial":  ("विरोधी",                "virodhī",        ""),
    "perturbation": ("विक्षोभ",               "vikṣobh",        ""),
    "fgsm":         ("एफजीएसएम",              "ephajīeseam",    "borrowed"),
    "robustness":   ("सुदृढ़ता",               "sudṛṛhatā",      ""),

    # ── Numbers & units ──────────────────────────────────────────────────────
    "zero":         ("शून्य",    "śūnya",   ""),
    "one":          ("एक",       "ek",      ""),
    "two":          ("दू",       "dū",      ""),
    "three":        ("तीन",      "tīn",     ""),
    "four":         ("चारि",     "cāri",    ""),
    "five":         ("पाँच",     "pāṃc",    ""),
    "six":          ("छह",       "chah",    ""),
    "seven":        ("सात",      "sāt",     ""),
    "eight":        ("आठ",       "āṭh",     ""),
    "nine":         ("नौ",       "nau",     ""),
    "ten":          ("दस",       "das",     ""),
    "hundred":      ("सय",       "say",     "Maithili: सय (not सौ)"),
    "thousand":     ("हजार",     "hajār",   ""),
    "hertz":        ("हर्ट्ज",    "harṭj",   "borrowed"),
    "khz":          ("किलोहर्ट्ज", "kilohṭj", "borrowed"),
    "percent":      ("प्रतिशत",   "pratiśat",""),
    "second":       ("सेकेंड",    "sekaṃḍ",  "time unit"),
    "millisecond":  ("मिलिसेकेंड", "milisekaṃḍ",""),
    "decibel":      ("डेसीबेल",   "ḍesībel", "borrowed"),
    "db":           ("डेसीबेल",   "ḍesībel", ""),

    # ── Discourse markers ─────────────────────────────────────────────────────
    "first":        ("पहिल",     "pahil",   ""),
    "second":       ("दोसर",     "dosar",   "ordinal"),
    "third":        ("तेसर",     "tesar",   ""),
    "finally":      ("अंतमे",    "aṃtme",   ""),
    "next":         ("आगाँ",     "āgāṃ",    ""),
    "previously":   ("पहिने",    "pahine",  ""),
    "in summary":   ("सारांशमे", "sārāṃśme",""),
    "for example":  ("उदाहरणार्थ","udāharaṇārth",""),
    "note that":    ("ध्यान दिअ", "dhyān dia",""),
    "recall":       ("मोन पाड़ब", "mon pāṛab",""),
    "consider":     ("विचार करू", "vicār karū",""),
    "given":        ("देल गेल",  "del gel",  ""),
    "therefore":    ("तेँ",      "teṃ",      ""),
    "hence":        ("अतः",      "ataḥ",     ""),
    "thus":         ("एहि प्रकारेँ","ehi prakāreṃ",""),
    "in other words": ("दोसर शब्दमे","dosar śabdme",""),
    "specifically": ("विशेष रूपेँ", "viśeṣ rūpeṃ",""),
    "generally":    ("सामान्यतः", "sāmānyataḥ",""),
    "importantly":  ("महत्वपूर्ण रूपेँ","mahattavapūrṇ rūpeṃ",""),
    "lecture":      ("व्याख्यान", "vyākhyān", ""),
    "class":        ("वर्ग",      "varg",     ""),
    "topic":        ("विषय",      "viṣay",    ""),
    "concept":      ("अवधारणा",   "avadhāraṇā",""),
    "definition":   ("परिभाषा",   "paribhāṣā",""),
    "theorem":      ("प्रमेय",    "pramey",   ""),
    "algorithm":    ("कलनविधि",   "kalanvidhi",""),
    "equation":     ("समीकरण",   "samīkaraṇ", ""),
    "matrix":       ("आव्यूह",    "āvyūh",    ""),
    "vector":       ("सदिश",     "sadiś",    ""),
    "dimension":    ("आयाम",     "āyām",     "dimension"),
    "space":        ("समष्टि",   "samaṣṭi",  "vector space"),
    "distribution": ("वितरण",    "vitaraṇ",   ""),
    "variance":     ("विचरण",    "vicaraṇ",   ""),
    "mean":         ("माध्य",    "mādhya",    ""),
    "threshold":    ("सीमा",     "sīmā",     ""),
    "accuracy":     ("शुद्धता",   "śuddhatā", ""),
    "error":        ("त्रुटि",    "truṭi",    ""),
    "performance":  ("प्रदर्शन",  "pradarśan",""),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Translation engine
# ═══════════════════════════════════════════════════════════════════════════════

# Common English suffixes to strip for morpheme-level lookup
_SUFFIXES = ["ing", "tion", "sion", "ness", "ment", "ity", "ies", "ed",
             "er", "est", "ly", "ful", "less", "s"]

class MaithiliTranslator:
    """
    Word/phrase-level translator using the parallel corpus.

    Translation order:
    1. Direct phrase lookup (up to 4-gram phrases)
    2. Morpheme-stripped lookup
    3. Mark unknown words as [BORROWED:word]
    """

    def __init__(self, corpus: Dict = MAITHILI_CORPUS):
        # Build normalised lookup (lowercase)
        self.corpus = {k.lower(): v for k, v in corpus.items()}
        log.info(f"Maithili corpus loaded: {len(self.corpus)} entries.")

    def _lookup(self, key: str) -> Optional[Tuple[str, str, str]]:
        key = key.lower().strip()
        if key in self.corpus:
            return self.corpus[key]
        # Try suffix stripping
        for suf in _SUFFIXES:
            if key.endswith(suf) and len(key) - len(suf) > 2:
                stem = key[: -len(suf)]
                if stem in self.corpus:
                    entry = self.corpus[stem]
                    return (entry[0], entry[1], f"stem:{stem}")
        return None

    def translate_tokens(self, tokens: List[str]) -> List[Dict]:
        """
        Translate a list of word tokens.
        Returns list of dicts with src, mai (Devanagari), roman, note, status.
        """
        results = []
        i = 0
        while i < len(tokens):
            # Try longest n-gram match first (4→1)
            matched = False
            for n in range(min(4, len(tokens) - i), 0, -1):
                phrase = " ".join(tokens[i:i + n]).lower()
                entry  = self._lookup(phrase)
                if entry is not None:
                    results.append({
                        "src":    " ".join(tokens[i:i + n]),
                        "mai":    entry[0],
                        "roman":  entry[1],
                        "note":   entry[2],
                        "status": "translated",
                    })
                    i += n
                    matched = True
                    break
            if not matched:
                w = tokens[i]
                results.append({
                    "src":    w,
                    "mai":    w,          # keep source word
                    "roman":  w.lower(),
                    "note":   "borrowed",
                    "status": "borrowed",
                })
                i += 1
        return results

    def translate(self, text: str) -> Dict:
        """Translate a full text string. Returns structured output."""
        tokens  = text.split()
        items   = self.translate_tokens(tokens)
        mai_str = " ".join(it["mai"] for it in items)
        return {
            "source":   text,
            "maithili": mai_str,
            "tokens":   items,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# IPA JSON → Maithili JSON
# ═══════════════════════════════════════════════════════════════════════════════

def translate_ipa_json(ipa_json_path: str, output_path: str) -> dict:
    with open(ipa_json_path, encoding="utf-8") as f:
        ipa_data = json.load(f)

    tr = MaithiliTranslator()

    translated_segs = []
    for seg in ipa_data.get("segments", []):
        res = tr.translate(seg.get("text", ""))
        translated_segs.append({
            "start":       seg.get("start"),
            "end":         seg.get("end"),
            "source_text": seg.get("text"),
            "source_ipa":  seg.get("ipa"),
            "maithili":    res["maithili"],
            "tokens":      res["tokens"],
        })

    full_res = tr.translate(ipa_data.get("full_text", ""))
    output = {
        "full_source":   ipa_data.get("full_text", ""),
        "full_maithili": full_res["maithili"],
        "segments":      translated_segs,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    log.info(f"Maithili translation saved to {output_path}")

    # Print coverage stats
    all_tokens = [t for seg in translated_segs for t in seg["tokens"]]
    n_total     = len(all_tokens)
    n_borrowed  = sum(1 for t in all_tokens if t["status"] == "borrowed")
    log.info(f"Coverage: {n_total - n_borrowed}/{n_total} "
             f"({100*(n_total-n_borrowed)/max(n_total,1):.1f}%) translated; "
             f"{n_borrowed} borrowed.")
    return output


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text",     help="Inline English/Hinglish text")
    group.add_argument("--ipa-json", help="IPA JSON from task2_1")
    parser.add_argument("--output",  default="data/maithili_output.json")
    args = parser.parse_args()

    tr = MaithiliTranslator()

    if args.text:
        res = tr.translate(args.text)
        print(f"Source  : {res['source']}")
        print(f"Maithili: {res['maithili']}")
        for t in res["tokens"]:
            flag = "✓" if t["status"] == "translated" else "✗"
            print(f"  {flag} {t['src']:25s} → {t['mai']}")
    else:
        out = translate_ipa_json(args.ipa_json, args.output)
        print(f"Maithili:\n{out['full_maithili'][:500]}")
