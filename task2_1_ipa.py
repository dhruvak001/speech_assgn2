#!/usr/bin/env python3
"""
task2_1_ipa.py
==============
Task 2.1 – Unified IPA Representation for Code-Switched Hinglish

Problem
───────
Standard G2P tools handle one language at a time and fail on code-switched
input like "aaj hum log MFCC features ki baat karenge stochastic process ke
baare mein".  We need a single pipeline that:

    1. Detects the language/script of each word (Devanagari, Roman Hindi,
       Roman English, or digit/symbol).
    2. Routes the word to the correct phonemic rule-set.
    3. Emits IPA symbols for the entire utterance.

IPA rule sets implemented
──────────────────────────
• English (Roman) – CMU Pronouncing Dictionary lookup with a regex fallback
  for out-of-vocabulary words.
• Hindi (Devanagari) – direct Unicode → IPA mapping for all ISCII/Devanagari
  consonants, vowels, and matras.
• Romanized Hindi (Hinglish) – custom phonological rules for common
  transliteration conventions used in Indian social-media / lecture speech
  (e.g. "aaj" → /aːdʒ/, "ki" → /kiː/, "karna" → /kərnaː/).

Design choice (non-obvious)
────────────────────────────
Standard IPA converters (espeak, phonemizer) treat the entire input as one
language and collapse code-switching boundaries.  Our implementation uses
**word-level language routing**: each whitespace token is independently
classified (script detection + vocabulary look-up), then processed by the
matching rule-set.  This preserves fine-grained phonological differences at
the switch boundary — critical for downstream TTS synthesis which must adjust
the vocal tract model mid-utterance.

Usage
─────
    python task2_1_ipa.py --text "aaj hum MFCC features ke baare mein padh rahe hain"
    python task2_1_ipa.py --transcript data/transcript.json --output data/ipa_output.json
"""

import os
import re
import json
import unicodedata
import argparse
import logging
from typing import List, Tuple, Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1 – Language / script detection
# ═══════════════════════════════════════════════════════════════════════════════

_DEVANAGARI = re.compile(r"[\u0900-\u097F]+")
_ENGLISH    = re.compile(r"[a-zA-Z]{2,}")   # ≥2 ASCII letters
_DIGIT      = re.compile(r"^\d+$")

# Romanized Hindi function words / particles (closed set)
_ROMAN_HINDI_VOCAB = {
    "aaj","kal","aap","tum","hum","mai","mera","tera","uska","unka","hamara",
    "aur","ya","ki","ke","ka","ko","se","mein","par","ne","hai","hain","tha",
    "the","thi","ho","hoga","hogi","hoge","kar","karna","karein","karte",
    "karenge","kiya","ki","kee","lekin","magar","par","phir","toh","na","nahi",
    "nahin","haa","han","kya","kyun","kaise","kaun","kab","kahan","ek","do",
    "teen","char","paanch","baat","cheez","kaam","log","sab","kuch","bahut",
    "thoda","zyada","sirf","bas","ab","phir","jab","tab","agar","nahi","bilkul",
    "matlab","yani","samajh","samjhe","dekho","suno","padh","likh","bolo",
    "isko","usse","inko","unhe","yeh","woh","yahan","wahan","iska","uska",
    "poora","aadha","shuru","khatam","pehle","baad","saath","bina","liye",
    "zaroor","chahiye","sakta","sakti","sakte","chahte","chahti",
}

def detect_word_lang(word: str) -> str:
    """
    Classify a single whitespace-token.

    Returns: "devanagari" | "roman_hindi" | "english" | "digit" | "mixed"
    """
    clean = re.sub(r"[^\w\u0900-\u097F]", "", word)
    if not clean:
        return "symbol"
    if _DIGIT.match(clean):
        return "digit"
    if _DEVANAGARI.search(clean):
        return "devanagari"
    if clean.lower() in _ROMAN_HINDI_VOCAB:
        return "roman_hindi"
    # Heuristic: ends in typical Hindi suffixes?
    if re.search(r"(aa|ii|oo|uu|ai|au|iya|iye|enge|ogi|oge|ogi|ogi)$",
                 clean.lower()):
        return "roman_hindi"
    return "english"


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2 – Devanagari → IPA
# ═══════════════════════════════════════════════════════════════════════════════

# Consonant map (Unicode codepoint → IPA)
_DEVA_CONSONANT = {
    "क": "k", "ख": "kʰ", "ग": "ɡ", "घ": "ɡʰ", "ङ": "ŋ",
    "च": "tʃ", "छ": "tʃʰ", "ज": "dʒ", "झ": "dʒʰ", "ञ": "ɲ",
    "ट": "ʈ", "ठ": "ʈʰ", "ड": "ɖ", "ढ": "ɖʰ", "ण": "ɳ",
    "त": "t̪", "थ": "t̪ʰ", "द": "d̪", "ध": "d̪ʰ", "न": "n",
    "प": "p", "फ": "pʰ", "ब": "b", "भ": "bʰ", "म": "m",
    "य": "j", "र": "r", "ल": "l", "व": "ʋ",
    "श": "ʃ", "ष": "ʂ", "स": "s", "ह": "ɦ",
    "क्ष": "kʃ", "त्र": "t̪r", "ज्ञ": "dʒɲ",
    "ड़": "ɽ", "ढ़": "ɽʰ",
}

# Vowel / matra map
_DEVA_VOWEL = {
    "अ": "ə", "आ": "aː", "इ": "ɪ", "ई": "iː",
    "उ": "ʊ", "ऊ": "uː", "ऋ": "rɪ", "ए": "eː",
    "ऐ": "æː", "ओ": "oː", "औ": "ɔː",
    # matras
    "ा": "aː", "ि": "ɪ", "ी": "iː",
    "ु": "ʊ", "ू": "uː", "ृ": "rɪ",
    "े": "eː", "ै": "æː", "ो": "oː", "ौ": "ɔː",
    "ं": "◌̃",   # anusvara (nasalisation)
    "ः": "ɦ",   # visarga
    "ँ": "◌̃",   # chandrabindu
    "्": "",    # halant (virama – removes inherent /ə/)
    "ॅ": "æ",  # candra e
    "ॊ": "ɔ",
}

def devanagari_to_ipa(text: str) -> str:
    """Convert a Devanagari word/string to IPA."""
    ipa = []
    i   = 0
    chars = list(text)
    while i < len(chars):
        c = chars[i]
        # Check for two-char consonant cluster (क्ष etc.)
        if i + 1 < len(chars):
            two = c + chars[i + 1]
            if two in _DEVA_CONSONANT:
                ipa.append(_DEVA_CONSONANT[two])
                i += 2
                continue
        if c in _DEVA_CONSONANT:
            ipa.append(_DEVA_CONSONANT[c])
            # Check for following matra
            if i + 1 < len(chars) and chars[i + 1] in _DEVA_VOWEL:
                matra = chars[i + 1]
                if matra == "्":     # virama: suppress inherent vowel
                    pass
                else:
                    ipa.append(_DEVA_VOWEL[matra])
                i += 2
            else:
                # Add inherent vowel /ə/ unless followed by halant
                # (already handled above if next is halant)
                ipa.append("ə")
                i += 1
        elif c in _DEVA_VOWEL:
            ipa.append(_DEVA_VOWEL[c])
            i += 1
        else:
            i += 1   # skip unknown
    return "".join(ipa)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3 – Romanised Hindi → IPA
# ═══════════════════════════════════════════════════════════════════════════════

_ROMAN_HINDI_MAP: List[Tuple[re.Pattern, str]] = [
    # Long vowels (must come BEFORE short rules)
    (re.compile(r"aa"),  "aː"),
    (re.compile(r"ii"),  "iː"),
    (re.compile(r"uu"),  "uː"),
    (re.compile(r"ee"),  "eː"),
    (re.compile(r"oo"),  "oː"),
    (re.compile(r"au"),  "ɔː"),
    (re.compile(r"ai"),  "æː"),
    # Aspirated stops
    (re.compile(r"kh"),  "kʰ"),
    (re.compile(r"gh"),  "ɡʰ"),
    (re.compile(r"ch"),  "tʃ"),
    (re.compile(r"jh"),  "dʒʰ"),
    (re.compile(r"th"),  "t̪ʰ"),
    (re.compile(r"dh"),  "d̪ʰ"),
    (re.compile(r"ph"),  "pʰ"),
    (re.compile(r"bh"),  "bʰ"),
    # Retroflex
    (re.compile(r"[tT]"),  "ʈ"),
    (re.compile(r"[dD]"),  "ɖ"),
    (re.compile(r"[nN]"),  "ɳ"),
    (re.compile(r"[rR]"),  "r"),
    # Sibilants
    (re.compile(r"sh"),  "ʃ"),
    (re.compile(r"s"),   "s"),
    # Simple consonants
    (re.compile(r"k"),   "k"),
    (re.compile(r"g"),   "ɡ"),
    (re.compile(r"j"),   "dʒ"),
    (re.compile(r"t"),   "t̪"),
    (re.compile(r"d"),   "d̪"),
    (re.compile(r"n"),   "n"),
    (re.compile(r"p"),   "p"),
    (re.compile(r"b"),   "b"),
    (re.compile(r"m"),   "m"),
    (re.compile(r"y"),   "j"),
    (re.compile(r"l"),   "l"),
    (re.compile(r"v|w"), "ʋ"),
    (re.compile(r"h"),   "ɦ"),
    (re.compile(r"f"),   "f"),
    (re.compile(r"z"),   "z"),
    # Short vowels (AFTER consonants to avoid mis-matching)
    (re.compile(r"a"),   "ə"),
    (re.compile(r"i"),   "ɪ"),
    (re.compile(r"u"),   "ʊ"),
    (re.compile(r"e"),   "eː"),
    (re.compile(r"o"),   "oː"),
]

def roman_hindi_to_ipa(word: str) -> str:
    """
    Convert a Romanised Hindi word to IPA using ordered regex substitutions.
    Rules are applied left-to-right (longest match first where possible).
    """
    text = word.lower()
    out  = []
    i    = 0
    while i < len(text):
        matched = False
        for pattern, ipa_sym in _ROMAN_HINDI_MAP:
            m = pattern.match(text, i)
            if m:
                out.append(ipa_sym)
                i += m.end() - m.start()
                matched = True
                break
        if not matched:
            out.append(text[i])
            i += 1
    return "".join(out)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4 – English → IPA (CMU dict + regex fallback)
# ═══════════════════════════════════════════════════════════════════════════════

# Minimal CMU-style phoneme → IPA symbol table
_CMU_TO_IPA = {
    "AA": "ɑː", "AE": "æ",  "AH": "ʌ",  "AO": "ɔː", "AW": "aʊ",
    "AY": "aɪ", "B":  "b",  "CH": "tʃ", "D":  "d",   "DH": "ð",
    "EH": "ɛ",  "ER": "ɝ",  "EY": "eɪ", "F":  "f",   "G":  "ɡ",
    "HH": "h",  "IH": "ɪ",  "IY": "iː", "JH": "dʒ",  "K":  "k",
    "L":  "l",  "M":  "m",  "N":  "n",  "NG": "ŋ",   "OW": "oʊ",
    "OY": "ɔɪ", "P":  "p",  "R":  "r",  "S":  "s",   "SH": "ʃ",
    "T":  "t",  "TH": "θ",  "UH": "ʊ",  "UW": "uː",  "V":  "v",
    "W":  "w",  "Y":  "j",  "Z":  "z",  "ZH": "ʒ",
}

_CMU_DICT: Optional[Dict[str, List[str]]] = None

def _load_cmu() -> Dict[str, List[str]]:
    global _CMU_DICT
    if _CMU_DICT is not None:
        return _CMU_DICT
    try:
        import nltk
        try:
            nltk.data.find("corpora/cmudict")
        except LookupError:
            nltk.download("cmudict", quiet=True)
        from nltk.corpus import cmudict
        raw = cmudict.dict()
        _CMU_DICT = {w: phones for w, phones_list in raw.items()
                     for phones in [phones_list[0]]}
        log.info(f"CMU dict loaded: {len(_CMU_DICT)} entries.")
    except Exception as e:
        log.warning(f"CMU dict unavailable ({e}), using regex fallback only.")
        _CMU_DICT = {}
    return _CMU_DICT

def _cmu_phones_to_ipa(phones: List[str]) -> str:
    """Convert a CMU phoneme sequence to IPA string."""
    ipa = []
    for p in phones:
        p_base = re.sub(r"\d", "", p)   # strip stress numbers
        ipa.append(_CMU_TO_IPA.get(p_base, p_base.lower()))
    return "".join(ipa)

# English regex fallback (simplified grapheme→phoneme)
_ENG_REGEX: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"tion"),   "ʃən"),
    (re.compile(r"sion"),   "ʒən"),
    (re.compile(r"ough"),   "oʊ"),
    (re.compile(r"ight"),   "aɪt"),
    (re.compile(r"igh"),    "aɪ"),
    (re.compile(r"ea"),     "iː"),
    (re.compile(r"ee"),     "iː"),
    (re.compile(r"oo"),     "uː"),
    (re.compile(r"ou"),     "aʊ"),
    (re.compile(r"ow"),     "oʊ"),
    (re.compile(r"aw"),     "ɔː"),
    (re.compile(r"ay"),     "eɪ"),
    (re.compile(r"ai"),     "eɪ"),
    (re.compile(r"au"),     "ɔː"),
    (re.compile(r"ck"),     "k"),
    (re.compile(r"ph"),     "f"),
    (re.compile(r"wh"),     "w"),
    (re.compile(r"th"),     "ð"),
    (re.compile(r"ch"),     "tʃ"),
    (re.compile(r"sh"),     "ʃ"),
    (re.compile(r"ng"),     "ŋ"),
    (re.compile(r"nk"),     "ŋk"),
    (re.compile(r"[aeiou]"), "ə"),
    (re.compile(r"[bcdfghjklmnpqrstvwxyz]"), lambda m: m.group()),
]

def english_to_ipa(word: str) -> str:
    """CMU dict lookup with regex fallback."""
    cmu = _load_cmu()
    lw  = word.lower()
    if lw in cmu:
        return _cmu_phones_to_ipa(cmu[lw])
    # Regex fallback
    text = lw
    out  = []
    i    = 0
    while i < len(text):
        matched = False
        for pattern, sub in _ENG_REGEX:
            m = pattern.match(text, i)
            if m:
                if callable(sub):
                    out.append(sub(m))
                else:
                    out.append(sub)
                i += m.end() - m.start()
                matched = True
                break
        if not matched:
            out.append(text[i])
            i += 1
    return "".join(out)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5 – Main HinglishG2P interface
# ═══════════════════════════════════════════════════════════════════════════════

class HinglishG2P:
    """
    Unified Grapheme-to-Phoneme converter for code-switched Hinglish text.

    Returns a list of (word, language_tag, ipa) tuples.
    """

    def convert_word(self, word: str) -> Tuple[str, str, str]:
        lang = detect_word_lang(word)
        if lang == "devanagari":
            ipa = devanagari_to_ipa(word)
        elif lang == "roman_hindi":
            ipa = roman_hindi_to_ipa(word)
        elif lang == "english":
            ipa = english_to_ipa(word)
        elif lang == "digit":
            ipa = word   # keep digits as-is
        else:
            ipa = word
        return (word, lang, ipa)

    def convert(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Convert a full text string to a list of (word, lang, IPA) tuples.
        """
        tokens = text.split()
        return [self.convert_word(tok) for tok in tokens]

    def to_ipa_string(self, text: str) -> str:
        """Return full utterance as a single IPA string (space-separated)."""
        return " ".join(ipa for _, _, ipa in self.convert(text))


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6 – Transcript → IPA JSON
# ═══════════════════════════════════════════════════════════════════════════════

def process_transcript(transcript_path: str, output_path: str) -> dict:
    """
    Read a transcript JSON (from task1_2) and add IPA fields to each segment.
    """
    with open(transcript_path, encoding="utf-8") as f:
        transcript = json.load(f)

    g2p = HinglishG2P()
    ipa_segs = []

    for seg in transcript.get("segments", []):
        text = seg.get("text", "").strip()
        items = g2p.convert(text)
        ipa_segs.append({
            "start":   seg.get("start"),
            "end":     seg.get("end"),
            "text":    text,
            "ipa":     " ".join(ipa for _, _, ipa in items),
            "tokens":  [{"word": w, "lang": l, "ipa": i} for w, l, i in items],
        })

    output = {
        "full_text": transcript.get("full_text", ""),
        "full_ipa":  g2p.to_ipa_string(transcript.get("full_text", "")),
        "segments":  ipa_segs,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    log.info(f"IPA output saved to {output_path}")
    return output


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text",       help="Inline text to convert")
    group.add_argument("--transcript", help="Transcript JSON from task1_2")
    parser.add_argument("--output", default="data/ipa_output.json")
    args = parser.parse_args()

    g2p = HinglishG2P()

    if args.text:
        result = g2p.convert(args.text)
        for word, lang, ipa in result:
            print(f"{word:20s}  [{lang:12s}]  /{ipa}/")
        print("\nFull IPA:", g2p.to_ipa_string(args.text))
    else:
        out = process_transcript(args.transcript, args.output)
        print(f"Full IPA:\n{out['full_ipa'][:500]}")
