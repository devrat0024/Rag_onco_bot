
#!/usr/bin/env python3
"""
translator_local.py — Local-only translator (no API providers)
---------------------------------------------------------------
Priority:
  1) Local NLLB-200 (facebook/nllb-200-distilled-600M) if installed
  2) deep_translator.GoogleTranslator (no API key required)
  3) Fallback to original text (tagged)

Notes:
  - No external API calls that require keys.
  - Includes glossary pre/post to stabilize domain terms.
  - Optional language detection via langdetect (safe to omit).

Install (optional deps):
  pip install transformers torch sentencepiece  # for NLLB
  pip install deep-translator                  # fallback
  pip install langdetect                       # optional detection
"""
import os, re, json, time, typing as T, warnings
from dataclasses import dataclass, field
from difflib import SequenceMatcher

# Optional deps
_has_torch = _has_hf = False
try:
    import torch
    from transformers import NllbTokenizer, AutoModelForSeq2SeqLM
    _has_torch = True
    _has_hf = True
except Exception:
    pass

try:
    from langdetect import detect as _detect_lang
except Exception:
    _detect_lang = None

try:
    from deep_translator import GoogleTranslator as _GT
    _has_deep_translator = True
except Exception:
    _has_deep_translator = False

NLLB_MODEL_NAME = os.getenv("NLLB_MODEL_NAME", "facebook/nllb-200-distilled-600M")

# Label/code → NLLB code map (extend as needed)
NLLB_MAP: T.Dict[str, str] = {
    "EN": "eng_Latn", "English": "eng_Latn",
    "HI": "hin_Deva", "Hindi": "hin_Deva",
    "BN": "ben_Beng", "Bengali": "ben_Beng",
    "GU": "guj_Gujr", "Gujarati": "guj_Gujr",
    "KN": "kan_Knda", "Kannada": "kan_Knda",
    "ML": "mal_Mlym", "Malayalam": "mal_Mlym",
    "MR": "mar_Deva", "Marathi": "mar_Deva",
    "PA": "pan_Guru", "Punjabi": "pan_Guru",
    "TA": "tam_Taml", "Tamil": "tam_Taml",
    "TE": "tel_Telu", "Telugu": "tel_Telu",
    "UR": "urd_Arab", "Urdu": "urd_Arab",
    "OR": "ory_Orya", "Odia": "ory_Orya",
    "AS": "asm_Beng", "Assamese": "asm_Beng",
}


def _to_nllb_code(x: str) -> T.Optional[str]:
    if not x:
        return None
    x = x.strip()
    return NLLB_MAP.get(x, NLLB_MAP.get(x.upper()))


def _compile_glossary(glossary: T.Dict[str, str] = None):
    glossary = glossary or {}
    pairs = []
    for k, v in glossary.items():
        pat = re.compile(r"\b" + re.escape(k) + r"\b", flags=re.IGNORECASE)
        pairs.append((pat, v))
    return pairs


def _apply_glossary(text: str, compiled_pairs) -> str:
    if not text or not compiled_pairs:
        return text
    out = text
    for pat, repl in compiled_pairs:
        out = pat.sub(repl, out)
    return out


@dataclass
class TranslationOutput:
    translated_text: str
    provider: str
    source_lang: str
    target_lang: str
    detected_source_lang: T.Optional[str] = None
    latency_ms: int = 0
    quality_flags: T.Dict[str, T.Any] = field(default_factory=dict)


class LocalTranslator:
    def __init__(self, glossary: T.Optional[T.Dict[str, str]] = None, round_trip_check: bool = False, round_trip_threshold: float = 0.72):
        self._glossary_pairs = _compile_glossary(glossary or {})
        self.round_trip_check = round_trip_check
        self.round_trip_threshold = round_trip_threshold

        # Lazy NLLB
        self._tok = None
        self._model = None
        self._device = None

    def detect_language(self, text: str) -> str:
        if _detect_lang is None:
            return "unknown"
        try:
            return _detect_lang(text)
        except Exception:
            return "unknown"

    def _ensure_nllb(self):
        if self._model is not None:
            return
        if not (_has_torch and _has_hf):
            raise RuntimeError("Install torch + transformers + sentencepiece for local NLLB support.")
        self._tok = NllbTokenizer.from_pretrained(NLLB_MODEL_NAME)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_NAME)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = dev
        self._model.to(dev).eval()

    def _nllb_translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        self._ensure_nllb()
        src = _to_nllb_code(src_lang) or _to_nllb_code("EN")
        tgt = _to_nllb_code(tgt_lang) or _to_nllb_code("EN")
        if not src or not tgt:
            raise ValueError("Unsupported language for NLLB")
        self._tok.src_lang = src
        encoded = self._tok(text, return_tensors="pt").to(self._device)
        forced_bos_token_id = self._tok.convert_tokens_to_ids(tgt)
        with torch.no_grad():
            out = self._model.generate(**encoded, forced_bos_token_id=forced_bos_token_id, max_length=256)
        return self._tok.decode(out[0], skip_special_tokens=True)

    def translate(self, text: str, target_lang: str, source_lang: T.Optional[str] = None, use_glossary: bool = True) -> TranslationOutput:
        if not text or not target_lang:
            raise ValueError("text and target_lang are required")
        detected = None
        if not source_lang:
            detected = self.detect_language(text)
        src_lang = source_lang or detected or "unknown"

        # glossary pre
        in_text = _apply_glossary(text, self._glossary_pairs) if use_glossary else text

        t0 = time.time()
        provider_used = None
        translated = None

        # 1) Try local NLLB
        try:
            translated = self._nllb_translate(in_text, src_lang, target_lang)
            provider_used = "nllb-local"
        except Exception as e:
            # 2) deep_translator fallback (no API keys)
            if _has_deep_translator:
                try:
                    translated = _GT(source='auto', target=target_lang).translate(in_text)
                    provider_used = "deep_translator"
                except Exception:
                    translated = None
            if translated is None:
                translated = f"[UNTRANSLATED to {target_lang}] {in_text}"
                provider_used = "none"

        latency = int((time.time() - t0) * 1000)

        # glossary post
        out_text = _apply_glossary(translated, self._glossary_pairs) if use_glossary else translated

        return TranslationOutput(
            translated_text=out_text,
            provider=provider_used,
            source_lang=src_lang,
            target_lang=target_lang,
            detected_source_lang=detected,
            latency_ms=latency
        )






