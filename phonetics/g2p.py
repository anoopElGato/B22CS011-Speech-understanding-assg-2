"""
phonetics/g2p.py
================
Rule-based + dictionary-based G2P for Hinglish text.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

DEVANAGARI_PATTERN = re.compile(r"[\u0900-\u097F]")
LATIN_PATTERN = re.compile(r"[A-Za-z]")
TOKEN_PATTERN = re.compile(
    r"[\u0900-\u097F]+|[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\w\s]",
    flags=re.UNICODE,
)


class G2PConverter:
    """Convert Hinglish text into an IPA string."""

    _INDEPENDENT_VOWELS: Dict[str, str] = {
        "अ": "ə",
        "आ": "aː",
        "इ": "ɪ",
        "ई": "iː",
        "उ": "ʊ",
        "ऊ": "uː",
        "ऋ": "rɪ",
        "ए": "eː",
        "ऐ": "ɛː",
        "ओ": "oː",
        "औ": "ɔː",
        "ऑ": "ɔ",
    }
    _MATRAS: Dict[str, str] = {
        "ा": "aː",
        "ि": "ɪ",
        "ी": "iː",
        "ु": "ʊ",
        "ू": "uː",
        "ृ": "rɪ",
        "े": "eː",
        "ै": "ɛː",
        "ो": "oː",
        "ौ": "ɔː",
        "ॉ": "ɔ",
        "ॅ": "æ",
    }
    _CONSONANTS: Dict[str, str] = {
        "क": "k",
        "ख": "kʰ",
        "ग": "ɡ",
        "घ": "ɡʱ",
        "ङ": "ŋ",
        "च": "tʃ",
        "छ": "tʃʰ",
        "ज": "dʒ",
        "झ": "dʒʱ",
        "ञ": "ɲ",
        "ट": "ʈ",
        "ठ": "ʈʰ",
        "ड": "ɖ",
        "ढ": "ɖʱ",
        "ण": "ɳ",
        "त": "t̪",
        "थ": "t̪ʰ",
        "द": "d̪",
        "ध": "d̪ʱ",
        "न": "n",
        "प": "p",
        "फ": "pʰ",
        "ब": "b",
        "भ": "bʱ",
        "म": "m",
        "य": "j",
        "र": "r",
        "ल": "l",
        "व": "ʋ",
        "श": "ʃ",
        "ष": "ʂ",
        "स": "s",
        "ह": "ɦ",
        "ड़": "ɽ",
        "ढ़": "ɽʱ",
        "ऴ": "ɭ",
        "क़": "q",
        "ख़": "x",
        "ग़": "ɣ",
        "ज़": "z",
        "फ़": "f",
        "ऱ": "ɾ",
    }
    _DIACRITICS: Dict[str, str] = {
        "ं": "̃",
        "ँ": "̃",
        "ः": "h",
        "ऽ": "",
    }
    _ARPABET_TO_IPA: Dict[str, str] = {
        "AA": "ɑ",
        "AE": "æ",
        "AH": "ʌ",
        "AO": "ɔ",
        "AW": "aʊ",
        "AY": "aɪ",
        "B": "b",
        "CH": "tʃ",
        "D": "d",
        "DH": "ð",
        "EH": "ɛ",
        "ER": "ɝ",
        "EY": "eɪ",
        "F": "f",
        "G": "ɡ",
        "HH": "h",
        "IH": "ɪ",
        "IY": "i",
        "JH": "dʒ",
        "K": "k",
        "L": "l",
        "M": "m",
        "N": "n",
        "NG": "ŋ",
        "OW": "oʊ",
        "OY": "ɔɪ",
        "P": "p",
        "R": "ɹ",
        "S": "s",
        "SH": "ʃ",
        "T": "t",
        "TH": "θ",
        "UH": "ʊ",
        "UW": "u",
        "V": "v",
        "W": "w",
        "Y": "j",
        "Z": "z",
        "ZH": "ʒ",
    }
    _ENGLISH_DIGRAPHS: Sequence[Tuple[str, str]] = (
        ("tion", "ʃən"),
        ("sion", "ʒən"),
        ("ough", "oʊ"),
        ("eigh", "eɪ"),
        ("igh", "aɪ"),
        ("dge", "dʒ"),
        ("tch", "tʃ"),
        ("ch", "tʃ"),
        ("sh", "ʃ"),
        ("ph", "f"),
        ("th", "θ"),
        ("dh", "ð"),
        ("ng", "ŋ"),
        ("qu", "kw"),
        ("ck", "k"),
        ("wh", "w"),
        ("oo", "uː"),
        ("ee", "iː"),
        ("ea", "iː"),
        ("ai", "eɪ"),
        ("ay", "eɪ"),
        ("oa", "oʊ"),
        ("ow", "aʊ"),
        ("oi", "ɔɪ"),
        ("oy", "ɔɪ"),
        ("au", "ɔː"),
        ("ou", "aʊ"),
    )
    _ENGLISH_SINGLE: Dict[str, str] = {
        "a": "æ",
        "b": "b",
        "c": "k",
        "d": "d",
        "e": "ɛ",
        "f": "f",
        "g": "ɡ",
        "h": "h",
        "i": "ɪ",
        "j": "dʒ",
        "k": "k",
        "l": "l",
        "m": "m",
        "n": "n",
        "o": "ɒ",
        "p": "p",
        "q": "k",
        "r": "ɹ",
        "s": "s",
        "t": "t",
        "u": "ʌ",
        "v": "v",
        "w": "w",
        "x": "ks",
        "y": "j",
        "z": "z",
    }
    _ROMANIZED_HINDI: Dict[str, str] = {
        "acha": "ətʃʰaː",
        "achha": "ətʃʰaː",
        "arey": "əreː",
        "arre": "əreː",
        "aur": "ɔːr",
        "bhai": "bʱaːi",
        "chai": "tʃaːi",
        "chalo": "tʃəloː",
        "didi": "diːdiː",
        "dost": "d̪oːst̪",
        "guru": "ɡʊruː",
        "hai": "ɦɛː",
        "haan": "ɦaː̃",
        "jaldi": "dʒəldiː",
        "kaise": "kɛːseː",
        "kya": "kjaː",
        "namaste": "nəməsteː",
        "namaskar": "nəməskɑːr",
        "nahi": "nəɦiː",
        "nahin": "nəɦĩː",
        "paani": "paːniː",
        "sahi": "səɦiː",
        "shukriya": "ʃʊkrɪjaː",
        "yaar": "jaːr",
    }
    _CUSTOM_OVERRIDES: Dict[str, str] = {
        "ios": "aɪoʊɛs",
        "wifi": "waɪfaɪ",
        "youtube": "juːtjuːb",
        "google": "ɡuːɡəl",
        "openai": "oʊpənaɪ",
        "whatsapp": "wɒtsæp",
    }

    def __init__(
        self,
        boundary_marker: str = " | ",
        unknown_token: str = "?",
        custom_loanwords: Optional[Dict[str, str]] = None,
        custom_dictionary: Optional[Dict[str, str]] = None,
    ) -> None:
        self.boundary_marker = boundary_marker
        self.unknown_token = unknown_token
        self.loanwords = {**self._ROMANIZED_HINDI, **(custom_loanwords or {})}
        self.custom_dictionary = {
            key.lower(): value for key, value in (custom_dictionary or {}).items()
        }
        self.cmudict = self._load_cmudict()

    def convert(self, text: str) -> str:
        tokens = self.tokenize(text)
        ipa_parts: List[str] = []
        previous_lang: Optional[str] = None

        for token in tokens:
            ipa_token, token_lang, is_word = self._convert_token(token)
            if not ipa_token:
                continue

            if (
                is_word
                and previous_lang
                and token_lang
                and token_lang != previous_lang
                and ipa_parts
            ):
                ipa_parts.append(self.boundary_marker)

            ipa_parts.append(ipa_token)
            if is_word and token_lang:
                previous_lang = token_lang

        return self._format_output(ipa_parts)

    def __call__(self, text: str) -> str:
        return self.convert(text)

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return TOKEN_PATTERN.findall(text)

    def _convert_token(self, token: str) -> Tuple[str, Optional[str], bool]:
        if DEVANAGARI_PATTERN.search(token):
            return self._convert_mixed_script_token(token), "hi", True
        if LATIN_PATTERN.search(token):
            return self._convert_latin_token(token), self._classify_latin_token(token), True
        if token.isdigit():
            return token, None, False
        return token, None, False

    def _convert_mixed_script_token(self, token: str) -> str:
        if DEVANAGARI_PATTERN.fullmatch(token):
            return self.devanagari_to_ipa(token)

        parts: List[str] = []
        current = [token[0]]
        current_script = self._script_of(token[0])
        for ch in token[1:]:
            script = self._script_of(ch)
            if script == current_script:
                current.append(ch)
            else:
                parts.append(self._convert_script_run("".join(current), current_script))
                current = [ch]
                current_script = script
        parts.append(self._convert_script_run("".join(current), current_script))
        return " ".join(part for part in parts if part)

    def _convert_script_run(self, text: str, script: str) -> str:
        if script == "hi":
            return self.devanagari_to_ipa(text)
        if script == "en":
            return self._convert_latin_token(text)
        return text

    @staticmethod
    def _script_of(ch: str) -> str:
        if DEVANAGARI_PATTERN.match(ch):
            return "hi"
        if LATIN_PATTERN.match(ch):
            return "en"
        return "other"

    def _classify_latin_token(self, token: str) -> str:
        lowered = token.lower()
        if lowered in self.loanwords:
            return "hi"
        if lowered in self.custom_dictionary or lowered in self._CUSTOM_OVERRIDES:
            return "en"
        if lowered in self.cmudict:
            return "en"

        hindi_hints = ("aa", "ai", "bh", "dh", "gh", "jh", "kh", "ph", "th", "ya")
        if any(hint in lowered for hint in hindi_hints) and lowered.endswith(
            ("a", "e", "i", "o", "u")
        ):
            return "hi"
        return "en"

    def _convert_latin_token(self, token: str) -> str:
        lowered = token.lower()
        if lowered in self.loanwords:
            return self.loanwords[lowered]
        if lowered in self.custom_dictionary:
            return self.custom_dictionary[lowered]
        if lowered in self._CUSTOM_OVERRIDES:
            return self._CUSTOM_OVERRIDES[lowered]
        if lowered in self.cmudict:
            return self._cmu_to_ipa(self.cmudict[lowered][0])
        return self.english_to_ipa(lowered)

    def devanagari_to_ipa(self, text: str) -> str:
        ipa_units: List[str] = []
        chars = list(text)
        i = 0
        while i < len(chars):
            ch = chars[i]

            if ch in self._INDEPENDENT_VOWELS:
                ipa_units.append(self._INDEPENDENT_VOWELS[ch])
                i += 1
                continue

            if ch in self._CONSONANTS:
                base = self._CONSONANTS[ch]
                next_ch = chars[i + 1] if i + 1 < len(chars) else ""

                if next_ch == "्":
                    ipa_units.append(base)
                    i += 2
                    continue
                if next_ch in self._MATRAS:
                    ipa_units.append(base + self._MATRAS[next_ch])
                    i += 2
                    continue

                ipa_units.append(base + "ə")
                i += 1
                continue

            if ch in self._MATRAS:
                ipa_units.append(self._MATRAS[ch])
                i += 1
                continue

            if ch in self._DIACRITICS:
                ipa_units.append(self._DIACRITICS[ch])
                i += 1
                continue

            if ch == "्":
                i += 1
                continue

            ipa_units.append(ch)
            i += 1

        return self._postprocess_hindi_ipa("".join(ipa_units))

    def english_to_ipa(self, word: str) -> str:
        if not word:
            return self.unknown_token

        silent_e = word.endswith("e") and len(word) > 2 and word[-2] not in "aeiou"
        source = word[:-1] if silent_e else word

        parts: List[str] = []
        i = 0
        while i < len(source):
            matched = False
            for grapheme, ipa in self._ENGLISH_DIGRAPHS:
                if source.startswith(grapheme, i):
                    parts.append(ipa)
                    i += len(grapheme)
                    matched = True
                    break
            if matched:
                continue

            ch = source[i]
            if ch == "c" and i + 1 < len(source) and source[i + 1] in "eiy":
                parts.append("s")
            elif ch == "g" and i + 1 < len(source) and source[i + 1] in "eiy":
                parts.append("dʒ")
            else:
                parts.append(self._ENGLISH_SINGLE.get(ch, self.unknown_token))
            i += 1

        return self._squash_repetitions("".join(parts))

    def _postprocess_hindi_ipa(self, ipa: str) -> str:
        ipa = re.sub(r"ə$", "", ipa)
        ipa = re.sub(r"ə([aːɪiːʊuːeːɛːoːɔːæ])", r"\1", ipa)
        return ipa

    def _cmu_to_ipa(self, phonemes: Iterable[str]) -> str:
        parts: List[str] = []
        for phoneme in phonemes:
            base = re.sub(r"\d", "", phoneme)
            parts.append(self._ARPABET_TO_IPA.get(base, self.unknown_token))
        return "".join(parts)

    @staticmethod
    def _squash_repetitions(ipa: str) -> str:
        return re.sub(r"(.)\1{2,}", r"\1", ipa)

    @staticmethod
    def _format_output(parts: Sequence[str]) -> str:
        text = " ".join(parts)
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        text = re.sub(r"\s+\|\s+", " | ", text)
        return re.sub(r"\s{2,}", " ", text).strip()

    @staticmethod
    def _load_cmudict() -> Dict[str, List[List[str]]]:
        try:
            import cmudict  # type: ignore

            data = cmudict.dict()
            log.info("Loaded CMUdict with %d entries", len(data))
            return data
        except Exception:
            log.warning("CMUdict unavailable; using built-in fallback pronunciations.")
            return {
                "hello": [["HH", "AH0", "L", "OW1"]],
                "language": [["L", "AE1", "NG", "G", "W", "IH0", "JH"]],
                "model": [["M", "AA1", "D", "AH0", "L"]],
                "speech": [["S", "P", "IY1", "CH"]],
                "test": [["T", "EH1", "S", "T"]],
                "world": [["W", "ER1", "L", "D"]],
            }
