"""
translation/translator.py
=========================
Dictionary-based Gujarati translator with IPA-first transliteration fallback.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

from phonetics.g2p import G2PConverter

from .lexicon import GUJARATI_LEXICON

log = logging.getLogger(__name__)

TOKEN_PATTERN = re.compile(
    r"[\u0900-\u097F]+|[\u0A80-\u0AFF]+|[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\w\s]",
    re.UNICODE,
)


class Translator:
    """Translate Hinglish text into Gujarati using lookup plus transliteration."""

    _DEVANAGARI_TO_GUJARATI = {
        "अ": "અ", "आ": "આ", "इ": "ઇ", "ई": "ઈ", "उ": "ઉ", "ऊ": "ઊ", "ऋ": "ઋ",
        "ए": "એ", "ऐ": "ઐ", "ओ": "ઓ", "औ": "ઔ",
        "क": "ક", "ख": "ખ", "ग": "ગ", "घ": "ઘ", "ङ": "ઙ",
        "च": "ચ", "छ": "છ", "ज": "જ", "झ": "ઝ", "ञ": "ઞ",
        "ट": "ટ", "ठ": "ઠ", "ड": "ડ", "ढ": "ઢ", "ण": "ણ",
        "त": "ત", "थ": "થ", "द": "દ", "ध": "ધ", "न": "ન",
        "प": "પ", "फ": "ફ", "ब": "બ", "भ": "ભ", "म": "મ",
        "य": "ય", "र": "ર", "ल": "લ", "व": "વ", "श": "શ",
        "ष": "ષ", "स": "સ", "ह": "હ", "ळ": "ળ",
        "ा": "ા", "ि": "િ", "ी": "ી", "ु": "ુ", "ू": "ૂ", "ृ": "ૃ",
        "े": "ે", "ै": "ૈ", "ो": "ો", "ौ": "ૌ", "ं": "ં", "ँ": "ઁ", "ः": "ઃ",
        "्": "્", "़": "", "ऽ": "",
        "०": "૦", "१": "૧", "२": "૨", "३": "૩", "४": "૪",
        "५": "૫", "६": "૬", "७": "૭", "८": "૮", "९": "૯",
    }
    _LATIN_DIGRAPHS: Tuple[Tuple[str, str], ...] = (
        ("ksh", "ક્ષ"),
        ("gny", "જ્ઞ"),
        ("chh", "છ"),
        ("sh", "શ"),
        ("kh", "ખ"),
        ("gh", "ઘ"),
        ("ch", "ચ"),
        ("jh", "ઝ"),
        ("th", "થ"),
        ("dh", "ધ"),
        ("ph", "ફ"),
        ("bh", "ભ"),
        ("aa", "આ"),
        ("ee", "ઈ"),
        ("ii", "ઈ"),
        ("oo", "ઊ"),
        ("uu", "ઊ"),
        ("ai", "ઐ"),
        ("au", "ઔ"),
    )
    _LATIN_SINGLE = {
        "a": "અ", "b": "બ", "c": "ક", "d": "દ", "e": "એ", "f": "ફ", "g": "ગ",
        "h": "હ", "i": "ઇ", "j": "જ", "k": "ક", "l": "લ", "m": "મ", "n": "ન",
        "o": "ઓ", "p": "પ", "q": "ક", "r": "ર", "s": "સ", "t": "ત", "u": "ઉ",
        "v": "વ", "w": "વ", "x": "ક્સ", "y": "ય", "z": "ઝ",
    }
    _IPA_DIGRAPHS: Tuple[Tuple[str, str], ...] = (
        ("tʃʰ", "છ"),
        ("dʒʱ", "ઝ"),
        ("tʃ", "ચ"),
        ("dʒ", "જ"),
        ("kʰ", "ખ"),
        ("ɡʱ", "ઘ"),
        ("t̪ʰ", "થ"),
        ("d̪ʱ", "ધ"),
        ("t̪", "ત"),
        ("d̪", "દ"),
        ("ʈʰ", "ઠ"),
        ("ɖʱ", "ઢ"),
        ("ʈ", "ટ"),
        ("ɖ", "ડ"),
        ("pʰ", "ફ"),
        ("bʱ", "ભ"),
        ("aɪ", "આઇ"),
        ("aʊ", "આઉ"),
        ("eɪ", "એઇ"),
        ("oʊ", "ઓ"),
        ("ɔɪ", "ઓઇ"),
        ("aː", "આ"),
        ("iː", "ઈ"),
        ("uː", "ઊ"),
        ("eː", "એ"),
        ("ɛː", "એ"),
        ("oː", "ઓ"),
        ("ɔː", "ઓ"),
        ("rɪ", "રિ"),
        ("ks", "ક્સ"),
    )
    _IPA_SINGLE = {
        "ɑ": "આ",
        "æ": "એ",
        "ə": "અ",
        "ʌ": "અ",
        "ɒ": "ઓ",
        "ɔ": "ઓ",
        "ɛ": "એ",
        "ɪ": "ઇ",
        "i": "ઈ",
        "ʊ": "ઉ",
        "u": "ઊ",
        "b": "બ",
        "d": "દ",
        "ð": "ધ",
        "f": "ફ",
        "ɡ": "ગ",
        "h": "હ",
        "j": "ય",
        "k": "ક",
        "l": "લ",
        "m": "મ",
        "n": "ન",
        "ŋ": "ઙ",
        "ɲ": "ઞ",
        "ɳ": "ણ",
        "p": "પ",
        "ɹ": "ર",
        "r": "ર",
        "ɾ": "ર",
        "ɽ": "ડ",
        "s": "સ",
        "ʃ": "શ",
        "ʂ": "ષ",
        "t": "ટ",
        "θ": "થ",
        "ʋ": "વ",
        "v": "વ",
        "w": "વ",
        "x": "ખ",
        "ɣ": "ગ",
        "z": "ઝ",
        "ʒ": "ઝ",
        "ɦ": "હ",
        "̃": "ં",
    }

    def __init__(self, custom_dictionary: Optional[Dict[str, str]] = None) -> None:
        self.dictionary = self.load_dictionary(custom_dictionary)
        self.g2p = G2PConverter(boundary_marker=" ", unknown_token="?")

    def load_dictionary(self, custom_dictionary: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Load the base Gujarati lexicon with optional overrides."""
        dictionary = dict(GUJARATI_LEXICON)
        if custom_dictionary:
            dictionary.update({key.lower(): value for key, value in custom_dictionary.items()})
        return dictionary

    def translate_text(self, text: str) -> str:
        """Translate text word-by-word, transliterating unknown tokens."""
        tokens = TOKEN_PATTERN.findall(text)
        translated: List[str] = []

        for token in tokens:
            if re.fullmatch(r"[^\w\s]", token):
                translated.append(token)
                continue

            key = token.lower()
            if key in self.dictionary:
                translated.append(self.dictionary[key])
                continue

            if token.isdigit():
                translated.append(self._convert_digits(token))
                continue

            translated.append(self.transliterate_unknown_words(token))

        return self._join_tokens(translated)

    def transliterate_unknown_words(self, word: str) -> str:
        """Fallback transliteration for OOV words into Gujarati script."""
        if re.search(r"[\u0A80-\u0AFF]", word):
            return word

        ipa_result = self._transliterate_via_ipa(word)
        if ipa_result is not None:
            return ipa_result

        if re.search(r"[\u0900-\u097F]", word):
            return "".join(self._DEVANAGARI_TO_GUJARATI.get(ch, ch) for ch in word)
        return self._latin_to_gujarati(word)

    def _transliterate_via_ipa(self, word: str) -> Optional[str]:
        """Try IPA from the phonetics module before falling back further."""
        try:
            ipa = self.g2p.convert(word).strip()
        except Exception:
            log.debug("IPA conversion failed for %r", word, exc_info=True)
            return None

        if not ipa or "?" in ipa:
            return None

        if not re.search(r"[A-Za-zəɑæɔɛɪʊʌʃʒʰʱŋɲɳɽɦ̃ːɡʋɹɾθð]", ipa):
            return None

        transliterated = self._ipa_to_gujarati(ipa)
        return transliterated or None

    def _ipa_to_gujarati(self, ipa: str) -> str:
        text = ipa.replace("|", " ").replace(" ", "")
        output: List[str] = []
        i = 0
        while i < len(text):
            matched = False
            for source, target in self._IPA_DIGRAPHS:
                if text.startswith(source, i):
                    output.append(target)
                    i += len(source)
                    matched = True
                    break
            if matched:
                continue

            symbol = text[i]
            if symbol == "ː":
                i += 1
                continue

            mapped = self._IPA_SINGLE.get(symbol)
            if mapped is None:
                return ""
            output.append(mapped)
            i += 1
        return "".join(output)

    def _latin_to_gujarati(self, word: str) -> str:
        text = word.lower()
        output: List[str] = []
        i = 0
        while i < len(text):
            matched = False
            for source, target in self._LATIN_DIGRAPHS:
                if text.startswith(source, i):
                    output.append(target)
                    i += len(source)
                    matched = True
                    break
            if matched:
                continue
            output.append(self._LATIN_SINGLE.get(text[i], text[i]))
            i += 1
        return "".join(output)

    @staticmethod
    def _convert_digits(token: str) -> str:
        digits = str.maketrans("0123456789", "૦૧૨૩૪૫૬૭૮૯")
        return token.translate(digits)

    @staticmethod
    def _join_tokens(tokens: List[str]) -> str:
        text = " ".join(tokens)
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        text = re.sub(r"\(\s+", "(", text)
        text = re.sub(r"\s+\)", ")", text)
        return re.sub(r"\s{2,}", " ", text).strip()
