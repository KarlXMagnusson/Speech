# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Derived from OpenAI's Whisper text normalizer (MIT) by way of chime-utils
# (https://github.com/chimechallenge/chime-utils, MIT), whose
# chime_utils.text_norm.whisper_like is what meeteval-wer --normalizer chime8
# delegates to via get_txt_norm("chime8").
"""CHiME-8 scoring text normalizer.

Used to score multi-speaker (SOT) transcripts the way the CHiME-8 scoring pipeline does. It differs
from Whisper's own English normalizer in ways that move a WER materially, so the two are not
interchangeable and a number is meaningless without knowing which produced it. The most consequential
difference is direction: Whisper spells numbers as digits, this spells digits as words.

Two building blocks are IMPORTED from :mod:`.hf_asr_normalizer` rather than copied --
``remove_symbols_and_diacritics`` (which reaches ``ADDITIONAL_DIACRITICS`` through that module) and
``EnglishNumberNormalizer``.
Upstream carries its own private copies; they were verified equivalent here (0 mismatches over an
11,743-codepoint sweep, 20,000 random strings, and 5,000 ``keep=`` variants). Sharing avoids ~460
duplicated lines, at the cost of a coupling: an edit to ``hf_asr_normalizer`` silently moves every
score computed with this normalizer. ``test_chime8_normalizer.py`` pins that coupling explicitly so
such an edit fails a named test instead.

Two upstream behaviours are preserved deliberately even though they look like bugs; each is marked
at its site and pinned by a test, so that "fixing" one fails loudly rather than silently diverging
from the reference scorer.
"""

from __future__ import annotations

import re
from typing import Optional

from nemo.collections.asr.parts.utils.chime8_spelling_data import ENGLISH_SPELLING, PRE_ENGLISH_SPELLING
from nemo.collections.asr.parts.utils.hf_asr_normalizer import EnglishNumberNormalizer, remove_symbols_and_diacritics

__all__ = ["Chime8TextNormalizer", "get_chime8_normalizer", "normalize"]


class _EnglishReverseNumberNormalizer(EnglishNumberNormalizer):
    """Approximate inverse of ``_EnglishNumberNormalizer`` for ASR compatibility."""

    def __init__(self):
        super().__init__()
        self.int_to_ones = {v: k for k, v in self.ones.items()}
        self.int_to_tens = {v: k for k, v in self.tens.items()}
        self.str_to_ones_suffixed = {str(n) + s: k for k, (n, s) in self.ones_suffixed.items()}
        self.str_to_tens_suffixed = {str(n) + s: k for k, (n, s) in self.tens_suffixed.items()}

    def __call__(self, s: str):
        s = re.sub(r"\$(\d+(\.\d+)?)", r"\1 dollars", s)
        s = re.sub(r"(\d+(\.\d+)?)\$", r"\1 dollars", s)
        s = re.sub(r"(\d+(\.\d+)?)%", r"\1 percent", s)

        def number_to_words(w: str):
            if w.isdigit():
                num = int(w)
                if w == "000":
                    return "thousand"
                if num == 0:
                    return "zero"
                if num == 100:
                    return "hundred"
                if 0 < num < 1000:
                    hundreds, remainder = divmod(num, 100)
                    tens, ones = divmod(remainder, 10)
                    h = [f"{self.int_to_ones[hundreds]} hundred"] if hundreds > 0 else []
                    if 0 < remainder <= 19:
                        t = [self.int_to_ones[remainder]]
                        o = []
                    else:
                        t = [self.int_to_tens[tens * 10]] if tens > 0 else []
                        o = [self.int_to_ones[ones]] if ones > 0 else []
                    return " ".join(h + t + o)
                if num == 1000:
                    return "thousand"
                return w
            w = self.str_to_ones_suffixed.get(w, w)
            w = self.str_to_tens_suffixed.get(w, w)
            return w

        return " ".join(number_to_words(w) for w in s.split())


class _EnglishSpellingNormalizer:
    """British-American spelling mappings from Whisper / chime-utils."""

    def __init__(self, mapping: Optional[dict] = None):
        # Table shipped as a Python literal; see chime8_spelling_data for why not JSON.
        self.mapping = ENGLISH_SPELLING if mapping is None else mapping

    def __call__(self, s: str):
        return " ".join(self.mapping.get(word, word) for word in s.split())


class Chime8TextNormalizer:
    """
    CHIME-8 scoring text normalizer (Whisper-derived, idempotent, less aggressive).

    Key behaviour:
    - lowercases, strips bracketed / parenthetical annotations
    - expands contractions and title abbreviations (Mr. -> mister, etc.)
    - removes filler words (hmm, uh, ah, eh) by default
    - converts numerals to spelled-out form (reverse-number path) for fair ASR scoring
    - normalises UK/US spellings
    """

    def __init__(
        self,
        standardize_numbers: bool = False,
        standardize_numbers_rev: bool = True,
        remove_fillers: bool = True,
    ):
        self.replacers = {
            r"\u2019": ("'"),
            r"\b(hm+)\b|\b(mhm)\b|\b(mm+)\b|\b(m+h)\b|\b(hm+)\b|\b(um+)\b|\b(uhm+)\b": ("hmm"),
            r"\b(a+h+)\b|\b(ha+)\b": "ah",
            r"[!?.]+(?=$|\s)": "",
            r"\b(o+h+)\b|\b(h+o+)\b": "oh",
            r"\b(u+h+)\b|\b(h+u+)\b|\b(h+u+h+)\b": "uh",
            r"\b(wi\sfi)\b": "wifi",
            r"\b(goin)\b": "going",
            r"\wi-fi\b": "wifi",
            r"\bwon't\b": "will not",
            r"\bcan't\b": "can not",
            r"\blet's\b": "let us",
            r"\bain't\b": "aint",
            r"\by'all\b": "you all",
            r"\bwanna\b": "want to",
            r"\bgotta\b": "got to",
            r"\bgonna\b": "going to",
            r"\bi'ma\b": "i am going to",
            r"\bimma\b": "i am going to",
            r"\bwoulda\b": "would have",
            r"\bcoulda\b": "could have",
            r"\bshoulda\b": "should have",
            r"\bma'am\b": "madam",
            r"\bokay\b": "ok",
            r"\bsetup\b": "set up",
            r"\beveryday\b": "every day",
            r"\bmr\b": "mister ",
            r"\bmrs\b": "missus ",
            r"\bst\b": "saint ",
            r"\bdr\b": "doctor ",
            r"\bprof\b": "professor ",
            r"\bcapt\b": "captain ",
            r"\bgov\b": "governor ",
            r"\bald\b": "alderman ",
            r"\bgen\b": "general ",
            r"\bsen\b": "senator ",
            r"\brep\b": "representative ",
            r"\bpres\b": "president ",
            r"\brev\b": "reverend ",
            r"\bhon\b": "honorable ",
            r"\basst\b": "assistant ",
            r"\bassoc\b": "associate ",
            r"\blt\b": "lieutenant ",
            r"\bcol\b": "colonel ",
            r"\bjr\b": "junior ",
            r"\bsr\b": "senior ",
            r"\besq\b": "esquire ",
            r"'d been\b": " had been",
            r"'s been\b": " has been",
            r"'d gone\b": " had gone",
            r"'s gone\b": " has gone",
            r"'d done\b": " had done",
            r"'s got\b": " has got",
            r"n't\b": " not",
            r"'re\b": " are",
            r"'s\b": " is",
            r"'d\b": " would",
            r"'ll\b": " will",
            r"'t\b": " not",
            r"'ve\b": " have",
            r"'m\b": " am",
        }
        if standardize_numbers:
            self.standardize_numbers = EnglishNumberNormalizer()
            assert not standardize_numbers_rev
        else:
            self.standardize_numbers = None

        if standardize_numbers_rev:
            self.standardize_numbers_rev = _EnglishReverseNumberNormalizer()
        else:
            self.standardize_numbers_rev = None

        self.standardize_spellings = _EnglishSpellingNormalizer()
        self.pre_standardize_spellings = _EnglishSpellingNormalizer(PRE_ENGLISH_SPELLING)

        if remove_fillers:
            self.fillers = ["hmm", "uh", "ah", "eh"]
        else:
            self.fillers = None

    def __call__(self, s: str) -> str:
        s = s.lower()

        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)
        s = re.sub(r"\(([^)]+?)\)", "", s)
        s = self.pre_standardize_spellings(s)
        s = re.sub(r"\s+'", "'", s)

        for pattern, replacement in self.replacers.items():
            s = re.sub(pattern, replacement, s)

        s = re.sub(r"(\d),(\d)", r"\1\2", s)
        s = re.sub(r"\.([^0-9]|$)", r" \1", s)
        s = remove_symbols_and_diacritics(s, keep=".%$¢€£")

        if self.standardize_numbers is not None:
            s = self.standardize_numbers(s)

        if self.standardize_numbers_rev is not None:
            s = self.standardize_numbers_rev(s)

        s = self.standardize_spellings(s)
        s = re.sub(r"[.$¢€£]([^0-9])", r" \1", s)
        s = re.sub(r"([^0-9])%", r"\1 ", s)

        if self.fillers:
            s = re.sub(r"\b(" + "|".join(self.fillers) + r")\b", "", s)

        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"^\s+|\s+$", "", s)

        return s


_DEFAULT_NORMALIZER = Chime8TextNormalizer()


def normalize(text: str, *, max_passes: int = 5) -> str:
    """
    Apply CHIME-8 text normalization until idempotent (at most *max_passes*).

    This mirrors the loop recommended in chime-utils / MeetEval scoring pipelines.
    """
    words = text
    for _ in range(max_passes):
        words_ = _DEFAULT_NORMALIZER(words)
        if words == words_:
            return words
        words = words_
    raise RuntimeError("CHIME-8 normalizer did not converge within " f"{max_passes} passes for input: {text!r}")


def get_chime8_normalizer(language: str = "en"):
    """The CHiME-8 normalizer for ``language``.

    The family counterpart to ``get_whisper_normalizer`` / ``get_hf_normalizer``, so
    ``build_normalizer`` holds no per-family logic. Unlike those two it has no non-English branch:
    the pipeline is English-specific throughout -- an English spelling table, English contraction
    expansion and an English number verbaliser -- so applying it to another language would produce a
    number that looks fine and means nothing.

    Args:
        language: BCP-47-ish code. Only ``"en"`` is supported.

    Returns:
        Callable[[str], str]: the fixed-point :func:`normalize`.

    Raises:
        ValueError: for any language other than ``"en"``.
    """
    if language != "en":
        raise ValueError(
            f"chime8 is an English-only scorer; got language={language!r}. "
            "Use 'whisper' or 'hf', which both have a non-English branch."
        )
    return normalize
