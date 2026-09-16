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
"""Select a text normalizer by FAMILY, with the language choosing within it.

Evaluation scripts pick a normalizer with two settings -- a family name and a language code -- and
every family answers the language question the same way, through its own
``get_<family>_normalizer(language)``. :func:`build_normalizer` is only the name-to-family table.

The split matters because the families disagree on the same text, so a number is meaningless without
knowing which one produced it. Whisper's English normalizer expands contractions and spells numbers
one way; the Open ASR Leaderboard fork additionally collapses acronyms and rewrites compounds.
"""

from typing import Callable, Optional

from whisper_normalizer.basic import BasicTextNormalizer
from whisper_normalizer.english import EnglishTextNormalizer

from nemo.collections.asr.parts.utils.hf_asr_normalizer import get_hf_normalizer

__all__ = ["build_normalizer", "get_whisper_normalizer", "NORMALIZER_NAMES"]

#: Accepted ``use_normalizer`` values. Family names, not class names: the language selects the
#: concrete normalizer within a family.
NORMALIZER_NAMES = ("whisper", "hf", "none")


def build_normalizer(name: Optional[str], language: str = "en") -> Callable[[str], str]:
    """Resolve a normalizer family by name; the language selects within it.

    Raises on an unrecognised name rather than falling back to identity. The fallback it replaces
    was silent, so a typo (``use_normalizer=whsiper``) reported un-normalized numbers that looked
    plausible -- a wrong number is worse than a failed run.

    Args:
        name: family name, case-insensitive. ``whisper`` (Whisper's own normalizers), ``hf`` (the
            Open ASR Leaderboard fork, required to reproduce leaderboard WER), or ``none``.
            ``None``, ``"none"``, and an empty or whitespace-only string all mean identity.
            Surrounding whitespace is stripped, so a stray space in a config value is not a typo.
        language: BCP-47-ish code, e.g. ``"en"``, ``"de"``. Selects within the chosen family; it is
            NOT specific to ``hf``. Ignored by ``none``.

    Returns:
        Callable[[str], str]: maps a raw string to its normalized form.

    Raises:
        ValueError: if ``name`` is not one of :data:`NORMALIZER_NAMES`, with the accepted values
            listed.
    """
    key = name.strip().lower() if isinstance(name, str) else ""
    key = key or "none"  # unset, empty, or whitespace all mean "do not normalize"
    if key == "whisper":
        return get_whisper_normalizer(language)
    if key == "hf":
        return get_hf_normalizer(language)
    if key == "none":
        return _identity
    raise ValueError(f"Unknown normalizer {name!r}. Accepted: {', '.join(NORMALIZER_NAMES)} (or None for identity).")


def get_whisper_normalizer(language: str = "en") -> Callable[[str], str]:
    """Whisper's own normalizer for ``language``.

    Mirrors upstream Whisper's evaluation dispatch: English gets
    :class:`~whisper_normalizer.english.EnglishTextNormalizer`, which expands contractions, spells
    numbers and applies an English spelling table; every other language gets
    :class:`~whisper_normalizer.basic.BasicTextNormalizer`, which only lowercases and strips
    bracket spans, symbols and diacritics.

    The counterpart to :func:`~nemo.collections.asr.parts.utils.hf_asr_normalizer.get_hf_normalizer`,
    which makes the same English-versus-other split for the leaderboard fork.

    Args:
        language: BCP-47-ish code. Only ``"en"`` selects the English normalizer.

    Returns:
        Callable[[str], str]: maps a raw string to its normalized form.
    """
    return EnglishTextNormalizer() if language == "en" else BasicTextNormalizer()


def _identity(text: str) -> str:
    return text
