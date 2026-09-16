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
"""Pin the vendored CHiME-8 normalizer.

Three kinds of guard, all of them about the same hazard: this normalizer decides the denominator of
every score computed with it, so a silent change to it silently moves published numbers.

1. The spelling tables are pinned by SHA-256. Nobody hand-edits a 1737-entry dict, but a
   well-meaning reformat or a "fix" to one entry would otherwise pass unnoticed.
2. The three blocks SHARED with ``hf_asr_normalizer`` are pinned by source hash. Sharing saved ~460
   duplicated lines; the cost is that editing that file changes this normalizer at a distance. The
   pin converts that from silent to a named failure.
3. Two upstream behaviours that look like bugs are pinned as-is, because matching the reference
   scorer is the point -- "fixing" them here would make our number disagree with everyone else's.
"""

import hashlib
import inspect
import json

import pytest

from nemo.collections.asr.parts.utils import hf_asr_normalizer as hf
from nemo.collections.asr.parts.utils.chime8_normalizer import Chime8TextNormalizer, get_chime8_normalizer, normalize
from nemo.collections.asr.parts.utils.chime8_spelling_data import ENGLISH_SPELLING, PRE_ENGLISH_SPELLING
from nemo.collections.asr.parts.utils.text_normalizers import build_normalizer


def _hash_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _hash_src(obj) -> str:
    return hashlib.sha256(inspect.getsource(obj).encode()).hexdigest()


# --------------------------------------------------------------------------------------------
# 1. the vendored tables
# --------------------------------------------------------------------------------------------


@pytest.mark.unit
def test_english_spelling_table_is_unchanged():
    assert len(ENGLISH_SPELLING) == 1737
    assert _hash_obj(ENGLISH_SPELLING) == "cedbc325b3d3b2cc8b09919b83c1b046625f1b3027d4cb9bc8f7bf614a7a4872"


@pytest.mark.unit
def test_pre_english_spelling_table_is_unchanged():
    assert len(PRE_ENGLISH_SPELLING) == 3
    assert _hash_obj(PRE_ENGLISH_SPELLING) == "2dfc6b88d24f654e02f11e7806e24da23543d76181fbc6a991197a103d11a4b6"


# --------------------------------------------------------------------------------------------
# 2. the coupling to hf_asr_normalizer
# --------------------------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "name,expected",
    [
        ("ADDITIONAL_DIACRITICS", "50ebc0d5fd4676f04c32290eb01cefb8771895df0bb97b3c430fe4769fbcca75"),
        ("remove_symbols_and_diacritics", "120f9411f65c7e5d9325c4310e860509e8ffc79f40906d4e14f457840dd909a2"),
        ("EnglishNumberNormalizer", "9938dc1cfebea63e9cb8c5da35ea547703380da41ea204e9f52e34fea77576e0"),
    ],
)
def test_shared_blocks_from_hf_normalizer_are_unchanged(name, expected):
    """chime8 imports these three. Editing hf_asr_normalizer moves every score computed here.

    If this fails, the edit may still be correct -- but it is NOT free, and the new number must be
    re-verified against the reference scorer before the hash is updated.
    """
    obj = getattr(hf, name)
    actual = _hash_obj(obj) if isinstance(obj, dict) else _hash_src(obj)
    assert actual == expected, f"{name} changed in hf_asr_normalizer; re-verify chime8 parity first"


# --------------------------------------------------------------------------------------------
# 3. deliberately preserved upstream behaviour
# --------------------------------------------------------------------------------------------


@pytest.mark.unit
def test_aint_expands_wrongly_and_that_is_deliberate():
    """``ain't`` -> ``ain not`` is wrong English but is what the reference scorer does."""
    assert normalize("ain't") == "ain not"


@pytest.mark.unit
def test_sci_fi_is_mangled_by_the_wifi_rule_and_that_is_deliberate():
    """The ``wi-fi`` -> ``wifi`` rewrite over-fires on ``sci-fi``. Reference behaviour, kept."""
    assert normalize("sci-fi") == "swifi"


@pytest.mark.unit
def test_reverse_number_normalization_is_lossy_for_round_thousands():
    """``1000`` loses its leading "one"; 1500 is left as digits. Both are upstream behaviour."""
    assert normalize("1000") == "thousand"
    assert normalize("1500") == "1500"


# --------------------------------------------------------------------------------------------
# 4. the normalizer's contract
# --------------------------------------------------------------------------------------------


@pytest.mark.unit
def test_normalize_is_idempotent():
    for probe in ["It cost one thousand five hundred dollars.", "um well uh yeah", "THE B B C said so"]:
        once = normalize(probe)
        assert normalize(once) == once


@pytest.mark.unit
def test_non_convergence_raises_rather_than_returning_a_half_normalized_string():
    class _Flapping(Chime8TextNormalizer):
        def __call__(self, s):  # alternates forever, so the fixed point is never reached
            return "b" if s == "a" else "a"

    import nemo.collections.asr.parts.utils.chime8_normalizer as mod

    original = mod._DEFAULT_NORMALIZER
    mod._DEFAULT_NORMALIZER = _Flapping()
    try:
        with pytest.raises(RuntimeError, match="did not converge"):
            normalize("a")
    finally:
        mod._DEFAULT_NORMALIZER = original


@pytest.mark.unit
def test_empty_and_whitespace_are_safe():
    assert normalize("") == ""
    assert normalize("   ") == ""


# --------------------------------------------------------------------------------------------
# 5. family dispatch
# --------------------------------------------------------------------------------------------


@pytest.mark.unit
def test_chime8_is_english_only():
    assert get_chime8_normalizer("en") is normalize
    for lang in ("fr", "de", "zh"):
        with pytest.raises(ValueError, match="English-only"):
            get_chime8_normalizer(lang)


@pytest.mark.unit
def test_build_normalizer_routes_chime8():
    assert build_normalizer("chime8", "en")("It cost 1000 dollars") == normalize("It cost 1000 dollars")
    with pytest.raises(ValueError, match="English-only"):
        build_normalizer("chime8", "fr")


@pytest.mark.unit
def test_chime8_and_whisper_disagree_in_the_direction_that_matters():
    """The reason both exist: Whisper spells numbers as digits, chime8 spells digits as words.

    If this ever passes trivially, one of the two families has silently changed.
    """
    probe = "It cost 1000 dollars"
    assert build_normalizer("chime8", "en")(probe) == "it cost thousand dollars"
    assert build_normalizer("whisper", "en")(probe) == "it cost $1000"
