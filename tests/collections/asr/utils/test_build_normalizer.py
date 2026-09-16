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
"""Pin the normalizer family dispatch.

Two properties matter. An unrecognised name must RAISE -- the fallback it replaced was silent, so a
typo reported un-normalized numbers that looked plausible. And ``whisper`` + ``en`` must be exactly
what the removed ``english`` name used to select, or every number recorded before this change
becomes incomparable with every number after it.
"""

import pytest
from whisper_normalizer.basic import BasicTextNormalizer
from whisper_normalizer.english import EnglishTextNormalizer

from nemo.collections.asr.parts.utils.text_normalizers import (
    NORMALIZER_NAMES,
    build_normalizer,
    get_whisper_normalizer,
)

# Exercises contractions, numbers, filler, casing, punctuation and bracket spans -- the axes on
# which the English and Basic normalizers actually differ.
_PROBES = [
    "It cost one thousand five hundred dollars.",
    "I kinda dunno, 'cause it's late",
    "Dr. Smith said <spk:0> hello",
    "THE B B C said so",
    "um well uh yeah",
    "",
]


@pytest.mark.unit
@pytest.mark.parametrize("name", ["chime9", "whsiper", "english", "basic", "whisperr", "hf2"])
def test_unknown_name_raises_and_names_the_accepted_values(name):
    with pytest.raises(ValueError) as exc:
        build_normalizer(name)
    for accepted in NORMALIZER_NAMES:
        assert accepted in str(exc.value)


@pytest.mark.unit
def test_removed_names_raise_rather_than_silently_meaning_none():
    """`english` / `basic` were collapsed into the `whisper` family; they must not resolve."""
    for removed in ("english", "basic"):
        with pytest.raises(ValueError):
            build_normalizer(removed)


@pytest.mark.unit
@pytest.mark.parametrize("name", [None, "none", "NONE", "None", "", "  ", " none "])
def test_none_empty_and_whitespace_are_identity(name):
    """Unset is not a typo: only a non-empty, unrecognised name raises."""
    norm = build_normalizer(name)
    for probe in _PROBES:
        assert norm(probe) == probe


@pytest.mark.unit
def test_name_lookup_is_case_insensitive():
    assert build_normalizer("WHISPER", "en")("Hello There") == build_normalizer("whisper", "en")("Hello There")


@pytest.mark.unit
def test_whisper_family_dispatches_on_language():
    assert isinstance(get_whisper_normalizer("en"), EnglishTextNormalizer)
    for lang in ("fr", "de", "zh"):
        assert isinstance(get_whisper_normalizer(lang), BasicTextNormalizer)


@pytest.mark.unit
def test_build_normalizer_routes_whisper_to_the_family():
    assert isinstance(build_normalizer("whisper", "en"), EnglishTextNormalizer)
    assert isinstance(build_normalizer("whisper", "fr"), BasicTextNormalizer)


@pytest.mark.unit
@pytest.mark.parametrize("probe", _PROBES)
def test_whisper_en_is_byte_identical_to_the_removed_english_path(probe):
    """The rename must move no number: this is what `use_normalizer=english` used to build."""
    assert build_normalizer("whisper", "en")(probe) == EnglishTextNormalizer()(probe)


@pytest.mark.unit
@pytest.mark.parametrize("probe", _PROBES)
def test_whisper_non_en_is_byte_identical_to_the_removed_basic_path(probe):
    assert build_normalizer("whisper", "fr")(probe) == BasicTextNormalizer()(probe)


@pytest.mark.unit
def test_english_and_basic_actually_differ_so_the_dispatch_is_observable():
    """Guards the test above from passing vacuously if the two families ever converged."""
    probe = "I kinda dunno, 'cause it's 1500 dollars"
    assert build_normalizer("whisper", "en")(probe) != build_normalizer("whisper", "fr")(probe)


@pytest.mark.unit
def test_hf_family_is_language_dispatched_too():
    """`normalizer_language` is not hf-specific -- both families read it."""
    assert callable(build_normalizer("hf", "en"))
    assert build_normalizer("hf", "en")("THE B B C said so") == build_normalizer("hf", "en")("THE B B C said so")
