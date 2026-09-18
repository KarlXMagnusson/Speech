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
"""Pin that the scoring surface works on the config object Hydra actually hands over.

`@hydra_runner` passes a ``DictConfig``, not the dataclass -- so dataclass *methods* are not bound
and ``cfg.effective_normalizer()`` raises ``ConfigAttributeError``. Every other test in this suite
constructs the dataclass directly, so the entire CLI path went unexercised: the inference script
ran a full GPU inference and then died at the reporting step.

`CpWER.from_config` already documents its argument as "any object carrying the CpWERScoringConfig
fields". These tests hold it to that.
"""

import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.speechlm2.streaming_stt_generate import StreamingSTTEvalConfig  # noqa: E402
from examples.speechlm2.streaming_stt_score import CpWERScoreConfig  # noqa: E402
from nemo.collections.speechlm2.parts.metrics import CpWER  # noqa: E402
from nemo.collections.speechlm2.parts.metrics.cpwer_report import axis_fingerprint, cpwer_metrics_dict  # noqa: E402

_ROWS = [("<spk:0> it cost one thousand dollars", "<spk:0> it cost one thousand dollars")]


def _as_hydra(cls, **overrides):
    """Exactly what `@hydra_runner(schema=...)` produces: a DictConfig, not the dataclass."""
    return OmegaConf.merge(OmegaConf.structured(cls), OmegaConf.create(overrides))


@pytest.mark.unit
@pytest.mark.parametrize("cls", [StreamingSTTEvalConfig, CpWERScoreConfig])
def test_report_renders_from_a_dictconfig(cls):
    """The crash: a full inference run reached this line and died after all the GPU work."""
    cfg = _as_hydra(cls)
    metrics = cpwer_metrics_dict({}, {}, cfg)
    assert "cpwer_axes" in metrics


@pytest.mark.unit
def test_axis_fingerprint_resolves_the_normalizer_from_a_dictconfig():
    cfg = _as_hydra(StreamingSTTEvalConfig, cpwer_normalizer="chime8", use_normalizer="whisper")
    assert "cpwer_normalizer='chime8'" in axis_fingerprint(cfg)["resolved"]


@pytest.mark.unit
def test_metric_builds_from_a_dictconfig():
    cfg = _as_hydra(StreamingSTTEvalConfig)
    assert CpWER.from_config(cfg) is not None


@pytest.mark.unit
def test_cpwer_normalizer_is_not_silently_ignored():
    """The worse bug: a wrong number rather than an error.

    The inference script built ONE normalizer from `use_normalizer` and handed it to cpWER, so
    `cpwer_normalizer` was inert inline while the axis stamp still claimed it had been applied.
    chime8 spells numbers as words and whisper as digits, so the reference word count differs and
    the two cannot be confused for rounding.
    """
    ref, hyp = _ROWS[0]
    scores = {}
    for name in ("whisper", "chime8"):
        cfg = _as_hydra(StreamingSTTEvalConfig, cpwer_normalizer=name, use_normalizer="whisper")
        metric = CpWER.from_config(cfg)
        scores[name] = metric.score_session(ref, hyp).ref_words
    assert (
        scores["whisper"] != scores["chime8"]
    ), f"cpwer_normalizer had no effect: both normalizers gave {scores['whisper']} reference words"
