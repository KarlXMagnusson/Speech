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
"""Pin that streaming inference is deterministic.

``BatchedCacheFeatureBufferer`` builds its own preprocessor rather than borrowing the model's, so
it is not an ``nn.Module`` child of anything and ``model.eval()`` does not reach it. A freshly
constructed module defaults to training mode, and ``FilterbankFeatures.forward`` adds dither when
``self.training`` -- so leaving it unset injects noise into every inference run, which shows up as
a model that silently returns a different transcript each time it is given the same audio.
"""

import pytest
import torch
from omegaconf import DictConfig

from nemo.collections.asr.inference.streaming.buffering.cache_feature_bufferer import BatchedCacheFeatureBufferer
from nemo.collections.asr.inference.streaming.framing.request import Frame

SAMPLE_RATE = 16000
CHUNK_SECS = 0.5

#: Dither deliberately non-zero: the point is that the bufferer stays deterministic anyway, by
#: virtue of eval mode, rather than by the caller having had to zero it out first.
PREPROCESSOR_CFG = DictConfig(
    {
        "_target_": "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor",
        "sample_rate": SAMPLE_RATE,
        "features": 80,
        "n_fft": 512,
        "window_size": 0.025,
        "window_stride": 0.01,
        "window": "hann",
        "log": True,
        "dither": 1e-05,
        "normalize": None,
        "pad_to": 0,
        "pad_value": 0.0,
        "frame_splicing": 1,
    }
)


def _bufferer(num_slots=1):
    return BatchedCacheFeatureBufferer(
        num_slots=num_slots,
        sample_rate=SAMPLE_RATE,
        buffer_size_in_secs=CHUNK_SECS,
        chunk_size_in_secs=CHUNK_SECS,
        preprocessor_cfg=PREPROCESSOR_CFG,
        device=torch.device("cpu"),
    )


def _audio(seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(int(SAMPLE_RATE * CHUNK_SECS), generator=generator)


@pytest.mark.unit
def test_preprocessor_is_in_eval_mode():
    """The direct contract. `.to(device)` does not change training mode; only `.eval()` does."""
    assert _bufferer().preprocessor.training is False


@pytest.mark.unit
def test_same_audio_gives_bit_identical_features():
    """The regression. Dither is live in training mode, so this differs run to run without the fix."""
    samples = _audio()
    outputs = []
    for _ in range(2):
        features, _ = _bufferer().update([Frame(samples=samples, stream_id=0, length=samples.shape[0])])
        outputs.append(features[0].clone())
    assert torch.equal(outputs[0], outputs[1]), (
        f"identical audio produced different features; max|diff| = "
        f"{(outputs[0] - outputs[1]).abs().max().item():.3e}"
    )


@pytest.mark.unit
def test_dither_config_is_left_alone():
    """Fixed by mode, not by rewriting the caller's config -- so a training-mode user still gets it."""
    bufferer = _bufferer()
    assert bufferer.preprocessor.featurizer.dither == PREPROCESSOR_CFG.dither
    bufferer.preprocessor.train()
    samples = _audio()
    a, _ = bufferer.preprocess([samples], torch.zeros(1, dtype=torch.long), expected_feat_len=50)
    b, _ = bufferer.preprocess([samples], torch.zeros(1, dtype=torch.long), expected_feat_len=50)
    assert not torch.equal(a, b), "dither should still apply when the preprocessor is put in training mode"
