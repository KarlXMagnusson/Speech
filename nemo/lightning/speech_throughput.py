# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamic, modality-aware throughput measurements for NeMo Speech."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

__all__ = [
    "ASRThroughputPolicy",
    "AudioCodecThroughputPolicy",
    "AudioThroughputPolicy",
    "DiarizationThroughputPolicy",
    "DuplexSTTThroughputPolicy",
    "SALMAutomodelThroughputPolicy",
    "SALMSequenceThroughputPolicy",
    "SALMThroughputPolicy",
    "SpeechThroughputPolicy",
    "SpeechToSpeechThroughputPolicy",
    "TTSThroughputPolicy",
    "ThroughputValue",
    "select_throughput_policy",
]


@dataclass(frozen=True)
class ThroughputValue:
    """A deferred additive value and its conversion to the reported unit."""

    value: Any
    scale: float = 1.0


class SpeechThroughputPolicy:
    """Extract additive work units from one dynamic speech batch.

    Returned values remain tensors until a reporting window closes. This lets
    the callback avoid a host/device synchronization on every batch.
    """

    name = "speech"

    def measure(self, model: Any, batch: Any) -> dict[str, Any]:
        del model, batch
        return {}


class ASRThroughputPolicy(SpeechThroughputPolicy):
    """Measure input audio and target text processed by an ASR model."""

    name = "asr"

    def measure(self, model: Any, batch: Any) -> dict[str, Any]:
        measurements = super().measure(model, batch)
        _add_audio_seconds(
            measurements,
            "input_audio_seconds",
            _asr_audio_lengths(batch),
            _sample_rate(model),
        )
        _add_sum(measurements, "target_text_tokens", _asr_text_lengths(batch))
        return measurements


class TTSThroughputPolicy(SpeechThroughputPolicy):
    """Measure input text and target audio processed by a TTS model."""

    name = "tts"

    def measure(self, model: Any, batch: Any) -> dict[str, Any]:
        measurements = super().measure(model, batch)
        _add_sum(measurements, "input_text_tokens", _tts_text_lengths(batch))
        _add_audio_seconds(
            measurements,
            "output_audio_seconds",
            _audio_lengths(batch),
            _sample_rate(model),
        )
        return measurements


class AudioCodecThroughputPolicy(SpeechThroughputPolicy):
    """Measure waveform duration processed by an audio codec model."""

    name = "audio_codec"

    def measure(self, model: Any, batch: Any) -> dict[str, Any]:
        measurements = super().measure(model, batch)
        _add_audio_seconds(
            measurements,
            "input_audio_seconds",
            _audio_lengths(batch),
            _sample_rate(model),
        )
        return measurements


class DiarizationThroughputPolicy(SpeechThroughputPolicy):
    """Measure input audio processed by a diarization model."""

    name = "diarization"

    def measure(self, model: Any, batch: Any) -> dict[str, Any]:
        measurements = super().measure(model, batch)
        _add_audio_seconds(
            measurements,
            "input_audio_seconds",
            _audio_lengths(batch),
            _sample_rate(model),
        )
        return measurements


class AudioThroughputPolicy(SpeechThroughputPolicy):
    """Measure input audio processed by enhancement and audio-to-audio models."""

    name = "audio"

    def measure(self, model: Any, batch: Any) -> dict[str, Any]:
        measurements = super().measure(model, batch)
        _add_audio_seconds(
            measurements,
            "input_audio_seconds",
            _audio_lengths(batch),
            _sample_rate(model),
        )
        return measurements


class SALMThroughputPolicy(SpeechThroughputPolicy):
    """Measure input audio processed by SALM variants."""

    name = "salm"

    def measure(self, model: Any, batch: Any) -> dict[str, Any]:
        measurements = super().measure(model, batch)
        _add_audio_seconds(
            measurements,
            "input_audio_seconds",
            _value(batch, "audio_lens"),
            _sample_rate(model),
        )
        return measurements


class SALMSequenceThroughputPolicy(SALMThroughputPolicy):
    """Measure SALM audio and post-expansion mixed-modality positions."""

    def measure(self, model: Any, batch: Any) -> dict[str, Any]:
        measurements = super().measure(model, batch)
        sequence_positions = getattr(model, "_last_batch_num_tokens", None)
        if sequence_positions is not None:
            _add_value(measurements, "model_sequence_positions", sequence_positions)
        return measurements


class SALMAutomodelThroughputPolicy(SALMSequenceThroughputPolicy):
    """Use the SALM position schema with a distinct Automodel identity."""

    name = "salm_automodel"


class DuplexSTTThroughputPolicy(SpeechThroughputPolicy):
    """Measure audio and text-only work in a mixed DuplexSTT batch."""

    name = "duplex_stt"

    def measure(self, model: Any, batch: Any) -> dict[str, Any]:
        measurements = super().measure(model, batch)
        audio_batch = _value(batch, "audio_data")
        text_batch = _value(batch, "text_data")
        if audio_batch is not None:
            _add_audio_seconds(
                measurements,
                "input_audio_seconds",
                _value(audio_batch, "source_audio_lens"),
                _source_sample_rate(model),
            )
        if text_batch is not None:
            _add_sum(measurements, "text_tokens", _value(text_batch, "text_token_lens"))
        return measurements


class SpeechToSpeechThroughputPolicy(SpeechThroughputPolicy):
    """Measure each direction of a duplex speech batch independently."""

    name = "speech_to_speech"

    def measure(self, model: Any, batch: Any) -> dict[str, Any]:
        measurements = super().measure(model, batch)
        _add_audio_seconds(
            measurements,
            "input_audio_seconds",
            _value(batch, "source_audio_lens"),
            _source_sample_rate(model),
        )
        _add_audio_seconds(
            measurements,
            "output_audio_seconds",
            _value(batch, "target_audio_lens"),
            _target_sample_rate(model),
        )
        return measurements


def select_throughput_policy(model: Any) -> SpeechThroughputPolicy | None:
    """Select a policy without importing collection model modules.

    Collection imports here would create cycles because the callback is
    initialized from NeMo's core model machinery. An explicit model policy is
    supported for specialized or downstream models; otherwise collection and
    class identity provide a small, dependency-free default dispatch.
    """

    explicit = getattr(model, "one_logger_throughput_policy", None)
    if explicit is not None:
        return explicit() if isinstance(explicit, type) else explicit

    model_type = type(model)
    module = model_type.__module__
    name = model_type.__name__.lower()
    if module.startswith("nemo.collections.speechlm2"):
        if name == "salmautomodel":
            return SALMAutomodelThroughputPolicy()
        if name == "salm":
            return SALMSequenceThroughputPolicy()
        if name.startswith("salm"):
            return SALMThroughputPolicy()
        if name == "duplexsttmodel":
            return DuplexSTTThroughputPolicy()
        if name in _SPEECH_TO_SPEECH_MODELS:
            return SpeechToSpeechThroughputPolicy()
        return None
    if module.startswith("nemo.collections.tts"):
        if name == "audiocodecmodel":
            return AudioCodecThroughputPolicy()
        if module.rsplit(".", 1)[-1] in _TTS_MODEL_MODULES:
            return TTSThroughputPolicy()
        return None
    if module.startswith("nemo.collections.asr"):
        if name == "sortformerenclabelmodel":
            return DiarizationThroughputPolicy()
        if module.rsplit(".", 1)[-1] in _ASR_TRANSCRIPTION_MODULES:
            return ASRThroughputPolicy()
        return None
    if module.rsplit(".", 1)[-1] in _AUDIO_MODEL_MODULES:
        return AudioThroughputPolicy()
    return None


def _value(batch: Any, *names: str) -> Any:
    for name in names:
        if isinstance(batch, Mapping) and name in batch:
            return batch[name]
        if hasattr(batch, name):
            return getattr(batch, name)
    return None


def _asr_audio_lengths(batch: Any) -> Any:
    if hasattr(batch, "has_processed_signal"):
        return None if batch.has_processed_signal else batch[1]
    signal = _value(batch, "audio", "audio_signal", "input_signal")
    lengths = _value(batch, "audio_lens", "audio_lengths", "audio_len", "input_length")
    if isinstance(batch, (tuple, list)) and len(batch) > 1:
        signal, lengths = batch[0], batch[1]
    if torch.is_tensor(signal) and signal.ndim <= 2:
        return lengths
    return None


def _audio_lengths(batch: Any) -> Any:
    lengths = _value(batch, "audio_lens", "audio_len", "input_length")
    if lengths is not None:
        return lengths
    if isinstance(batch, (tuple, list)) and len(batch) > 1:
        return batch[1]
    return None


def _asr_text_lengths(batch: Any) -> Any:
    lengths = _value(
        batch,
        "text_token_lengths",
        "transcript_lens",
        "transcript_len",
        "prompted_transcript_lens",
        "token_lens",
    )
    if lengths is not None:
        return lengths
    if hasattr(batch, "has_processed_signal"):
        return batch[3]
    if isinstance(batch, (tuple, list)) and len(batch) > 3:
        return batch[3]
    return None


def _tts_text_lengths(batch: Any) -> Any:
    lengths = _value(batch, "text_lens", "text_len", "token_lens")
    if lengths is not None:
        return lengths
    if isinstance(batch, (tuple, list)) and len(batch) > 3:
        return batch[3]
    return None


def _sample_rate(model: Any) -> float | None:
    return _first_positive(
        model,
        "sampling_rate",
        "sample_rate",
        "preprocessor._sample_rate",
        "preprocessor.sample_rate",
        "preprocessor._cfg.sample_rate",
        "perception.preprocessor.featurizer.sample_rate",
        "perception.preprocessor._sample_rate",
        "perception.preprocessor._cfg.sample_rate",
        "cfg.sample_rate",
        "cfg.preprocessor.sample_rate",
    )


def _source_sample_rate(model: Any) -> float | None:
    return _first_positive(
        model,
        "source_sample_rate",
        "perception.preprocessor.featurizer.sample_rate",
        "perception.preprocessor._sample_rate",
        "perception.preprocessor._cfg.sample_rate",
    )


def _target_sample_rate(model: Any) -> float | None:
    return _first_positive(
        model,
        "target_sample_rate",
        "audio_codec.sample_rate",
        "audio_codec.output_sample_rate",
    )


def _first_positive(obj: Any, *paths: str) -> float | None:
    for path in paths:
        value = obj
        for part in path.split("."):
            if isinstance(value, Mapping):
                value = value.get(part)
            else:
                value = getattr(value, part, None)
            if value is None:
                break
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None


def _add_audio_seconds(measurements: dict[str, Any], name: str, lengths: Any, sample_rate: float | None) -> None:
    if lengths is not None and sample_rate is not None:
        _add_value(measurements, name, lengths, scale=1.0 / sample_rate)


def _add_sum(measurements: dict[str, Any], name: str, value: Any) -> None:
    if value is not None:
        _add_value(measurements, name, value)


def _add_value(measurements: dict[str, Any], name: str, value: Any, scale: float = 1.0) -> None:
    if value is not None:
        value = value.detach() if torch.is_tensor(value) else value
        measurements[name] = ThroughputValue(value, scale)


_SPEECH_TO_SPEECH_MODELS = {
    "duplexeartts",
    "duplexs2smodel",
    "duplexs2sspeechdecodermodel",
}
_ASR_TRANSCRIPTION_MODULES = {
    "aed_multitask_models",
    "asr_eou_models",
    "ctc_bpe_models",
    "ctc_models",
    "hybrid_rnnt_ctc_bpe_models",
    "hybrid_rnnt_ctc_bpe_models_prompt",
    "hybrid_rnnt_ctc_models",
    "multitalker_asr_models",
    "rnnt_bpe_models",
    "rnnt_bpe_models_prompt",
    "rnnt_models",
    "transformer_bpe_models",
}
_TTS_MODEL_MODULES = {
    "aligner",
    "easy_magpietts",
    "easy_magpietts_cfg_distillation",
    "easy_magpietts_preference_optimization",
    "fastpitch",
    "fastpitch_ssl",
    "hifigan",
    "magpietts",
    "magpietts_cfg_distillation",
    "magpietts_preference_optimization",
}
_AUDIO_MODEL_MODULES = {"audio_to_audio", "enhancement"}
