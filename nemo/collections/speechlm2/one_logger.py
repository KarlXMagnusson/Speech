# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OneLogger throughput policies for the SpeechLM2 collection."""

from typing import Any

from nemo.lightning.speech_throughput import (
    SpeechThroughputPolicy,
    ThroughputMeasurements,
    add_audio_seconds,
    add_sum,
    add_value,
    batch_value,
    first_positive,
)

__all__ = [
    "DuplexSTTThroughputPolicy",
    "SALMAutomodelThroughputPolicy",
    "SALMSequenceThroughputPolicy",
    "SALMThroughputPolicy",
    "SpeechToSpeechThroughputPolicy",
]


class SALMThroughputPolicy(SpeechThroughputPolicy):
    """Measure input waveform duration for SALM variants."""

    name = "salm"

    def measure(self, model: Any, batch: Any) -> ThroughputMeasurements:
        measurements = {}
        add_audio_seconds(
            measurements,
            "input_audio_seconds",
            batch_value(batch, "audio_lens"),
            _sample_rate(model),
        )
        return measurements


class SALMSequenceThroughputPolicy(SALMThroughputPolicy):
    """Measure SALM audio and exact post-expansion mixed-modality positions."""

    def measure(self, model: Any, batch: Any) -> ThroughputMeasurements:
        measurements = super().measure(model, batch)
        add_value(measurements, "model_sequence_positions", getattr(model, "_last_batch_num_tokens", None))
        return measurements


class SALMAutomodelThroughputPolicy(SALMSequenceThroughputPolicy):
    """Use the SALM position schema with a distinct Automodel identity."""

    name = "salm_automodel"


class DuplexSTTThroughputPolicy(SpeechThroughputPolicy):
    """Measure audio and text-only work in a mixed DuplexSTT batch."""

    name = "duplex_stt"

    def measure(self, model: Any, batch: Any) -> ThroughputMeasurements:
        measurements = {}
        audio_batch = batch_value(batch, "audio_data")
        text_batch = batch_value(batch, "text_data")
        if audio_batch is not None:
            add_audio_seconds(
                measurements,
                "input_audio_seconds",
                batch_value(audio_batch, "source_audio_lens"),
                _source_sample_rate(model),
            )
        if text_batch is not None:
            add_sum(measurements, "text_tokens", batch_value(text_batch, "text_token_lens"))
        return measurements


class SpeechToSpeechThroughputPolicy(SpeechThroughputPolicy):
    """Measure each direction of a duplex speech batch independently."""

    name = "speech_to_speech"

    def measure(self, model: Any, batch: Any) -> ThroughputMeasurements:
        measurements = {}
        add_audio_seconds(
            measurements,
            "input_audio_seconds",
            batch_value(batch, "source_audio_lens"),
            _source_sample_rate(model),
        )
        add_audio_seconds(
            measurements,
            "output_audio_seconds",
            batch_value(batch, "target_audio_lens"),
            _target_sample_rate(model),
        )
        return measurements


def _sample_rate(model: Any) -> float | None:
    return first_positive(
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
    return first_positive(
        model,
        "source_sample_rate",
        "perception.preprocessor.featurizer.sample_rate",
        "perception.preprocessor._sample_rate",
        "perception.preprocessor._cfg.sample_rate",
    )


def _target_sample_rate(model: Any) -> float | None:
    return first_positive(
        model,
        "target_sample_rate",
        "audio_codec.sample_rate",
        "audio_codec.output_sample_rate",
    )
