# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from nemo.collections.asr.one_logger import ASRThroughputPolicy, DiarizationThroughputPolicy
from nemo.collections.audio.one_logger import AudioThroughputPolicy
from nemo.collections.speechlm2.one_logger import (
    DuplexSTTThroughputPolicy,
    SALMAutomodelThroughputPolicy,
    SALMSequenceThroughputPolicy,
    SALMThroughputPolicy,
    SpeechToSpeechThroughputPolicy,
)
from nemo.collections.tts.one_logger import AudioCodecThroughputPolicy, TTSThroughputPolicy
from nemo.lightning.speech_throughput import (
    SpeechThroughputPolicy,
    register_throughput_policy,
    select_throughput_policy,
)


def _model(module: str, name: str, **attributes):
    model_type = type(name, (), {"__module__": module})
    model = model_type()
    for key, value in attributes.items():
        setattr(model, key, value)
    return model


def _total(measurement):
    value = measurement.value
    if torch.is_tensor(value):
        value = value.sum().item()
    elif isinstance(value, (tuple, list)):
        value = sum(value)
    return value * measurement.scale


@pytest.mark.parametrize(
    "policy_type",
    [
        ASRThroughputPolicy,
        DiarizationThroughputPolicy,
        AudioThroughputPolicy,
        TTSThroughputPolicy,
        AudioCodecThroughputPolicy,
        SALMThroughputPolicy,
        SALMSequenceThroughputPolicy,
        SALMAutomodelThroughputPolicy,
        DuplexSTTThroughputPolicy,
        SpeechToSpeechThroughputPolicy,
    ],
)
def test_registered_policy_is_selected_through_inheritance(policy_type):
    @register_throughput_policy(policy_type)
    class ModelBase:
        pass

    class Model(ModelBase):
        pass

    assert isinstance(select_throughput_policy(Model()), policy_type)


def test_specialized_subclass_can_override_inherited_policy():
    @register_throughput_policy(ASRThroughputPolicy)
    class ModelBase:
        pass

    @register_throughput_policy(TTSThroughputPolicy)
    class Model(ModelBase):
        pass

    assert isinstance(select_throughput_policy(Model()), TTSThroughputPolicy)


def test_unregistered_model_is_not_reported():
    assert select_throughput_policy(object()) is None


def test_explicit_policy_instance_is_supported_for_downstream_models():
    policy = ASRThroughputPolicy()
    model = _model("downstream.models", "CustomModel", one_logger_throughput_policy=policy)

    assert select_throughput_policy(model) is policy


def test_collection_trunks_register_representative_model_policies():
    from nemo.collections.asr.parts.mixins.diarization import SpkDiarizationMixin
    from nemo.collections.asr.parts.mixins.transcription import ASRTranscriptionMixin
    from nemo.collections.audio.models.audio_to_audio import AudioToAudioModel
    from nemo.collections.speechlm2.models.duplex_s2s_model import DuplexS2SModel
    from nemo.collections.speechlm2.models.duplex_stt_model import DuplexSTTModel
    from nemo.collections.speechlm2.models.salm import SALM
    from nemo.collections.speechlm2.models.salm_automodel import SALMAutomodel
    from nemo.collections.tts.models.audio_codec import AudioCodecModel
    from nemo.collections.tts.models.fastpitch import FastPitchModel

    registrations = {
        ASRTranscriptionMixin: ASRThroughputPolicy,
        SpkDiarizationMixin: DiarizationThroughputPolicy,
        AudioToAudioModel: AudioThroughputPolicy,
        FastPitchModel: TTSThroughputPolicy,
        AudioCodecModel: AudioCodecThroughputPolicy,
        SALM: SALMSequenceThroughputPolicy,
        SALMAutomodel: SALMAutomodelThroughputPolicy,
        DuplexSTTModel: DuplexSTTThroughputPolicy,
        DuplexS2SModel: SpeechToSpeechThroughputPolicy,
    }

    for model_type, policy_type in registrations.items():
        assert model_type.one_logger_throughput_policy is policy_type


def test_asr_measures_dynamic_waveform_and_target_text_without_reducing_them():
    model = SimpleNamespace(preprocessor=SimpleNamespace(_sample_rate=16000))
    audio_lengths = torch.tensor([8000, 32000])
    text_lengths = torch.tensor([3, 8])
    batch = (torch.zeros(2, 32000), audio_lengths, torch.zeros(2, 8, dtype=torch.long), text_lengths)

    measured = ASRThroughputPolicy().measure(model, batch)

    assert measured.keys() == {"input_audio_seconds", "target_text_tokens"}
    assert measured["input_audio_seconds"].value.data_ptr() == audio_lengths.data_ptr()
    assert measured["target_text_tokens"].value.data_ptr() == text_lengths.data_ptr()
    assert _total(measured["input_audio_seconds"]) == 2.5
    assert _total(measured["target_text_tokens"]) == 11


def test_asr_eou_batch_uses_its_named_waveform_and_text_lengths():
    model = SimpleNamespace(preprocessor=SimpleNamespace(_sample_rate=16000))
    batch = SimpleNamespace(
        audio_signal=torch.zeros(2, 24000),
        audio_lengths=torch.tensor([8000, 24000]),
        text_token_lengths=torch.tensor([4, 6]),
    )

    measured = ASRThroughputPolicy().measure(model, batch)

    assert _total(measured["input_audio_seconds"]) == 2.0
    assert _total(measured["target_text_tokens"]) == 10


def test_prompted_asr_reports_answer_tokens_not_prompt_plus_answer_positions():
    model = SimpleNamespace(preprocessor=SimpleNamespace(_sample_rate=16000))
    batch = SimpleNamespace(
        audio=torch.zeros(2, 16000),
        audio_lens=torch.tensor([16000, 8000]),
        transcript_lens=torch.tensor([4, 6]),
        prompted_transcript_lens=torch.tensor([12, 15]),
    )

    measured = ASRThroughputPolicy().measure(model, batch)

    assert _total(measured["target_text_tokens"]) == 10


class _DALIOutputs:
    def __init__(self, processed: bool):
        self.has_processed_signal = processed
        self.values = (
            torch.zeros(2, 80, 20) if processed else torch.zeros(2, 32000),
            torch.tensor([20, 15]) if processed else torch.tensor([16000, 32000]),
            torch.zeros(2, 5),
            torch.tensor([5, 3]),
        )

    def __getitem__(self, index):
        return self.values[index]


def test_asr_dali_reports_text_but_never_interprets_feature_frames_as_audio_samples():
    model = SimpleNamespace(preprocessor=SimpleNamespace(_sample_rate=16000))

    processed = ASRThroughputPolicy().measure(model, _DALIOutputs(processed=True))
    raw = ASRThroughputPolicy().measure(model, _DALIOutputs(processed=False))

    assert processed.keys() == {"target_text_tokens"}
    assert _total(processed["target_text_tokens"]) == 8
    assert _total(raw["input_audio_seconds"]) == 3.0


def test_asr_feature_tuple_omits_audio_duration_instead_of_treating_frames_as_samples():
    model = SimpleNamespace(preprocessor=SimpleNamespace(_sample_rate=16000))
    batch = (torch.zeros(2, 80, 100), torch.tensor([100, 80]), torch.zeros(2, 4), torch.tensor([4, 3]))

    measured = ASRThroughputPolicy().measure(model, batch)

    assert measured.keys() == {"target_text_tokens"}


def test_tts_measures_input_text_and_output_audio():
    model = SimpleNamespace(sample_rate=22050)
    batch = (
        torch.zeros(2, 22050),
        torch.tensor([22050, 11025]),
        torch.zeros(2, 10, dtype=torch.long),
        torch.tensor([10, 6]),
        None,
        None,
        None,
    )

    measured = TTSThroughputPolicy().measure(model, batch)

    assert _total(measured["input_text_tokens"]) == 16
    assert _total(measured["output_audio_seconds"]) == 1.5


def test_speech_to_speech_reads_distinct_real_model_rate_locations():
    model = SimpleNamespace(
        perception=SimpleNamespace(preprocessor=SimpleNamespace(_sample_rate=16000)),
        audio_codec=SimpleNamespace(sample_rate=24000, output_sample_rate=48000),
    )
    batch = {
        "source_audio_lens": torch.tensor([16000, 8000]),
        "target_audio_lens": torch.tensor([24000, 48000]),
    }

    measured = SpeechToSpeechThroughputPolicy().measure(model, batch)

    assert _total(measured["input_audio_seconds"]) == 1.5
    assert _total(measured["output_audio_seconds"]) == 3.0


def test_duplex_stt_measures_nested_audio_and_text_only_sub_batches():
    model = SimpleNamespace(source_sample_rate=16000)
    batch = {
        "audio_data": {"source_audio_lens": torch.tensor([16000, 8000])},
        "text_data": {"text_token_lens": torch.tensor([7, 5, 4])},
    }

    measured = DuplexSTTThroughputPolicy().measure(model, batch)

    assert _total(measured["input_audio_seconds"]) == 1.5
    assert _total(measured["text_tokens"]) == 16


def test_audio_codec_uses_input_sample_rate_when_output_rate_differs():
    model = SimpleNamespace(sample_rate=16000, output_sample_rate=24000)
    batch = {"audio_lens": torch.tensor([16000, 8000])}

    measured = AudioCodecThroughputPolicy().measure(model, batch)

    assert measured.keys() == {"input_audio_seconds"}
    assert _total(measured["input_audio_seconds"]) == 1.5


def test_salm_automodel_uses_exact_mixed_modality_positions_for_packed_sequences():
    model = SimpleNamespace(sampling_rate=16000, _last_batch_num_tokens=37)
    audio_lengths = torch.tensor([16000, 8000])
    batch = {
        "packed_audio_samples": torch.zeros(24000),
        "audio_lens": audio_lengths,
        "input_ids": torch.arange(12),
        "text_cu_seqlens": torch.tensor([0, 5, 12]),
    }

    measured = SALMAutomodelThroughputPolicy().measure(model, batch)

    assert measured.keys() == {"input_audio_seconds", "model_sequence_positions"}
    assert measured["input_audio_seconds"].value.data_ptr() == audio_lengths.data_ptr()
    assert _total(measured["input_audio_seconds"]) == 1.5
    assert _total(measured["model_sequence_positions"]) == 37


def test_salm_variants_without_mixed_position_counter_keep_stable_audio_schema():
    model = SimpleNamespace(sampling_rate=16000, text_pad_id=0, audio_locator_tag_id=99)
    batch = {
        "audio_lens": torch.empty(0, dtype=torch.long),
        "input_ids": torch.tensor([5, 99, 6, 0, 7]),
        "text_cu_seqlens": torch.tensor([0, 2, 5]),
    }

    measured = SALMThroughputPolicy().measure(model, batch)

    assert measured.keys() == {"input_audio_seconds"}
    assert _total(measured["input_audio_seconds"]) == 0


def test_missing_sample_rate_omits_duration_instead_of_guessing():
    batch = (torch.zeros(1, 20), torch.tensor([20]), torch.ones(1, 2), torch.tensor([2]))

    measured = ASRThroughputPolicy().measure(object(), batch)

    assert measured.keys() == {"target_text_tokens"}
    assert _total(measured["target_text_tokens"]) == 2


def test_base_policy_never_reports_batch_size():
    batch = {"audio_lens": torch.tensor([10, 20, 30])}

    assert SpeechThroughputPolicy().measure(object(), batch) == {}
