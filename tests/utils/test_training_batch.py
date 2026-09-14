# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from nemo.utils.training_batch import infer_batch_size


def test_infer_batch_size_from_asr_tuple():
    batch = (torch.zeros(3, 16000), torch.tensor([16000, 12000, 8000]), torch.zeros(3, 20))

    assert infer_batch_size(batch) == 3


def test_infer_batch_size_from_tts_mapping_fallback():
    batch = {"spectrogram": torch.zeros(4, 80, 100), "speaker": torch.arange(4)}

    assert infer_batch_size(batch) == 4


def test_infer_batch_size_sums_duplex_modalities():
    batch = {
        "audio_data": {"audio_lens": torch.tensor([100, 80])},
        "text_data": {"text_tokens": torch.zeros(3, 20, dtype=torch.long)},
    }

    assert infer_batch_size(batch) == 5


def test_infer_batch_size_from_sample_id_list():
    assert infer_batch_size({"sample_ids": ["a", "b", "c"]}) == 3


def test_infer_batch_size_does_not_treat_packed_tokens_as_examples():
    batch = {"input_ids": torch.arange(100), "attention_mask": torch.ones(100)}

    assert infer_batch_size(batch) is None
