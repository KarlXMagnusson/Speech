# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from nemo.utils.training_batch import all_data_parallel_ranks_true, infer_batch_size, reduce_batch_size


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


def test_infer_packed_salm_batch_uses_conversations_not_audio_segments():
    batch = {
        "input_ids": torch.arange(100),
        "text_cu_seqlens": torch.tensor([0, 60, 100]),
        "audio_lens": torch.tensor([16000, 12000, 8000, 4000]),
    }

    assert infer_batch_size(batch) == 2


def test_reduce_batch_size_collects_invalid_ranks_instead_of_skipping_collective(monkeypatch):
    module = type("Module", (), {"device": torch.device("cpu")})()
    seen = []

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)

    def fake_all_reduce(counts, op=None, group=None):
        seen.append(counts.tolist())
        # The remote rank inferred a valid batch of size three.
        counts += torch.tensor([3, 1])

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    assert reduce_batch_size(None, module) is None
    assert seen == [[0, 0]]


def test_all_data_parallel_ranks_true_requires_consensus(monkeypatch):
    module = type("Module", (), {"device": torch.device("cpu")})()

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def fake_all_reduce(flag, op=None, group=None):
        flag.zero_()  # Another rank cannot reuse the shared reduction.

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    assert not all_data_parallel_ranks_true(True, module)
