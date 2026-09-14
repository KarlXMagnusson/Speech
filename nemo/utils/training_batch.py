# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime accounting helpers shared by training telemetry callbacks."""

from collections.abc import Mapping
from typing import Any

import torch
import torch.distributed as dist
from lightning.pytorch import LightningModule

__all__ = ["get_data_parallel_group", "infer_batch_size", "reduce_batch_counts"]


def infer_batch_size(batch: Any) -> int | None:
    """Infer the number of examples in a realized, already-collated batch.

    Sampler configuration such as ``batch_duration``, ``batch_tokens``, or
    per-bucket batch sizes describes a limit, not the batch that was actually
    produced. This helper therefore only inspects runtime data.
    """
    if isinstance(batch, Mapping):
        combined = [batch.get(key) for key in ("audio_data", "text_data") if batch.get(key) is not None]
        if combined:
            sizes = [infer_batch_size(part) for part in combined]
            return sum(sizes) if all(size is not None for size in sizes) else None

        for key in (
            "audio_lens",
            "audio_len",
            "audio_signal_length",
            "input_signal_length",
            "input_lengths",
            "sample_id",
            "sample_ids",
        ):
            if key in batch:
                size = infer_batch_size(batch[key])
                if size is not None:
                    return size
        for key in ("input_ids", "text_tokens", "tokens"):
            if key in batch:
                value = batch[key]
                shape = getattr(value, "shape", None)
                if shape is not None and len(shape) < 2:
                    # A flattened packed sequence has no sample dimension.
                    return None
                return infer_batch_size(value)
        for value in batch.values():
            size = infer_batch_size(value)
            if size is not None:
                return size
        return None
    if torch.is_tensor(batch):
        return int(batch.shape[0]) if batch.ndim > 0 else None
    shape = getattr(batch, "shape", None)
    if shape is not None:
        return int(shape[0]) if len(shape) > 0 else None
    if isinstance(batch, (list, tuple)):
        if not batch:
            return None
        if all(isinstance(item, Mapping) for item in batch):
            return len(batch)
        if all(not isinstance(item, (Mapping, list, tuple)) and not torch.is_tensor(item) for item in batch):
            return len(batch)
        return infer_batch_size(batch[0])
    values = getattr(batch, "__dict__", None)
    if isinstance(values, Mapping):
        return infer_batch_size(values)
    return None


def get_data_parallel_group(pl_module: LightningModule):
    """Return a DP-only process group, or the default group for plain DDP."""
    device_mesh = getattr(pl_module, "_device_mesh", None)
    if device_mesh is None:
        device_mesh = getattr(pl_module, "device_mesh", None)
    if device_mesh is None:
        trainer = getattr(pl_module, "trainer", None)
        trainer_model = getattr(trainer, "model", None)
        device_mesh = getattr(trainer_model, "device_mesh", None)
    if device_mesh is None:
        return None

    names = device_mesh.mesh_dim_names or ()
    if "data_parallel" in names:
        return device_mesh["data_parallel"].get_group()

    try:
        from nemo_automodel.components.distributed.mesh_utils import get_flat_mesh

        return get_flat_mesh(device_mesh, "dp").get_group()
    except (ImportError, KeyError, RuntimeError, ValueError):
        # Automodel is optional; fall through to other DeviceMesh conventions.
        pass

    try:
        return device_mesh["dp"].get_group()
    except (KeyError, RuntimeError, ValueError):
        # The mesh may use explicit replicate/shard dimension names below.
        pass

    if "dp_shard" in names and "dp_replicate" in names:
        return device_mesh["dp_replicate", "dp_shard"].get_group()
    if "dp_shard" in names:
        return device_mesh["dp_shard"].get_group()
    return None


def reduce_batch_counts(local_tokens: int, local_examples: int, pl_module: LightningModule) -> tuple[int, int]:
    """Sum runtime token/example counts across data-parallel replicas."""
    if not (dist.is_available() and dist.is_initialized()):
        return local_tokens, local_examples
    counts = torch.tensor([local_tokens, local_examples], dtype=torch.long, device=pl_module.device)
    dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=get_data_parallel_group(pl_module))
    global_tokens, global_examples = counts.tolist()
    return int(global_tokens), int(global_examples)
