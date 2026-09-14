# SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""
OneLogger callback for NeMo training.

This module provides a callback that integrates OneLogger telemetry with NeMo training.
"""

import os
import time
from collections.abc import Mapping
from typing import Any

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from nv_one_logger.api.config import OneLoggerConfig, OneLoggerErrorHandlingStrategy
from nv_one_logger.training_telemetry.api.callbacks import (
    on_app_start,
    on_save_checkpoint_end,
    on_save_checkpoint_start,
    on_save_checkpoint_success,
    on_train_end,
    on_train_start,
    on_training_single_iteration_end,
    on_training_single_iteration_start,
)
from nv_one_logger.training_telemetry.api.config import TrainingTelemetryConfig
from nv_one_logger.training_telemetry.api.training_telemetry_provider import TrainingTelemetryProvider
from nv_one_logger.training_telemetry.integration.pytorch_lightning import TimeEventCallback as OneLoggerPTLCallback

from nemo.lightning.base_callback import BaseCallback
from nemo.utils.training_batch import all_data_parallel_ranks_true, infer_batch_size, reduce_batch_size

# Export all symbols for testing and usage
__all__ = ["OneLoggerNeMoCallback"]


def _get_env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value in (None, ""):
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _as_positive_int(value: Any) -> int | None:
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _get_env_positive_int(name: str) -> int | None:
    return _as_positive_int(os.environ.get(name))


def _get_world_size() -> int:
    return _get_env_positive_int("WORLD_SIZE") or _get_env_positive_int("SLURM_NTASKS") or 1


def _get_job_name() -> str:
    return os.environ.get("EXP_NAME") or os.environ.get("SLURM_JOB_NAME") or "nemo-run"


def _get_config_value(config: Any, *path: str) -> Any:
    if not isinstance(config, Mapping):
        return None
    value = config
    try:
        for key in path:
            if value is None:
                return None
            value = value.get(key) if hasattr(value, "get") else getattr(value, key, None)
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    return value


def _has_validation_data(trainer: Any) -> bool:
    num_val_batches = getattr(trainer, "num_val_batches", None)
    if isinstance(num_val_batches, int) and num_val_batches > 0:
        return True
    if isinstance(num_val_batches, (list, tuple)) and any(num_val_batches):
        return True

    configs = (
        getattr(getattr(trainer, "datamodule", None), "cfg", None),
        getattr(getattr(trainer, "lightning_module", None), "cfg", None),
    )
    return any(_get_config_value(config, "validation_ds") is not None for config in configs)


def get_one_logger_init_config() -> dict[str, Any]:
    """Generate minimal configuration for OneLogger initialization.

    This function provides the absolute minimal configuration needed for OneLogger initialization.
    It only includes the required fields and uses defaults for everything else to avoid
    dependencies on exp_manager during early import.

    Returns:
        Dictionary containing minimal initialization configuration
    """
    session_tag = _get_job_name()
    world_size = _get_world_size()

    # Minimal configuration - required fields only
    init_config = {
        # Required fields (from OneLoggerConfig) - no defaults
        "application_name": "nemo-speech",
        "session_tag_or_fn": session_tag,
        # Important fields with defaults - provide if available from config
        "enable_for_current_rank": _should_enable_for_current_rank(),
        "world_size_or_fn": world_size,
        "error_handling_strategy": OneLoggerErrorHandlingStrategy.DISABLE_QUIETLY_AND_REPORT_METRIC_ERROR,
    }

    return init_config


def _get_base_callback_config(
    trainer: Any,
    global_batch_size: Any,
    seq_length: int | None,
) -> dict[str, Any]:
    """Build telemetry config from measured work or explicit fixed overrides."""
    job_name = _get_job_name()
    world_size = _get_world_size()
    max_steps = _as_positive_int(getattr(trainer, "max_steps", None))
    log_every_n_steps = _as_positive_int(getattr(trainer, "log_every_n_steps", 10)) or 1
    fixed_global_batch_size = _as_positive_int(global_batch_size) if not callable(global_batch_size) else None
    if fixed_global_batch_size is None and not callable(global_batch_size):
        raise ValueError("OneLogger global batch size must be measured at runtime or explicitly configured")
    seq_length = _as_positive_int(seq_length)

    perf_version_tag = os.environ.get("PERF_VERSION_TAG", "0.0.0")
    default_perf_tag = f"{job_name}_{perf_version_tag}"
    if fixed_global_batch_size is not None:
        default_perf_tag += f"_bf{fixed_global_batch_size}"
    if seq_length is not None:
        default_perf_tag += f"_se{seq_length}"
    default_perf_tag += f"_ws{world_size}"

    checkpoint_callbacks = [
        callback
        for callback in trainer.callbacks
        if isinstance(callback, ModelCheckpoint) and getattr(callback, "_nemo_one_logger_instrumented", False)
    ]
    save_checkpoint_strategy = "sync"
    strategy = getattr(trainer, "strategy", None)
    if (isinstance(strategy, dict) and strategy.get("async_save", False)) or getattr(strategy, "async_save", False):
        save_checkpoint_strategy = "async"
    if any(getattr(callback, "async_save", False) for callback in checkpoint_callbacks):
        save_checkpoint_strategy = "async"

    return {
        "perf_tag_or_fn": os.environ.get("NEMO_ONE_LOGGER_PERF_TAG", default_perf_tag),
        "global_batch_size_or_fn": global_batch_size,
        # OneLogger derives cumulative tokens as current sequence length times
        # cumulative samples. That is correct only for an explicitly fixed length.
        "seq_length_or_fn": seq_length,
        "train_iterations_target_or_fn": max_steps if getattr(trainer, "accumulate_grad_batches", 1) == 1 else None,
        "train_samples_target_or_fn": (
            max_steps * fixed_global_batch_size
            if max_steps is not None
            and fixed_global_batch_size
            and getattr(trainer, "accumulate_grad_batches", 1) == 1
            else None
        ),
        "log_every_n_train_iterations": log_every_n_steps,
        "is_validation_iterations_enabled_or_fn": _has_validation_data(trainer),
        "is_save_checkpoint_enabled_or_fn": bool(checkpoint_callbacks),
        "save_checkpoint_strategy": save_checkpoint_strategy,
    }


def get_nemo_v1_callback_config(
    trainer: Any,
    global_batch_size: Any = None,
) -> dict[str, Any]:
    """Generate OneLogger config without guessing from speech sampler budgets.

    ``batch_duration``, ``batch_tokens``, ``bucket_batch_size``, and related
    settings are limits or bucket policies. They do not describe the realized
    batch. The callback supplies callables backed by runtime measurements unless
    fixed values are explicitly provided through the legacy environment knobs.
    """
    fixed_global_batch_size = _get_env_positive_int("NEMO_ONE_LOGGER_GLOBAL_BATCH_SIZE")
    global_batch_size = fixed_global_batch_size or global_batch_size
    return _get_base_callback_config(
        trainer=trainer,
        global_batch_size=global_batch_size,
        seq_length=_get_env_positive_int("NEMO_ONE_LOGGER_SEQUENCE_LENGTH"),
    )


def _get_rank() -> int | None:
    """Resolve a global rank from torchrun, Slurm, or Lightning launcher metadata."""
    rank = _get_env_int("RANK")
    if rank is None:
        rank = _get_env_int("SLURM_PROCID")
    if rank is None:
        local_rank = _get_env_int("LOCAL_RANK")
        if local_rank is not None:
            node_rank = _get_env_int("NODE_RANK") or 0
            local_world_size = _get_env_positive_int("LOCAL_WORLD_SIZE") or 1
            rank = node_rank * local_world_size + local_rank
    return rank


def _is_explicitly_disabled() -> bool:
    enabled = os.environ.get("NEMO_ONE_LOGGER_ENABLED")
    return enabled is not None and enabled.lower() in {"0", "false", "no", "off"}


def _should_enable_for_current_rank() -> bool:
    """Return whether this process should export OneLogger telemetry."""
    if _is_explicitly_disabled():
        return False
    rank = _get_rank()
    if rank is None:
        enabled = os.environ.get("NEMO_ONE_LOGGER_ENABLED")
        return enabled is not None and enabled.lower() in {"1", "true", "yes", "on"}
    return rank == 0


def _should_participate_on_current_rank() -> bool:
    """Return whether this rank must join dynamic batch-count collectives."""
    return not _is_explicitly_disabled() and (_get_rank() is not None or _should_enable_for_current_rank())


class OneLoggerNeMoCallback(OneLoggerPTLCallback, BaseCallback):
    """OneLogger adapter with exact runtime accounting for dynamic speech batches."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        init_config = get_one_logger_init_config()
        self.enabled_for_current_rank = init_config.get("enable_for_current_rank", True)
        self.participates_on_all_ranks = _should_participate_on_current_rank()
        one_logger_config = OneLoggerConfig(**init_config)
        provider = TrainingTelemetryProvider.instance()
        provider.with_base_config(one_logger_config).with_export_config().configure_provider()
        self._provider = provider
        super().__init__(self._provider, call_on_app_start=False)
        if self.enabled_for_current_rank:
            on_app_start()

        self.num_samples_total = 0
        self._current_global_batch_size = 0
        self._train_start_time_msec = None
        self._batch_start_time_msec = None
        self._batch_token = 0
        self._train_iterations_start = 0
        self._reuse_training_stats = False
        self._training_started = False
        self._iteration_started = False
        self._configured = self._provider.config.telemetry_config is not None
        self._fixed_global_batch_size = _get_env_positive_int("NEMO_ONE_LOGGER_GLOBAL_BATCH_SIZE")
        self._initialized = True

    def state_dict(self) -> dict[str, Any]:
        return {"num_samples_total": int(self.num_samples_total)}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.num_samples_total = int(state_dict.get("num_samples_total", 0))

    def setup(self, trainer: Trainer, pl_module: Any, stage: str) -> None:
        if stage == "fit":
            if self._fixed_global_batch_size is None:
                callbacks = list(getattr(trainer, "callbacks", ()))
                callback_index = callbacks.index(self) if self in callbacks else 0
                stats_runs_first = any(
                    type(cb).__module__ == "nemo.utils.callbacks.training_stats"
                    and type(cb).__name__ == "TrainingStatsCallback"
                    for cb in callbacks[:callback_index]
                )
                # One consensus at setup guarantees every rank takes the same
                # per-batch path: reuse TrainingStats, or perform our own reduction.
                self._reuse_training_stats = all_data_parallel_ranks_true(stats_runs_first, pl_module)
            self.update_config(nemo_version="lightning", trainer=trainer)

    def update_config(self, nemo_version: str, trainer: Trainer, **kwargs) -> None:
        del nemo_version, kwargs
        if self._provider.config.telemetry_config is not None:
            self._configured = True
            return
        if self._fixed_global_batch_size is not None:
            self._current_global_batch_size = self._fixed_global_batch_size
            self._configure(trainer, fixed=True)

    def _configure(self, trainer: Trainer, fixed: bool = False) -> None:
        if self._configured:
            return
        global_batch_size = self._fixed_global_batch_size if fixed else lambda: self._current_global_batch_size
        config = get_nemo_v1_callback_config(
            trainer=trainer,
            global_batch_size=global_batch_size,
        )
        self._provider.set_training_telemetry_config(TrainingTelemetryConfig(**config))
        self._configured = True

    def _measure_batch(self, batch: Any, pl_module: Any) -> bool:
        if self._reuse_training_stats:
            marker_matches = getattr(pl_module, "_last_batch_stats_batch_token", None) == self._batch_token
            global_examples = (
                _as_positive_int(getattr(pl_module, "_last_batch_global_num_examples", None))
                if marker_matches
                else None
            )
        else:
            local_examples = _as_positive_int(getattr(pl_module, "_last_batch_num_examples", None))
            if local_examples is None:
                local_examples = infer_batch_size(batch)
            # Every DP rank enters this collective, even when local inference
            # failed, preventing asymmetric batch structures from deadlocking.
            global_examples = reduce_batch_size(local_examples, pl_module)
        if global_examples is None or global_examples <= 0:
            return False
        self._current_global_batch_size = global_examples
        return True

    def _start_training(self) -> None:
        if self._training_started or not self._configured or not self.enabled_for_current_rank:
            return
        on_train_start(
            train_iterations_start=self._train_iterations_start,
            train_samples_start=self.num_samples_total,
            start_time_msec=self._train_start_time_msec,
        )
        self._training_started = True

    def on_train_start(self, trainer: Trainer, pl_module: Any) -> None:
        self._train_start_time_msec = time.time() * 1000
        self._train_iterations_start = trainer.global_step
        if self._fixed_global_batch_size is not None:
            self._start_training()

    def on_train_batch_start(self, trainer: Trainer, pl_module: Any, batch: Any, batch_idx: int) -> None:
        del batch, batch_idx
        self._batch_start_time_msec = time.time() * 1000
        self._batch_token += 1
        setattr(pl_module, "_one_logger_batch_token", self._batch_token)
        setattr(pl_module, "_last_batch_stats_batch_token", None)
        if self._fixed_global_batch_size is not None:
            self._start_training()
            if self._training_started:
                on_training_single_iteration_start(start_time_msec=self._batch_start_time_msec)
                self._iteration_started = True

    def on_train_batch_end(self, trainer: Trainer, pl_module: Any, outputs: Any, batch: Any, batch_idx: int) -> None:
        """Finish telemetry for a measured training batch."""
        del outputs, batch_idx
        if self._fixed_global_batch_size is None:
            if not self._measure_batch(batch, pl_module):
                return
            if not self._configured:
                self._configure(trainer)
        self._start_training()
        if self._training_started and not self._iteration_started:
            # Dynamic batch sizes are known only now. Preserve the true start
            # timestamp while avoiding an open span if measurement fails.
            on_training_single_iteration_start(start_time_msec=self._batch_start_time_msec)
            self._iteration_started = True
        if self._iteration_started:
            on_training_single_iteration_end()
            self._iteration_started = False
        if self._configured and self._current_global_batch_size > 0:
            self.num_samples_total += self._current_global_batch_size

    def on_train_end(self, trainer: Trainer, pl_module: Any) -> None:
        del trainer, pl_module
        if self._training_started and self.enabled_for_current_rank:
            on_train_end()

    def on_validation_start(self, trainer: Trainer, pl_module: Any) -> None:
        if self._configured and self.enabled_for_current_rank:
            super().on_validation_start(trainer, pl_module)

    def on_validation_end(self, trainer: Trainer, pl_module: Any) -> None:
        if self._configured and self.enabled_for_current_rank:
            super().on_validation_end(trainer, pl_module)

    def on_validation_batch_start(
        self,
        trainer: Trainer,
        pl_module: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self._configured and self.enabled_for_current_rank:
            super().on_validation_batch_start(trainer, pl_module, batch, batch_idx, dataloader_idx)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Finish validation batch telemetry after training configuration exists."""
        if self._configured and self.enabled_for_current_rank:
            super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_save_checkpoint_start(self, global_step: int) -> None:
        if self._configured and self.enabled_for_current_rank:
            on_save_checkpoint_start(global_step)

    def on_save_checkpoint_success(self, global_step: int) -> None:
        if self._configured and self.enabled_for_current_rank:
            on_save_checkpoint_success(global_step)

    def on_save_checkpoint_end(self, global_step: int | None = None) -> None:
        del global_step
        if self._configured and self.enabled_for_current_rank:
            on_save_checkpoint_end()
