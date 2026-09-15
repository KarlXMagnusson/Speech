# SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OneLogger lifecycle tracing for NeMo Speech."""

import os
from typing import Any

from nv_one_logger.api.config import OneLoggerConfig, OneLoggerErrorHandlingStrategy
from nv_one_logger.core.attributes import Attributes
from nv_one_logger.core.event import Event
from nv_one_logger.training_telemetry.api.callbacks import on_app_end, on_app_start
from nv_one_logger.training_telemetry.api.training_telemetry_provider import TrainingTelemetryProvider

from nemo.lightning.base_callback import BaseCallback

__all__ = ["OneLoggerNeMoCallback"]

_SPAN_MODEL_INIT = "nemo_speech.model_initialization"
_SPAN_DATALOADER_INIT = "nemo_speech.data_loader_initialization"
_SPAN_OPTIMIZER_INIT = "nemo_speech.optimizer_initialization"
_SPAN_CHECKPOINT_LOAD = "nemo_speech.checkpoint_load"
_SPAN_CHECKPOINT_SAVE = "nemo_speech.checkpoint_save"
_SPAN_TRAINING = "nemo_speech.training"
_SPAN_VALIDATION = "nemo_speech.validation"


def _get_env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value in (None, ""):
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _get_world_size() -> int:
    for name in ("WORLD_SIZE", "SLURM_NTASKS"):
        value = _get_env_int(name)
        if value is not None and value > 0:
            return value
    return 1


def _get_job_name() -> str:
    return os.environ.get("EXP_NAME") or os.environ.get("SLURM_JOB_NAME") or "nemo-run"


def _get_rank() -> int | None:
    """Resolve a global rank from torchrun, Slurm, or Lightning launcher metadata."""
    rank = _get_env_int("RANK")
    if rank is None:
        rank = _get_env_int("SLURM_PROCID")
    if rank is None:
        local_rank = _get_env_int("LOCAL_RANK")
        if local_rank is not None:
            node_rank = _get_env_int("NODE_RANK") or 0
            local_world_size = _get_env_int("LOCAL_WORLD_SIZE") or 1
            rank = node_rank * local_world_size + local_rank
    return rank


def _is_explicitly_disabled() -> bool:
    enabled = os.environ.get("NEMO_ONE_LOGGER_ENABLED")
    return enabled is not None and enabled.lower() in {"0", "false", "no", "off"}


def _should_enable_for_current_rank() -> bool:
    """Export on rank zero; require explicit opt-in when no launcher rank exists."""
    if _is_explicitly_disabled():
        return False
    rank = _get_rank()
    if rank is not None:
        return rank == 0
    enabled = os.environ.get("NEMO_ONE_LOGGER_ENABLED")
    return enabled is not None and enabled.lower() in {"1", "true", "yes", "on"}


def get_one_logger_init_config() -> dict[str, Any]:
    """Build modality-independent OneLogger application configuration."""
    return {
        "application_name": "nemo-speech",
        "session_tag_or_fn": _get_job_name(),
        "enable_for_current_rank": _should_enable_for_current_rank(),
        "world_size_or_fn": _get_world_size(),
        "error_handling_strategy": OneLoggerErrorHandlingStrategy.DISABLE_QUIETLY_AND_REPORT_METRIC_ERROR,
    }


class OneLoggerNeMoCallback(BaseCallback):
    """Trace NeMo Speech lifecycle without imposing an LLM workload schema.

    NeMo Speech batches may contain waveforms, frames, text, codec tokens, or a
    mixture of modalities, often with dynamic batching and gradient
    accumulation. OneLogger's training-progress schema requires a universal
    batch size and optionally derives token counts from one sequence length, so
    this adapter intentionally reports lifecycle timing only.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        init_config = get_one_logger_init_config()
        self.enabled_for_current_rank = init_config["enable_for_current_rank"]
        self._provider = None
        self._application_span = None
        self._active_spans = []
        self._initialized = True
        if not self.enabled_for_current_rank:
            return

        provider = TrainingTelemetryProvider.instance()
        provider.with_base_config(OneLoggerConfig(**init_config)).with_export_config().configure_provider()
        self._provider = provider
        self._application_span = on_app_start()

    def _start_span(self, name: str, attributes: Attributes | None = None) -> None:
        if self._provider is None:
            return
        span = self._provider.recorder.start(name, span_attributes=attributes)
        if span is not None:
            self._active_spans.append((name, span))

    def _current_span(self, name: str):
        return next((span for span_name, span in reversed(self._active_spans) if span_name == name), None)

    def _stop_span(self, name: str) -> None:
        if self._provider is None:
            return
        for index in range(len(self._active_spans) - 1, -1, -1):
            span_name, span = self._active_spans[index]
            if span_name == name:
                self._active_spans.pop(index)
                self._provider.recorder.stop(span)
                return

    def on_app_end(self) -> None:
        if self._provider is None:
            return
        while self._active_spans:
            _, span = self._active_spans.pop()
            self._provider.recorder.stop(span)
        on_app_end()

    def on_model_init_start(self) -> None:
        self._start_span(_SPAN_MODEL_INIT)

    def on_model_init_end(self) -> None:
        self._stop_span(_SPAN_MODEL_INIT)

    def on_dataloader_init_start(self) -> None:
        self._start_span(_SPAN_DATALOADER_INIT)

    def on_dataloader_init_end(self) -> None:
        self._stop_span(_SPAN_DATALOADER_INIT)

    def on_optimizer_init_start(self) -> None:
        self._start_span(_SPAN_OPTIMIZER_INIT)

    def on_optimizer_init_end(self) -> None:
        self._stop_span(_SPAN_OPTIMIZER_INIT)

    def on_load_checkpoint_start(self) -> None:
        self._start_span(_SPAN_CHECKPOINT_LOAD)

    def on_load_checkpoint_end(self) -> None:
        self._stop_span(_SPAN_CHECKPOINT_LOAD)

    def on_save_checkpoint_start(self, global_step: int, async_save: bool = False) -> None:
        self._start_span(
            _SPAN_CHECKPOINT_SAVE,
            Attributes({"global_step": global_step, "asynchronous": async_save}),
        )

    def on_save_checkpoint_success(self, global_step: int) -> None:
        self._checkpoint_event("nemo_speech.checkpoint_save_success", global_step)

    def on_save_checkpoint_failure(self, global_step: int) -> None:
        self._checkpoint_event("nemo_speech.checkpoint_save_failure", global_step)

    def _checkpoint_event(self, name: str, global_step: int) -> None:
        if self._provider is None:
            return
        span = self._current_span(_SPAN_CHECKPOINT_SAVE) or self._application_span
        if span is not None:
            self._provider.recorder.event(span, Event.create(name, Attributes({"global_step": global_step})))

    def on_save_checkpoint_end(self, global_step: int | None = None) -> None:
        del global_step
        self._stop_span(_SPAN_CHECKPOINT_SAVE)

    def on_train_start(self, trainer: Any, pl_module: Any) -> None:
        del trainer, pl_module
        self._start_span(_SPAN_TRAINING)

    def on_train_end(self, trainer: Any, pl_module: Any) -> None:
        del trainer, pl_module
        self._stop_span(_SPAN_TRAINING)

    def on_validation_start(self, trainer: Any, pl_module: Any) -> None:
        del trainer, pl_module
        self._start_span(_SPAN_VALIDATION)

    def on_validation_end(self, trainer: Any, pl_module: Any) -> None:
        del trainer, pl_module
        self._stop_span(_SPAN_VALIDATION)
