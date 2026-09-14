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
from collections.abc import Mapping
from typing import Any, Dict, Optional

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from nv_one_logger.api.config import OneLoggerConfig, OneLoggerErrorHandlingStrategy
from nv_one_logger.training_telemetry.api.callbacks import on_app_start
from nv_one_logger.training_telemetry.api.config import TrainingTelemetryConfig
from nv_one_logger.training_telemetry.api.training_telemetry_provider import TrainingTelemetryProvider
from nv_one_logger.training_telemetry.integration.pytorch_lightning import TimeEventCallback as OneLoggerPTLCallback

from nemo.lightning.base_callback import BaseCallback

# Export all symbols for testing and usage
__all__ = ['OneLoggerNeMoCallback']


def _get_env_int(name: str) -> Optional[int]:
    value = os.environ.get(name)
    if value in (None, ''):
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _as_positive_int(value: Any) -> Optional[int]:
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _get_env_positive_int(name: str) -> Optional[int]:
    return _as_positive_int(os.environ.get(name))


def _get_world_size() -> int:
    return _get_env_positive_int('WORLD_SIZE') or _get_env_positive_int('SLURM_NTASKS') or 1


def _get_job_name() -> str:
    return os.environ.get('EXP_NAME') or os.environ.get('SLURM_JOB_NAME') or 'nemo-run'


def _get_config_value(config: Any, *path: str) -> Any:
    if not isinstance(config, Mapping):
        return None
    value = config
    try:
        for key in path:
            if value is None:
                return None
            value = value.get(key) if hasattr(value, 'get') else getattr(value, key, None)
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    return value


def _has_validation_data(trainer: Any) -> bool:
    num_val_batches = getattr(trainer, 'num_val_batches', None)
    if isinstance(num_val_batches, int) and num_val_batches > 0:
        return True
    if isinstance(num_val_batches, (list, tuple)) and any(num_val_batches):
        return True

    configs = (
        getattr(getattr(trainer, 'datamodule', None), 'cfg', None),
        getattr(getattr(trainer, 'lightning_module', None), 'cfg', None),
    )
    return any(_get_config_value(config, 'validation_ds') is not None for config in configs)


def get_one_logger_init_config() -> Dict[str, Any]:
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
        "application_name": "nemo",
        "session_tag_or_fn": session_tag,
        # Important fields with defaults - provide if available from config
        "enable_for_current_rank": _should_enable_for_current_rank(),
        "world_size_or_fn": world_size,
        "error_handling_strategy": OneLoggerErrorHandlingStrategy.DISABLE_QUIETLY_AND_REPORT_METRIC_ERROR,
    }

    return init_config


def _get_base_callback_config(
    trainer: Any,
    global_batch_size: int,
    seq_length: Optional[int],
    micro_batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Generate base configuration for OneLogger training telemetry.

    This function provides the common configuration needed for both NeMo v1 and v2.
    It extracts basic training information from trainer object and uses provided
    batch size and sequence length values.

    Args:
        trainer: PyTorch Lightning trainer instance
        global_batch_size: Global batch size (calculated by version-specific function)
        seq_length: Sequence length (calculated by version-specific function)

    Returns:
        Dictionary containing base training callback configuration
    """
    job_name = _get_job_name()
    world_size = _get_world_size()
    max_steps = _as_positive_int(getattr(trainer, 'max_steps', 1))
    log_every_n_steps = _as_positive_int(getattr(trainer, 'log_every_n_steps', 10)) or 1
    global_batch_size = _as_positive_int(global_batch_size) or 1
    micro_batch_size = _as_positive_int(micro_batch_size) or max(global_batch_size // world_size, 1)
    seq_length = _as_positive_int(seq_length)
    # Get PERF_VERSION_TAG from environment
    perf_version_tag = os.environ.get('PERF_VERSION_TAG', '0.0.0')

    # Calculate performance tag
    default_perf_tag = f"{job_name}_{perf_version_tag}_bf{global_batch_size}"
    if seq_length is not None:
        default_perf_tag += f"_se{seq_length}"
    default_perf_tag += f"_ws{world_size}"
    perf_tag = os.environ.get('NEMO_ONE_LOGGER_PERF_TAG', default_perf_tag)

    # Unknown or epoch-based targets should remain unset instead of reporting negative values.
    train_iterations_target = max_steps
    train_samples_target = max_steps * global_batch_size if max_steps is not None else None

    # Fallback values
    is_save_checkpoint_enabled = False
    is_validation_iterations_enabled = False
    save_checkpoint_strategy = "sync"

    checkpoint_callbacks = [
        cb
        for cb in trainer.callbacks
        if isinstance(cb, ModelCheckpoint) and getattr(cb, '_nemo_one_logger_instrumented', False)
    ]
    is_save_checkpoint_enabled = len(checkpoint_callbacks) > 0

    is_validation_iterations_enabled = _has_validation_data(trainer)

    # Check for async_save in trainer strategy (handle both dict and object cases)
    if hasattr(trainer, 'strategy') and trainer.strategy is not None:
        if isinstance(trainer.strategy, dict):
            if trainer.strategy.get('async_save', False):
                save_checkpoint_strategy = "async"
        else:
            if hasattr(trainer.strategy, 'async_save') and trainer.strategy.async_save:
                save_checkpoint_strategy = "async"

    for callback in checkpoint_callbacks:
        if hasattr(callback, 'async_save') and callback.async_save:
            save_checkpoint_strategy = "async"
            break

    # Base training telemetry configuration
    base_config = {
        # Performance tag (REQUIRED in TrainingTelemetryConfig)
        "perf_tag_or_fn": perf_tag,
        # Batch information (REQUIRED in TrainingTelemetryConfig)
        "global_batch_size_or_fn": global_batch_size,
        "micro_batch_size_or_fn": micro_batch_size,
        "seq_length_or_fn": seq_length,
        # Training targets
        "train_iterations_target_or_fn": train_iterations_target,
        "train_samples_target_or_fn": train_samples_target,
        # Logging frequency
        "log_every_n_train_iterations": log_every_n_steps,
        'is_validation_iterations_enabled_or_fn': is_validation_iterations_enabled,
        'is_save_checkpoint_enabled_or_fn': is_save_checkpoint_enabled,
        'save_checkpoint_strategy': save_checkpoint_strategy,
    }

    return base_config


def get_nemo_v1_callback_config(trainer: Any) -> Dict[str, Any]:
    """Generate OneLogger training configuration for any NeMo Lightning module.

    This function provides NeMo v1 specific configuration by extracting values from
    the exp_manager_config object and trainer object.

    Args:
        trainer: PyTorch Lightning trainer instance

    Returns:
        Dictionary containing NeMo v1 training callback configuration
    """
    model_cfg = getattr(getattr(trainer, 'lightning_module', None), 'cfg', None)
    datamodule = getattr(trainer, 'datamodule', None)
    data_cfg = getattr(datamodule, 'cfg', None)

    micro_batch_size = _get_env_positive_int('NEMO_ONE_LOGGER_MICRO_BATCH_SIZE')
    if micro_batch_size is None:
        for config in (data_cfg, model_cfg):
            micro_batch_size = _as_positive_int(_get_config_value(config, 'train_ds', 'batch_size'))
            if micro_batch_size is not None:
                break

            bucket_batch_sizes = _get_config_value(config, 'train_ds', 'bucket_batch_size')
            if bucket_batch_sizes:
                try:
                    micro_batch_size = _as_positive_int(sum(bucket_batch_sizes) / len(bucket_batch_sizes))
                except TypeError:
                    micro_batch_size = None
                if micro_batch_size is not None:
                    break

    micro_batch_size = micro_batch_size or 1

    data_parallel_size = _get_world_size()
    get_data_parallel_size = getattr(datamodule, '_get_world_size', None)
    if isinstance(data_cfg, Mapping) and callable(get_data_parallel_size):
        try:
            data_parallel_size = int(get_data_parallel_size())
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
            pass

    global_batch_size = _get_env_positive_int('NEMO_ONE_LOGGER_GLOBAL_BATCH_SIZE')
    if global_batch_size is None:
        global_batch_size = micro_batch_size * data_parallel_size

    seq_length = _get_env_positive_int('NEMO_ONE_LOGGER_SEQUENCE_LENGTH')
    if seq_length is None:
        for config in (data_cfg, model_cfg):
            for field in ('seq_length', 'max_seq_length'):
                value = _get_config_value(config, 'train_ds', field)
                seq_length = _as_positive_int(value)
                if seq_length is not None:
                    break
            if seq_length is not None:
                break

    # Get base configuration with calculated values
    config = _get_base_callback_config(
        trainer=trainer,
        global_batch_size=global_batch_size,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
    )

    return config


def _should_enable_for_current_rank() -> bool:
    """Determine if OneLogger should be enabled for the current rank.

    Uses environment variables instead of torch.distributed to avoid circular imports.
    In distributed training, typically only rank 0 (or the last rank) should
    enable OneLogger to avoid duplicate telemetry data.

    Returns:
        True if OneLogger should be enabled for the current rank, False otherwise
    """
    enabled = os.environ.get('NEMO_ONE_LOGGER_ENABLED')
    if enabled is not None and enabled.lower() in {'0', 'false', 'no', 'off'}:
        return False

    rank = _get_env_int('RANK')
    if rank is None:
        rank = _get_env_int('SLURM_PROCID')
    if rank is None:
        local_rank = _get_env_int('LOCAL_RANK')
        if local_rank is not None:
            node_rank = _get_env_int('NODE_RANK') or 0
            local_world_size = _get_env_positive_int('LOCAL_WORLD_SIZE') or 1
            rank = node_rank * local_world_size + local_rank
    if rank is None:
        return enabled is not None and enabled.lower() in {'1', 'true', 'yes', 'on'}
    return rank == 0


class OneLoggerNeMoCallback(OneLoggerPTLCallback, BaseCallback):
    """Adapter extending OneLogger's PTL callback with init + config update.

    __init__ configures the provider from meta info, then calls super().__init__.
    update_config computes TrainingTelemetryConfig and applies it.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, '_initialized', False):
            return
        init_config = get_one_logger_init_config()
        self.enabled_for_current_rank = init_config.get("enable_for_current_rank", True)
        one_logger_config = OneLoggerConfig(**init_config)
        TrainingTelemetryProvider.instance().with_base_config(
            one_logger_config
        ).with_export_config().configure_provider()
        # Initialize underlying OneLogger PTL callback
        super().__init__(TrainingTelemetryProvider.instance(), call_on_app_start=False)
        # Explicitly signal application start after provider configuration.
        if self.enabled_for_current_rank:
            on_app_start()
        self._initialized = True

    def setup(self, trainer: Trainer, pl_module: Any, stage: str) -> None:
        if stage == 'fit':
            self.update_config(nemo_version='lightning', trainer=trainer)

    def update_config(self, nemo_version: str, trainer: Trainer, **kwargs) -> None:
        # Avoid this function being called multiple times
        if TrainingTelemetryProvider.instance().config.telemetry_config is not None:
            return
        else:
            config = get_nemo_v1_callback_config(trainer=trainer)
        training_telemetry_config = TrainingTelemetryConfig(**config)
        TrainingTelemetryProvider.instance().set_training_telemetry_config(training_telemetry_config)
