# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""Unit tests for OneLoggerNeMoCallback."""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from lightning.pytorch.callbacks import Callback as PTLCallback
from omegaconf import OmegaConf

from nemo.core.classes.modelPT import ModelPT
from nemo.lightning.base_callback import BaseCallback
from nemo.lightning.callback_group import CallbackGroup, callback_context, with_model_init_callbacks
from nemo.lightning.one_logger_callback import (
    OneLoggerNeMoCallback,
    _get_base_callback_config,
    _should_enable_for_current_rank,
    get_nemo_v1_callback_config,
    get_one_logger_init_config,
)
from nemo.utils.callbacks.nemo_model_checkpoint import NeMoModelCheckpoint


@pytest.fixture(autouse=True)
def reset_one_logger_callback_singleton():
    """Isolate singleton initialization assertions from import-time setup."""
    previous_instance = OneLoggerNeMoCallback._instance
    OneLoggerNeMoCallback._instance = None
    yield
    OneLoggerNeMoCallback._instance = previous_instance


class TestOneLoggerNeMoCallback:
    """Test suite for OneLoggerNeMoCallback."""

    def test_inheritance(self):
        """Test that OneLoggerNeMoCallback properly inherits from both parent classes."""
        with (
            patch('nemo.lightning.one_logger_callback.OneLoggerPTLCallback') as mock_ptl_callback,
            patch('nemo.lightning.one_logger_callback.TrainingTelemetryProvider') as mock_provider,
            patch('nemo.lightning.one_logger_callback.get_one_logger_init_config') as mock_get_config,
            patch('nemo.lightning.one_logger_callback.OneLoggerConfig') as mock_config_class,
            patch('nemo.lightning.one_logger_callback.on_app_start') as mock_on_app_start,
        ):

            # Setup mocks
            mock_get_config.return_value = {"application_name": "test", "session_tag_or_fn": "test-session"}
            mock_config_instance = MagicMock()
            mock_config_class.return_value = mock_config_instance
            mock_provider_instance = MagicMock()
            mock_provider_instance.config = MagicMock()
            mock_provider_instance.config.telemetry_config = None
            mock_provider.instance.return_value = mock_provider_instance
            mock_ptl_callback_instance = MagicMock()
            mock_ptl_callback.return_value = mock_ptl_callback_instance

            # Create callback instance
            callback = OneLoggerNeMoCallback()

            # Test inheritance
            assert isinstance(callback, BaseCallback)
            assert isinstance(callback, PTLCallback)

    @patch('nemo.lightning.one_logger_callback.TrainingTelemetryProvider')
    @patch('nemo.lightning.one_logger_callback.get_one_logger_init_config')
    @patch('nemo.lightning.one_logger_callback.OneLoggerConfig')
    @patch('nemo.lightning.one_logger_callback.on_app_start')
    @patch('nemo.lightning.one_logger_callback.OneLoggerPTLCallback.__init__', return_value=None)
    def test_init_configures_provider(
        self, mock_ptl_callback_init, mock_on_app_start, mock_config_class, mock_get_config, mock_provider
    ):
        """Test that __init__ properly configures the OneLogger provider."""
        # Setup mocks
        mock_init_config = {
            "application_name": "nemo",
            "session_tag_or_fn": "test-session",
            "enable_for_current_rank": True,
            "world_size_or_fn": 1,
        }
        mock_get_config.return_value = mock_init_config

        mock_config_instance = MagicMock()
        mock_config_class.return_value = mock_config_instance

        mock_provider_instance = MagicMock()
        mock_provider_instance.config = MagicMock()
        mock_provider_instance.config.telemetry_config = None
        mock_provider.instance.return_value = mock_provider_instance

        # Create callback instance
        OneLoggerNeMoCallback()

        # Verify initialization sequence
        mock_get_config.assert_called_once()
        mock_config_class.assert_called_once_with(**mock_init_config)
        mock_provider_instance.with_base_config.assert_called_once_with(mock_config_instance)
        mock_provider_instance.with_base_config.return_value.with_export_config.assert_called_once()
        mock_provider_instance.with_base_config.return_value.with_export_config.return_value.configure_provider.assert_called_once()
        mock_ptl_callback_init.assert_called_once_with(mock_provider_instance, call_on_app_start=False)
        mock_on_app_start.assert_called_once_with()


class TestOneLoggerCallback:
    """Test cases for one_logger_callback utility functions."""

    @pytest.mark.unit
    def test_get_one_logger_init_config(self):
        """Test get_one_logger_init_config returns correct minimal configuration."""
        with patch.dict(os.environ, {"SLURM_JOB_NAME": "test_job", "WORLD_SIZE": "4"}):
            config = get_one_logger_init_config()

            assert isinstance(config, dict)
            assert config["application_name"] == "nemo-speech"
            assert config["session_tag_or_fn"] == "test_job"
            assert "enable_for_current_rank" in config
            assert config["world_size_or_fn"] == 4

    @pytest.mark.unit
    def test_get_one_logger_init_config_no_slurm(self):
        """Test get_one_logger_init_config when SLURM_JOB_NAME is not set."""
        with patch.dict(os.environ, {"WORLD_SIZE": "1"}, clear=True):
            config = get_one_logger_init_config()

            assert config["session_tag_or_fn"] == "nemo-run"
            assert config["world_size_or_fn"] == 1

    @pytest.mark.unit
    def test_invalid_rank_environment_is_disabled_safely(self):
        """Malformed scheduler metadata must not break application startup."""
        with patch.dict(os.environ, {"RANK": "not-an-integer"}, clear=True):
            assert _should_enable_for_current_rank() is False

    @pytest.mark.unit
    def test_get_base_callback_config(self):
        """Test _get_base_callback_config with basic trainer setup."""
        trainer = MagicMock()
        trainer.max_steps = 1000
        trainer.callbacks = []
        trainer.val_check_interval = 1.0
        trainer.strategy = None
        trainer.log_every_n_steps = 10
        trainer.accumulate_grad_batches = 1
        trainer.datamodule = SimpleNamespace(cfg=OmegaConf.create({"validation_ds": {"batch_size": 2}}))

        with patch.dict(os.environ, {"SLURM_JOB_NAME": "test_job", "WORLD_SIZE": "4", "PERF_VERSION_TAG": "1.0.0"}):
            config = _get_base_callback_config(trainer=trainer, global_batch_size=32, seq_length=512)

            assert config["perf_tag_or_fn"] == "test_job_1.0.0_bf32_se512_ws4"
            assert config["global_batch_size_or_fn"] == 32
            assert config["micro_batch_size_or_fn"] == 8
            assert config["seq_length_or_fn"] == 512
            assert config["train_iterations_target_or_fn"] == 1000
            assert config["train_samples_target_or_fn"] == 32000
            assert config["log_every_n_train_iterations"] == 10
            assert config["is_validation_iterations_enabled_or_fn"] is True
            assert config["is_save_checkpoint_enabled_or_fn"] is False
            assert config["save_checkpoint_strategy"] == "sync"

    @pytest.mark.unit
    def test_get_base_callback_config_with_checkpoint_callback(self):
        """Test _get_base_callback_config when checkpoint callback is present."""
        trainer = MagicMock()
        trainer.max_steps = 1000
        trainer.val_check_interval = 0

        checkpoint_callback = NeMoModelCheckpoint(dirpath=".", save_top_k=-1)
        trainer.callbacks = [checkpoint_callback]

        with patch.dict(os.environ, {"SLURM_JOB_NAME": "test_job", "WORLD_SIZE": "2"}):
            config = _get_base_callback_config(trainer=trainer, global_batch_size=16, seq_length=256)

            assert config["is_save_checkpoint_enabled_or_fn"] is True
            assert config["is_validation_iterations_enabled_or_fn"] is False

    @pytest.mark.unit
    def test_get_base_callback_config_async_save(self):
        """Test _get_base_callback_config with async save strategy."""
        trainer = MagicMock()
        trainer.max_steps = 1000
        trainer.callbacks = []
        trainer.val_check_interval = 0  # Set to 0 to avoid validation

        # Mock strategy with async_save
        strategy = MagicMock()
        strategy.async_save = True
        trainer.strategy = strategy

        with patch.dict(os.environ, {"WORLD_SIZE": "1"}):
            config = _get_base_callback_config(trainer=trainer, global_batch_size=8, seq_length=128)

            assert config["save_checkpoint_strategy"] == "async"

    @pytest.mark.unit
    def test_get_base_callback_config_dict_strategy(self):
        """Test _get_base_callback_config with dict strategy."""
        trainer = MagicMock()
        trainer.max_steps = 1000
        trainer.callbacks = []
        trainer.val_check_interval = 0  # Set to 0 to avoid validation
        trainer.strategy = {"async_save": True}

        with patch.dict(os.environ, {"WORLD_SIZE": "1"}):
            config = _get_base_callback_config(trainer=trainer, global_batch_size=8, seq_length=128)

            assert config["save_checkpoint_strategy"] == "async"

    @pytest.mark.unit
    def test_get_nemo_v1_callback_config(self):
        """Runtime batch callables are used instead of train_ds batch-size hints."""
        trainer = MagicMock(max_steps=500, callbacks=[], strategy=None, log_every_n_steps=10)
        trainer.lightning_module.cfg = OmegaConf.create({"train_ds": {"batch_size": 8}})
        current = {"global": 7, "local": 3}

        with patch.dict(os.environ, {"WORLD_SIZE": "2"}, clear=True):
            config = get_nemo_v1_callback_config(
                trainer,
                global_batch_size=lambda: current["global"],
                micro_batch_size=lambda: current["local"],
            )

        assert config["global_batch_size_or_fn"]() == 7
        assert config["micro_batch_size_or_fn"]() == 3
        assert config["seq_length_or_fn"] is None
        assert config["train_samples_target_or_fn"] is None
        assert "_bf" not in config["perf_tag_or_fn"]

    @pytest.mark.unit
    def test_get_nemo_v1_callback_config_bucket_batch_size(self):
        """Per-bucket limits must never be averaged into a reported batch size."""
        trainer = MagicMock(max_steps=1000, callbacks=[], strategy=None, log_every_n_steps=10)
        trainer.lightning_module.cfg = OmegaConf.create(
            {"train_ds": {"bucket_batch_size": [4, 8, 12], "bucket_duration_bins": [5, 10, 20]}}
        )

        config = get_nemo_v1_callback_config(trainer, global_batch_size=lambda: 4, micro_batch_size=lambda: 4)

        assert callable(config["global_batch_size_or_fn"])
        assert config["global_batch_size_or_fn"]() == 4
        assert config["train_samples_target_or_fn"] is None

    @pytest.mark.unit
    def test_get_nemo_v1_callback_config_fallback(self):
        """An unknown batch size is rejected instead of silently reporting one."""
        trainer = MagicMock(max_steps=100, callbacks=[], strategy=None, log_every_n_steps=10)

        with pytest.raises(ValueError, match="must be measured at runtime"):
            get_nemo_v1_callback_config(trainer)

    @pytest.mark.unit
    def test_explicit_fixed_batch_overrides_remain_supported(self):
        trainer = MagicMock(max_steps=100, callbacks=[], strategy=None, log_every_n_steps=10)
        trainer.accumulate_grad_batches = 1
        env = {
            "WORLD_SIZE": "4",
            "NEMO_ONE_LOGGER_GLOBAL_BATCH_SIZE": "16",
            "NEMO_ONE_LOGGER_MICRO_BATCH_SIZE": "4",
            "NEMO_ONE_LOGGER_SEQUENCE_LENGTH": "256",
        }

        with patch.dict(os.environ, env, clear=True):
            config = get_nemo_v1_callback_config(trainer)

        assert config["global_batch_size_or_fn"] == 16
        assert config["micro_batch_size_or_fn"] == 4
        assert config["seq_length_or_fn"] == 256
        assert config["train_samples_target_or_fn"] == 1600

    @pytest.mark.unit
    def test_should_enable_for_current_rank_single_process(self):
        """Test _should_enable_for_current_rank if rank is not set."""
        with patch.dict(os.environ, {}, clear=True):
            result = _should_enable_for_current_rank()
            assert result is False

    @pytest.mark.unit
    def test_should_enable_for_current_rank_slurm_rank0(self):
        """Native Slurm launches expose SLURM_PROCID instead of RANK."""
        with patch.dict(os.environ, {"SLURM_PROCID": "0", "SLURM_NTASKS": "4"}, clear=True):
            assert _should_enable_for_current_rank() is True

    @pytest.mark.unit
    def test_should_disable_for_nonzero_slurm_rank(self):
        """Only one native Slurm rank should emit telemetry."""
        with patch.dict(os.environ, {"SLURM_PROCID": "1", "SLURM_NTASKS": "4"}, clear=True):
            assert _should_enable_for_current_rank() is False

    @pytest.mark.unit
    def test_should_enable_for_lightning_local_rank0(self):
        """Lightning subprocess launch metadata should enable only global rank zero."""
        env = {"LOCAL_RANK": "0", "LOCAL_WORLD_SIZE": "8", "NODE_RANK": "0", "WORLD_SIZE": "16"}
        with patch.dict(os.environ, env, clear=True):
            assert _should_enable_for_current_rank() is True

    @pytest.mark.unit
    def test_should_disable_for_lightning_rank_on_second_node(self):
        env = {"LOCAL_RANK": "0", "LOCAL_WORLD_SIZE": "8", "NODE_RANK": "1", "WORLD_SIZE": "16"}
        with patch.dict(os.environ, env, clear=True):
            assert _should_enable_for_current_rank() is False

    @pytest.mark.unit
    def test_should_enable_for_current_rank_distributed_rank0(self):
        """Test _should_enable_for_current_rank for rank 0 in distributed training."""
        with patch.dict(os.environ, {"RANK": "0", "WORLD_SIZE": "4"}):
            result = _should_enable_for_current_rank()
            assert result is True

    @pytest.mark.unit
    def test_should_enable_for_current_rank_distributed_middle_rank(self):
        """Test _should_enable_for_current_rank for middle rank in distributed training."""
        with patch.dict(os.environ, {"RANK": "1", "WORLD_SIZE": "4"}):
            result = _should_enable_for_current_rank()
            assert result is False

    @pytest.mark.unit
    def test_validation_disabled_without_validation_data(self):
        trainer = MagicMock()
        trainer.max_steps = 10
        trainer.callbacks = []
        trainer.val_check_interval = 1.0
        trainer.strategy = None
        trainer.log_every_n_steps = 10
        trainer.datamodule = SimpleNamespace(cfg=OmegaConf.create({"train_ds": {"batch_size": 2}}))

        config = _get_base_callback_config(trainer=trainer, global_batch_size=2, seq_length=None)

        assert config["is_validation_iterations_enabled_or_fn"] is False

    @pytest.mark.unit
    def test_validation_enabled_from_datamodule_config(self):
        trainer = MagicMock()
        trainer.max_steps = 10
        trainer.callbacks = []
        trainer.val_check_interval = 1.0
        trainer.strategy = None
        trainer.log_every_n_steps = 10
        trainer.datamodule = SimpleNamespace(
            cfg=OmegaConf.create({"train_ds": {"batch_size": 2}, "validation_ds": {"batch_size": 2}})
        )

        config = _get_base_callback_config(trainer=trainer, global_batch_size=2, seq_length=None)

        assert config["is_validation_iterations_enabled_or_fn"] is True

    @pytest.mark.unit
    def test_sequence_length_is_only_reported_when_explicit(self):
        trainer = MagicMock(max_steps=10, callbacks=[], strategy=None, log_every_n_steps=10)
        trainer.lightning_module.cfg = OmegaConf.create(
            {"train_ds": {"max_seq_length": 512, "batch_tokens": 4000}, "encoder": {"d_model": 512}}
        )

        config = get_nemo_v1_callback_config(trainer, global_batch_size=lambda: 2)
        assert config["seq_length_or_fn"] is None

        with patch.dict(os.environ, {"NEMO_ONE_LOGGER_SEQUENCE_LENGTH": "256"}, clear=True):
            config = get_nemo_v1_callback_config(trainer, global_batch_size=lambda: 2)
        assert config["seq_length_or_fn"] == 256

    @pytest.mark.unit
    def test_callback_group_attaches_callback_once(self):
        group = CallbackGroup.__new__(CallbackGroup)
        callback = BaseCallback()
        group._callbacks = [callback]
        trainer = SimpleNamespace(callbacks=[])

        group.attach_to_trainer(trainer)
        group.attach_to_trainer(trainer)

        assert trainer.callbacks == [callback]

    @pytest.mark.unit
    def test_disabled_callback_is_not_attached(self):
        group = CallbackGroup.__new__(CallbackGroup)
        callback = BaseCallback()
        callback.enabled_for_current_rank = False
        group._callbacks = [callback]
        trainer = SimpleNamespace(callbacks=[])

        group.attach_to_trainer(trainer)

        assert trainer.callbacks == []

    @pytest.mark.unit
    def test_disabled_collective_participant_is_attached(self):
        group = CallbackGroup.__new__(CallbackGroup)
        callback = BaseCallback()
        callback.enabled_for_current_rank = False
        callback.participates_on_all_ranks = True
        group._callbacks = [callback]
        trainer = SimpleNamespace(callbacks=[])

        group.attach_to_trainer(trainer)

        assert trainer.callbacks == [callback]

    @pytest.mark.unit
    def test_modelpt_subclass_init_emits_one_paired_span(self):
        class ParentModel(ModelPT):
            def __init__(self):
                super().__init__(cfg=OmegaConf.create({}))

            @classmethod
            def list_available_models(cls):
                return []

            def setup_training_data(self, train_data_config):
                pass

            def setup_validation_data(self, val_data_config):
                pass

        class ChildModel(ParentModel):
            def __init__(self):
                super().__init__()

        group = MagicMock()
        with patch('nemo.lightning.callback_group.CallbackGroup.get_instance', return_value=group):
            ChildModel()

        group.on_model_init_start.assert_called_once_with()
        group.on_model_init_end.assert_called_once_with()

    @pytest.mark.unit
    def test_model_class_decorator_emits_one_paired_span(self):
        @with_model_init_callbacks
        class Model:
            def __init__(self):
                pass

        group = MagicMock()
        with patch('nemo.lightning.callback_group.CallbackGroup.get_instance', return_value=group):
            Model()

        group.on_model_init_start.assert_called_once_with()
        group.on_model_init_end.assert_called_once_with()

    @pytest.mark.unit
    def test_callback_context_always_emits_end(self):
        group = MagicMock()
        error = None
        with patch('nemo.lightning.callback_group.CallbackGroup.get_instance', return_value=group):
            try:
                with callback_context('on_model_init_start', 'on_model_init_end'):
                    raise RuntimeError("boom")
            except RuntimeError as caught_error:
                error = caught_error

        assert str(error) == "boom"
        group.on_model_init_start.assert_called_once_with()
        group.on_model_init_end.assert_called_once_with()

    @pytest.mark.unit
    @patch('nemo.utils.callbacks.nemo_model_checkpoint.CallbackGroup.get_instance')
    def test_checkpoint_lifecycle_success(self, mock_get_group, tmp_path):
        group = mock_get_group.return_value
        callback = NeMoModelCheckpoint(dirpath=tmp_path, save_top_k=-1)
        callback.set_checkpoint_unfinished_marker = MagicMock()
        callback.remove_checkpoint_unfinished_marker = MagicMock()
        trainer = SimpleNamespace(
            global_step=7,
            callbacks=[],
            is_global_zero=False,
            loggers=[],
            save_checkpoint=MagicMock(),
        )

        callback._save_checkpoint(trainer, str(tmp_path / "model.ckpt"))

        group.on_save_checkpoint_start.assert_called_once_with(7)
        group.on_save_checkpoint_success.assert_called_once_with(7)
        group.on_save_checkpoint_end.assert_called_once_with()

    @pytest.mark.unit
    @patch('nemo.utils.callbacks.nemo_model_checkpoint.CallbackGroup.get_instance')
    def test_checkpoint_lifecycle_ends_on_failure(self, mock_get_group, tmp_path):
        group = mock_get_group.return_value
        callback = NeMoModelCheckpoint(dirpath=tmp_path, save_top_k=-1)
        callback.set_checkpoint_unfinished_marker = MagicMock()
        trainer = SimpleNamespace(
            global_step=9,
            callbacks=[],
            save_checkpoint=MagicMock(side_effect=RuntimeError("save failed")),
        )

        with pytest.raises(RuntimeError, match="save failed"):
            callback._save_checkpoint(trainer, str(tmp_path / "model.ckpt"))

        group.on_save_checkpoint_start.assert_called_once_with(9)
        group.on_save_checkpoint_success.assert_not_called()
        group.on_save_checkpoint_end.assert_called_once_with()

    @patch('nemo.lightning.one_logger_callback.on_training_single_iteration_end')
    @patch('nemo.lightning.one_logger_callback.on_training_single_iteration_start')
    @patch('nemo.lightning.one_logger_callback.on_train_start')
    @patch('nemo.lightning.one_logger_callback.TrainingTelemetryProvider')
    @patch('nemo.lightning.one_logger_callback.TrainingTelemetryConfig')
    @patch('nemo.lightning.one_logger_callback.get_one_logger_init_config')
    @patch('nemo.lightning.one_logger_callback.OneLoggerConfig')
    @patch('nemo.lightning.one_logger_callback.on_app_start')
    @patch('nemo.lightning.one_logger_callback.OneLoggerPTLCallback.__init__', return_value=None)
    def test_dynamic_batches_are_measured_and_accumulated(
        self,
        mock_ptl_init,
        mock_app_start,
        mock_config_class,
        mock_get_init_config,
        mock_training_config_class,
        mock_provider,
        mock_train_start,
        mock_iteration_start,
        mock_iteration_end,
    ):
        mock_get_init_config.return_value = {"application_name": "nemo-speech", "enable_for_current_rank": True}
        provider = MagicMock()
        provider.config.telemetry_config = None
        mock_provider.instance.return_value = provider
        callback = OneLoggerNeMoCallback()
        trainer = SimpleNamespace(global_step=4, max_steps=20, callbacks=[], strategy=None, log_every_n_steps=5)
        module = SimpleNamespace(device=torch.device("cpu"))

        callback.update_config(nemo_version="lightning", trainer=trainer)
        assert provider.set_training_telemetry_config.call_count == 0
        callback.load_state_dict({"num_samples_total": 9})
        callback.on_train_start(trainer, module)

        first_batch = {"input_signal": torch.zeros(2, 20), "input_signal_length": torch.tensor([20, 15])}
        callback.on_train_batch_start(trainer, module, first_batch, 0)
        callback.on_train_batch_end(trainer, module, None, first_batch, 0)
        configured = mock_training_config_class.call_args.kwargs
        assert configured["global_batch_size_or_fn"]() == 2
        assert configured["micro_batch_size_or_fn"]() == 2
        assert configured["seq_length_or_fn"] is None
        assert configured["train_samples_target_or_fn"] is None

        second_batch = {"input_signal": torch.zeros(3, 20), "input_signal_length": torch.tensor([20, 15, 10])}
        callback.on_train_batch_start(trainer, module, second_batch, 1)
        callback.on_train_batch_end(trainer, module, None, second_batch, 1)

        assert configured["global_batch_size_or_fn"]() == 3
        assert callback.num_samples_total == 14
        mock_train_start.assert_called_once_with(
            train_iterations_start=4,
            train_samples_start=9,
            start_time_msec=callback._train_start_time_msec,
        )
        assert mock_iteration_start.call_count == 2
        assert mock_iteration_end.call_count == 2

    def test_export_all_symbols(self):
        """Test that __all__ contains the expected symbols."""
        from nemo.lightning.one_logger_callback import __all__

        assert 'OneLoggerNeMoCallback' in __all__

    @patch.dict(os.environ, {'EXP_NAME': 'test-experiment', 'WORLD_SIZE': '4'})
    @patch('nemo.lightning.one_logger_callback.TrainingTelemetryProvider')
    @patch('nemo.lightning.one_logger_callback.get_one_logger_init_config')
    @patch('nemo.lightning.one_logger_callback.OneLoggerConfig')
    @patch('nemo.lightning.one_logger_callback.on_app_start')
    @patch('nemo.lightning.one_logger_callback.OneLoggerPTLCallback')
    def test_init_with_environment_variables(
        self, mock_ptl_callback, mock_on_app_start, mock_config_class, mock_get_config, mock_provider
    ):
        """Test initialization with environment variables set."""
        # Setup mocks
        mock_get_config.return_value = {
            "application_name": "nemo",
            "session_tag_or_fn": "test-experiment",
            "world_size_or_fn": 4,
        }
        mock_config_class.return_value = MagicMock()
        mock_provider_instance = MagicMock()
        mock_provider_instance.config = MagicMock()
        mock_provider_instance.config.telemetry_config = None
        mock_provider.instance.return_value = mock_provider_instance
        mock_ptl_callback.return_value = MagicMock()

        # Create callback instance
        OneLoggerNeMoCallback()

        # Verify that get_one_logger_init_config was called
        mock_get_config.assert_called_once()

        # Verify that the config was created with the environment-based values
        mock_config_class.assert_called_once()
        call_args = mock_config_class.call_args[1]
        assert call_args['session_tag_or_fn'] == 'test-experiment'
        assert call_args['world_size_or_fn'] == 4

    def test_callback_instantiation_without_mocks_raises_import_error(self):
        """Test that callback instantiation without proper mocks raises appropriate errors."""
        # This test verifies that the callback properly depends on external libraries
        # and will raise import errors if they're not available
        with patch(
            'nemo.lightning.one_logger_callback.OneLoggerPTLCallback.__init__',
            side_effect=Exception("with_base_config can be called only before configure_provider is called."),
        ):
            with pytest.raises(
                Exception, match="with_base_config can be called only before configure_provider is called."
            ):
                OneLoggerNeMoCallback()

    @patch('nemo.lightning.one_logger_callback.TrainingTelemetryProvider')
    @patch('nemo.lightning.one_logger_callback.get_one_logger_init_config')
    @patch('nemo.lightning.one_logger_callback.OneLoggerConfig')
    @patch('nemo.lightning.one_logger_callback.on_app_start')
    @patch('nemo.lightning.one_logger_callback.OneLoggerPTLCallback.__init__', return_value=None)
    def test_init_provider_chain_calls(
        self, mock_ptl_callback_init, mock_on_app_start, mock_config_class, mock_get_config, mock_provider
    ):
        """Test that the provider configuration chain is called in correct order."""
        # Setup mocks
        mock_get_config.return_value = {"application_name": "test"}
        mock_config_instance = MagicMock()
        mock_config_class.return_value = mock_config_instance
        mock_provider_instance = MagicMock()
        mock_provider_instance.config = MagicMock()
        mock_provider_instance.config.telemetry_config = None
        mock_provider.instance.return_value = mock_provider_instance

        # Create callback instance
        OneLoggerNeMoCallback()

        # Verify the provider configuration chain
        mock_provider_instance.with_base_config.assert_called_once_with(mock_config_instance)
        chain_result = mock_provider_instance.with_base_config.return_value
        chain_result.with_export_config.assert_called_once()
        chain_result.with_export_config.return_value.configure_provider.assert_called_once()

        # Verify PTL callback was initialized with provider instance and explicit on_app_start was called
        mock_ptl_callback_init.assert_called_once_with(mock_provider_instance, call_on_app_start=False)
        mock_on_app_start.assert_called_once_with()
