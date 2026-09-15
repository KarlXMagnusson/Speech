# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for NeMo Speech OneLogger lifecycle tracing."""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
from lightning.pytorch.callbacks import Callback as PTLCallback
from omegaconf import OmegaConf

from nemo.core.classes.modelPT import ModelPT
from nemo.lightning.base_callback import BaseCallback
from nemo.lightning.callback_group import CallbackGroup, callback_context, with_model_init_callbacks
from nemo.lightning.one_logger_callback import (
    OneLoggerNeMoCallback,
    _should_enable_for_current_rank,
    get_one_logger_init_config,
)
from nemo.utils.callbacks.dist_ckpt_io import AsyncFinalizableCheckpointIO
from nemo.utils.callbacks.nemo_model_checkpoint import NeMoModelCheckpoint


@pytest.fixture(autouse=True)
def reset_one_logger_callback_singleton():
    previous_instance = OneLoggerNeMoCallback._instance
    OneLoggerNeMoCallback._instance = None
    yield
    OneLoggerNeMoCallback._instance = previous_instance


def _enabled_callback():
    provider = MagicMock()
    provider.recorder.start.side_effect = lambda name, **kwargs: SimpleNamespace(name=name, attributes=kwargs)
    init_config = {
        "application_name": "nemo-speech",
        "session_tag_or_fn": "test",
        "enable_for_current_rank": True,
        "world_size_or_fn": 1,
    }
    patches = (
        patch('nemo.lightning.one_logger_callback.get_one_logger_init_config', return_value=init_config),
        patch('nemo.lightning.one_logger_callback.TrainingTelemetryProvider.instance', return_value=provider),
        patch('nemo.lightning.one_logger_callback.OneLoggerConfig'),
        patch(
            'nemo.lightning.one_logger_callback.on_app_start',
            return_value=SimpleNamespace(name='application'),
        ),
    )
    for active_patch in patches:
        active_patch.start()
    callback = OneLoggerNeMoCallback()
    for active_patch in reversed(patches):
        active_patch.stop()
    return callback, provider


class TestOneLoggerNeMoCallback:
    def test_inherits_only_nemo_callback(self):
        callback, _ = _enabled_callback()

        assert isinstance(callback, BaseCallback)
        assert isinstance(callback, PTLCallback)
        assert type(callback).__bases__ == (BaseCallback,)

    @patch('nemo.lightning.one_logger_callback.on_app_start')
    @patch('nemo.lightning.one_logger_callback.OneLoggerConfig')
    @patch('nemo.lightning.one_logger_callback.TrainingTelemetryProvider')
    @patch('nemo.lightning.one_logger_callback.get_one_logger_init_config')
    def test_init_configures_application_without_training_schema(
        self, mock_get_config, mock_provider_class, mock_config_class, mock_app_start
    ):
        init_config = {
            "application_name": "nemo-speech",
            "session_tag_or_fn": "test",
            "enable_for_current_rank": True,
            "world_size_or_fn": 2,
        }
        mock_get_config.return_value = init_config
        provider = mock_provider_class.instance.return_value

        callback = OneLoggerNeMoCallback()

        mock_config_class.assert_called_once_with(**init_config)
        provider.with_base_config.assert_called_once_with(mock_config_class.return_value)
        provider.with_base_config.return_value.with_export_config.assert_called_once_with()
        provider.with_base_config.return_value.with_export_config.return_value.configure_provider.assert_called_once_with()
        provider.set_training_telemetry_config.assert_not_called()
        mock_app_start.assert_called_once_with()
        assert callback._provider is provider

    @patch('nemo.lightning.one_logger_callback.on_app_start')
    @patch('nemo.lightning.one_logger_callback.OneLoggerConfig')
    @patch('nemo.lightning.one_logger_callback.TrainingTelemetryProvider')
    @patch('nemo.lightning.one_logger_callback.get_one_logger_init_config')
    def test_disabled_rank_does_not_initialize_onelogger(
        self, mock_get_config, mock_provider_class, mock_config_class, mock_app_start
    ):
        mock_get_config.return_value = {
            "application_name": "nemo-speech",
            "session_tag_or_fn": "test",
            "enable_for_current_rank": False,
            "world_size_or_fn": 8,
        }

        callback = OneLoggerNeMoCallback()

        assert callback.enabled_for_current_rank is False
        assert callback._provider is None
        mock_provider_class.instance.assert_not_called()
        mock_config_class.assert_not_called()
        mock_app_start.assert_not_called()

    def test_lifecycle_spans_are_paired_by_identity(self):
        callback, provider = _enabled_callback()
        lifecycle = (
            (callback.on_model_init_start, callback.on_model_init_end, "nemo_speech.model_initialization"),
            (
                callback.on_dataloader_init_start,
                callback.on_dataloader_init_end,
                "nemo_speech.data_loader_initialization",
            ),
            (callback.on_optimizer_init_start, callback.on_optimizer_init_end, "nemo_speech.optimizer_initialization"),
            (callback.on_load_checkpoint_start, callback.on_load_checkpoint_end, "nemo_speech.checkpoint_load"),
        )

        for start, end, name in lifecycle:
            start()
            span = callback._current_span(name)
            end()
            provider.recorder.stop.assert_any_call(span)

        assert callback._active_spans == []

    def test_nested_same_name_spans_are_paired_last_in_first_out(self):
        callback, provider = _enabled_callback()

        callback.on_model_init_start()
        outer = callback._current_span("nemo_speech.model_initialization")
        callback.on_model_init_start()
        inner = callback._current_span("nemo_speech.model_initialization")
        callback.on_model_init_end()
        callback.on_model_init_end()

        assert inner is not outer
        assert provider.recorder.stop.call_args_list == [call(inner), call(outer)]
        assert callback._active_spans == []

    def test_training_and_validation_report_only_loop_timing(self):
        callback, provider = _enabled_callback()
        trainer = module = object()

        callback.on_train_start(trainer, module)
        training_span = callback._current_span("nemo_speech.training")
        callback.on_validation_start(trainer, module)
        validation_span = callback._current_span("nemo_speech.validation")
        callback.on_validation_end(trainer, module)
        callback.on_train_end(trainer, module)

        assert provider.recorder.start.call_args_list == [
            call("nemo_speech.training", span_attributes=None),
            call("nemo_speech.validation", span_attributes=None),
        ]
        assert provider.recorder.stop.call_args_list == [call(validation_span), call(training_span)]

    def test_batches_are_ignored_regardless_of_modality_or_packing(self):
        callback, provider = _enabled_callback()
        packed_batch = {"input_ids": object(), "text_cu_seqlens": object()}

        callback.on_train_batch_start(object(), object(), packed_batch, 0)
        callback.on_train_batch_end(object(), object(), None, packed_batch, 0)
        callback.on_validation_batch_start(object(), object(), packed_batch, 0)
        callback.on_validation_batch_end(object(), object(), None, packed_batch, 0)

        provider.recorder.start.assert_not_called()
        provider.recorder.stop.assert_not_called()

    @patch('nemo.lightning.one_logger_callback.Event.create')
    def test_checkpoint_outcomes_are_explicit_and_async_safe(self, mock_event_create):
        callback, provider = _enabled_callback()
        mock_event_create.side_effect = lambda name, attributes: (name, attributes.to_json())

        callback.on_save_checkpoint_start(3)
        success_span = callback._current_span("nemo_speech.checkpoint_save")
        callback.on_save_checkpoint_success(3)
        callback.on_save_checkpoint_end()

        callback.on_save_checkpoint_start(4, async_save=True)
        async_span = callback._current_span("nemo_speech.checkpoint_save")
        callback.on_save_checkpoint_end()
        callback.on_save_checkpoint_success(4)

        callback.on_save_checkpoint_start(5)
        failed_span = callback._current_span("nemo_speech.checkpoint_save")
        callback.on_save_checkpoint_failure(5)
        callback.on_save_checkpoint_end()

        assert [
            (args[0], kwargs["span_attributes"].to_json()) for args, kwargs in provider.recorder.start.call_args_list
        ] == [
            ("nemo_speech.checkpoint_save", {"global_step": 3, "asynchronous": False}),
            ("nemo_speech.checkpoint_save", {"global_step": 4, "asynchronous": True}),
            ("nemo_speech.checkpoint_save", {"global_step": 5, "asynchronous": False}),
        ]
        assert provider.recorder.event.call_args_list == [
            call(success_span, ("nemo_speech.checkpoint_save_success", {"global_step": 3})),
            call(callback._application_span, ("nemo_speech.checkpoint_save_success", {"global_step": 4})),
            call(failed_span, ("nemo_speech.checkpoint_save_failure", {"global_step": 5})),
        ]
        assert provider.recorder.stop.call_args_list == [call(success_span), call(async_span), call(failed_span)]

    @patch('nemo.lightning.one_logger_callback.on_app_end')
    def test_app_end_closes_unfinished_spans(self, mock_app_end):
        callback, provider = _enabled_callback()
        callback.on_train_start(object(), object())
        callback.on_validation_start(object(), object())
        validation_span = callback._current_span("nemo_speech.validation")
        training_span = callback._current_span("nemo_speech.training")

        callback.on_app_end()

        assert provider.recorder.stop.call_args_list == [call(validation_span), call(training_span)]
        mock_app_end.assert_called_once_with()


class TestOneLoggerConfiguration:
    def test_init_config_is_modality_independent(self):
        with patch.dict(os.environ, {"SLURM_JOB_NAME": "speech-job", "WORLD_SIZE": "4", "RANK": "0"}, clear=True):
            config = get_one_logger_init_config()

        assert config["application_name"] == "nemo-speech"
        assert config["session_tag_or_fn"] == "speech-job"
        assert config["world_size_or_fn"] == 4
        assert config["enable_for_current_rank"] is True
        assert "telemetry_config" not in config
        assert all("batch" not in key and "sequence" not in key and "token" not in key for key in config)

    @pytest.mark.parametrize("value", ["0", "false", "no", "off"])
    def test_explicit_disable(self, value):
        with patch.dict(os.environ, {"NEMO_ONE_LOGGER_ENABLED": value, "RANK": "0"}, clear=True):
            assert not _should_enable_for_current_rank()

    def test_explicit_single_process_enable(self):
        with patch.dict(os.environ, {"NEMO_ONE_LOGGER_ENABLED": "true"}, clear=True):
            assert _should_enable_for_current_rank()

    def test_distributed_rank_selection(self):
        with patch.dict(os.environ, {"RANK": "1", "WORLD_SIZE": "4"}, clear=True):
            assert not _should_enable_for_current_rank()
        with patch.dict(os.environ, {"RANK": "0", "WORLD_SIZE": "4"}, clear=True):
            assert _should_enable_for_current_rank()


class TestCallbackGroup:
    @pytest.mark.unit
    def test_attaches_callback_once_after_existing_callbacks(self):
        group = CallbackGroup.__new__(CallbackGroup)
        callback = BaseCallback()
        existing = PTLCallback()
        group._callbacks = [callback]
        trainer = SimpleNamespace(callbacks=[existing])

        group.attach_to_trainer(trainer)
        group.attach_to_trainer(trainer)

        assert trainer.callbacks == [existing, callback]

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
    def test_eager_dataloader_failure_closes_dataloader_and_model_spans(self):
        class BrokenModel(ModelPT):
            @classmethod
            def list_available_models(cls):
                return []

            def setup_training_data(self, train_data_config):
                raise RuntimeError("dataloader failed")

            def setup_validation_data(self, val_data_config):
                pass

        group = MagicMock()
        with (
            patch('nemo.lightning.callback_group.CallbackGroup.get_instance', return_value=group),
            pytest.raises(RuntimeError, match="dataloader failed"),
        ):
            BrokenModel(cfg=OmegaConf.create({"train_ds": {}}))

        assert group.method_calls == [
            call.on_model_init_start(),
            call.on_dataloader_init_start(),
            call.on_dataloader_init_end(),
            call.on_model_init_end(),
        ]

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
        with patch('nemo.lightning.callback_group.CallbackGroup.get_instance', return_value=group):
            with pytest.raises(RuntimeError, match="boom"):
                with callback_context('on_model_init_start', 'on_model_init_end'):
                    raise RuntimeError("boom")

        group.on_model_init_start.assert_called_once_with()
        group.on_model_init_end.assert_called_once_with()


class TestCheckpointIntegration:
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

        group.on_save_checkpoint_start.assert_called_once_with(7, async_save=False)
        group.on_save_checkpoint_success.assert_called_once_with(7)
        group.on_save_checkpoint_end.assert_called_once_with()

    @pytest.mark.unit
    @patch('nemo.utils.callbacks.nemo_model_checkpoint.CallbackGroup.get_instance')
    def test_async_checkpoint_reports_success_only_after_finalization(self, mock_get_group, tmp_path):
        group = mock_get_group.return_value
        callback = NeMoModelCheckpoint(dirpath=tmp_path, save_top_k=-1, async_save=True)
        callback.set_checkpoint_unfinished_marker = MagicMock()
        callback.remove_checkpoint_unfinished_marker = MagicMock()
        trainer = SimpleNamespace(
            global_step=8,
            callbacks=[],
            is_global_zero=False,
            loggers=[],
            strategy=SimpleNamespace(checkpoint_io=MagicMock(spec=AsyncFinalizableCheckpointIO)),
            save_checkpoint=MagicMock(),
        )

        callback._save_checkpoint(trainer, str(tmp_path / "model.ckpt"))

        group.on_save_checkpoint_start.assert_called_once_with(8, async_save=True)
        group.on_save_checkpoint_success.assert_not_called()
        group.on_save_checkpoint_failure.assert_not_called()
        group.on_save_checkpoint_end.assert_called_once_with()

        finalize_fn = trainer.save_checkpoint.call_args.kwargs["storage_options"]["finalize_fn"]
        finalize_fn()

        group.on_save_checkpoint_success.assert_called_once_with(8)
        group.on_save_checkpoint_failure.assert_not_called()

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

        group.on_save_checkpoint_start.assert_called_once_with(9, async_save=False)
        group.on_save_checkpoint_success.assert_not_called()
        group.on_save_checkpoint_failure.assert_called_once_with(9)
        group.on_save_checkpoint_end.assert_called_once_with()


def test_export_all_symbols():
    from nemo.lightning.one_logger_callback import __all__

    assert __all__ == ["OneLoggerNeMoCallback"]
