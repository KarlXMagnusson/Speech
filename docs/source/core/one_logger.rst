.. _onelogger-integration:

OneLogger Integration
=====================

NeMo Speech can emit training lifecycle and throughput telemetry through
OneLogger. The integration is opt-in and is inactive unless it is explicitly
enabled.

Enabling OneLogger
------------------

Set ``NEMO_ONE_LOGGER_ENABLED`` to a true value before starting a training job:

.. code-block:: bash

    export NEMO_ONE_LOGGER_ENABLED=true
    python <training-script> <training-options>

Training entry points that call :func:`nemo.utils.exp_manager.exp_manager`
attach the callback automatically. In distributed jobs, only global rank zero
exports telemetry. In a single-process job, the same environment variable
enables the local process.

Unset ``NEMO_ONE_LOGGER_ENABLED``, or set it to ``false``, to leave the
integration disabled. OneLogger exporter destinations and credentials use the
standard configuration supported by the installed OneLogger packages.

Reporting cadence
-----------------

Throughput is reported every 100 training batches by default. If
``trainer.log_every_n_steps`` is larger, that value is used instead. Override
the cadence with a positive batch count:

.. code-block:: bash

    export NEMO_ONE_LOGGER_THROUGHPUT_INTERVAL=250

A reporting window is closed before validation and checkpointing so those
operations are not included in training throughput. GPU timing and aggregation
are asynchronous during training.

Reported telemetry
------------------

Lifecycle spans use the ``nemo_speech`` namespace and cover model, data loader,
and optimizer initialization; checkpoint load and save; training; and
validation. Checkpoint save success and failure are emitted as explicit events.

Throughput is reported as the ``nemo_speech.throughput`` event. Every event
contains:

* ``policy``: the model-specific measurement policy;
* ``rank`` and ``scope``: the rank-local origin of the measurement;
* ``global_step``;
* ``window_batches`` and ``window_seconds``;
* each available work-unit total; and
* a corresponding ``<unit>_per_second`` rate.

The available work units depend on the model and batch schema:

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Model family
     - Input work
     - Output or target work
   * - ASR
     - ``input_audio_seconds``
     - ``target_text_tokens``
   * - TTS
     - ``input_text_tokens``
     - ``output_audio_seconds``
   * - Audio codec
     - ``input_audio_seconds``
     -
   * - Diarization
     - ``input_audio_seconds``
     -
   * - Audio-to-audio
     - ``input_audio_seconds``
     -
   * - SALM and SALMAutomodel
     - ``input_audio_seconds``
     - ``model_sequence_positions`` when the model exposes its exact
       post-expansion counter
   * - DuplexSTT
     - ``input_audio_seconds``
     - ``text_tokens``
   * - Speech-to-speech
     - ``input_audio_seconds``
     - ``output_audio_seconds``

Measurements use the length tensors from the actual dynamic batch. Packed SALM
batches use the model's exact post-expansion mixed-modality position counter.
NeMo Speech does not infer or report micro batch size, global batch size, or a
static sequence length. Audio durations are omitted when a trustworthy sample
rate or waveform length is unavailable, rather than estimated from unrelated
configuration.

Throughput events are rank-local measurements, not distributed global
estimates. If a model has no registered policy or its batch does not expose a
safe measurement, no throughput event is emitted for that batch. Telemetry
setup and reporting failures disable telemetry without stopping training.
