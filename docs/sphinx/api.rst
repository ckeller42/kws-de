Firmware C API reference
==========================

Generated from the Doxygen comments in ``firmware/main/*.h`` via
`Breathe <https://breathe.readthedocs.io/>`_. Requires
``cd firmware && doxygen Doxyfile`` to have been run first (produces
``firmware/docs/doxygen/xml``); if that XML is absent this page is
excluded from the build (see ``conf.py``).

Audio capture
-------------

.. doxygenfile:: audio.h
   :project: kws_de_fw

Storage
-------

.. doxygenfile:: storage.h
   :project: kws_de_fw

Voice activity detection
--------------------------

.. doxygenfile:: vad.h
   :project: kws_de_fw

Guided recorder
----------------

Implements :need:`REQ_FW_RECORD_TWO_TAKES`, :need:`REQ_FW_RECORD_CAPS`,
:need:`REQ_FW_RECORD_CLIP_REJECT`, :need:`REQ_FW_RECORD_SPEAKER_ID` (see
``record_start``/``record_post``/``record_get_status`` below).

.. doxygenfile:: record.h
   :project: kws_de_fw

MFCC front end
---------------

Implements :need:`REQ_FW_MFCC_PARITY` (``mfcc_compute``) and
:need:`REQ_FW_MFCC_QUANTIZE` (``mfcc_quantize``).

.. doxygenfile:: mfcc.h
   :project: kws_de_fw

Streaming detector
--------------------

Implements :need:`REQ_FW_DETECTOR_PARAMS` (``stream_push``).

.. doxygenfile:: stream.h
   :project: kws_de_fw

WAV writer
----------

Implements :need:`REQ_FW_RECORD_WAV_FORMAT` (``wav_write_header``).

.. doxygenfile:: wav.h
   :project: kws_de_fw

Prompts
-------

Implements :need:`REQ_FW_PROMPT_SHUFFLE_SEED` (``prompt_session_init``)
and :need:`REQ_FW_RECORD_FILENAME_SLUG` (``prompt_slug``).

.. doxygenfile:: prompts.h
   :project: kws_de_fw

USB mass storage
-----------------

Implements :need:`REQ_FW_USB_SINGLE_OWNER`.

.. doxygenfile:: usb_drive.h
   :project: kws_de_fw

Recogniser
----------

Implements :need:`REQ_FW_TFLM_OPSET`, :need:`REQ_FW_23_CLASSES`,
:need:`REQ_FW_RECOGNISE_LOG`.

.. doxygenfile:: recognise.h
   :project: kws_de_fw

UI
--

.. doxygenfile:: ui.h
   :project: kws_de_fw
