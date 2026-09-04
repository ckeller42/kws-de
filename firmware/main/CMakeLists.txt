# Vendored TFLite-Micro audio frontend + kissfft (see microfrontend/LICENSE.apache-2.0).
# Copied verbatim from pymicro-features so the on-device features stay bit-identical
# to the ones microWakeWord trained on — listed file by file rather than globbed, so
# a re-vendor has to be a deliberate, reviewable change.
set(MICROFRONTEND_LIB "microfrontend/tensorflow/lite/experimental/microfrontend/lib")
set(MICROFRONTEND_SRCS
    "${MICROFRONTEND_LIB}/fft.cc"
    "${MICROFRONTEND_LIB}/fft_util.cc"
    "${MICROFRONTEND_LIB}/filterbank.cc"
    "${MICROFRONTEND_LIB}/filterbank_util.cc"
    "${MICROFRONTEND_LIB}/frontend.cc"
    "${MICROFRONTEND_LIB}/frontend_util.cc"
    "${MICROFRONTEND_LIB}/kiss_fft_int16.cc"
    "${MICROFRONTEND_LIB}/log_lut.cc"
    "${MICROFRONTEND_LIB}/log_scale.cc"
    "${MICROFRONTEND_LIB}/log_scale_util.cc"
    "${MICROFRONTEND_LIB}/noise_reduction.cc"
    "${MICROFRONTEND_LIB}/noise_reduction_util.cc"
    "${MICROFRONTEND_LIB}/pcan_gain_control.cc"
    "${MICROFRONTEND_LIB}/pcan_gain_control_util.cc"
    "${MICROFRONTEND_LIB}/window.cc"
    "${MICROFRONTEND_LIB}/window_util.cc"
    "microfrontend/kissfft/kiss_fft.cc"
    "microfrontend/kissfft/tools/kiss_fftr.cc")

idf_component_register(
  SRCS "main.c" "audio.c" "storage.c" "vad.c" "record.c" "mfcc.c" "mfcc_fft.cc" "stream.c" "wav.c" "prompts.c" "recognise.cc"
       "wake.cc" "wakefront.c" "gen/wake_infer.c" "gen/command_infer.c" "assist_gate.c" "beep.c" "console.c" ${MICROFRONTEND_SRCS}
       "ui/ui_menu.c" "ui/ui_record.c" "ui/ui_usb.c" "ui/ui_recognise.c" "ui/ui_wake.c" "ui/ui_assist.c" "ui/font_prompt_28.c" "usb_drive.c"
  INCLUDE_DIRS "." "microfrontend" "microfrontend/kissfft"
  REQUIRES fatfs wear_levelling nvs_flash esp_timer esp_partition esp_tinyusb driver sdmmc
  PRIV_REQUIRES esp-tflite-micro esp-nn)
