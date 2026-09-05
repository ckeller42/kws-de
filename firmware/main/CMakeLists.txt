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
       "wake.cc" "wakefront.c" "gen/wake_infer.c" "gen/command_infer.c" "infer_lock.c" "assist_gate.c" "field.c" "intent.c" "beep.c" "console.c" ${MICROFRONTEND_SRCS}
       "ui/ui_menu.c" "ui/ui_record.c" "ui/ui_usb.c" "ui/ui_recognise.c" "ui/ui_wake.c" "ui/ui_assist.c" "ui/font_prompt_28.c" "usb_drive.c"
  INCLUDE_DIRS "." "microfrontend" "microfrontend/kissfft"
  REQUIRES fatfs wear_levelling nvs_flash esp_timer esp_partition esp_tinyusb driver sdmmc
  PRIV_REQUIRES esp-tflite-micro esp-nn)

# Both generated models work in one esp-nn scratch region, kws_infer_scratch,
# defined by infer_lock.c at the larger of their two reserves — esp-nn's scratch
# pointers are file-static globals, so a region per model would be handed to the
# other model's kernels (see infer_lock.h). Without this define each generated
# file falls back to a private region, which is right when it is linked on its
# own (firmware/test's host harness) and wrong here.
set_source_files_properties("gen/wake_infer.c" "gen/command_infer.c" PROPERTIES
    COMPILE_DEFINITIONS "KWS_INFER_SHARED_SCRATCH")

# With CONFIG_KWS_INFER_GENERATED=n nothing calls into either generated file and
# nothing references kws_infer_scratch, so -ffunction-sections/-fdata-sections
# plus --gc-sections drop all three whole: the fallback build pays no RAM for
# the path it does not use. Checked with xtensa-esp32s3-elf-nm on both builds.

# Arena placement (Kconfig choice KWS_INFER_COMMAND_ARENA). The generated file
# declares its arena as `COMMAND_INFER_ARENA_ATTR static int8_t arena[...]` with
# the macro defaulting to empty, so pointing it at EXT_RAM_BSS_ATTR is all it
# takes to move 51 KB out of internal SRAM. -include esp_attr.h because the
# generated C is platform-neutral and pulls in no ESP-IDF header of its own.
if(CONFIG_KWS_INFER_COMMAND_ARENA_PSRAM)
  set_source_files_properties("gen/command_infer.c" PROPERTIES COMPILE_OPTIONS
      "-include;esp_attr.h;-DCOMMAND_INFER_ARENA_ATTR=EXT_RAM_BSS_ATTR")
endif()
