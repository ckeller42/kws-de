idf_component_register(
  SRCS "main.c" "audio.c" "storage.c" "vad.c" "record.c" "mfcc.c" "stream.c" "wav.c" "prompts.c" "ui/ui_record.c" "usb_drive.c" "ui/ui_usb.c"
  INCLUDE_DIRS "."
  REQUIRES fatfs wear_levelling nvs_flash esp_timer esp_partition esp_tinyusb)
