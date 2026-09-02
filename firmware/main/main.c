#include "bsp/esp-bsp.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "ff.h"
#include "audio.h"
#include "record.h"
#include "storage.h"
#include "usb_drive.h"
#include "ui/ui.h"

static const char *TAG = "main";
static ui_mode_t s_mode = UI_MODE_RECORD;

/* wait up to 1s for the recorder task to drain to idle before pulling storage out from under it */
static void wait_record_idle(void)
{
    record_status_t st;
    for (int i = 0; i < 10; i++) {
        record_get_status(&st);
        if (st.phase == REC_IDLE) return;
        vTaskDelay(pdMS_TO_TICKS(100));
    }
}

void app_set_mode(ui_mode_t m)
{
    if (m == s_mode) return;
    ESP_LOGI(TAG, "mode %d -> %d", s_mode, m);
    if (s_mode == UI_MODE_RECORD) record_post(REC_CMD_PAUSE);
    if (s_mode == UI_MODE_USB) ESP_ERROR_CHECK(usb_drive_exit());
    /* Task 7 adds: recognise_pause/resume */
    s_mode = m;
    if (m == UI_MODE_USB) {
        ui_show_usb();
        wait_record_idle();                /* no fopen races the unmount */
        ESP_ERROR_CHECK(usb_drive_enter());
    }
    if (m == UI_MODE_RECORD) { ui_show_record(); record_post(REC_CMD_RESUME); }
}

void app_main(void)
{
    ESP_ERROR_CHECK(nvs_flash_init());
    bsp_i2c_init();
    bsp_display_start();
    bsp_display_backlight_on();
    ESP_ERROR_CHECK(storage_mount());
    /* volume label is a FAT property, not a USB descriptor: set once. Only one FAT
     * drive is ever mounted on this device, so its FatFs pdrv is always 0.
     * ponytail: hardcoded "0:" drive, revisit if a second FAT volume is ever added */
    char label[12] = {0};
    if (f_getlabel("0:", label, NULL) == FR_OK && label[0] == '\0') f_setlabel("0:KWSREC");
    audio_start();
    ui_show_record();
    record_start();
    record_post(REC_CMD_RESUME);
}
