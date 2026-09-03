#include "bsp/esp-bsp.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "ff.h"
#include "audio.h"
#include "record.h"
#include "storage.h"
#include "wake.h"
#include "usb_drive.h"
#include "console.h"
#include "gen/model_config.h"
#include "gen/wake_model_config.h"
#include "ui/ui.h"

static const char *TAG = "main";
static ui_mode_t s_mode = UI_MODE_MENU;

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
    if (s_mode == UI_MODE_RECORD || s_mode == UI_MODE_RECORD_WAKE) record_post(REC_CMD_PAUSE);
    if (s_mode == UI_MODE_USB) ESP_ERROR_CHECK(usb_drive_exit());
    if (s_mode == UI_MODE_RECOGNISE) recognise_set_active(false);
    if (s_mode == UI_MODE_WAKE) wake_set_active(false);
    if (s_mode == UI_MODE_ASSIST) { wake_set_active(false); recognise_set_active(false); }
    s_mode = m;
    if (m == UI_MODE_MENU) ui_show_menu();
    if (m == UI_MODE_USB) {
        ui_show_usb();
        wait_record_idle();                /* no fopen races the unmount */
        ESP_ERROR_CHECK(usb_drive_enter());
    }
    /* Entering RECORD always starts a fresh guided session: new speaker id,
       sentences first, negatives auto-chained on completion (record.c). */
    if (m == UI_MODE_RECORD) { ui_show_record(); record_post(REC_CMD_START_SESSION); }
    /* RECORD_WAKE reuses the record screen for a "Hey Bus"-only session: new
       speaker id, PROMPT_WAKE only, no negatives chained on completion. */
    if (m == UI_MODE_RECORD_WAKE) { ui_show_record(); record_post(REC_CMD_START_WAKE_SESSION); }
    if (m == UI_MODE_RECOGNISE) { ui_show_recognise(); recognise_set_active(true); }
    /* Wake mode measures the wake model alone: the command recogniser stays off
       so nothing else competes for the mic, the CPU, or the screen. */
    if (m == UI_MODE_WAKE) { ui_show_wake(); recognise_set_active(false); wake_set_active(true); }
    /* Assist: the wake model runs continuously and opens a window for the
       recogniser on each fire, so the recogniser starts OFF and the wake
       task turns it on. See assist_gate.h. */
    if (m == UI_MODE_ASSIST) { ui_show_assist(); recognise_set_active(false); wake_set_active(true); }
}

ui_mode_t app_get_mode(void) { return s_mode; }

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
    /* Which models this image actually carries. A firmware binary outlives the
       checkout that built it, so the stamp (name@sha8 date, generated into the
       model config headers) is the only way to answer that from the device. */
    ESP_LOGI(TAG, "models: command %s, wake %s", KWS_MODEL_ID, KWS_WAKE_MODEL_ID);
    audio_start();
    /* Wake before recognise: only one TFLM arena fits internal SRAM, and the
       first caller takes it. The wake model is the always-on one and gains far
       more from being there (3x a step) than the recogniser does (5%), so it
       gets first claim rather than whichever happened to start first. Taking
       the smaller arena first also leaves room for both tasks' 16 KB stacks. */
    wake_start();
    recognise_start();
    record_start();                    /* starts paused (REC_IDLE) until Record is chosen */
    console_start();
    ui_show_menu();
}
