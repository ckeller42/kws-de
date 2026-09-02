#include "bsp/esp-bsp.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "audio.h"
#include "record.h"
#include "storage.h"
#include "ui/ui.h"

static const char *TAG = "main";
static ui_mode_t s_mode = UI_MODE_RECORD;

void app_set_mode(ui_mode_t m)
{
    if (m == s_mode) return;
    ESP_LOGI(TAG, "mode %d -> %d", s_mode, m);
    if (s_mode == UI_MODE_RECORD) record_post(REC_CMD_PAUSE);
    /* Tasks 6/7 add: usb_drive_enter/exit, recognise_pause/resume */
    s_mode = m;
    if (m == UI_MODE_RECORD) { ui_show_record(); record_post(REC_CMD_RESUME); }
}

void app_main(void)
{
    ESP_ERROR_CHECK(nvs_flash_init());
    bsp_i2c_init();
    bsp_display_start();
    bsp_display_backlight_on();
    ESP_ERROR_CHECK(storage_mount());
    audio_start();
    ui_show_record();
    record_start();
    record_post(REC_CMD_RESUME);
}
