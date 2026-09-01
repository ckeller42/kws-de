#include <stdio.h>
#include "bsp/esp-bsp.h"
#include "esp_log.h"
#include "lvgl.h"

static const char *TAG = "main";

void app_main(void)
{
    bsp_i2c_init();
    lv_display_t *disp = bsp_display_start();
    bsp_display_backlight_on();
    (void)disp;

    bsp_display_lock(0);
    lv_obj_t *label = lv_label_create(lv_screen_active());
    lv_label_set_text(label, "kws-de firmware");
    lv_obj_center(label);
    bsp_display_unlock();

    ESP_LOGI(TAG, "boot ok, free heap %lu", (unsigned long)esp_get_free_heap_size());
}
