#include "ui.h"
#include "bsp/esp-bsp.h"
#include "lvgl.h"

static void on_back(lv_event_t *e) { (void)e; app_set_mode(UI_MODE_MENU); }

void ui_show_usb(void)
{
    bsp_display_lock(0);
    lv_obj_t *scr = lv_obj_create(NULL);
    lv_obj_t *l = lv_label_create(scr);
    lv_label_set_text(l, "USB drive mode\n\nKWSREC is mounted on your computer.\nRun scripts/pull-recordings.sh,\nthen tap Menu.");
    lv_obj_set_style_text_align(l, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_align(l, LV_ALIGN_TOP_MID, 0, 30);
    lv_obj_t *b = lv_button_create(scr);
    lv_obj_set_size(b, 120, 44); lv_obj_align(b, LV_ALIGN_BOTTOM_MID, 0, -16);
    lv_obj_add_event_cb(b, on_back, LV_EVENT_CLICKED, NULL);
    lv_obj_t *bl = lv_label_create(b); lv_label_set_text(bl, "Menu"); lv_obj_center(bl);
    lv_screen_load(scr);
    bsp_display_unlock();
}
