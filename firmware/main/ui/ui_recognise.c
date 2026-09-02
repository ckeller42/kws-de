#include "ui.h"
#include <stdio.h>
#include "bsp/esp-bsp.h"
#include "lvgl.h"
#include "recognise.h"

static lv_obj_t *l_word, *l_stats;
static void on_back(lv_event_t *e) { (void)e; app_set_mode(UI_MODE_RECORD); }

void ui_show_recognise(void)
{
    bsp_display_lock(0);
    lv_obj_t *scr = lv_obj_create(NULL);
    l_word = lv_label_create(scr);
    lv_obj_set_style_text_font(l_word, &lv_font_montserrat_28, 0);
    lv_label_set_text(l_word, "...");
    lv_obj_align(l_word, LV_ALIGN_CENTER, 0, -30);
    l_stats = lv_label_create(scr);
    lv_obj_align(l_stats, LV_ALIGN_CENTER, 0, 30);
    lv_obj_t *b = lv_button_create(scr);
    lv_obj_set_size(b, 120, 44); lv_obj_align(b, LV_ALIGN_BOTTOM_MID, 0, -16);
    lv_obj_add_event_cb(b, on_back, LV_EVENT_CLICKED, NULL);
    lv_obj_t *bl = lv_label_create(b); lv_label_set_text(bl, "Record"); lv_obj_center(bl);
    lv_screen_load(scr);
    bsp_display_unlock();
}

void ui_recognise_refresh(const recognise_status_t *st)
{
    char buf[80];
    if (!bsp_display_lock(50)) return;
    lv_label_set_text(l_word, st->word[0] ? st->word : "...");
    snprintf(buf, sizeof buf, "conf %.2f   %lu ms   arena %lu B   fired %lu",
             st->conf, (unsigned long)st->infer_ms, (unsigned long)st->arena_used, (unsigned long)st->fired_count);
    lv_label_set_text(l_stats, buf);
    bsp_display_unlock();
}
