#include "ui.h"
#include <stdio.h>
#include "assist_gate.h"
#include "bsp/esp-bsp.h"
#include "lvgl.h"

/* Assistant screen: the wake-gated duty cycle made visible. Idle shows the live
   wake probability, exactly as the wake test screen does. A fire flashes green
   and switches the big label to the recognised word for as long as the gate
   stays open, which is the recognise screen's job. One screen, two states, so
   the difference between "listening for the wake word" and "listening for a
   command" is never ambiguous from across the van. */
#define UI_ASSIST_BG 0x12161c

LV_FONT_DECLARE(font_prompt_28);

static lv_obj_t *scr, *l_state, *l_big, *l_stats;
static uint32_t s_last_fire;
static uint32_t s_flash_until;
static bool s_listening;

static void on_back(lv_event_t *e) { (void)e; app_set_mode(UI_MODE_MENU); }

void ui_show_assist(void)
{
    bsp_display_lock(0);
    scr = lv_obj_create(NULL);
    lv_obj_set_style_bg_color(scr, lv_color_hex(UI_ASSIST_BG), 0);
    lv_obj_set_style_text_color(scr, lv_color_hex(0xe6eaef), 0);
    s_last_fire = 0;
    s_flash_until = 0;
    s_listening = false;

    l_state = lv_label_create(scr);
    lv_label_set_text(l_state, "Assistent - sag \"Hey Bus\"");
    lv_obj_set_style_text_font(l_state, &font_prompt_28, 0);
    lv_obj_set_style_text_color(l_state, lv_color_hex(0x8a94a0), 0);
    lv_obj_align(l_state, LV_ALIGN_TOP_MID, 0, 12);

    l_big = lv_label_create(scr);
    lv_obj_set_style_text_font(l_big, &font_prompt_28, 0);
    lv_obj_set_style_text_color(l_big, lv_color_white(), 0);
    lv_label_set_text(l_big, "0.00");
    lv_obj_align(l_big, LV_ALIGN_CENTER, 0, -10);

    l_stats = lv_label_create(scr);
    lv_obj_set_style_text_color(l_stats, lv_color_hex(0x8a94a0), 0);
    lv_label_set_text(l_stats, "");
    lv_obj_align(l_stats, LV_ALIGN_CENTER, 0, 30);

    lv_obj_t *b = lv_button_create(scr);
    lv_obj_set_size(b, 120, 44);
    lv_obj_align(b, LV_ALIGN_BOTTOM_MID, 0, -16);
    lv_obj_add_event_cb(b, on_back, LV_EVENT_CLICKED, NULL);
    lv_obj_t *bl = lv_label_create(b);
    lv_label_set_text(bl, "Menu");
    lv_obj_center(bl);

    lv_screen_load(scr);
    bsp_display_unlock();
}

void ui_assist_refresh(const wake_status_t *wst, const recognise_status_t *rst, bool listening)
{
    char buf[96];
    if (!bsp_display_lock(50)) return;      /* skip a frame rather than block a model task */

    uint32_t now = lv_tick_get();
    if (wst->fired_at_ms && wst->fired_at_ms != s_last_fire) {
        s_last_fire = wst->fired_at_ms;
        s_flash_until = now + WAKE_FLASH_MS;
        lv_obj_set_style_bg_color(scr, lv_palette_main(LV_PALETTE_GREEN), 0);
    } else if (s_flash_until && (int32_t)(now - s_flash_until) >= 0) {
        s_flash_until = 0;
        lv_obj_set_style_bg_color(scr, lv_color_hex(UI_ASSIST_BG), 0);
    }

    if (listening != s_listening) {
        s_listening = listening;
        lv_label_set_text(l_state, listening ? "Ich hoere zu" : "Assistent - sag \"Hey Bus\"");
    }
    if (listening) {
        snprintf(buf, sizeof buf, "%s", rst->word[0] ? rst->word : "...");
        lv_label_set_text(l_big, buf);
        snprintf(buf, sizeof buf, "%.2f   %lu ms", (double)rst->conf, (unsigned long)rst->infer_ms);
    } else {
        snprintf(buf, sizeof buf, "%.2f", (double)wst->prob);
        lv_label_set_text(l_big, buf);
        snprintf(buf, sizeof buf, "%lu Anfragen   %lu ms", (unsigned long)wst->fired_count,
                 (unsigned long)wst->infer_ms);
    }
    lv_label_set_text(l_stats, buf);
    bsp_display_unlock();
}
