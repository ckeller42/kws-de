#include "ui.h"
#include <stdio.h>
#include "bsp/esp-bsp.h"
#include "lvgl.h"
#include "wake.h"

/* Wake test screen: nothing but the wake model's own output. Dark base, a big
   live probability, and a full-screen green flash on every fire so a detection
   is unmissable from across the van. */
#define UI_WAKE_BG 0x12161c

static lv_obj_t *scr, *l_prob, *l_stats;
static uint32_t s_last_fire;      /* fired_at_ms of the fire we are already showing */
static uint32_t s_flash_until;    /* lv_tick when the green flash ends, 0 = not flashing */

static void on_back(lv_event_t *e) { (void)e; app_set_mode(UI_MODE_MENU); }

void ui_show_wake(void)
{
    bsp_display_lock(0);
    scr = lv_obj_create(NULL);
    lv_obj_set_style_bg_color(scr, lv_color_hex(UI_WAKE_BG), 0);
    lv_obj_set_style_text_color(scr, lv_color_hex(0xe6eaef), 0);
    s_last_fire = 0;
    s_flash_until = 0;

    lv_obj_t *title = lv_label_create(scr);
    lv_label_set_text(title, "Hey Bus?");
    lv_obj_set_style_text_font(title, &lv_font_montserrat_28, 0);
    lv_obj_align(title, LV_ALIGN_TOP_MID, 0, 12);

    l_prob = lv_label_create(scr);
    lv_obj_set_style_text_font(l_prob, &lv_font_montserrat_28, 0);
    lv_obj_set_style_text_color(l_prob, lv_color_white(), 0);
    lv_label_set_text(l_prob, "0.00");
    lv_obj_align(l_prob, LV_ALIGN_CENTER, 0, -10);

    l_stats = lv_label_create(scr);
    lv_obj_set_style_text_color(l_stats, lv_color_hex(0x8a94a0), 0);
    lv_label_set_text(l_stats, "fired 0");
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

void ui_wake_refresh(const wake_status_t *st)
{
    char buf[80];
    if (!bsp_display_lock(50)) return;      /* skip a frame rather than block the wake task */

    /* A new fire is one with a fired_at_ms we have not painted yet. The flash is
       cleared by the next refresh past its deadline; refreshes arrive every
       inference step (~30 ms), so the 600 ms window is honoured closely. */
    uint32_t now = lv_tick_get();
    if (st->fired_at_ms && st->fired_at_ms != s_last_fire) {
        s_last_fire = st->fired_at_ms;
        s_flash_until = now + WAKE_FLASH_MS;
        lv_obj_set_style_bg_color(scr, lv_palette_main(LV_PALETTE_GREEN), 0);
    } else if (s_flash_until && (int32_t)(now - s_flash_until) >= 0) {
        s_flash_until = 0;
        lv_obj_set_style_bg_color(scr, lv_color_hex(UI_WAKE_BG), 0);
    }

    snprintf(buf, sizeof buf, "%.2f", st->prob);
    lv_label_set_text(l_prob, buf);
    snprintf(buf, sizeof buf, "fired %lu   %lu ms   arena %lu B",
             (unsigned long)st->fired_count, (unsigned long)st->infer_ms,
             (unsigned long)st->arena_used);
    lv_label_set_text(l_stats, buf);
    bsp_display_unlock();
}
