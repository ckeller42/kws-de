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

static lv_obj_t *scr, *l_state, *l_big, *l_stats, *l_rec, *sw_field;
static lv_obj_t *card, *l_card_text, *l_card_words;
static lv_timer_t *s_card_timer;
static uint32_t s_last_fire;
static uint32_t s_flash_until;
static bool s_listening;
static bool s_field_on;              /* last toggle state painted, see ui_assist_refresh() */
static float s_field_thresh;         /* last capture threshold painted on the badge */

/** How long the result card stays up before it hides itself again. */
#define UI_ASSIST_CARD_MS 3000

/* "REC", or "REC 0.60" when capture is running a LOOSER gate than production.
   The takes on the card came from whichever detector was live, so the screen has
   to say which one — a plain "REC" means what was recorded is exactly what the
   shipped gate would have fired on. */
static void rec_badge(lv_obj_t *l, float thresh)
{
    char buf[16];
    if (thresh < WAKE_THRESHOLD) snprintf(buf, sizeof buf, "REC %.2f", (double)thresh);
    else snprintf(buf, sizeof buf, "REC");
    lv_label_set_text(l, buf);
}

static void on_back(lv_event_t *e) { (void)e; app_set_mode(UI_MODE_MENU); }

/* The toggle reads its own state rather than the event target: LVGL 9 hands
   back a void*, and the switch is a file static anyway. The badge is not touched
   here — ui_assist_refresh() paints it from wake_field_get(), so the tap and the
   console command take exactly the same path to the screen. */
static void on_field(lv_event_t *e)
{
    (void)e;
    wake_field_set(lv_obj_has_state(sw_field, LV_STATE_CHECKED));
}

void ui_show_assist(void)
{
    bsp_display_lock(0);
    if (s_card_timer) { lv_timer_delete(s_card_timer); s_card_timer = NULL; }
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

    /* Opt-in field capture: a switch the user must turn on once, and a "REC"
       badge that is the only visible difference while it is on. */
    l_rec = lv_label_create(scr);
    s_field_thresh = wake_field_thresh_get();
    rec_badge(l_rec, s_field_thresh);
    lv_obj_set_style_text_color(l_rec, lv_palette_main(LV_PALETTE_RED), 0);
    lv_obj_align(l_rec, LV_ALIGN_TOP_RIGHT, -12, 12);

    lv_obj_t *lf = lv_label_create(scr);
    lv_label_set_text(lf, "Aufnahme");
    lv_obj_set_style_text_color(lf, lv_color_hex(0x8a94a0), 0);
    lv_obj_align(lf, LV_ALIGN_BOTTOM_LEFT, 12, -24);

    sw_field = lv_switch_create(scr);
    lv_obj_align(sw_field, LV_ALIGN_BOTTOM_LEFT, 100, -28);
    lv_obj_add_event_cb(sw_field, on_field, LV_EVENT_VALUE_CHANGED, NULL);
    s_field_on = wake_field_get();
    if (s_field_on) lv_obj_add_state(sw_field, LV_STATE_CHECKED);
    else lv_obj_add_flag(l_rec, LV_OBJ_FLAG_HIDDEN);

    lv_obj_t *b = lv_button_create(scr);
    lv_obj_set_size(b, 120, 44);
    lv_obj_align(b, LV_ALIGN_BOTTOM_MID, 0, -16);
    lv_obj_add_event_cb(b, on_back, LV_EVENT_CLICKED, NULL);
    lv_obj_t *bl = lv_label_create(b);
    lv_label_set_text(bl, "Menu");
    lv_obj_center(bl);

    /* Result card: shown for UI_ASSIST_CARD_MS by ui_assist_show_result() at
       each window's close, hidden the rest of the time. */
    card = lv_obj_create(scr);
    lv_obj_set_size(card, 280, 140);
    lv_obj_center(card);
    lv_obj_set_style_border_width(card, 0, 0);
    lv_obj_set_style_radius(card, 12, 0);
    lv_obj_add_flag(card, LV_OBJ_FLAG_HIDDEN);

    l_card_text = lv_label_create(card);
    lv_obj_set_style_text_font(l_card_text, &font_prompt_28, 0);
    lv_obj_set_style_text_color(l_card_text, lv_color_black(), 0);
    lv_label_set_long_mode(l_card_text, LV_LABEL_LONG_WRAP);
    lv_obj_set_width(l_card_text, 256);
    lv_obj_set_style_text_align(l_card_text, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_align(l_card_text, LV_ALIGN_TOP_MID, 0, 14);

    l_card_words = lv_label_create(card);
    lv_obj_set_style_text_color(l_card_words, lv_color_hex(0x2a2f36), 0);
    lv_label_set_long_mode(l_card_words, LV_LABEL_LONG_WRAP);
    lv_obj_set_width(l_card_words, 256);
    lv_obj_set_style_text_align(l_card_words, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_align(l_card_words, LV_ALIGN_BOTTOM_MID, 0, -10);

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

    /* The badge follows the toggle, not the switch that was last tapped: `field
       on` over the console must never leave the device recording with nothing
       on screen to say so. Edge-triggered, so the common frame does no work. */
    bool fon = wake_field_get();
    float fth = wake_field_thresh_get();
    if (fth != s_field_thresh) {
        s_field_thresh = fth;
        rec_badge(l_rec, fth);
    }
    if (fon != s_field_on) {
        s_field_on = fon;
        if (fon) {
            lv_obj_clear_flag(l_rec, LV_OBJ_FLAG_HIDDEN);
            lv_obj_add_state(sw_field, LV_STATE_CHECKED);
        } else {
            lv_obj_add_flag(l_rec, LV_OBJ_FLAG_HIDDEN);
            lv_obj_remove_state(sw_field, LV_STATE_CHECKED);
        }
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

/* Runs on the LVGL task's own timer, not a model task: the card's 3 s
   lifetime is entirely LVGL-side once ui_assist_show_result() has shown it. */
static void card_hide_cb(lv_timer_t *t)
{
    lv_obj_add_flag(card, LV_OBJ_FLAG_HIDDEN);
    s_card_timer = NULL;
    lv_timer_delete(t);
}

void ui_assist_show_result(bool valid, const char *text, const char *heard_words)
{
    if (!bsp_display_lock(50)) return;      /* skip a frame rather than block the wake task */
    lv_obj_set_style_bg_color(card, valid ? lv_palette_main(LV_PALETTE_GREEN) : lv_color_hex(0x8a94a0), 0);
    lv_label_set_text(l_card_text, valid ? text : "nicht verstanden");
    lv_label_set_text(l_card_words, valid ? "" : (heard_words && *heard_words ? heard_words : "(nichts gehoert)"));
    lv_obj_clear_flag(card, LV_OBJ_FLAG_HIDDEN);
    if (s_card_timer) lv_timer_delete(s_card_timer);
    s_card_timer = lv_timer_create(card_hide_cb, UI_ASSIST_CARD_MS, NULL);
    lv_timer_set_repeat_count(s_card_timer, 1);
    bsp_display_unlock();
}
