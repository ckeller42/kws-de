#include "ui.h"
#include <stdint.h>
#include <stdio.h>
#include "bsp/esp-bsp.h"
#include "lvgl.h"
#include "record.h"

static lv_obj_t *scr, *l_set, *l_counter, *l_speaker, *l_prompt, *bar, *l_phase, *b_next;

static void on_cmd(lv_event_t *e) { record_post((record_cmd_t)(intptr_t)lv_event_get_user_data(e)); }
static void on_mode_recognise(lv_event_t *e) { (void)e; app_set_mode(UI_MODE_RECOGNISE); }
static void on_mode_usb(lv_event_t *e) { (void)e; app_set_mode(UI_MODE_USB); }

static lv_obj_t *button(lv_obj_t *parent, const char *txt, lv_event_cb_t cb, void *ud, int x, int y, int w)
{
    lv_obj_t *b = lv_button_create(parent);
    lv_obj_set_size(b, w, 40);
    lv_obj_set_pos(b, x, y);
    lv_obj_add_event_cb(b, cb, LV_EVENT_CLICKED, ud);
    lv_obj_t *l = lv_label_create(b);
    lv_label_set_text(l, txt);
    lv_obj_center(l);
    return b;
}

void ui_show_record(void)
{
    bsp_display_lock(0);
    scr = lv_obj_create(NULL);
    l_set = lv_label_create(scr);     lv_obj_set_pos(l_set, 8, 6);
    l_counter = lv_label_create(scr); lv_obj_set_pos(l_counter, 200, 6);
    l_speaker = lv_label_create(scr); lv_obj_set_pos(l_speaker, 264, 6);
    l_prompt = lv_label_create(scr);  lv_obj_set_width(l_prompt, 304); lv_obj_set_pos(l_prompt, 8, 50);
    lv_obj_set_style_text_font(l_prompt, &lv_font_montserrat_28, 0);
    lv_label_set_long_mode(l_prompt, LV_LABEL_LONG_WRAP);
    lv_obj_set_style_text_align(l_prompt, LV_TEXT_ALIGN_CENTER, 0);
    bar = lv_bar_create(scr); lv_obj_set_size(bar, 200, 14); lv_obj_set_pos(bar, 60, 128);
    lv_bar_set_range(bar, -60, 0);
    l_phase = lv_label_create(scr);   lv_obj_set_pos(l_phase, 8, 148);
    button(scr, "Redo", on_cmd, (void *)REC_CMD_REDO, 8, 176, 70);
    button(scr, "Skip", on_cmd, (void *)REC_CMD_SKIP, 92, 176, 70);
    b_next = button(scr, "Next", on_cmd, (void *)REC_CMD_NEXT, 236, 176, 76);
    button(scr, "Recog", on_mode_recognise, NULL, 8, 222, 60);
    button(scr, "USB", on_mode_usb, NULL, 76, 222, 50);
    button(scr, "+Spk", on_cmd, (void *)REC_CMD_NEW_SPEAKER, 134, 222, 56);
    button(scr, "W", on_cmd, (void *)REC_CMD_SET_WORDS, 198, 222, 34);
    button(scr, "S", on_cmd, (void *)REC_CMD_SET_SENTENCES, 238, 222, 34);
    button(scr, "N", on_cmd, (void *)REC_CMD_SET_NEGS, 278, 222, 34);
    lv_screen_load(scr);
    bsp_display_unlock();
}

void ui_record_refresh(const record_status_t *st)
{
    static const char *setname[] = {"words", "sentences", "negatives"};
    static const char *phase[] = {"paused", "listening...", "recording", "saved", "CLIPPED - redo", "no speech - redo", "flash full", "set complete"};
    char buf[48];
    if (!bsp_display_lock(50)) return;          /* skip a frame rather than block the recorder */
    snprintf(buf, sizeof buf, "%s \xc2\xb7 seed %lu", setname[st->set], (unsigned long)st->seed); lv_label_set_text(l_set, buf);
    snprintf(buf, sizeof buf, "%d/%d", st->index + 1, st->count);                        lv_label_set_text(l_counter, buf);
    lv_label_set_text(l_speaker, st->speaker);
    lv_label_set_text(l_prompt, st->prompt);
    lv_label_set_text(l_phase, phase[st->phase]);
    lv_bar_set_value(bar, (int)st->level_dbfs, LV_ANIM_OFF);
    lv_obj_set_style_bg_color(bar, st->phase == REC_CLIPPED ? lv_palette_main(LV_PALETTE_RED) : lv_palette_main(LV_PALETTE_GREEN), LV_PART_INDICATOR);
    bsp_display_unlock();
}
