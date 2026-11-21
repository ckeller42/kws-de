#include "ui.h"
#include <stdio.h>
#include "bsp/esp-bsp.h"
#include "lvgl.h"

/* Success-screen text carries the speaker id and, potentially, German text;
   reuse the umlaut-subset font already built for the record/recognise screens. */
LV_FONT_DECLARE(font_prompt_28);

#define UI_MENU_BG  0x12161c
#define UI_MENU_FG  0xe6eaef
#define UI_MENU_DIM 0x8a94a0
#define UI_MENU_BTN 0x2a3340

static lv_obj_t *big_button(lv_obj_t *parent, const char *txt, lv_event_cb_t cb, int x, int y, int w, int h)
{
    lv_obj_t *b = lv_button_create(parent);
    lv_obj_set_size(b, w, h);
    lv_obj_set_pos(b, x, y);
    lv_obj_set_style_bg_color(b, lv_color_hex(UI_MENU_BTN), 0);
    lv_obj_add_event_cb(b, cb, LV_EVENT_CLICKED, NULL);
    lv_obj_t *l = lv_label_create(b);
    lv_label_set_text(l, txt);
    lv_obj_center(l);
    return b;
}

static void on_recognise(lv_event_t *e)   { (void)e; app_set_mode(UI_MODE_RECOGNISE); }
static void on_wake(lv_event_t *e)        { (void)e; app_set_mode(UI_MODE_WAKE); }
static void on_assist(lv_event_t *e)      { (void)e; app_set_mode(UI_MODE_ASSIST); }
static void on_record(lv_event_t *e)      { (void)e; app_set_mode(UI_MODE_RECORD); }
static void on_record_wake(lv_event_t *e) { (void)e; app_set_mode(UI_MODE_RECORD_WAKE); }
static void on_record_elicit(lv_event_t *e) { (void)e; app_set_mode(UI_MODE_RECORD_ELICIT); }
static void on_record_scene(lv_event_t *e) { (void)e; app_set_mode(UI_MODE_RECORD_SCENE); }
static void on_record_words(lv_event_t *e) { (void)e; app_set_mode(UI_MODE_RECORD_WORDS); }
static void on_record_light(lv_event_t *e) { (void)e; app_set_mode(UI_MODE_RECORD_LIGHT); }
static void on_usb(lv_event_t *e)         { (void)e; app_set_mode(UI_MODE_USB); }
static void on_menu(lv_event_t *e)        { (void)e; app_set_mode(UI_MODE_MENU); }

/* Boot screen and every mode's "back" destination: a small title line over a
   single column of ten full-width touch targets, one per mode. Dark theme
   matching the record screen. Fits 320x240: rows 24..233 x 10..310, 20 px
   tall with a 1 px gap. Assistant first: it is the one that is meant to be
   used, the others are measurement and data-collection modes. */
void ui_show_menu(void)
{
    bsp_display_lock(0);
    lv_obj_t *scr = lv_obj_create(NULL);
    lv_obj_set_style_bg_color(scr, lv_color_hex(UI_MENU_BG), 0);
    lv_obj_set_style_text_color(scr, lv_color_hex(UI_MENU_FG), 0);

    lv_obj_t *title = lv_label_create(scr);
    lv_label_set_text(title, "kws-de");
    lv_obj_set_style_text_color(title, lv_color_hex(UI_MENU_DIM), 0);
    lv_obj_align(title, LV_ALIGN_TOP_MID, 0, 4);

    static const struct { const char *txt; lv_event_cb_t cb; } rows[] = {
        {"Assistent", on_assist},
        {"Recognition", on_recognise},
        {"Hey Bus", on_wake},
        {"Record", on_record},
        {"Hey Bus aufnehmen", on_record_wake},
        {"Situationen", on_record_elicit},
        {"Szenen aufnehmen", on_record_scene},
        {"Einzelwort aufnehmen", on_record_words},
        {"Licht aufnehmen", on_record_light},
        {"USB", on_usb},
    };
    for (int i = 0; i < (int)(sizeof rows / sizeof *rows); i++)
        big_button(scr, rows[i].txt, rows[i].cb, 10, 24 + i * 21, 300, 20);

    lv_screen_load(scr);
    bsp_display_unlock();
}

/* Shown by ui_record_refresh() when the recorder reaches REC_SESSION_DONE:
   the sentence and negative sets have both been completed for one speaker. */
void ui_show_success(const char *speaker, int saved_takes)
{
    bsp_display_lock(0);
    lv_obj_t *scr = lv_obj_create(NULL);
    lv_obj_set_style_bg_color(scr, lv_color_hex(UI_MENU_BG), 0);
    lv_obj_set_style_text_color(scr, lv_color_hex(UI_MENU_FG), 0);

    lv_obj_t *l = lv_label_create(scr);
    lv_obj_set_style_text_font(l, &font_prompt_28, 0);
    lv_obj_set_style_text_align(l, LV_TEXT_ALIGN_CENTER, 0);
    char buf[96];
    snprintf(buf, sizeof buf, "Fertig - danke!\n%s\n%d takes saved", speaker, saved_takes);
    lv_label_set_text(l, buf);
    lv_obj_align(l, LV_ALIGN_CENTER, 0, -20);

    big_button(scr, "Menu", on_menu, 60, 190, 200, 40);

    lv_screen_load(scr);
    bsp_display_unlock();
}
