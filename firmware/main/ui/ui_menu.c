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
static void on_usb(lv_event_t *e)         { (void)e; app_set_mode(UI_MODE_USB); }
static void on_menu(lv_event_t *e)        { (void)e; app_set_mode(UI_MODE_MENU); }

/* Boot screen and every mode's "back" destination: a small title line over a
   single column of six full-width touch targets, one per mode. Dark theme
   matching the record screen. Fits 320x240: rows 26..238 x 10..310, 32 px
   tall with a 4 px gap. Assistant first: it is the one that is meant to be
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
    lv_obj_align(title, LV_ALIGN_TOP_MID, 0, 6);

    big_button(scr, "Assistent",          on_assist,      10, 26,  300, 32);
    big_button(scr, "Recognition",        on_recognise,   10, 62,  300, 32);
    big_button(scr, "Hey Bus",            on_wake,        10, 98,  300, 32);
    big_button(scr, "Record",             on_record,      10, 134, 300, 32);
    big_button(scr, "Hey Bus aufnehmen",  on_record_wake, 10, 170, 300, 32);
    big_button(scr, "USB",                on_usb,         10, 206, 300, 32);

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
