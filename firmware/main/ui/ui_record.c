#include "ui.h"
#include <stdint.h>
#include <stdio.h>
#include "bsp/esp-bsp.h"
#include "esp_heap_caps.h"
#include "lvgl.h"
#include "record.h"

/* Montserrat-28 subset that includes the German umlauts (a-o-u umlaut, sharp s);
   the built-in lv_font_montserrat_28 is ASCII-only, so prompts like the umlaut
   words showed boxes. Generated with lv_font_conv, committed as source. */
LV_FONT_DECLARE(font_prompt_28);

static lv_obj_t *scr, *l_counter, *l_prompt, *bar, *l_phase;

/* app_set_mode() already pauses the recorder whenever RECORD is left (see
   main.c), so aborting is just a mode switch — same pattern as every other
   screen's back button. */
static void on_abort(lv_event_t *e) { (void)e; app_set_mode(UI_MODE_MENU); }

static lv_obj_t *button(lv_obj_t *parent, const char *txt, lv_event_cb_t cb, void *ud, int x, int y, int w)
{
    lv_obj_t *b = lv_button_create(parent);
    lv_obj_set_size(b, w, 34);
    lv_obj_set_pos(b, x, y);
    lv_obj_add_event_cb(b, cb, LV_EVENT_CLICKED, ud);
    lv_obj_t *l = lv_label_create(b);
    lv_label_set_text(l, txt);
    lv_obj_center(l);
    return b;
}

/* Debug tool (off by default): stream the live screen over the serial console so
   UI colours/layout can be reviewed from a laptop without a camera. Build with
   -DKWS_UI_SCREENSHOT=1 to enable. On each phase change it snapshots the screen
   to a PSRAM buffer (LVGL mem is only 64 KB, so we hand it our own), RLE-packs
   the RGB565, and base64s it between [SHOT w h RLE16 n]…[/SHOT] markers; a host
   script decodes it to PNG. */
#ifndef KWS_UI_SCREENSHOT
#define KWS_UI_SCREENSHOT 0
#endif
#if KWS_UI_SCREENSHOT
static uint8_t *s_shot, *s_rle;
#define SHOT_SZ (320 * 240 * 2 + 4096)
#define RLE_SZ (320 * 240 * 4)          /* worst case (no runs); PSRAM is plentiful */

static void b64_print(const uint8_t *p, size_t n)
{
    static const char T[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    char line[80]; int c = 0;
    for (size_t i = 0; i < n; i += 3) {
        uint32_t v = (uint32_t)p[i] << 16;
        if (i + 1 < n) v |= (uint32_t)p[i + 1] << 8;
        if (i + 2 < n) v |= p[i + 2];
        line[c++] = T[(v >> 18) & 63];
        line[c++] = T[(v >> 12) & 63];
        line[c++] = (i + 1 < n) ? T[(v >> 6) & 63] : '=';
        line[c++] = (i + 2 < n) ? T[v & 63] : '=';
        if (c >= 76) { line[c] = 0; puts(line); c = 0; }
    }
    if (c) { line[c] = 0; puts(line); }
}

/* Per-row RLE of RGB565: (uint16 count, uint16 pixel) LE pairs, runs never cross
   a row. A flat UI screen collapses to a few KB, so it streams over the 115200
   console in well under a second. Decoded host-side by expanding the pairs. */
static void ui_shot_maybe(int phase)
{
    static int last = -1;
    if (phase == last) return;                 /* one frame per state change */
    last = phase;
    if (!s_shot && !(s_shot = heap_caps_malloc(SHOT_SZ, MALLOC_CAP_SPIRAM))) return;
    if (!s_rle && !(s_rle = heap_caps_malloc(RLE_SZ, MALLOC_CAP_SPIRAM))) return;
    lv_image_dsc_t dsc;
    if (lv_snapshot_take_to_buf(scr, LV_COLOR_FORMAT_RGB565, &dsc, s_shot, SHOT_SZ) != LV_RESULT_OK) return;
    uint32_t w = dsc.header.w, h = dsc.header.h, stride = dsc.header.stride;
    size_t r = 0;
    for (uint32_t y = 0; y < h; y++) {
        const uint16_t *row = (const uint16_t *)(dsc.data + y * stride);
        for (uint32_t x = 0; x < w;) {
            uint16_t px = row[x]; uint32_t run = 1;
            while (x + run < w && row[x + run] == px && run < 65535) run++;
            s_rle[r++] = run & 0xff; s_rle[r++] = run >> 8;
            s_rle[r++] = px & 0xff;  s_rle[r++] = px >> 8;
            x += run;
        }
    }
    printf("[SHOT %u %u RLE16 %u]\n", (unsigned)w, (unsigned)h, (unsigned)r);
    b64_print(s_rle, r);
    printf("[/SHOT]\n");
}
#else
static void ui_shot_maybe(int phase) { (void)phase; }
#endif

/* CoreS3 screen is 320x240. Just the guided prompt now — mode switching and
   set selection moved to the menu and the automatic session flow, so the
   only control left is Abbrechen (abort -> menu). */
void ui_show_record(void)
{
    bsp_display_lock(0);
    scr = lv_obj_create(NULL);
    /* Constant dark charcoal base, light text — high contrast, no full-screen
       colour flashing. Only the status pill (l_phase) changes colour per phase. */
    lv_obj_set_style_bg_color(scr, lv_color_hex(0x12161c), 0);
    lv_obj_set_style_text_color(scr, lv_color_hex(0xe6eaef), 0);   /* inherited by labels */
    l_prompt = lv_label_create(scr);  lv_obj_set_width(l_prompt, 304); lv_obj_set_pos(l_prompt, 8, 12);
    lv_obj_set_style_text_font(l_prompt, &font_prompt_28, 0);
    lv_obj_set_style_text_color(l_prompt, lv_color_white(), 0);
    lv_label_set_long_mode(l_prompt, LV_LABEL_LONG_WRAP);
    lv_obj_set_style_text_align(l_prompt, LV_TEXT_ALIGN_CENTER, 0);
    l_counter = lv_label_create(scr); lv_obj_set_width(l_counter, 304); lv_obj_set_pos(l_counter, 8, 86);
    lv_obj_set_style_text_align(l_counter, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_set_style_text_color(l_counter, lv_color_hex(0x8a94a0), 0);  /* dim secondary text */
    bar = lv_bar_create(scr); lv_obj_set_size(bar, 200, 10); lv_obj_set_pos(bar, 60, 112);
    lv_bar_set_range(bar, -60, 0);
    l_phase = lv_label_create(scr);   lv_obj_set_pos(l_phase, 8, 132); lv_obj_set_size(l_phase, 304, LV_SIZE_CONTENT);
    lv_obj_set_style_text_align(l_phase, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_set_style_text_font(l_phase, &lv_font_montserrat_28, 0);
    lv_obj_set_style_text_color(l_phase, lv_color_white(), 0);
    lv_obj_set_style_radius(l_phase, 8, 0);
    lv_obj_set_style_pad_all(l_phase, 6, 0);
    lv_obj_set_style_bg_opa(l_phase, LV_OPA_COVER, 0);
    button(scr, "Abbrechen", on_abort, NULL, 60, 190, 200);
    lv_screen_load(scr);
    bsp_display_unlock();
}

void ui_record_refresh(const record_status_t *st)
{
    if (st->phase == REC_SESSION_DONE) { ui_show_success(st->speaker, st->saved_takes); return; }
    static const char *setname[] = {"words", "sentences", "negatives"};
    static const char *phase[] = {"paused", "listening...", "recording", "saved", "CLIPPED - redo", "no speech - redo", "flash full", "set complete", "get ready..."};
    char buf[64];
    if (!bsp_display_lock(50)) return;          /* skip a frame rather than block the recorder */
    /* "<set> <n>/<N> - read <r>/<takes>" — one progress line replaces the old
       three-label header row (set/seed, counter, speaker). */
    snprintf(buf, sizeof buf, "%s %d/%d - read %d/%d", setname[st->set], st->index + 1, st->count, st->take, st->takes);
    lv_label_set_text(l_counter, buf);
    lv_label_set_text(l_prompt, st->prompt);
    /* The background stays a constant dark charcoal (set once in ui_show_record);
       only the status *pill* changes colour, so the screen never flashes garish
       full-frame colours and every label stays readable. Green pill + "SPEAK NOW"
       while armed/recording, amber while getting ready, red on a bad take. */
    lv_color_t pill = lv_color_hex(0x2a3340);           /* neutral slate for idle states */
    const char *big = phase[st->phase];
    switch (st->phase) {
    case REC_LISTENING:
    case REC_CAPTURING: pill = lv_palette_main(LV_PALETTE_GREEN);  big = "SPEAK NOW"; break;
    case REC_GETREADY:  pill = lv_palette_main(LV_PALETTE_AMBER);  big = "get ready"; break;
    case REC_CLIPPED:
    case REC_TIMEOUT:   pill = lv_palette_main(LV_PALETTE_RED);     break;
    default: break;
    }
    lv_label_set_text(l_phase, big);
    lv_obj_set_style_bg_color(l_phase, pill, 0);
    ui_shot_maybe(st->phase);
    lv_bar_set_value(bar, (int)st->level_dbfs, LV_ANIM_OFF);
    lv_obj_set_style_bg_color(bar, st->phase == REC_CLIPPED ? lv_palette_main(LV_PALETTE_RED) : lv_palette_main(LV_PALETTE_GREEN), LV_PART_INDICATOR);
    bsp_display_unlock();
}
