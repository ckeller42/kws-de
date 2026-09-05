/**
 * @file ui.h
 * @brief LVGL-based UI: selection menu / record / USB-drive / recognise / wake screens.
 */
#pragma once
#include "record.h"
#include "recognise.h"
#include "wake.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Build the UI. Call after bsp_display_start(). */
void ui_init(void);
/** @brief Switch the display to the mode-selection menu (the app's home screen). */
void ui_show_menu(void);
/** @brief Switch the display to the record screen. */
void ui_show_record(void);
/** @brief Refresh the record screen from a status snapshot. Called from the record task; takes bsp_display_lock.
 * When @p st reports REC_SESSION_DONE this shows the success screen instead (see ui_show_success()). */
void ui_record_refresh(const record_status_t *st);
/** @brief Switch the display to the session-complete success screen. Called by ui_record_refresh(). */
void ui_show_success(const char *speaker, int saved_takes);
/** @brief Switch the display to the USB-drive screen. */
void ui_show_usb(void);
/** @brief Switch the display to the recognise screen. */
void ui_show_recognise(void);
/** @brief Refresh the recognise screen from a status snapshot. Called from the recognise task; takes bsp_display_lock. */
void ui_recognise_refresh(const recognise_status_t *st);
/** @brief Switch the display to the wake-word test screen. */
void ui_show_wake(void);
/** @brief Refresh the wake screen from a status snapshot. Called from the wake task; takes bsp_display_lock. */
void ui_wake_refresh(const wake_status_t *st);
/** @brief Switch the display to the assistant (wake-gated) screen. */
void ui_show_assist(void);
/** @brief Refresh the assistant screen: wake probability while idle, the recognised
 * word while the gate is open. Called from the wake and recognise tasks; takes bsp_display_lock. */
void ui_assist_refresh(const wake_status_t *wst, const recognise_status_t *rst, bool listening);
/** @brief Show the assist screen's result card for ~3 s: a valid intent in
 * green (@p text, the formatted "Licht Küche → an"), or grey "nicht verstanden"
 * plus @p heard_words (the raw fired words) when @p valid is false. Called
 * once per window, from the wake task at the window's close; takes
 * bsp_display_lock. */
void ui_assist_show_result(bool valid, const char *text, const char *heard_words);

/** @brief Top-level app mode, one screen/consumer task active at a time.
 * UI_MODE_RECORD_WAKE is the guided-recorder's "Hey Bus"-only session (PROMPT_WAKE);
 * UI_MODE_WAKE is the unrelated wake-*model* test screen ("Hey Bus" demo button).
 * UI_MODE_ASSIST is the deployment shape both of those measure: the wake model
 * runs continuously and a fire opens a short window for the recogniser. */
typedef enum { UI_MODE_RECORD, UI_MODE_RECORD_WAKE, UI_MODE_USB, UI_MODE_RECOGNISE, UI_MODE_WAKE, UI_MODE_ASSIST, UI_MODE_MENU } ui_mode_t;
/** @brief Switch app mode. Defined in main.c; the only place that suspends/resumes consumer tasks. */
void app_set_mode(ui_mode_t m);
/** @brief Current app mode. Defined in main.c; used by the serial console's `status` command. */
ui_mode_t app_get_mode(void);

#ifdef __cplusplus
}
#endif
