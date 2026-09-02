/**
 * @file ui.h
 * @brief LVGL-based UI: record / USB-drive / recognise / wake screens.
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
/** @brief Switch the display to the record screen. */
void ui_show_record(void);
/** @brief Refresh the record screen from a status snapshot. Called from the record task; takes bsp_display_lock. */
void ui_record_refresh(const record_status_t *st);
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

/** @brief Top-level app mode, one screen/consumer task active at a time. */
typedef enum { UI_MODE_RECORD, UI_MODE_USB, UI_MODE_RECOGNISE, UI_MODE_WAKE } ui_mode_t;
/** @brief Switch app mode. Defined in main.c; the only place that suspends/resumes consumer tasks. */
void app_set_mode(ui_mode_t m);

#ifdef __cplusplus
}
#endif
