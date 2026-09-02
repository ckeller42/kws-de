#pragma once
/* ui/ui.h */
#include "record.h"

void ui_init(void);                                       /* after bsp_display_start */
void ui_show_record(void);
void ui_record_refresh(const record_status_t *st);        /* called from record task, takes bsp_display_lock */
void ui_show_usb(void);                                   /* Task 6 */
void ui_show_recognise(void);                              /* Task 7 */

typedef enum { UI_MODE_RECORD, UI_MODE_USB, UI_MODE_RECOGNISE } ui_mode_t;
void app_set_mode(ui_mode_t m);                           /* in main.c; the only place that suspends/resumes consumers */
