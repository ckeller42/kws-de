#pragma once
/* usb_drive.h */
#include "esp_err.h"

esp_err_t usb_drive_enter(void);   /* unmount /rec from the app, expose partition as MSC "KWSREC" */
esp_err_t usb_drive_exit(void);    /* stop USB, remount /rec */
