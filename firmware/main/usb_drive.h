/**
 * @file usb_drive.h
 * @brief Expose the /rec storage partition to a host PC as a USB mass-storage drive.
 */
#pragma once
#include "esp_err.h"

/** @brief Unmount /rec from the app and expose the partition as a USB MSC drive ("KWSREC"). */
esp_err_t usb_drive_enter(void);
/** @brief Stop the USB MSC device and remount /rec for the app. */
esp_err_t usb_drive_exit(void);
