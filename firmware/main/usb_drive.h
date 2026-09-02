/**
 * @file usb_drive.h
 * @brief Expose the /rec storage partition to a host PC as a USB mass-storage
 *        drive, alongside a CDC-ACM serial port that keeps the console
 *        (console.c's `mode`/`status` commands) reachable while it is mounted.
 */
#pragma once
#include "esp_err.h"

/**
 * @brief Unmount /rec, then bring up a composite USB device: MSC ("KWSREC")
 *        plus one CDC-ACM port, with stdio (the console task's stdin/stdout)
 *        redirected onto the CDC port for the duration of USB mode.
 *
 * The device's normal console port rides the same USB PHY that TinyUSB takes
 * over here, so it disappears from the host for as long as USB mode is
 * active; the CDC-ACM port is what replaces it (a different serial device
 * node on the host - see firmware/README.md "USB mode").
 */
esp_err_t usb_drive_enter(void);
/** @brief Stop the USB MSC+CDC device, restore stdio to the normal console port, and remount /rec for the app. */
esp_err_t usb_drive_exit(void);
