/**
 * @file console.h
 * @brief Line-based remote-control console over the serial port (stdin/stdout).
 *
 * Lets a laptop drive the device during automated data-ingest sessions, e.g.
 * `echo 'mode usb' > /dev/cu.usbmodemNNN`. See firmware/README.md "Serial
 * commands" for the protocol.
 */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Create the console task (low priority; reads line commands from stdin). */
void console_start(void);

#ifdef __cplusplus
}
#endif
