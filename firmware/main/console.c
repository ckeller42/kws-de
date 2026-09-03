#include "console.h"
#include <stdio.h>
#include <string.h>
#include "driver/usb_serial_jtag.h"
#include "driver/usb_serial_jtag_vfs.h"
#include "esp_log.h"
#include "tusb_cdc_acm.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "gen/model_config.h"
#include "gen/wake_model_config.h"
#include "record.h"
#include "storage.h"
#include "ui/ui.h"

static const char *mode_name(ui_mode_t m)
{
    switch (m) {
    case UI_MODE_MENU:        return "menu";
    case UI_MODE_RECORD:      return "record";
    case UI_MODE_RECORD_WAKE: return "recordwake";
    case UI_MODE_USB:         return "usb";
    case UI_MODE_RECOGNISE:   return "recognise";
    case UI_MODE_WAKE:        return "wake";
    case UI_MODE_ASSIST:      return "assist";
    }
    return "?";
}

static const char *phase_name(record_phase_t p)
{
    static const char *names[] = {
        "idle", "listening", "capturing", "saved", "clipped",
        "timeout", "full", "done", "getready", "session_done",
    };
    return (unsigned)p < sizeof names / sizeof names[0] ? names[p] : "?";
}

/* "mode <name>" -> app_set_mode(); "status" -> current mode (+ recorder detail
 * in record mode). Anything else -> err. Every command ends with ok/err so a
 * host script can tell when it finished. */
static void handle_line(char *line)
{
    line[strcspn(line, "\r\n")] = 0;
    char *cmd = strtok(line, " ");
    if (!cmd) return;
    if (strcmp(cmd, "mode") == 0) {
        char *arg = strtok(NULL, " ");
        ui_mode_t m;
        if (!arg) { printf("err missing mode\n"); return; }
        if (strcmp(arg, "menu") == 0) m = UI_MODE_MENU;
        else if (strcmp(arg, "record") == 0) m = UI_MODE_RECORD;
        else if (strcmp(arg, "recordwake") == 0) m = UI_MODE_RECORD_WAKE;
        else if (strcmp(arg, "recognise") == 0) m = UI_MODE_RECOGNISE;
        else if (strcmp(arg, "wake") == 0) m = UI_MODE_WAKE;
        else if (strcmp(arg, "assist") == 0) m = UI_MODE_ASSIST;
        else if (strcmp(arg, "usb") == 0) m = UI_MODE_USB;
        else { printf("err unknown mode %s\n", arg); return; }
        app_set_mode(m);
        printf("ok\n");
    } else if (strcmp(cmd, "wakefire") == 0) {
        wake_inject_fire();                /* measurement hook: see wake.h */
        printf("ok\n");
    } else if (strcmp(cmd, "status") == 0) {
        ui_mode_t m = app_get_mode();
        printf("mode %s\n", mode_name(m));
        printf("models command=%s wake=%s\n", KWS_MODEL_ID, KWS_WAKE_MODEL_ID);
        /* Which volume the takes land on, and how much of a session still fits.
           MB for a card, KB for the 12 MB flash partition — the units are the
           quickest tell of which medium is live. */
        unsigned div = storage_is_sdcard() ? 1024 * 1024 : 1024;
        printf("storage %s %llu/%llu %s\n", storage_is_sdcard() ? "sd" : "flash",
               storage_free_bytes() / div, storage_total_bytes() / div,
               storage_is_sdcard() ? "MB" : "KB");
        if (m == UI_MODE_RECORD || m == UI_MODE_RECORD_WAKE) {
            record_status_t st;
            record_get_status(&st);
            printf("phase %s index %d count %d speaker %s\n",
                   phase_name(st.phase), st.index, st.count, st.speaker);
        }
        printf("ok\n");
    } else {
        printf("err unknown command\n");
    }
}

/* Pull whatever bytes are waiting on the active input. The CoreS3 has no
   UART bridge: its USB-C is the ESP32-S3's own USB-Serial-JTAG peripheral, so
   host bytes arrive there (IDF only *mirrors* stdout onto it as the secondary
   console; stdin stays on the unconnected UART0 - reading stdin never sees a
   host command). In USB-drive mode TinyUSB owns the PHY and the JTAG port is
   gone, so the CDC-ACM port is read instead. Both reads are bounded, so this
   task never parks across the mode switch and never drops a partial line. */
static size_t console_read(uint8_t *buf, size_t cap)
{
    if (app_get_mode() == UI_MODE_USB) {
        size_t n = 0;
        if (tinyusb_cdcacm_read(TINYUSB_CDC_ACM_0, buf, cap, &n) != ESP_OK) n = 0;
        if (n == 0) vTaskDelay(pdMS_TO_TICKS(20));
        return n;
    }
    int n = usb_serial_jtag_read_bytes(buf, cap, pdMS_TO_TICKS(50));
    return n > 0 ? (size_t)n : 0;
}

static void console_task(void *arg)
{
    (void)arg;
    static char line[64];
    size_t len = 0;
    uint8_t buf[32];
    for (;;) {
        size_t n = console_read(buf, sizeof buf);
        for (size_t i = 0; i < n; i++) {
            char c = (char)buf[i];
            if (c == '\n' || c == '\r') {
                if (len) { line[len] = 0; handle_line(line); len = 0; }
                continue;
            }
            if (len < sizeof line - 1) line[len++] = c;   /* overlong line: keep the tail */
        }
    }
}

void console_start(void)
{
    /* Interrupt-driven USB-Serial-JTAG driver: console_read() polls its ring
       with a bounded wait, and the VFS switch keeps stdout (logs, the replies
       printed by handle_line()) flowing through the same driver. In USB-drive
       mode usb_drive.c redirects stdout onto the CDC port instead. */
    usb_serial_jtag_driver_config_t cfg = USB_SERIAL_JTAG_DRIVER_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(usb_serial_jtag_driver_install(&cfg));
    usb_serial_jtag_vfs_use_driver();
    xTaskCreatePinnedToCore(console_task, "console", 4096, NULL, 1, NULL, 0);
}
