#include "console.h"
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include "driver/uart.h"
#include "driver/uart_vfs.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "record.h"
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
        else if (strcmp(arg, "usb") == 0) m = UI_MODE_USB;
        else { printf("err unknown mode %s\n", arg); return; }
        app_set_mode(m);
        printf("ok\n");
    } else if (strcmp(cmd, "status") == 0) {
        ui_mode_t m = app_get_mode();
        printf("mode %s\n", mode_name(m));
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

static void console_task(void *arg)
{
    (void)arg;
    char line[64];
    for (;;) {
        if (fgets(line, sizeof line, stdin)) {
            handle_line(line);
        } else {
            /* No line ready (stdin is non-blocking - see console_start()) or a
               transient read error; clearerr() so a stuck error indicator
               can't wedge every future fgets() on this stream. */
            clearerr(stdin);
        }
        vTaskDelay(pdMS_TO_TICKS(20));
    }
}

void console_start(void)
{
    /* The console UART starts in the ROM's polling driver (busy-waits on
       every byte, no FreeRTOS yield). Installing the interrupt-driven UART
       driver and switching the VFS to it makes fgets() a normal read that
       sleeps the task instead of spinning the CPU. */
    uart_driver_install((uart_port_t)CONFIG_ESP_CONSOLE_UART_NUM, 256, 0, 0, NULL, 0);
    uart_vfs_dev_use_driver(CONFIG_ESP_CONSOLE_UART_NUM);
    setvbuf(stdin, NULL, _IONBF, 0);
    /* Non-blocking stdin, not blocking: usb_drive.c's usb_drive_enter()/exit()
       can freopen() stdin/stdout out from under this task (moving the console
       onto/off the CDC-ACM port for USB mode) from a different task (e.g. the
       UI task, when USB mode is entered by tapping the menu rather than over
       a serial 'mode usb' command). The UART VFS driver's blocking read
       parks the caller in a FreeRTOS queue wait with no timeout
       (uart_read_bytes(..., portMAX_DELAY) - see esp-idf's uart_vfs.c); if
       this task were blocked in that wait when the redirect happens, it
       would never return (no more bytes ever arrive on the old fd), and USB
       mode's CDC console would never get read. The CDC-ACM VFS read
       (esp_tinyusb's vfs_tinyusb.c) is already always non-blocking, so
       matching that here means this loop's fgets() call can never block past
       one vTaskDelay(20ms) tick on either port, on either side of the
       switch. */
    fcntl(fileno(stdin), F_SETFL, O_NONBLOCK);
    xTaskCreatePinnedToCore(console_task, "console", 4096, NULL, 1, NULL, 0);
}
