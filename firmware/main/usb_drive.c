#include "usb_drive.h"
#include "esp_log.h"
#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_check.h"
#include "storage.h"
#include "tinyusb.h"
#include "tusb_msc_storage.h"
#include "tusb_cdc_acm.h"
#include "tusb_console.h"

static const char *TAG = "usb";

esp_err_t usb_drive_enter(void)
{
    ESP_RETURN_ON_ERROR(storage_unmount(), TAG, "unmount");
    /* Composite MSC + CDC-ACM device. Default descriptors are enough: esp_tinyusb
       builds the composite config descriptor (and picks the IAD-compatible
       device class) from CONFIG_TINYUSB_{MSC,CDC}_ENABLED, so no custom
       descriptor set is needed here. The media exported is whichever one
       storage.c registered with esp_tinyusb — the microSD if a card is in the
       slot, the flash partition otherwise — and the label "KWSREC" is a FAT
       property forced on both at mount (storage.c). */
    const tinyusb_config_t cfg = {0};
    ESP_RETURN_ON_ERROR(tinyusb_driver_install(&cfg), TAG, "tinyusb install");

    const tinyusb_config_cdcacm_t acm_cfg = {
        .usb_dev = TINYUSB_USBDEV_0,
        .cdc_port = TINYUSB_CDC_ACM_0,
    };
    ESP_RETURN_ON_ERROR(tusb_cdc_acm_init(&acm_cfg), TAG, "cdc init");
    /* The console's normal port (the S3's USB-Serial-JTAG) rides the same USB
       PHY TinyUSB just took over (see usb_drive.h), so move stdout onto the
       new CDC-ACM port for the duration of USB mode. Input is not affected:
       console.c polls the CDC port directly while this mode is active. */
    ESP_RETURN_ON_ERROR(esp_tusb_init_console(TINYUSB_CDC_ACM_0), TAG, "cdc console");

    ESP_LOGI(TAG, "exposed %s as MSC, console moved to CDC-ACM", storage_root());
    return ESP_OK;
}

esp_err_t usb_drive_exit(void)
{
    ESP_RETURN_ON_ERROR(esp_tusb_deinit_console(TINYUSB_CDC_ACM_0), TAG, "cdc console teardown");
    ESP_RETURN_ON_ERROR(tusb_cdc_acm_deinit(TINYUSB_CDC_ACM_0), TAG, "cdc deinit");
    ESP_RETURN_ON_ERROR(tinyusb_driver_uninstall(), TAG, "tinyusb uninstall");
    ESP_RETURN_ON_ERROR(storage_mount(), TAG, "remount");
    /* ponytail: restart instead of re-initialising the USB PHY. Once TinyUSB
       has released the PHY the USB-Serial-JTAG console does not re-enumerate
       on the host (observed: no USB device at all until a re-plug), so the
       device would be unreachable for the next remote ingest. A restart lands
       on the menu in ~2 s with the JTAG console back - exactly what leaving
       USB mode means. Upgrade path: usb_phy re-attach for the JTAG peripheral. */
    ESP_LOGI(TAG, "leaving USB mode: restarting so the JTAG console re-enumerates");
    vTaskDelay(pdMS_TO_TICKS(200));
    esp_restart();
    return ESP_OK;
}
