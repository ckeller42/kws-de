#include "usb_drive.h"
#include "esp_log.h"
#include "esp_check.h"
#include "storage.h"
#include "tinyusb.h"
#include "tusb_msc_storage.h"

static const char *TAG = "usb";

esp_err_t usb_drive_enter(void)
{
    ESP_RETURN_ON_ERROR(storage_unmount(), TAG, "unmount");
    const tinyusb_config_t cfg = {0};          /* default descriptors; label "KWSREC" is a FAT property, set at first mount */
    ESP_RETURN_ON_ERROR(tinyusb_driver_install(&cfg), TAG, "tinyusb install");
    ESP_LOGI(TAG, "exposed /rec as MSC");
    return ESP_OK;
}

esp_err_t usb_drive_exit(void)
{
    ESP_RETURN_ON_ERROR(tinyusb_driver_uninstall(), TAG, "tinyusb uninstall");
    return storage_mount();
}
