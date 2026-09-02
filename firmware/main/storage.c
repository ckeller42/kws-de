#include "storage.h"
#include "esp_log.h"
#include "esp_vfs_fat.h"
#include "esp_partition.h"
#include "wear_levelling.h"
#include "tusb_msc_storage.h"          /* esp_tinyusb: MSC and FAT share s_wl */

static const char *TAG = "storage";
static wl_handle_t s_wl = WL_INVALID_HANDLE;

esp_err_t storage_mount(void)
{
    static bool inited;
    if (!inited) {
        const esp_partition_t *part = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_DATA_FAT, "storage");
        ESP_ERROR_CHECK(wl_mount(part, &s_wl));
        tinyusb_msc_spiflash_config_t cfg = {.wl_handle = s_wl};
        ESP_ERROR_CHECK(tinyusb_msc_storage_init_spiflash(&cfg));
        inited = true;
    }
    esp_err_t err = tinyusb_msc_storage_mount("/rec");
    ESP_LOGI(TAG, "mount /rec: %s, free %llu KB", esp_err_to_name(err), storage_free_bytes() / 1024);
    return err;
}

esp_err_t storage_unmount(void)
{
    return tinyusb_msc_storage_unmount();
}

wl_handle_t storage_wl_handle(void) { return s_wl; }   /* Task 6 needs it for MSC */

uint64_t storage_free_bytes(void)
{
    uint64_t total = 0, free_b = 0;
    if (esp_vfs_fat_info("/rec", &total, &free_b) != ESP_OK) return 0;
    return free_b;
}
