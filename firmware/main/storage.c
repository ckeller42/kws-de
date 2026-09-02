#include "storage.h"
#include "esp_log.h"
#include "esp_vfs_fat.h"
#include "wear_levelling.h"

static const char *TAG = "storage";
static wl_handle_t s_wl = WL_INVALID_HANDLE;

esp_err_t storage_mount(void)
{
    esp_vfs_fat_mount_config_t cfg = {.max_files = 4, .format_if_mount_failed = true, .allocation_unit_size = 4096};
    esp_err_t err = esp_vfs_fat_spiflash_mount_rw_wl("/rec", "storage", &cfg, &s_wl);
    ESP_LOGI(TAG, "mount /rec: %s, free %llu KB", esp_err_to_name(err), storage_free_bytes() / 1024);
    return err;
}

esp_err_t storage_unmount(void)
{
    esp_err_t err = esp_vfs_fat_spiflash_unmount_rw_wl("/rec", s_wl);
    s_wl = WL_INVALID_HANDLE;
    return err;
}

wl_handle_t storage_wl_handle(void) { return s_wl; }   /* Task 6 needs it for MSC */

uint64_t storage_free_bytes(void)
{
    uint64_t total = 0, free_b = 0;
    if (esp_vfs_fat_info("/rec", &total, &free_b) != ESP_OK) return 0;
    return free_b;
}
