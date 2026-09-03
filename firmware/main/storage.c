#include "storage.h"
#include <stdio.h>
#include <string.h>
#include "bsp/esp-bsp.h"
#include "diskio_impl.h"
#include "diskio_sdmmc.h"
#include "diskio_wl.h"
#include "esp_check.h"
#include "esp_log.h"
#include "esp_partition.h"
#include "esp_vfs_fat.h"
#include "ff.h"
#include "tusb_msc_storage.h"          /* esp_tinyusb: MSC and FAT share one media handle */
#include "wear_levelling.h"

static const char *TAG = "storage";
static sdmmc_card_t *s_card;                       /* non-NULL once a microSD is the recording volume */
static wl_handle_t s_wl = WL_INVALID_HANDLE;

/* Four files can be open at once in the worst case: the take being written, its
   session.csv, wake.log and recognise.log. esp_tinyusb defaults to 2. */
#define STORAGE_MAX_FILES 5

/* Drop the VFS mount bsp_sdcard_mount() made, leaving the card powered and
   initialised, so the same card can be re-mounted through esp_tinyusb. Only the
   media esp_tinyusb holds can be exported over USB, and exporting anything but
   the volume the recorder writes to would be pointless — hence the round trip
   rather than keeping the BSP's own mount. */
static void sd_vfs_detach(void)
{
    BYTE pdrv = ff_diskio_get_pdrv_card(s_card);
    char drv[3] = {(char)('0' + pdrv), ':', 0};
    f_mount(NULL, drv, 0);
    ff_diskio_unregister(pdrv);
    esp_vfs_fat_unregister_path(BSP_SD_MOUNT_POINT);
}

/* Bring up the microSD, or fail: no card, a card the SPI probe cannot talk to,
   or one that carries no filesystem and cannot be formatted into one. A card
   that probes but has no readable filesystem is reformatted FAT on the spot —
   see CONFIG_BSP_SD_FORMAT_ON_MOUNT_FAIL in sdkconfig.defaults.
   ponytail: a card that can never be formatted (write-protected, dying, fake)
   pays that format attempt on every boot — ~25 s for a 64 GB card — before the
   fallback wins. An empty slot costs nothing and a good card pays it once, so
   this is a broken-card symptom rather than a standing cost. Avoiding it would
   mean probing raw sectors for writability first, which needs the card handle
   the BSP only hands out through a full mount. Upgrade path if dead cards turn
   out to be common: own the sdspi bring-up here instead of going via the BSP. */
static esp_err_t sd_init(void)
{
    ESP_RETURN_ON_ERROR(bsp_sdcard_mount(), TAG, "microSD mount");
    s_card = bsp_sdcard;
    sdmmc_card_print_info(stdout, s_card);
    sd_vfs_detach();
    const tinyusb_msc_sdmmc_config_t cfg = {
        .card = s_card,
        .mount_config = {.max_files = STORAGE_MAX_FILES},
    };
    return tinyusb_msc_storage_init_sdmmc(&cfg);
}

static esp_err_t flash_init(void)
{
    const esp_partition_t *part = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_DATA_FAT, "storage");
    ESP_RETURN_ON_FALSE(part, ESP_ERR_NOT_FOUND, TAG, "no storage partition");
    ESP_RETURN_ON_ERROR(wl_mount(part, &s_wl), TAG, "wl_mount");
    const tinyusb_msc_spiflash_config_t cfg = {
        .wl_handle = s_wl,
        .mount_config = {.max_files = STORAGE_MAX_FILES},
    };
    return tinyusb_msc_storage_init_spiflash(&cfg);
}

/* Write a file and read it back. A dying or counterfeit card can report every
   write as successful and quietly keep the old sectors — one such card is what
   this probe was written for — and mounting it would swallow a whole session in
   silence. A 3-byte round trip catches that. The flash partition is not probed:
   it is the fallback, and each probe would cost it an erase cycle. */
static bool sd_write_probe(void)
{
    char path[40];
    char back[4] = {0};
    snprintf(path, sizeof path, "%s/.probe", storage_root());
    FILE *f = fopen(path, "wb");
    if (!f) return false;
    bool ok = fwrite("kws", 1, 3, f) == 3;
    ok = (fclose(f) == 0) && ok;
    f = fopen(path, "rb");
    if (!f) return false;
    ok = ok && fread(back, 1, 3, f) == 3 && memcmp(back, "kws", 3) == 0;
    fclose(f);
    remove(path);
    return ok;
}

const char *storage_root(void) { return s_card ? BSP_SD_MOUNT_POINT : "/rec"; }

bool storage_is_sdcard(void) { return s_card != NULL; }

/* Force the FAT volume label. It is what the host mounts in USB-drive mode and
   what the ingest script probes for, so it is set on both media rather than
   only when blank: a pre-formatted card arrives carrying the vendor's label.
   The drive number has to be asked for, not assumed to be 0 — a failed microSD
   attempt before the flash mount can leave the flash volume on drive 1, and a
   label written to the wrong drive is silently dropped. */
static void set_label(void)
{
    BYTE pdrv = s_card ? ff_diskio_get_pdrv_card(s_card) : ff_diskio_get_pdrv_wl(s_wl);
    char drv[16];
    char label[12] = {0};
    snprintf(drv, sizeof drv, "%u:", pdrv);
    if (f_getlabel(drv, label, NULL) != FR_OK) { ESP_LOGW(TAG, "no label on drive %u:", pdrv); return; }
    if (strcmp(label, STORAGE_LABEL) == 0) return;
    snprintf(drv, sizeof drv, "%u:" STORAGE_LABEL, pdrv);
    FRESULT res = f_setlabel(drv);
    ESP_LOGI(TAG, "label \"%s\" -> \"%s\" on drive %u: (%d)", label, STORAGE_LABEL, pdrv, res);
}

static esp_err_t mount_root(void)
{
    esp_err_t err = tinyusb_msc_storage_mount(storage_root());
    if (err == ESP_OK) set_label();
    ESP_LOGI(TAG, "mount %s (%s): %s, %llu of %llu KB free", storage_root(),
             storage_is_sdcard() ? "microSD" : "flash", esp_err_to_name(err),
             storage_free_bytes() / 1024, storage_total_bytes() / 1024);
    return err;
}

esp_err_t storage_mount(void)
{
    static bool inited;
    if (inited) return mount_root();       /* re-mount after USB mode: media already chosen */
    inited = true;
    if (sd_init() == ESP_OK) {
        if (mount_root() == ESP_OK && sd_write_probe()) return ESP_OK;
        ESP_LOGE(TAG, "microSD mounted but does not keep what is written to it — ignoring the card");
        tinyusb_msc_storage_unmount();
        tinyusb_msc_storage_deinit();
        s_card = NULL;
    }
    ESP_LOGW(TAG, "recording to the flash partition (one guided session; insert a microSD for more)");
    ESP_ERROR_CHECK(flash_init());
    return mount_root();
}

esp_err_t storage_unmount(void)
{
    return tinyusb_msc_storage_unmount();
}

uint64_t storage_free_bytes(void)
{
    uint64_t total = 0, free_b = 0;
    if (esp_vfs_fat_info(storage_root(), &total, &free_b) != ESP_OK) return 0;
    return free_b;
}

uint64_t storage_total_bytes(void)
{
    uint64_t total = 0, free_b = 0;
    if (esp_vfs_fat_info(storage_root(), &total, &free_b) != ESP_OK) return 0;
    return total;
}

void storage_recheck(void)
{
    uint64_t total = 0, free_b = 0;
    if (esp_vfs_fat_info(storage_root(), &total, &free_b) == ESP_OK) return;
    /* ponytail: report and let the recorder refuse takes (storage_free_bytes()
       reads 0 -> REC_FULL). Switching back to the flash partition while the
       device runs would have to reopen the wake/recognise log handles and swap
       the MSC media under a live session; a reboot does it correctly and is
       what pulling a card asks for anyway. Upgrade path if cards get swapped
       mid-session: re-run the media choice in storage_mount(). */
    ESP_LOGE(TAG, "%s stopped responding (microSD removed?) — takes are refused until a restart", storage_root());
}
