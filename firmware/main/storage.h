/**
 * @file storage.h
 * @brief The volume recordings are written to: the microSD card when one is
 *        present, otherwise the wear-levelled FAT "storage" flash partition.
 *
 * Both media are mounted through esp_tinyusb, so USB-drive mode hands the host
 * the very volume the recorder writes to. Callers never name a mount point:
 * they build paths under storage_root().
 */
#pragma once
#include <stdbool.h>
#include <stdint.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Mount the recording volume — microSD if a card is usable, else the flash partition. Idempotent. */
esp_err_t storage_mount(void);
/** @brief Unmount the recording volume (e.g. before exposing it as a USB MSC drive). */
esp_err_t storage_unmount(void);
/** @brief Mount point of the recording volume ("/sdcard" or "/rec"); a literal, valid before storage_mount(). */
const char *storage_root(void);
/** @brief true when the recording volume is the microSD card rather than the flash partition. */
bool storage_is_sdcard(void);
/** @brief Free space on the recording volume, in bytes (0 on error). */
uint64_t storage_free_bytes(void);
/** @brief Total size of the recording volume, in bytes (0 on error). */
uint64_t storage_total_bytes(void);
/**
 * @brief Re-check the recording volume and log if it has stopped answering
 *        (a microSD pulled while the device runs). Called on every mode entry.
 *
 * Nothing else has to happen for the device to stay safe: with the volume gone
 * storage_free_bytes() reads 0, so the recorder refuses each take as REC_FULL
 * rather than writing into a dead mount.
 */
void storage_recheck(void);

#ifdef __cplusplus
}
#endif

#define STORAGE_MIN_FREE_BYTES (200 * 1024)  /**< Recording stops (REC_FULL) below this many free bytes. */
#define STORAGE_LABEL "KWSREC"               /**< FAT volume label; what the host mounts in USB-drive mode. */
