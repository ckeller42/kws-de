/**
 * @file storage.h
 * @brief Wear-levelled FAT storage partition, mounted at /rec.
 */
#pragma once
#include <stdint.h>
#include "esp_err.h"
#include "wear_levelling.h"

/** @brief Mount the wear-levelled FAT "storage" partition at /rec. Idempotent. */
esp_err_t   storage_mount(void);
/** @brief Unmount /rec (e.g. before exposing the partition as a USB MSC drive). */
esp_err_t   storage_unmount(void);
/** @brief Free space on /rec, in bytes (0 on error). */
uint64_t    storage_free_bytes(void);
/** @brief Wear-levelling handle backing /rec, needed to expose the partition as USB MSC. */
wl_handle_t storage_wl_handle(void);

#define STORAGE_MIN_FREE_BYTES (200 * 1024)  /**< Recording stops (REC_FULL) below this many free bytes. */
