#pragma once
/* storage.h */
#include <stdint.h>
#include "esp_err.h"
#include "wear_levelling.h"

esp_err_t   storage_mount(void);      /* wear-levelled FAT "storage" partition at /rec */
esp_err_t   storage_unmount(void);
uint64_t    storage_free_bytes(void);
wl_handle_t storage_wl_handle(void);  /* Task 6 needs it for MSC */

#define STORAGE_MIN_FREE_BYTES (200 * 1024)
