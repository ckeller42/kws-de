/**
 * @file arena.h
 * @brief Where a TFLM tensor arena lives: internal SRAM if it fits, PSRAM if not.
 */
#pragma once
#include <stddef.h>
#include <stdint.h>
#include "esp_heap_caps.h"
#include "esp_log.h"

/**
 * @brief Allocate a TFLM tensor arena, preferring internal SRAM.
 *
 * TFLM touches the arena on every Invoke, so where it lives is the single
 * biggest lever on inference time: internal SRAM is a direct access, PSRAM
 * goes over the cached octal bus. PSRAM is the fallback, not the default — the
 * device must still boot when internal RAM is tight, just slower; the WARN line
 * is how that shows up in the log rather than as a mystery.
 *
 * @param tag   ESP log tag of the calling module.
 * @param what  Short arena name for the log line.
 * @param bytes Arena size.
 * @return The arena, or NULL if neither heap could satisfy it.
 */
static inline uint8_t *arena_alloc(const char *tag, const char *what, size_t bytes)
{
    size_t before = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    uint8_t *p = (uint8_t *)heap_caps_malloc(bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (p) {
        ESP_LOGI(tag, "%s arena %u B internal; free internal %u -> %u", what, (unsigned)bytes,
                 (unsigned)before, (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
    } else {
        p = (uint8_t *)heap_caps_malloc(bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        ESP_LOGW(tag, "%s arena %u B does not fit internal RAM (free %u) — using PSRAM, "
                      "inference will be several times slower",
                 what, (unsigned)bytes, (unsigned)before);
    }
    return p;
}
