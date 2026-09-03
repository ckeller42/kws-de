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
 * goes over the shared data cache. PSRAM is the fallback, not the default — the
 * device must still boot when internal RAM is tight, just slower; the WARN line
 * is how that shows up in the log rather than as a mystery.
 *
 * How much slower depends on the data cache: at 32 KB, moving the command arena
 * to PSRAM cost 12.4 ms of a 53 ms Invoke; at 64 KB it costs about 2 ms. Only
 * one arena fits internal SRAM, so callers are ordered deliberately (main.c).
 *
 * @param tag   ESP log tag of the calling module.
 * @param what  Short arena name for the log line.
 * @param bytes Arena size.
 * @return The arena, or NULL if neither heap could satisfy it.
 */
static inline uint8_t *arena_alloc(const char *tag, const char *what, size_t bytes)
{
    size_t before = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    /* 16-byte aligned: esp-nn's assembly kernels take their scratch from inside
       the arena and several of them require a 16-byte-aligned buffer, and TFLM
       aligns its allocations relative to the arena base. */
    uint8_t *p = (uint8_t *)heap_caps_aligned_alloc(16, bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (p) {
        ESP_LOGI(tag, "%s arena %u B internal; free internal %u -> %u (largest block %u)",
                 what, (unsigned)bytes, (unsigned)before,
                 (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
                 (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL));
    } else {
        p = (uint8_t *)heap_caps_malloc(bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        /* Total free is not the constraint — a contiguous, 16-byte-aligned block
           is — so the largest block is logged too; that is what a failure means. */
        ESP_LOGW(tag, "%s arena %u B does not fit internal RAM (free %u, largest block %u) — "
                      "using PSRAM, inference will be slower",
                 what, (unsigned)bytes, (unsigned)before,
                 (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL));
    }
    return p;
}
