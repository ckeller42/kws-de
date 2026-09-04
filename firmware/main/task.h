/**
 * @file task.h
 * @brief Create a pinned FreeRTOS task, and say so in the log when it fails.
 */
#pragma once
#include <stdbool.h>
#include <stdint.h>
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

/**
 * @brief xTaskCreatePinnedToCore, with the failure logged instead of ignored.
 *
 * Every task here was created with the return value dropped, which is fine
 * until internal RAM runs short — a task stack must come from internal SRAM
 * (CONFIG_SPIRAM_ALLOW_STACK_EXTERNAL_MEMORY is off) and cannot fall back to
 * PSRAM the way arena_alloc() does. When that happened, the task simply never
 * existed: no recogniser, or a record queue nobody drained, with nothing in the
 * log to say why. Total free is not the constraint — one contiguous block is —
 * so the failure line carries both.
 *
 * Deliberately not fatal: losing one mode's task is better than refusing to
 * boot, and the modes that still have their tasks keep working.
 *
 * @return true if the task was created.
 */
static inline bool task_spawn(const char *tag, TaskFunction_t fn, const char *name,
                              uint32_t stack, void *arg, UBaseType_t prio, BaseType_t core)
{
    if (xTaskCreatePinnedToCore(fn, name, stack, arg, prio, NULL, core) == pdPASS) return true;
    ESP_LOGE(tag, "%s task (%u B stack) not created: free internal %u, largest block %u",
             name, (unsigned)stack, (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
             (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL));
    return false;
}
