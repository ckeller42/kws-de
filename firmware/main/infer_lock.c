#include "infer_lock.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"

/* Not EXT_RAM_BSS_ATTR, unlike the command model's arena: esp-nn's kernels hit
   the scratch on every row, and it is the one buffer of the three that both
   models touch. It stays in internal .bss. In a CONFIG_KWS_INFER_GENERATED=n
   build nothing references it and --gc-sections drops it whole. */
int8_t kws_infer_scratch[KWS_INFER_SCRATCH_BYTES] __attribute__((aligned(16)));

/* Created in app_main's call chain, before either task exists, so the tasks
   never race to create it. Taking a NULL handle trips FreeRTOS's own
   configASSERT rather than silently running unserialised. */
static SemaphoreHandle_t s_lock;

void kws_infer_lock_init(void)
{
    if (!s_lock) s_lock = xSemaphoreCreateMutex();
}

void kws_infer_lock(void)
{
    xSemaphoreTake(s_lock, portMAX_DELAY);
}

void kws_infer_unlock(void)
{
    xSemaphoreGive(s_lock);
}
