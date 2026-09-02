#pragma once
/* recognise.h — on-device TFLM inference: recognise mode public API */
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char word[24];
    float conf;
    uint32_t infer_ms;
    uint32_t arena_used;
    uint32_t fired_count;
} recognise_status_t;

void recognise_start(void);              /* create task, paused */
void recognise_set_active(bool on);
void recognise_get_status(recognise_status_t *out);

#ifdef __cplusplus
}
#endif
