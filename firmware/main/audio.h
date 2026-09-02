#pragma once
/* audio.h — always-on capture into a ring buffer */
#include <stdint.h>
#include "gen/features_config.h"

#define AUDIO_RING_SAMPLES (KWS_SAMPLE_RATE * 10)          /* 10 s */

void     audio_start(void);                                  /* codec init + task, never returns error silently: abort() on failure */
uint32_t audio_write_pos(void);                              /* monotonically increasing sample counter */
/* Copy `n` samples ending at absolute position `end` (end - n must be inside the ring). */
void     audio_read(uint32_t end, int16_t *dst, uint32_t n);
