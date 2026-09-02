/**
 * @file audio.h
 * @brief Always-on mono microphone capture into a ring buffer.
 */
#pragma once
#include <stdint.h>
#include "gen/features_config.h"

#define AUDIO_RING_SAMPLES (KWS_SAMPLE_RATE * 10)          /**< Ring buffer capacity, 10 s of audio. */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialise the codec and start the capture task.
 *
 * Allocates the ring buffer in PSRAM and starts continuous 16 kHz mono
 * capture (left channel of a stereo codec read) on a pinned task. Never
 * fails silently: aborts the process on codec/allocation failure.
 */
void     audio_start(void);
/** @brief Monotonically increasing count of samples written since audio_start(). */
uint32_t audio_write_pos(void);
/**
 * @brief Copy `n` samples ending at absolute position `end` out of the ring buffer.
 * @param end Absolute sample position (as from audio_write_pos()) one past the last sample to copy.
 * @param dst Destination buffer, at least `n` samples.
 * @param n   Number of samples to copy; `end - n` must still be inside the ring.
 */
void     audio_read(uint32_t end, int16_t *dst, uint32_t n);

#ifdef __cplusplus
}
#endif
