/**
 * @file vad.h
 * @brief Simple energy-based voice activity detector, pure C and host-testable.
 */
#pragma once
#include <stdint.h>

/** @brief VAD running state; zero-init or use vad_reset(). */
typedef struct {
    float noise;          /**< Exponentially tracked noise-floor RMS. */
    int   speech_frames;  /**< Consecutive frames above threshold. */
    int   silence_frames; /**< Consecutive frames at/below threshold. */
    int   in_speech;      /**< 1 while speech is considered active. */
} vad_t;

/** @brief Reset a vad_t to its initial (silent) state. */
void vad_reset(vad_t *v);
/**
 * @brief Feed one 20 ms frame (KWS_HOP samples) to the detector.
 *
 * Speech opens when rms > max(noise*4, 300) for 2 consecutive frames; the
 * noise floor tracks rms exponentially (alpha 0.05) only while not in
 * speech. Speech closes after VAD_TRAILING_FRAMES consecutive frames below
 * threshold.
 *
 * @return 1 while speech is active, 0 otherwise.
 */
int  vad_push(vad_t *v, const int16_t *frame, int n);

#define VAD_TRAILING_FRAMES 25  /**< Consecutive silent frames to close speech (25 * 20 ms = 500 ms). */
