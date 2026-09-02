/**
 * @file vad.h
 * @brief Simple energy-based voice activity detector, pure C and host-testable.
 */
#pragma once
#include <stdint.h>

/** @brief VAD running state; zero-init or use vad_reset(). */
typedef struct {
    float noise;           /**< Exponentially tracked noise-floor RMS. */
    int   speech_frames;   /**< Consecutive frames above threshold. */
    int   silence_frames;  /**< Consecutive frames at/below threshold. */
    int   in_speech;       /**< 1 while speech is considered active. */
    int   trailing_frames; /**< Consecutive silent frames required to close speech; set by vad_reset(). */
    int   speech_total;    /**< Total frames above threshold since the last vad_reset() (false-start filter). */
} vad_t;

/**
 * @brief Reset a vad_t to its initial (silent) state.
 * @param trailing_frames Consecutive silent frames required to close speech
 *     once open (see vad_push()); pass VAD_TRAILING_FRAMES for the default
 *     500 ms, or a set-specific value from prompt_hangover_ms() / 20.
 */
void vad_reset(vad_t *v, int trailing_frames);
/**
 * @brief Feed one 20 ms frame (KWS_HOP samples) to the detector.
 *
 * Speech opens when rms > max(noise*4, 300) for 2 consecutive frames; the
 * noise floor tracks rms exponentially (alpha 0.05) only while not in
 * speech. Speech closes after v->trailing_frames consecutive frames below
 * threshold. Every frame above threshold, open or not, adds to
 * v->speech_total for the caller's false-start check.
 *
 * @return 1 while speech is active, 0 otherwise.
 */
int  vad_push(vad_t *v, const int16_t *frame, int n);

#define VAD_TRAILING_FRAMES 25  /**< Default trailing hangover: 25 * 20 ms = 500 ms (words). */
