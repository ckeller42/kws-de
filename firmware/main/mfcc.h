/**
 * @file mfcc.h
 * @brief Sliding-window MFCC front end. No ESP-IDF dependencies (host-testable).
 */
#pragma once
#include <stdint.h>
#include "gen/features_config.h"

/** @brief Ring of per-frame log-mel rows, filled incrementally by mfcc_push_frame(). */
typedef struct {
    float logmel[KWS_N_FRAMES][KWS_N_MELS]; /**< Ring of per-frame log-mel rows. */
    int   head;                              /**< Index of the OLDEST row. */
    int   count;                             /**< Rows filled so far (<= KWS_N_FRAMES). */
} mfcc_state_t;

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Reset an mfcc_state_t to empty. */
void mfcc_init(mfcc_state_t *s);
/** @brief Push exactly KWS_WIN samples (one analysis window; caller advances by KWS_HOP between calls). */
void mfcc_push_frame(mfcc_state_t *s, const int16_t pcm[KWS_WIN]);
/**
 * @brief Finish the current window into MFCC features.
 *
 * Applies librosa's top_db clamp over the whole window and the DCT.
 * @param out [KWS_N_FRAMES][KWS_N_MFCC], oldest frame first. Rows not yet
 * filled are computed from zeros (matches Python zero-padding).
 */
void mfcc_finish(const mfcc_state_t *s, float out[KWS_N_FRAMES][KWS_N_MFCC]);
/** @brief One-shot MFCC over a whole 1 s clip (KWS_N_FRAMES*KWS_HOP + KWS_WIN - KWS_HOP samples = 16000). */
void mfcc_compute(const int16_t *pcm, float out[KWS_N_FRAMES][KWS_N_MFCC]);
/** @brief Quantise MFCC features to TFLite int8: q = round(x/scale) + zero_point, clamped. */
void mfcc_quantize(const float in[KWS_N_FRAMES][KWS_N_MFCC], int8_t *out, float scale, int zero_point);

#ifdef __cplusplus
}
#endif
