#pragma once
/* mfcc.h — sliding-window MFCC front end. No IDF dependencies. */
#include <stdint.h>
#include "gen/features_config.h"

typedef struct {
    float logmel[KWS_N_FRAMES][KWS_N_MELS]; /* ring of per-frame log-mel rows */
    int   head;                              /* index of the OLDEST row */
    int   count;                             /* rows filled (<= KWS_N_FRAMES) */
} mfcc_state_t;

#ifdef __cplusplus
extern "C" {
#endif

void mfcc_init(mfcc_state_t *s);
/* Push exactly KWS_WIN samples (one analysis window, caller advances by KWS_HOP). */
void mfcc_push_frame(mfcc_state_t *s, const int16_t pcm[KWS_WIN]);
/* Apply librosa's top_db clamp over the whole window and the DCT: out is [KWS_N_FRAMES][KWS_N_MFCC],
   oldest frame first. Rows not yet filled are computed from zeros (matches Python zero-padding). */
void mfcc_finish(const mfcc_state_t *s, float out[KWS_N_FRAMES][KWS_N_MFCC]);
/* One-shot: whole 1 s clip (KWS_N_FRAMES*KWS_HOP + KWS_WIN - KWS_HOP samples = 16000). */
void mfcc_compute(const int16_t *pcm, float out[KWS_N_FRAMES][KWS_N_MFCC]);
/* TFLite int8 quantisation: q = round(x/scale) + zero_point, clamped. */
void mfcc_quantize(const float in[KWS_N_FRAMES][KWS_N_MFCC], int8_t *out, float scale, int zero_point);

#ifdef __cplusplus
}
#endif
