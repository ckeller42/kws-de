#include "mfcc.h"
#include <math.h>
#include <string.h>

/* 480 is 2^5*3*5 — not a power of two, but an exact kissfft mixed radix, so the
   per-frame spectrum is a real FFT (mfcc_fft.cc) rather than the O(KWS_WIN *
   KWS_N_BINS) DFT this used to run: ~115k multiply-adds per frame became ~4k,
   and the streaming recogniser step dropped from ~170 ms to single digits.
   Zero-padding to 512 would have been the easier FFT and the wrong one — it
   changes the bin spacing, hence the mel energies the models were trained on. */

static void frame_logmel(const int16_t pcm[KWS_WIN], float logmel[KWS_N_MELS])
{
    float x[KWS_WIN], power[KWS_N_BINS];
    for (int n = 0; n < KWS_WIN; n++) x[n] = (pcm[n] / 32768.f) * KWS_WINDOW[n];
    mfcc_fft_power(x, power);
    for (int m = 0; m < KWS_N_MELS; m++) {
        float acc = 0.f;
        for (int k = 0; k < KWS_N_BINS; k++) acc += KWS_MEL[m][k] * power[k];
        logmel[m] = 10.f * log10f(acc > KWS_AMIN ? acc : KWS_AMIN);
    }
}

void mfcc_init(mfcc_state_t *s)
{
    mfcc_fft_init();
    memset(s, 0, sizeof *s);
}

void mfcc_push_frame(mfcc_state_t *s, const int16_t pcm[KWS_WIN])
{
    int slot = (s->head + s->count) % KWS_N_FRAMES;
    if (s->count == KWS_N_FRAMES) { slot = s->head; s->head = (s->head + 1) % KWS_N_FRAMES; }
    else s->count++;
    frame_logmel(pcm, s->logmel[slot]);
}

void mfcc_finish(const mfcc_state_t *s, float out[KWS_N_FRAMES][KWS_N_MFCC])
{
    static float zero_row[KWS_N_MELS];
    static int zero_ready;
    if (!zero_ready) {               /* log-mel of an all-zero frame = 10*log10(AMIN) */
        for (int m = 0; m < KWS_N_MELS; m++) zero_row[m] = 10.f * log10f(KWS_AMIN);
        zero_ready = 1;
    }
    const float *rows[KWS_N_FRAMES];
    float peak = -1e30f;
    for (int t = 0; t < KWS_N_FRAMES; t++) {
        rows[t] = t < s->count ? s->logmel[(s->head + t) % KWS_N_FRAMES] : zero_row;
        for (int m = 0; m < KWS_N_MELS; m++) if (rows[t][m] > peak) peak = rows[t][m];
    }
    float floor_db = peak - KWS_TOP_DB;
    for (int t = 0; t < KWS_N_FRAMES; t++)
        for (int c = 0; c < KWS_N_MFCC; c++) {
            float acc = 0.f;
            for (int m = 0; m < KWS_N_MELS; m++) {
                float v = rows[t][m] < floor_db ? floor_db : rows[t][m];
                acc += KWS_DCT[c][m] * v;
            }
            out[t][c] = acc;
        }
}

void mfcc_compute(const int16_t *pcm, float out[KWS_N_FRAMES][KWS_N_MFCC])
{
    mfcc_state_t s;
    mfcc_init(&s);
    for (int t = 0; t < KWS_N_FRAMES; t++) mfcc_push_frame(&s, pcm + t * KWS_HOP);
    mfcc_finish(&s, out);
}

void mfcc_quantize(const float in[KWS_N_FRAMES][KWS_N_MFCC], int8_t *out, float scale, int zero_point)
{
    for (int t = 0; t < KWS_N_FRAMES; t++)
        for (int c = 0; c < KWS_N_MFCC; c++) {
            long q = lroundf(in[t][c] / scale) + zero_point;
            if (q < -128) q = -128;
            if (q > 127) q = 127;
            *out++ = (int8_t)q;
        }
}
