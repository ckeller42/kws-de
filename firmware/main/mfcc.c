#include "mfcc.h"
#include <math.h>
#include <string.h>

#define KWS_PI 3.14159265358979323846f

/* 480 is 2^5*3*5 — not a power of two — so we do a plain DFT against a small 1-D
   twiddle table in fast internal RAM. The naive form indexed it with
   `(k*n) % KWS_WIN`; that integer modulo ran ~5.7 M times per one-shot inference
   and cost most of a ~1.2 s frame on the S3. Instead we walk the index
   incrementally — for fixed k it steps by k each sample, wrapping once — which
   is a couple of cheap ops and gives the identical index (host MFCC parity test
   still passes). ponytail: swap for esp-dsp mixed radix if it ever shows up hot. */
static float s_cos[KWS_WIN], s_sin[KWS_WIN];
static int s_twiddle_ready;

static void twiddle_init(void)
{
    if (s_twiddle_ready) return;
    for (int n = 0; n < KWS_WIN; n++) {
        s_cos[n] = cosf(2.f * KWS_PI * n / KWS_WIN);
        s_sin[n] = sinf(2.f * KWS_PI * n / KWS_WIN);
    }
    s_twiddle_ready = 1;
}

static void frame_logmel(const int16_t pcm[KWS_WIN], float logmel[KWS_N_MELS])
{
    float x[KWS_WIN], power[KWS_N_BINS];
    for (int n = 0; n < KWS_WIN; n++) x[n] = (pcm[n] / 32768.f) * KWS_WINDOW[n];
    for (int k = 0; k < KWS_N_BINS; k++) {
        float re = 0.f, im = 0.f;
        int idx = 0;                                   /* (k*n) % KWS_WIN, walked incrementally */
        for (int n = 0; n < KWS_WIN; n++) {
            re += x[n] * s_cos[idx];
            im -= x[n] * s_sin[idx];
            idx += k;
            if (idx >= KWS_WIN) idx -= KWS_WIN;         /* k < KWS_WIN so a single subtract suffices */
        }
        power[k] = re * re + im * im;
    }
    for (int m = 0; m < KWS_N_MELS; m++) {
        float acc = 0.f;
        for (int k = 0; k < KWS_N_BINS; k++) acc += KWS_MEL[m][k] * power[k];
        logmel[m] = 10.f * log10f(acc > KWS_AMIN ? acc : KWS_AMIN);
    }
}

void mfcc_init(mfcc_state_t *s)
{
    twiddle_init();
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
