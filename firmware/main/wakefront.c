#include "wakefront.h"
#include <assert.h>
#include <string.h>
#include "tensorflow/lite/experimental/microfrontend/lib/frontend.h"
#include "tensorflow/lite/experimental/microfrontend/lib/frontend_util.h"

static struct FrontendState s_state;
static bool s_inited;

/* Ring of the most recent rows plus a count of how many are new since the last
   wakefront_take(). One row per 10 ms, so this never has to hold much. */
static int8_t s_rows[WAKEFRONT_MAX_ROWS][WAKEFRONT_FEATURES];
static int s_head;   /* next slot to write */
static int s_fresh;  /* rows produced since the last take(), capped at MAX_ROWS */

static void fill_config(struct FrontendConfig *c)
{
    c->window.size_ms = WAKEFRONT_WINDOW_MS;
    c->window.step_size_ms = WAKEFRONT_STEP_MS;
    c->filterbank.num_channels = WAKEFRONT_FEATURES;
    c->filterbank.lower_band_limit = WAKEFRONT_LOWER_BAND_LIMIT;
    c->filterbank.upper_band_limit = WAKEFRONT_UPPER_BAND_LIMIT;
    c->noise_reduction.smoothing_bits = WAKEFRONT_NR_SMOOTHING_BITS;
    c->noise_reduction.even_smoothing = WAKEFRONT_NR_EVEN_SMOOTHING;
    c->noise_reduction.odd_smoothing = WAKEFRONT_NR_ODD_SMOOTHING;
    c->noise_reduction.min_signal_remaining = WAKEFRONT_NR_MIN_SIGNAL_REMAINING;
    c->pcan_gain_control.enable_pcan = WAKEFRONT_PCAN_ENABLE;
    c->pcan_gain_control.strength = WAKEFRONT_PCAN_STRENGTH;
    c->pcan_gain_control.offset = WAKEFRONT_PCAN_OFFSET;
    c->pcan_gain_control.gain_bits = WAKEFRONT_PCAN_GAIN_BITS;
    c->log_scale.enable_log = WAKEFRONT_LOG_ENABLE;
    c->log_scale.scale_shift = WAKEFRONT_LOG_SCALE_SHIFT;
}

/* uint16 frontend value -> int8 model input. See wakefront.h for the derivation;
   this must stay integer-exact, a float rewrite drifts by an LSB near ties. */
static int8_t to_int8(uint16_t v)
{
    int32_t x = ((int32_t)v * WAKEFRONT_VALUE_SCALE + WAKEFRONT_VALUE_DIV / 2) / WAKEFRONT_VALUE_DIV;
    x += -128;
    if (x < -128) x = -128;
    if (x > 127) x = 127;
    return (int8_t)x;
}

void wakefront_init(void)
{
    if (s_inited) return;
    struct FrontendConfig cfg;
    fill_config(&cfg);
    int ok = FrontendPopulateState(&cfg, &s_state, WAKEFRONT_SAMPLE_RATE);
    assert(ok);
    (void)ok;
    s_inited = true;
    s_head = 0;
    s_fresh = 0;
}

void wakefront_reset(void)
{
    if (!s_inited) return;
    FrontendReset(&s_state);
    s_head = 0;
    s_fresh = 0;
}

int wakefront_push(const int16_t *pcm, int n_samples)
{
    int produced = 0;
    size_t left = (size_t)n_samples;
    while (left > 0) {
        size_t read = 0;
        struct FrontendOutput out = FrontendProcessSamples(&s_state, pcm, left, &read);
        if (read == 0) break;                 /* needs more data than we hold; keep it buffered */
        pcm += read;
        left -= read;
        if (out.size == 0) continue;
        for (int i = 0; i < WAKEFRONT_FEATURES; i++) s_rows[s_head][i] = to_int8(out.values[i]);
        s_head = (s_head + 1) % WAKEFRONT_MAX_ROWS;
        if (s_fresh < WAKEFRONT_MAX_ROWS) s_fresh++;
        produced++;
    }
    return produced;
}

bool wakefront_ready(int frames) { return s_fresh >= frames; }

void wakefront_take(int frames, int8_t *dst)
{
    /* s_head points one past the newest row; walk back `frames` and copy forward. */
    int start = (s_head + WAKEFRONT_MAX_ROWS - frames) % WAKEFRONT_MAX_ROWS;
    for (int f = 0; f < frames; f++)
        memcpy(dst + (size_t)f * WAKEFRONT_FEATURES, s_rows[(start + f) % WAKEFRONT_MAX_ROWS],
               WAKEFRONT_FEATURES);
    s_fresh = 0;
}
