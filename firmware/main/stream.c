#include "stream.h"
#include <string.h>

void stream_reset(stream_t *s)
{
    memset(s, 0, sizeof *s);
    s->run_label = -1;
    s->last_fired_label = -1;
}

int stream_push(stream_t *s, const float posterior[KWS_NUM_LABELS])
{
    memcpy(s->hist[s->hist_pos], posterior, sizeof(float) * KWS_NUM_LABELS);
    s->hist_pos = (s->hist_pos + 1) % KWS_SMOOTH_WIN;
    if (s->hist_len < KWS_SMOOTH_WIN) s->hist_len++;

    float best = -1.f; int idx = 0;
    for (int k = 0; k < KWS_NUM_LABELS; k++) {
        float m = 0.f;
        for (int h = 0; h < s->hist_len; h++) m += s->hist[h][k];
        m /= s->hist_len;
        if (m > best) { best = m; idx = k; }
    }
    int candidate = (best >= KWS_THRESHOLD && idx != KWS_SILENCE_INDEX) ? idx : -1;

    if (candidate == s->run_label) s->run_len++;
    else { s->run_label = candidate; s->run_len = 1; s->run_fired = 0; }

    int fired = -1;
    if (candidate >= 0 && s->run_len >= KWS_MIN_CONSECUTIVE && !s->run_fired) {
        int gap_ok = candidate != s->last_fired_label || s->gap_since_last_fired >= KWS_GAP_STEPS;
        if (gap_ok) {
            fired = candidate;
            s->run_fired = 1;
            s->last_fired_label = candidate;
            s->gap_since_last_fired = 0;
        }
    }
    if (s->last_fired_label >= 0 && candidate != s->last_fired_label) s->gap_since_last_fired++;
    return fired;
}
