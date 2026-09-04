#include "field.h"
#include <string.h>

void field_reset(field_state_t *f)
{
    memset(f, 0, sizeof *f);
    f->thresh = FIELD_THRESH_DEFAULT;
}

bool field_set_thresh(field_state_t *f, float v)
{
    /* Written as a positive test, so a NaN out of strtof() is rejected too. */
    if (!(v >= FIELD_THRESH_MIN && v <= FIELD_THRESH_MAX)) return false;
    f->thresh = v;
    return true;
}

float field_gate_thresh(const field_state_t *f, bool assist, float prod)
{
    /* The loose gate exists to put near-misses and false alarms on the card, so
       it applies exactly where takes are written and nowhere else: wake mode,
       Assistent mode with capture off, and every other mode keep the production
       gate untouched. The comparison against `prod` is the "never tighter"
       guarantee in code rather than in prose. */
    if (!assist || !f->enabled || f->thresh > prod) return prod;
    return f->thresh;
}

void field_set_enabled(field_state_t *f, bool on)
{
    f->enabled = on;
    if (!on) f->armed = false;
}

void field_on_wake(field_state_t *f, uint32_t fire_pos)
{
    if (!f->enabled) return;
    /* A fire inside an open window extends it (assist_gate_on_wake) rather than
       starting a second one, so the take keeps the first fire's position: one
       interaction, one take, whose pre-roll holds the phrase actually spoken. */
    if (f->armed) return;
    f->armed = true;
    f->fire_pos = fire_pos;
}

bool field_take_span(const field_state_t *f, uint32_t window_ms,
                     uint32_t *start, uint32_t *len, bool *truncated)
{
    if (!f->enabled || !f->armed) return false;
    /* The window's real length, not ASSIST_WINDOW_MS: every fire inside an open
       window pushes the gate's deadline out, so the audio the recogniser
       listened to is as long as the gate actually stayed open. */
    uint32_t window = (uint32_t)((uint64_t)KWS_SAMPLE_RATE * window_ms / 1000u);
    /* ponytail: a fire less than the pre-roll into the ring is treated as a
       boot-time fire and the take is shortened to what exists. The same test is
       true once every 74 h, when the uint32 sample counter wraps; the cost is
       one truncated take, not a read of stale audio. Track the ring's own start
       position if that ever matters. */
    if (f->fire_pos < FIELD_PREROLL_SAMPLES) {
        *start = 0;
        *len = f->fire_pos + window;
    } else {
        *start = f->fire_pos - FIELD_PREROLL_SAMPLES;
        *len = FIELD_PREROLL_SAMPLES + window;
    }
    /* Cut at the end, never at the front: the pre-roll and the wake phrase are
       what make the take an interaction rather than a clip. */
    *truncated = *len > FIELD_MAX_TAKE_SAMPLES;
    if (*truncated) *len = FIELD_MAX_TAKE_SAMPLES;
    return true;
}

uint32_t field_clamp_len(uint32_t start, uint32_t len, uint32_t head)
{
    /* Signed difference, so this stays right across the uint32 wrap. */
    if ((int32_t)(start + len - head) <= 0) return len;
    uint32_t fit = head - start;
    /* fit > len can only mean head is BEHIND start (the same wrap, read the
       other way), i.e. the take's audio is no longer in the ring at all. */
    return fit < len ? fit : 0;
}

void field_disarm(field_state_t *f)
{
    f->armed = false;
}
