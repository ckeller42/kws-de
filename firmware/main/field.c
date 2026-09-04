#include "field.h"
#include <string.h>

void field_reset(field_state_t *f)
{
    memset(f, 0, sizeof *f);
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

bool field_take_span(const field_state_t *f, uint32_t *start, uint32_t *len)
{
    if (!f->enabled || !f->armed) return false;
    /* ponytail: a fire less than the pre-roll into the ring is treated as a
       boot-time fire and the take is shortened to what exists. The same test is
       true once every 74 h, when the uint32 sample counter wraps; the cost is
       one truncated take, not a read of stale audio. Track the ring's own start
       position if that ever matters. */
    if (f->fire_pos < FIELD_PREROLL_SAMPLES) {
        *start = 0;
        *len = f->fire_pos + FIELD_WINDOW_SAMPLES;
    } else {
        *start = f->fire_pos - FIELD_PREROLL_SAMPLES;
        *len = FIELD_TAKE_SAMPLES;
    }
    return true;
}

void field_disarm(field_state_t *f)
{
    f->armed = false;
}
