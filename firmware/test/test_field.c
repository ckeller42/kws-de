/* Host test for the field-capture window arithmetic (REQ_FW_FIELD_CAPTURE). */
#include "field.h"
#include <assert.h>
#include <stdio.h>

int main(void)
{
    field_state_t f;
    uint32_t start = 0, len = 0;

    /* Off by default: a wake fire arms nothing and no span is offered. Capture
       is opt-in — this is the assertion that says so. */
    field_reset(&f);
    assert(!f.enabled);
    field_on_wake(&f, 100000);
    assert(!field_take_span(&f, &start, &len));

    /* Enabled: the take is pre-roll + window and ends exactly at the window's close. */
    field_set_enabled(&f, true);
    field_on_wake(&f, 100000);
    assert(field_take_span(&f, &start, &len));
    assert(start == 100000 - FIELD_PREROLL_SAMPLES);
    assert(len == FIELD_TAKE_SAMPLES);
    assert(start + len == 100000 + FIELD_WINDOW_SAMPLES);

    /* A second fire inside an open window keeps the FIRST fire's position:
       assist_gate extends the window, so one interaction stays one take, and
       its pre-roll still holds the wake phrase the user actually said. */
    field_on_wake(&f, 120000);
    assert(field_take_span(&f, &start, &len));
    assert(start == 100000 - FIELD_PREROLL_SAMPLES);

    /* Disarmed once the take has been handed over: no second copy of it. */
    field_disarm(&f);
    assert(!field_take_span(&f, &start, &len));

    /* A fire in the first second after boot shortens the take instead of
       reading in front of the start of the ring. */
    field_reset(&f);
    field_set_enabled(&f, true);
    field_on_wake(&f, 8000);
    assert(field_take_span(&f, &start, &len));
    assert(start == 0);
    assert(len == 8000 + FIELD_WINDOW_SAMPLES);

    /* Turning capture off drops a pending take: the toggle is the control. */
    field_reset(&f);
    field_set_enabled(&f, true);
    field_on_wake(&f, 100000);
    field_set_enabled(&f, false);
    assert(!field_take_span(&f, &start, &len));

    printf("test_field OK\n");
    return 0;
}
