/* Host test for the field-capture window arithmetic (REQ_FW_FIELD_CAPTURE). */
#include "field.h"
#include <assert.h>
#include <stdio.h>

#define MS(ms) ((uint32_t)((uint64_t)KWS_SAMPLE_RATE * (ms) / 1000u))

int main(void)
{
    field_state_t f;
    uint32_t start = 0, len = 0;
    bool cut = true;

    /* The take has to hold the phrase the fire is an ANSWER to, whole. Put that
       phrase where the measured timings put it — it ends FIELD_WAKE_LATENCY_MS
       before the fire and runs FIELD_PHRASE_MS before that — and require the
       span to contain it. This is the assertion the 1.5 s pre-roll failed: it
       budgeted 0.2-0.3 s of model latency against a real 1.06-1.20 s, so the
       phrase sat in front of the take's first sample and 8 of 11 real takes
       carried no wake word at all. Everything else in this file is written in
       the macros and would pass at any pre-roll. */
    field_reset(&f);
    field_set_enabled(&f, true);
    const uint32_t fire = 30 * KWS_SAMPLE_RATE;                  /* 30 s into a session */
    const uint32_t phrase_end = fire - MS(FIELD_WAKE_LATENCY_MS);
    const uint32_t phrase_start = phrase_end - MS(FIELD_PHRASE_MS);
    field_on_wake(&f, fire);
    assert(field_take_span(&f, ASSIST_WINDOW_MS, &start, &len, &cut));
    assert(start <= phrase_start);            /* the onset is inside the take... */
    assert(start + len > phrase_end);         /* ...and so is the rest of it */

    /* kws_de/qc.py's FIELD_PREROLL_MS carries the same number for the
       truncation test and its WAKE_MAX_S is derived from it for the wake-clip
       cut; both have to move with this, and nothing else checks that they did. */
    assert(FIELD_PREROLL_MS == 2500);
    /* The ring still has room for the wider take (the cap is on the total). */
    assert(FIELD_TAKE_SAMPLES <= FIELD_MAX_TAKE_SAMPLES);

    /* Off by default: a wake fire arms nothing and no span is offered. Capture
       is opt-in — this is the assertion that says so. */
    field_reset(&f);
    assert(!f.enabled);
    field_on_wake(&f, 100000);
    assert(!field_take_span(&f, ASSIST_WINDOW_MS, &start, &len, &cut));

    /* Enabled: the take is pre-roll + window and ends exactly at the window's close. */
    field_set_enabled(&f, true);
    field_on_wake(&f, 100000);
    assert(field_take_span(&f, ASSIST_WINDOW_MS, &start, &len, &cut));
    assert(start == 100000 - FIELD_PREROLL_SAMPLES);
    assert(len == FIELD_TAKE_SAMPLES);
    assert(start + len == 100000 + FIELD_WINDOW_SAMPLES);
    assert(!cut);

    /* A second fire inside an open window keeps the FIRST fire's position:
       assist_gate extends the window, so one interaction stays one take, and
       its pre-roll still holds the wake phrase the user actually said. */
    field_on_wake(&f, 120000);
    assert(field_take_span(&f, ASSIST_WINDOW_MS, &start, &len, &cut));
    assert(start == 100000 - FIELD_PREROLL_SAMPLES);

    /* ...and the extended window is captured WHOLE, not cut back to
       ASSIST_WINDOW_MS: the span still ends at the gate's real close, so no
       fire the recogniser reported can fall past the end of the audio. */
    assert(field_take_span(&f, 2 * ASSIST_WINDOW_MS, &start, &len, &cut));
    assert(start == 100000 - FIELD_PREROLL_SAMPLES);
    assert(start + len == 100000 + 2 * FIELD_WINDOW_SAMPLES);
    assert(!cut);

    /* A window longer than the ring can hold is cut at the END and says so, so
       the caller knows to drop the prediction it can no longer place. */
    assert(field_take_span(&f, 60000, &start, &len, &cut));
    assert(cut);
    assert(len == FIELD_MAX_TAKE_SAMPLES);
    assert(start == 100000 - FIELD_PREROLL_SAMPLES);   /* the pre-roll survives */

    /* Disarmed once the take has been handed over: no second copy of it. */
    field_disarm(&f);
    assert(!field_take_span(&f, ASSIST_WINDOW_MS, &start, &len, &cut));

    /* A fire less than a pre-roll after boot shortens the take instead of
       reading in front of the start of the ring. */
    field_reset(&f);
    field_set_enabled(&f, true);
    field_on_wake(&f, 8000);
    assert(field_take_span(&f, ASSIST_WINDOW_MS, &start, &len, &cut));
    assert(start == 0);
    assert(len == 8000 + FIELD_WINDOW_SAMPLES);
    assert(!cut);

    /* Turning capture off drops a pending take: the toggle is the control. */
    field_reset(&f);
    field_set_enabled(&f, true);
    field_on_wake(&f, 100000);
    field_set_enabled(&f, false);
    assert(!field_take_span(&f, ASSIST_WINDOW_MS, &start, &len, &cut));

    /* The span's end is derived from the arming fire's ring position and the
       window's ms length, and those two were sampled one inference apart, so it
       can sit a few dozen samples IN FRONT of the write head. Copying that would
       read the ring's previous lap into the take's tail; the clamp cuts it back
       to what has actually been written. */
    assert(field_clamp_len(1000, 500, 2000) == 500);   /* ends behind the head: untouched */
    assert(field_clamp_len(1000, 500, 1500) == 500);   /* ends exactly at the head */
    assert(field_clamp_len(1000, 500, 1468) == 468);   /* 32 samples past: cut back */
    assert(field_clamp_len(1000, 500, 900) == 0);      /* head behind start: nothing to copy */
    /* ...and it holds across the uint32 sample-counter wrap (every ~74 h). */
    assert(field_clamp_len(0xFFFFFF00u, 500, 0xFFFFFF00u + 400u) == 400);
    assert(field_clamp_len(0xFFFFFF00u, 500, 0xFFFFFF00u + 500u) == 500);

    printf("test_field OK\n");
    return 0;
}
