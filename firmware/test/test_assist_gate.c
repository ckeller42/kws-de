/* Host test for the wake-gated duty cycle (REQ_FW_ASSIST_GATE). */
#include "assist_gate.h"
#include <assert.h>
#include <stdio.h>

int main(void)
{
    assist_gate_t g;

    /* Closed until a wake fire; ticking alone never opens it. */
    assist_gate_reset(&g);
    assert(!assist_gate_tick(&g, 0));
    assert(!assist_gate_tick(&g, 100000));
    assert(g.windows == 0);

    /* A fire opens exactly one window, which lasts ASSIST_WINDOW_MS. */
    assert(assist_gate_on_wake(&g, 1000));
    assert(g.windows == 1);
    assert(assist_gate_tick(&g, 1000 + ASSIST_WINDOW_MS - 1));
    assert(!assist_gate_tick(&g, 1000 + ASSIST_WINDOW_MS));
    assert(g.windows == 1);

    /* A fire inside an open window extends it instead of opening a second:
       one interaction, not two. */
    assist_gate_reset(&g);
    assist_gate_on_wake(&g, 1000);
    assist_gate_tick(&g, 2000);
    assist_gate_on_wake(&g, 2000);              /* extends to 4500 */
    assert(g.windows == 1);
    assert(assist_gate_tick(&g, 4499));
    assert(!assist_gate_tick(&g, 4500));

    /* Duty: one 2.5 s window in 10 s of wall time is 250 per mille. */
    assist_gate_reset(&g);
    assist_gate_on_wake(&g, 0);
    assist_gate_tick(&g, ASSIST_WINDOW_MS);
    assert(!g.open);
    assert(assist_gate_duty_permille(&g, 10000, 10000) == 250);

    /* Read mid-window: the open part counts, so the figure never understates. */
    assist_gate_reset(&g);
    assist_gate_on_wake(&g, 0);
    assist_gate_tick(&g, 1000);
    assert(g.open);
    assert(assist_gate_duty_permille(&g, 1000, 2000) == 500);

    /* Never reports more than 100%, and a zero window is not a divide by zero. */
    assert(assist_gate_duty_permille(&g, 1000, 0) == 0);
    assist_gate_reset(&g);
    assist_gate_on_wake(&g, 0);
    assert(assist_gate_duty_permille(&g, 10000, 100) == 1000);

    /* Survives the 32-bit millisecond wrap (49.7 days of uptime). */
    assist_gate_reset(&g);
    uint32_t near_wrap = 0xFFFFFFFFu - 1000u;
    assist_gate_on_wake(&g, near_wrap);
    assert(assist_gate_tick(&g, near_wrap + ASSIST_WINDOW_MS - 1));   /* wrapped */
    assert(!assist_gate_tick(&g, near_wrap + ASSIST_WINDOW_MS));
    assert(g.open_ms_total == ASSIST_WINDOW_MS);

    printf("test_assist_gate OK\n");
    return 0;
}
