#include "assist_gate.h"
#include <string.h>

/* Deadlines are compared as (int32_t)(now - deadline) >= 0 rather than
   now >= deadline, so the gate keeps working across the 32-bit millisecond
   wrap (49.7 days). The device is expected to run for weeks in a van. */
static bool expired(uint32_t now_ms, uint32_t deadline_ms)
{
    return (int32_t)(now_ms - deadline_ms) >= 0;
}

void assist_gate_reset(assist_gate_t *g)
{
    memset(g, 0, sizeof *g);
}

/* Accumulate the time the window has been open up to `now`, then restart the
   accounting clock. Called on every transition and every close so the total is
   exact regardless of tick spacing. */
static void accrue(assist_gate_t *g, uint32_t now_ms)
{
    if (g->open) g->open_ms_total += now_ms - g->last_open_ms;
    g->last_open_ms = now_ms;
}

bool assist_gate_on_wake(assist_gate_t *g, uint32_t now_ms)
{
    accrue(g, now_ms);
    /* A fire inside an open window extends it rather than starting a second
       one: the user is still talking to the device, and a re-triggered window
       is one interaction, not two. */
    if (!g->open) {
        g->open = true;
        g->windows++;
    }
    g->open_until_ms = now_ms + ASSIST_WINDOW_MS;
    return true;
}

bool assist_gate_tick(assist_gate_t *g, uint32_t now_ms)
{
    if (g->open && expired(now_ms, g->open_until_ms)) {
        accrue(g, now_ms);
        g->open = false;
    } else {
        accrue(g, now_ms);
    }
    return g->open;
}

uint32_t assist_gate_duty_permille(const assist_gate_t *g, uint32_t now_ms, uint32_t elapsed_ms)
{
    if (!elapsed_ms) return 0;
    /* Include the currently-open window's unaccrued part so a reading taken
       mid-window is not an underestimate. */
    uint32_t open_ms = g->open_ms_total + (g->open ? now_ms - g->last_open_ms : 0);
    if (open_ms > elapsed_ms) open_ms = elapsed_ms;
    return (uint32_t)((uint64_t)open_ms * 1000u / elapsed_ms);
}
