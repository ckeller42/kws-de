/**
 * @file assist_gate.h
 * @brief Wake-gated duty cycle: when the command recogniser is allowed to run.
 *
 * Assist mode is the deployment shape the always-on recognise mode is only a
 * measurement of. The wake model runs continuously at ~1.7 ms per 30 ms of
 * audio; the command recogniser, which costs ~46 ms per 100 ms step, runs only
 * inside a short window opened by a wake fire. The saving is the whole point,
 * so the gate that decides it is pure C with no FreeRTOS, no logging and no
 * globals — the state machine is testable on the host, and the task around it
 * only has to call two functions.
 *
 * Time is caller-supplied milliseconds since boot; the gate never reads a
 * clock. `assist_gate_tick()` is edge-free and idempotent, so calling it more
 * or less often changes nothing but the resolution of the window's end.
 */
#pragma once
#include <stdbool.h>
#include <stdint.h>

/** How long the recogniser stays enabled after a wake fire, in ms. */
#define ASSIST_WINDOW_MS 2500

/** How soon after the window opens a command fire is dropped, in ms (#64).
 *  The wake fire lands ~0.14 s after "Hey Bus" finishes, and the window's
 *  first classification is a full KWS_N_FRAMES (~1 s) retrospective slice
 *  primed from ring audio that reaches back before the window opened — so it
 *  can score the tail of "...Bus" itself as a command ("aus" was the device's
 *  first word in 12 of 17 real takes). A fire this soon in is that artefact,
 *  not a spoken command: drop it before it reaches window_words/device_words
 *  or the confirmation tone. The window itself is unaffected — it still runs
 *  the full ASSIST_WINDOW_MS.
 *  450, not 300: on the CoreS3 the artefact's own fire (the recogniser's
 *  fixed ~150-400 ms scheduling-plus-inference latency on window entry, not
 *  phrase content) lands at 374-386 ms, so 300 ms let it through; real
 *  commands in the same replay fired no earlier than 531 ms, so 450 clears
 *  the artefact with margin on both sides. */
#define ASSIST_WAKE_TAIL_MS 450

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Should a command fire this soon after the window opened be dropped (#64)?
 * @param ms_since_open Milliseconds between the window opening and the fire.
 * @return true while still inside the wake-word tail (drop it); host-tested
 *         directly since recognise.cc's own loop needs real hardware.
 */
static inline bool assist_gate_in_wake_tail(int64_t ms_since_open)
{
    return ms_since_open < ASSIST_WAKE_TAIL_MS;
}

/** @brief Gate state. Zero-initialise; `open_until_ms` is only meaningful while `open`. */
typedef struct {
    bool open;               /**< Is the recogniser currently enabled? */
    uint32_t open_until_ms;  /**< Deadline for the current window (valid while `open`). */
    uint32_t windows;        /**< Windows opened since reset — one per wake fire. */
    uint32_t open_ms_total;  /**< Accumulated milliseconds the gate has been open. */
    uint32_t last_open_ms;   /**< Internal: when the current window started. */
} assist_gate_t;

/** @brief Reset to closed with zeroed accounting. */
void assist_gate_reset(assist_gate_t *g);

/**
 * @brief A wake fire arrived: open the window, or extend it if already open.
 * @return true if the recogniser should be running after this call.
 */
bool assist_gate_on_wake(assist_gate_t *g, uint32_t now_ms);

/**
 * @brief Advance time and close the window once it has expired.
 * @return true if the recogniser should be running.
 */
bool assist_gate_tick(assist_gate_t *g, uint32_t now_ms);

/**
 * @brief Fraction of wall time the gate has been open, in parts per thousand.
 *
 * Integer per-mille rather than a float: this is the duty-cycle number the
 * paper reports, and it is computed inside a firmware task that has no reason
 * to pull in floating point. @p elapsed_ms of 0 reads as 0.
 */
uint32_t assist_gate_duty_permille(const assist_gate_t *g, uint32_t now_ms, uint32_t elapsed_ms);

#ifdef __cplusplus
}
#endif
