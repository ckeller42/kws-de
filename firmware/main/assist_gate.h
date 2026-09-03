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

#ifdef __cplusplus
extern "C" {
#endif

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
