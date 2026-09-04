/**
 * @file field.h
 * @brief Field capture in Assistent mode: which span of the audio ring one
 * real interaction occupies, and whether it may be written at all.
 *
 * Capture is opt-in and the toggle is persisted, so the decision "is there a
 * take, and where is it" is a tiny piece of state plus one subtraction. It
 * lives here, in pure C with no FreeRTOS, no clock and no globals, for the
 * same reason assist_gate.c does: it is the part worth testing on the host.
 *
 * Positions are absolute sample counts as returned by audio_write_pos().
 */
#pragma once
#include <stdbool.h>
#include <stdint.h>
#include "assist_gate.h"
#include "audio.h"
#include "gen/features_config.h"

/** Audio kept in front of the wake fire, so the take contains the wake phrase. */
#define FIELD_PREROLL_MS 1000
/** One take: the pre-roll plus the assist window the recogniser listened in. */
#define FIELD_TAKE_MS (FIELD_PREROLL_MS + ASSIST_WINDOW_MS)
#define FIELD_PREROLL_SAMPLES (KWS_SAMPLE_RATE * FIELD_PREROLL_MS / 1000)
#define FIELD_WINDOW_SAMPLES (KWS_SAMPLE_RATE * ASSIST_WINDOW_MS / 1000)
#define FIELD_TAKE_SAMPLES (FIELD_PREROLL_SAMPLES + FIELD_WINDOW_SAMPLES)
/** Worst case between the window closing and the record task starting the copy
 *  (a full recogniser step plus scheduling), rounded up to 0.2 s. */
#define FIELD_COPY_LATENCY_SAMPLES (KWS_SAMPLE_RATE / 5)

/* The ring must still hold the whole take when the copy starts. It is 10 s
   today, so this is a guard against a future shrink, not a constraint. */
_Static_assert(AUDIO_RING_SAMPLES >= FIELD_TAKE_SAMPLES + FIELD_COPY_LATENCY_SAMPLES,
               "audio ring must hold pre-roll + assist window + the copy latency");

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Capture state. Zero-initialise via field_reset(); `fire_pos` is only
 *  meaningful while `armed`. */
typedef struct {
    bool enabled;       /**< The user's toggle, restored from NVS at boot. */
    bool armed;         /**< A wake fire is waiting for its window to close. */
    uint32_t fire_pos;  /**< audio_write_pos() at the fire that armed us. */
    uint32_t taken;     /**< Field takes saved since boot. */
    uint32_t dropped;   /**< Field takes dropped because storage was low. */
} field_state_t;

/** @brief One take handed from the wake task to the record task. The device's
 *  own prediction travels with the audio; it is scored on the workstation and
 *  is never used as a label. */
typedef struct {
    uint32_t start;     /**< Absolute position of the take's first sample. */
    uint32_t len;       /**< Samples to copy out of the ring. */
    uint32_t fire_ms;   /**< ms since boot of the wake fire (names the file). */
    float wake_prob;    /**< Wake probability at that fire. */
    char intent[64];    /**< Ordered fired command words, space-joined; "" if none. */
    char words[96];     /**< The same fires as "<word>:<conf>", joined by '|'. */
} field_take_t;

/** @brief Reset to disabled with zeroed accounting. */
void field_reset(field_state_t *f);
/** @brief Turn capture on/off. Turning it off also drops a pending take. */
void field_set_enabled(field_state_t *f, bool on);
/** @brief A wake fire arrived at ring position @p fire_pos. No-op while disabled. */
void field_on_wake(field_state_t *f, uint32_t fire_pos);
/**
 * @brief The span to copy for the armed take.
 * @return false if capture is off or nothing is armed; otherwise @p start and
 *         @p len describe the take and `start + len` is the window's end.
 */
bool field_take_span(const field_state_t *f, uint32_t *start, uint32_t *len);
/** @brief Forget the armed take (it has been handed to the recorder). */
void field_disarm(field_state_t *f);

#ifdef __cplusplus
}
#endif
