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

/** Audio kept in front of the wake fire, so the take contains the wake phrase.
 *  1.5 s, not 1.0 s: the wake model fires ~0.2-0.3 s *after* the end of a
 *  ~0.7 s "Hey Bus", so 1.0 s back from the fire already lands inside the
 *  phrase and cuts its onset — only 3 of 11 real takes still transcribed with
 *  the wake phrase intact. The cap below is on the whole take, so widening the
 *  pre-roll costs the tail of a long chain of fires, nothing else. */
#define FIELD_PREROLL_MS 1500
#define FIELD_PREROLL_SAMPLES (KWS_SAMPLE_RATE * FIELD_PREROLL_MS / 1000)
#define FIELD_WINDOW_SAMPLES (KWS_SAMPLE_RATE * ASSIST_WINDOW_MS / 1000)
#define FIELD_TAKE_SAMPLES (FIELD_PREROLL_SAMPLES + FIELD_WINDOW_SAMPLES)
/** Worst case between the window closing and the record task starting the copy
 *  (a full recogniser step plus scheduling), rounded up to 0.2 s. */
#define FIELD_COPY_LATENCY_SAMPLES (KWS_SAMPLE_RATE / 5)
/** The most one take may hold: what is left of the ring once the copy latency is
 *  reserved. A wake fire inside an open window *extends* it
 *  (assist_gate_on_wake), so a window has no fixed length and a long enough
 *  chain of fires outgrows the ring; such a take is cut at the end and says so
 *  (field_take_span()'s `truncated`). */
#define FIELD_MAX_TAKE_SAMPLES (AUDIO_RING_SAMPLES - FIELD_COPY_LATENCY_SAMPLES)

/* The ring must still hold an un-extended take when the copy starts. It is 10 s
   today, so this is a guard against a future shrink, not a constraint. */
_Static_assert(FIELD_TAKE_SAMPLES <= FIELD_MAX_TAKE_SAMPLES,
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
} field_state_t;

/** @brief One take handed from the wake task to the record task. The device's
 *  own prediction travels with the audio; it is scored on the workstation and
 *  is never used as a label. */
typedef struct {
    uint32_t start;     /**< Absolute position of the take's first sample. */
    uint32_t len;       /**< Samples to copy out of the ring. */
    uint32_t fire_ms;   /**< ms since boot of the wake fire (names the file). */
    uint32_t window_ms; /**< How long the window was really open: the gate's close
                             time minus `fire_ms`. ASSIST_WINDOW_MS for a lone
                             fire, longer for every fire that extended it. */
    float wake_prob;    /**< Wake probability at that fire. */
    char intent[64];    /**< Ordered fired command words, space-joined; "" if none. */
    char words[96];     /**< The same fires as "<word>:<conf>", joined by '|'. */
} field_take_t;

/**
 * @brief Reset to disabled and unarmed.
 * @param f Capture state.
 */
void field_reset(field_state_t *f);
/**
 * @brief Turn capture on/off.
 * @param f  Capture state.
 * @param on The user's toggle.
 *
 * Turning it off drops an *armed* take — one whose window has not closed yet.
 * A take already handed to the recorder (window closed, write outstanding) is
 * still written; the toggle stops new capture, it does not reach into the
 * record queue.
 */
void field_set_enabled(field_state_t *f, bool on);
/**
 * @brief A wake fire arrived at ring position @p fire_pos. No-op while disabled.
 * @param f        Capture state.
 * @param fire_pos audio_write_pos() at the fire.
 */
void field_on_wake(field_state_t *f, uint32_t fire_pos);
/**
 * @brief The span to copy for the armed take.
 * @param f         Capture state.
 * @param window_ms How long the window was open (its close time minus the arming
 *        fire), so an extended window is captured whole instead of being cut to
 *        ASSIST_WINDOW_MS.
 * @param start Out: absolute position of the take's first sample.
 * @param len   Out: samples to copy out of the ring.
 * @param truncated Set true when the span did not fit FIELD_MAX_TAKE_SAMPLES and
 *        was cut at the end. The caller must then drop the take's device
 *        prediction: it cannot say which fires are still inside the audio.
 * @return false if capture is off or nothing is armed; otherwise @p start and
 *         @p len describe the take and, when not truncated, `start + len` is the
 *         window's close.
 */
bool field_take_span(const field_state_t *f, uint32_t window_ms,
                     uint32_t *start, uint32_t *len, bool *truncated);
/**
 * @brief Shorten a span so it cannot read past the ring's write head.
 * @param start Absolute position of the span's first sample.
 * @param len   Samples the span wants.
 * @param head  audio_write_pos() as read at copy time.
 * @return @p len when the span ends at or before @p head; otherwise what is
 *         actually written (0 if @p head is behind @p start).
 *
 * field_take_span() derives the end from the arming fire's ring position and
 * the window's ms length, but the two were sampled one inference apart, so the
 * computed end can sit a few dozen samples in FRONT of the head — which would
 * copy stale audio from the ring's previous lap into the take's tail. Comparing
 * as a signed difference keeps this correct across the uint32 wrap.
 */
uint32_t field_clamp_len(uint32_t start, uint32_t len, uint32_t head);
/**
 * @brief Forget the armed take (it has been handed to the recorder).
 * @param f Capture state.
 */
void field_disarm(field_state_t *f);

#ifdef __cplusplus
}
#endif
