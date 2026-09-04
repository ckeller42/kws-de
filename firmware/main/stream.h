/**
 * @file stream.h
 * @brief Streaming keyword decision smoothing/debouncing. Port of kws_de.stream.KeywordStream.
 */
#pragma once
#include "gen/labels.h"
#include "gen/features_config.h"

/** @brief Running state of the smoothing/debounce logic; use stream_reset() before first use. */
typedef struct {
    float hist[KWS_SMOOTH_WIN][KWS_NUM_LABELS]; /**< Ring of the last posteriors, for moving-average smoothing. */
    int   hist_len, hist_pos;
    int   run_label;            /**< Currently smoothed-winning label, or -1 = none. */
    int   run_len;               /**< Consecutive pushes with this run_label. */
    int   run_fired;             /**< Whether the current run has already fired. */
    int   last_fired_label;     /**< Last label that fired, or -1 = none. */
    int   gap_since_last_fired; /**< Pushes since last_fired_label stopped being the candidate. */
} stream_t;

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Reset a stream_t to its initial (no history, nothing fired) state. */
void stream_reset(stream_t *s);
/**
 * @brief Push one frame's posterior probabilities and update the smoothing/debounce state.
 *
 * Averages the last KWS_SMOOTH_WIN posteriors, tracks consecutive runs of the
 * winning label, and fires when a run reaches KWS_MIN_CONSECUTIVE and either
 * it's a different label than last fired or enough steps (KWS_GAP_STEPS)
 * have passed since.
 *
 * @return The fired label index, or -1 if nothing fired this push.
 */
int  stream_push(stream_t *s, const float posterior[KWS_NUM_LABELS]);

/**
 * @brief Did this stream_push() return name a real command?
 *
 * True only for a fire on one of the command words. `-1` (nothing fired),
 * `_unknown_` and `_silence_` are all rejections: the model heard something,
 * but not an instruction the van can act on. This is what the confirmation
 * tone is keyed off, so it is a pure function here rather than an inline test
 * at the call site — the host test is the only place that distinction is
 * checked, the device cannot be made to speak on demand.
 */
static inline int stream_is_command(int fired)
{
    return fired >= 0 && fired != KWS_UNKNOWN_INDEX && fired != KWS_SILENCE_INDEX;
}

#ifdef __cplusplus
}
#endif
