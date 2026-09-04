/**
 * @file recognise.h
 * @brief On-device TFLite Micro keyword-spotting inference: recognise-mode public API.
 */
#pragma once
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Snapshot of recognise-mode state, filled by recognise_get_status(). */
typedef struct {
    char word[24];          /**< Last fired keyword (kept on screen between fires). */
    float conf;              /**< Confidence of `word`, or of the current top-1 label if nothing has fired since going active. */
    uint32_t infer_ms;       /**< Duration of the last inference pass, in ms. */
    uint32_t arena_used;     /**< TFLite Micro tensor arena bytes actually used. */
    uint32_t fired_count;    /**< Total number of keyword detections since recognise_start(). */
    char window_intent[64];  /**< Command words fired since the last recognise_listen_for(), in order, space-joined ("Licht Küche an"); empty if none. */
    char window_words[96];   /**< The same fires as "<word>:<conf>" entries joined by '|' ("Licht:0.93|an:0.88"). */
} recognise_status_t;

/** @brief Create the recognise task (allocates the model arena). Starts inactive; call recognise_set_active(true) to run inference. */
void recognise_start(void);
/** @brief Enable/disable inference. When turning off, closes the detection log file. */
void recognise_set_active(bool on);
/**
 * @brief Run inference for at most @p ms, then switch off unaided.
 *
 * Assist mode's window. The deadline is enforced by the recognise task itself
 * rather than by whoever opened the window: the recogniser is the expensive
 * task, and if it could only be stopped by another task it would be able to
 * starve its own off switch. That is not hypothetical — with both model tasks
 * on one core the wake task stopped being scheduled, the window never closed,
 * and the recogniser ran until the watchdog fired.
 */
void recognise_listen_for(uint32_t ms);
/** @brief Copy the current recognise status under mutex. */
void recognise_get_status(recognise_status_t *out);

#ifdef __cplusplus
}
#endif
