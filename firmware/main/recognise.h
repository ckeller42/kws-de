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
} recognise_status_t;

/** @brief Create the recognise task (allocates the model arena). Starts inactive; call recognise_set_active(true) to run inference. */
void recognise_start(void);
/** @brief Enable/disable inference. When turning off, closes the detection log file. */
void recognise_set_active(bool on);
/** @brief Copy the current recognise status under mutex. */
void recognise_get_status(recognise_status_t *out);

#ifdef __cplusplus
}
#endif
