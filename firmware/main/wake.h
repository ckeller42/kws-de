/**
 * @file wake.h
 * @brief "Hey Bus" wake-word test mode: streaming microWakeWord inference.
 *
 * A deliberately isolated measurement mode. It runs ONLY the wake model
 * (`models/hey_bus.tflite`) over the microphone ring — the command recogniser
 * is switched off while it is active, so what the screen shows is the wake
 * model's behaviour and nothing else.
 *
 * The model is a *streaming* TFLite-Micro graph with resource variables: the
 * interpreter is created once and kept alive, and every Invoke() consumes the
 * next KWS_WAKE_FRAMES (3) 10 ms feature rows, i.e. one inference per 30 ms of
 * audio. Its variables are reset on every wake_set_active(true) so a previous
 * session cannot leak state into a new one.
 */
#pragma once
#include <stdbool.h>
#include <stdint.h>

/** @name Detector tunables
 * One utterance must produce exactly one fire: the probability has to clear
 * WAKE_THRESHOLD on WAKE_MIN_CONSECUTIVE inference steps in a row, and after a
 * fire the detector is deaf for WAKE_REFRACTORY_MS. Raise the threshold if the
 * device fires on speech that is not the wake phrase; lower it if real
 * utterances are missed. One step is 30 ms of audio.
 * @{ */
#define WAKE_THRESHOLD 0.85f       /**< Probability a step must reach to count (real voice peaks 0.83-0.99, noise <= 0.44 on the v4 model). */
#define WAKE_MIN_CONSECUTIVE 2     /**< Consecutive qualifying steps needed to fire. */
#define WAKE_REFRACTORY_MS 1500    /**< Deaf period after a fire, in ms. */
/** @} */

/** Milliseconds the UI keeps the screen green after a fire. */
#define WAKE_FLASH_MS 600

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Snapshot of wake-mode state, filled by wake_get_status(). */
typedef struct {
    float prob;              /**< Wake probability from the most recent inference step. */
    uint32_t fired_count;    /**< Total detections since wake_start(). */
    uint32_t infer_ms;       /**< Duration of the last inference pass, in ms. */
    uint32_t arena_used;     /**< TFLite Micro tensor arena bytes actually used. */
    uint32_t fired_at_ms;    /**< ms-since-boot of the last fire, 0 if none — drives the green flash. */
} wake_status_t;

/** @brief Create the wake task (allocates the model arena and the front-end). Starts inactive. */
void wake_start(void);
/** @brief Enable/disable wake inference. Turning on resets the model and front-end state; turning off closes the log. */
void wake_set_active(bool on);
/** @brief Copy the current wake status under mutex. */
void wake_get_status(wake_status_t *out);

#ifdef __cplusplus
}
#endif
