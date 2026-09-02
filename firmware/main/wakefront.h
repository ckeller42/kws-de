/**
 * @file wakefront.h
 * @brief microWakeWord's 40-channel audio front-end: PCM in, int8 model rows out.
 *
 * Thin wrapper around the vendored TFLite-Micro microfrontend
 * (`microfrontend/tensorflow/lite/experimental/microfrontend/lib`). It holds
 * the exact `FrontendConfig` microWakeWord trains with and applies the exact
 * integer requantisation microWakeWord's runtime applies, so the rows this
 * produces are bit-identical to the ones the model saw in training.
 *
 * Every constant below is part of the trained model's contract — changing one
 * silently invalidates `models/hey_bus.tflite`. They are cross-checked against
 * `pymicro_features` by the host parity test (`firmware/test/test_wakefront.c`)
 * over a golden vector in `gen/wake_test_vectors.h`.
 */
#pragma once
#include <stdbool.h>
#include <stdint.h>

/** @name microWakeWord front-end parameters
 * Source of truth: `pymicro_features` 2.0.2 `src/micro_features.cpp::init_cfg()`,
 * mirrored by ESPHome `micro_wake_word/preprocessor_settings.h`.
 * @{ */
#define WAKEFRONT_SAMPLE_RATE 16000    /**< Hz; the mic is already opened at this rate. */
#define WAKEFRONT_WINDOW_MS 30         /**< Analysis window (480 samples). */
#define WAKEFRONT_STEP_MS 10           /**< Hop between windows (160 samples) — one row per step. */
#define WAKEFRONT_FEATURES 40          /**< Filterbank channels = model input width. */
#define WAKEFRONT_LOWER_BAND_LIMIT 125.0f   /**< Filterbank low edge, Hz. */
#define WAKEFRONT_UPPER_BAND_LIMIT 7500.0f  /**< Filterbank high edge, Hz. */
#define WAKEFRONT_NR_SMOOTHING_BITS 10      /**< Noise reduction: fixed-point smoothing bits. */
#define WAKEFRONT_NR_EVEN_SMOOTHING 0.025f  /**< Noise reduction: even-channel smoothing. */
#define WAKEFRONT_NR_ODD_SMOOTHING 0.06f    /**< Noise reduction: odd-channel smoothing. */
#define WAKEFRONT_NR_MIN_SIGNAL_REMAINING 0.05f /**< Noise reduction: floor on retained signal. */
#define WAKEFRONT_PCAN_ENABLE 1        /**< Per-channel amplitude normalisation on. */
#define WAKEFRONT_PCAN_STRENGTH 0.95f  /**< PCAN exponent. */
#define WAKEFRONT_PCAN_OFFSET 80.0f    /**< PCAN offset. */
#define WAKEFRONT_PCAN_GAIN_BITS 21    /**< PCAN gain fixed-point bits. */
#define WAKEFRONT_LOG_ENABLE 1         /**< Log compression on. */
#define WAKEFRONT_LOG_SCALE_SHIFT 6    /**< Log output shift. */
/** @} */

/** Samples consumed per feature row (10 ms at 16 kHz). Feed PCM in this stride. */
#define WAKEFRONT_STRIDE (WAKEFRONT_SAMPLE_RATE / 1000 * WAKEFRONT_STEP_MS)

/** @name int8 requantisation
 * The frontend emits uint16 in roughly 0..670. Training divided that by 25.6 to
 * get floats in 0..26 (`pymicro_features` FLOAT32_SCALE = 1/25.6 = 0.0390625),
 * and the int8 model maps 0..26 onto -128..127. Both steps collapse into one
 * 32-bit integer expression, taken verbatim from ESPHome
 * `micro_wake_word.cpp::generate_features_()`:
 * `int8 = (v * 256 + 333) / 666 - 128`, clamped. 666 = round(25.6 * 26.0).
 * @{ */
#define WAKEFRONT_VALUE_SCALE 256
#define WAKEFRONT_VALUE_DIV 666
/** @} */

/** Rows buffered internally; the model consumes 3 per Invoke(). */
#define WAKEFRONT_MAX_ROWS 3

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Allocate the front-end state. Idempotent; aborts on allocation failure.
 *
 * Must be called once before the first wakefront_push(). Safe to call again;
 * later calls are no-ops (use wakefront_reset() to clear running state).
 */
void wakefront_init(void);

/**
 * @brief Clear the front-end's running state (noise estimate, PCAN gains, window).
 *
 * Call whenever wake mode is (re-)entered so a stale noise estimate from an old
 * session cannot bias the first detections.
 */
void wakefront_reset(void);

/**
 * @brief Feed PCM and emit feature rows.
 * @param pcm       16 kHz mono signed 16-bit samples.
 * @param n_samples Number of samples in @p pcm. Any length works; the front-end
 *                  buffers the remainder internally, but WAKEFRONT_STRIDE-sized
 *                  pushes are the intended cadence.
 * @return Number of new 40-feature rows produced by this call.
 */
int wakefront_push(const int16_t *pcm, int n_samples);

/**
 * @brief True when at least @p frames rows have been produced since the last wakefront_take().
 * @param frames Row count to test for, at most WAKEFRONT_MAX_ROWS.
 */
bool wakefront_ready(int frames);

/**
 * @brief Copy the most recent @p frames rows out and clear the new-row counter.
 * @param frames Rows to copy, at most WAKEFRONT_MAX_ROWS.
 * @param dst    Destination, at least `frames * WAKEFRONT_FEATURES` bytes.
 *               Filled oldest row first — the order the model expects.
 */
void wakefront_take(int frames, int8_t *dst);

#ifdef __cplusplus
}
#endif
