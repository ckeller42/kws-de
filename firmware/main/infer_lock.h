/**
 * @file infer_lock.h
 * @brief The one esp-nn scratch region both generated models work in, and the
 *        lock that stops them working in it at the same time.
 *
 * esp-nn's ESP32-S3 kernels do not allocate. The caller hands them a scratch
 * buffer through esp_nn_set_*_scratch_buf(), and those pointers are file-static
 * globals inside esp-nn — one per kernel family for the whole image, not one
 * per model. Two models with separate regions therefore do not get one each:
 * whichever set the pointer last owns it, and since scratch is a *write*
 * target, the other model's kernels then write into it — past the end, if it is
 * the smaller of the two. So there is one region, sized for the widest op of
 * either model, and the two inferences are serialised.
 */
#pragma once
#include <stdint.h>
#include "gen/command_infer.h"
#include "gen/wake_infer.h"

/** Big enough for whichever model asks for more (the generator emits both). */
#define KWS_INFER_SCRATCH_BYTES                                                        \
    (WAKE_INFER_SCRATCH_BYTES > COMMAND_INFER_SCRATCH_BYTES ? WAKE_INFER_SCRATCH_BYTES \
                                                            : COMMAND_INFER_SCRATCH_BYTES)

#ifdef __cplusplus
extern "C" {
#endif

/** Internal RAM, 16-byte aligned: esp-nn's S3 kernels want both. */
extern int8_t kws_infer_scratch[KWS_INFER_SCRATCH_BYTES];

/** Create the lock. Idempotent; call before spawning a task that infers. */
void kws_infer_lock_init(void);

/**
 * @brief Hold the scratch region for one generated inference.
 *
 * Only the wake and recognise tasks contend, and only inside an assist window —
 * outside one the recogniser is idle, and in recognise mode the wake task is.
 * When they do overlap the wait is bounded by a single command inference
 * (~29 ms), which the wake task can absorb: it runs every 30 ms against an
 * audio ring holding a second of history, so a step that starts late catches
 * up on the next iteration rather than dropping audio.
 */
void kws_infer_lock(void);
void kws_infer_unlock(void);

#ifdef __cplusplus
}
#endif
