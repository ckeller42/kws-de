/**
 * @file record.h
 * @brief Guided recorder state machine: prompts a speaker through word/sentence/negative
 * takes and saves them as WAV files under storage_root() (see storage.h).
 */
#pragma once
#include <stdint.h>
#include "field.h"
#include "prompts.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Commands posted to the record task from UI callbacks, via record_post(). */
typedef enum {
    REC_CMD_START_SESSION,      /**< Bump the speaker id, start the sentence set; negatives auto-chain on completion. */
    REC_CMD_START_WAKE_SESSION, /**< Bump the speaker id, start the "Hey Bus"-only wake set (PROMPT_WAKE);
                                      session ends when it's exhausted, no chaining into negatives. */
    REC_CMD_PAUSE,               /**< Pause the recorder (goes idle, waits for a command). */
    REC_CMD_FIELD_TAKE,          /**< Assist mode: copy one field take out of the audio ring and save it.
                                      The payload is set by record_post_field_take(). */
} record_cmd_t;
/**
 * @brief Recorder phase, as shown by the UI.
 *
 * REC_GETREADY and REC_SESSION_DONE are appended (not inserted) so existing
 * phase indices — and the UI's phase-label table indexed by them — stay put.
 */
typedef enum {
    REC_IDLE,      /**< Paused, waiting for a command. */
    REC_LISTENING, /**< Capturing, waiting for speech to start (VAD). */
    REC_CAPTURING, /**< Speech detected, recording the take. */
    REC_SAVED,     /**< Take saved; briefly shown before the next take. */
    REC_CLIPPED,   /**< Take discarded: input clipped, redo requested. */
    REC_TIMEOUT,   /**< Take discarded: no speech detected in time, redo requested. */
    REC_FULL,      /**< Take discarded: storage below STORAGE_MIN_FREE_BYTES. */
    REC_DONE,      /**< All prompts in the current set completed. */
    REC_GETREADY,  /**< "Get ready" beat shown before a take starts capturing. */
    REC_SESSION_DONE, /**< Sentences + negatives both completed for this speaker; session over. */
} record_phase_t;

/** @brief Snapshot of recorder state, filled by record_get_status(). */
typedef struct {
    record_phase_t phase;
    prompt_set_t set; uint32_t seed; int index, count;
    int take, takes;                       /**< Which read of the prompt (1-based) out of how many. */
    char prompt[96]; char speaker[8];      /**< Display text of the current prompt; speaker id, e.g. "spk03". */
    float level_dbfs;                      /**< Input level for the level bar, updated every 100 ms. */
    int saved_takes;                       /**< Takes saved since the last REC_CMD_START_SESSION. */
    uint32_t field_takes;                  /**< Field takes saved since boot. */
    uint32_t field_dropped;                /**< Field takes dropped: storage below STORAGE_MIN_FREE_BYTES. */
} record_status_t;

/** @brief Create the record task. Starts paused (REC_IDLE); call record_post(REC_CMD_START_SESSION) to begin. */
void record_start(void);
/** @brief Post a command to the record task from a UI callback. Non-blocking. */
void record_post(record_cmd_t cmd);
/** @brief Post one field take (assist mode). Copies @p t into the recorder's
 *  single pending slot and posts REC_CMD_FIELD_TAKE. Non-blocking. */
void record_post_field_take(const field_take_t *t);
/** @brief Copy the current recorder status under mutex. */
void record_get_status(record_status_t *out);

#ifdef __cplusplus
}
#endif
