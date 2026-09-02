/**
 * @file record.h
 * @brief Guided recorder state machine: prompts a speaker through word/sentence/negative
 * takes and saves them as WAV files under /rec.
 */
#pragma once
#include <stdint.h>
#include "prompts.h"

/** @brief Commands posted to the record task from UI callbacks, via record_post(). */
typedef enum {
    REC_CMD_REDO,          /**< Redo the current prompt from take 1. */
    REC_CMD_SKIP,          /**< Skip the current prompt, advance to the next. */
    REC_CMD_NEXT,          /**< Same as REC_CMD_SKIP: advance to the next prompt. */
    REC_CMD_NEW_SPEAKER,   /**< Bump the speaker id in NVS and reshuffle the current prompt set. */
    REC_CMD_SET_WORDS,     /**< Switch to the word prompt set with a fresh random order. */
    REC_CMD_SET_SENTENCES, /**< Switch to the sentence prompt set with a fresh random order. */
    REC_CMD_SET_NEGS,      /**< Switch to the negative prompt set with a fresh random order. */
    REC_CMD_PAUSE,         /**< Pause the recorder (goes idle, waits for a command). */
    REC_CMD_RESUME,        /**< Resume recording from take 1 of the current prompt. */
} record_cmd_t;
/**
 * @brief Recorder phase, as shown by the UI.
 *
 * REC_GETREADY is appended (not inserted) so existing phase indices — and the
 * UI's phase-label table indexed by them — stay put.
 */
typedef enum {
    REC_IDLE,      /**< Paused, waiting for a command. */
    REC_LISTENING, /**< Capturing, waiting for speech to start (VAD). */
    REC_CAPTURING, /**< Speech detected, recording the take. */
    REC_SAVED,     /**< Take saved to /rec; briefly shown before the next take. */
    REC_CLIPPED,   /**< Take discarded: input clipped, redo requested. */
    REC_TIMEOUT,   /**< Take discarded: no speech detected in time, redo requested. */
    REC_FULL,      /**< Take discarded: storage below STORAGE_MIN_FREE_BYTES. */
    REC_DONE,      /**< All prompts in the current set completed. */
    REC_GETREADY,  /**< "Get ready" beat shown before a take starts capturing. */
} record_phase_t;

/** @brief Snapshot of recorder state, filled by record_get_status(). */
typedef struct {
    record_phase_t phase;
    prompt_set_t set; uint32_t seed; int index, count;
    int take, takes;                       /**< Which read of the prompt (1-based) out of how many. */
    char prompt[96]; char speaker[8];      /**< Display text of the current prompt; speaker id, e.g. "spk03". */
    float level_dbfs;                      /**< Input level for the level bar, updated every 100 ms. */
} record_status_t;

/** @brief Create the record task. Starts paused (REC_IDLE); call record_post(REC_CMD_RESUME) to begin. */
void record_start(void);
/** @brief Post a command to the record task from a UI callback. Non-blocking. */
void record_post(record_cmd_t cmd);
/** @brief Copy the current recorder status under mutex. */
void record_get_status(record_status_t *out);
