#pragma once
/* record.h — guided recorder state machine */
#include <stdint.h>
#include "prompts.h"

typedef enum { REC_CMD_REDO, REC_CMD_SKIP, REC_CMD_NEXT, REC_CMD_NEW_SPEAKER, REC_CMD_SET_WORDS, REC_CMD_SET_SENTENCES, REC_CMD_SET_NEGS, REC_CMD_PAUSE, REC_CMD_RESUME } record_cmd_t;
typedef enum { REC_IDLE, REC_LISTENING, REC_CAPTURING, REC_SAVED, REC_CLIPPED, REC_TIMEOUT, REC_FULL, REC_DONE } record_phase_t;

typedef struct {
    record_phase_t phase;
    prompt_set_t set; uint32_t seed; int index, count;
    char prompt[96]; char speaker[8];      /* "spk03" */
    float level_dbfs;                      /* for the bar, updated every 100 ms */
} record_status_t;

void record_start(void);                        /* creates the task (starts paused) */
void record_post(record_cmd_t cmd);             /* from UI callbacks */
void record_get_status(record_status_t *out);   /* copy under mutex */
