#pragma once
/* stream.h — port of kws_de.stream.KeywordStream */
#include "gen/labels.h"
#include "gen/features_config.h"

typedef struct {
    float hist[KWS_SMOOTH_WIN][KWS_NUM_LABELS];
    int   hist_len, hist_pos;
    int   run_label;            /* -1 = None */
    int   run_len;
    int   run_fired;
    int   last_fired_label;     /* -1 = None */
    int   gap_since_last_fired;
} stream_t;

void stream_reset(stream_t *s);
/* Returns the fired label index, or -1. */
int  stream_push(stream_t *s, const float posterior[KWS_NUM_LABELS]);
