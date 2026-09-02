#pragma once
/* vad.h — energy VAD, pure C, host-testable (see task-5-brief.md ruling) */
#include <stdint.h>

typedef struct { float noise; int speech_frames; int silence_frames; int in_speech; } vad_t;

void vad_reset(vad_t *v);
/* One 20 ms frame (KWS_HOP samples). Returns 1 while speech is active. Speech opens at
   rms > max(noise*4, 300) for 2 consecutive frames; noise floor tracks rms exponentially (alpha 0.05)
   only while not in speech. Closes after VAD_TRAILING_FRAMES (=25 -> 500 ms) below threshold. */
int  vad_push(vad_t *v, const int16_t *frame, int n);

#define VAD_TRAILING_FRAMES 25
