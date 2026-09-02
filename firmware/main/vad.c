#include "vad.h"
#include <math.h>

void vad_reset(vad_t *v) { v->noise = 300.f; v->speech_frames = v->silence_frames = v->in_speech = 0; }

int vad_push(vad_t *v, const int16_t *frame, int n)
{
    double acc = 0;
    for (int i = 0; i < n; i++) acc += (double)frame[i] * frame[i];
    float rms = sqrtf((float)(acc / n));
    float thr = v->noise * 4.f > 300.f ? v->noise * 4.f : 300.f;
    if (!v->in_speech) v->noise += 0.05f * (rms - v->noise);
    if (rms > thr) { v->speech_frames++; v->silence_frames = 0; }
    else { v->silence_frames++; v->speech_frames = 0; }
    if (!v->in_speech && v->speech_frames >= 2) v->in_speech = 1;
    if (v->in_speech && v->silence_frames >= VAD_TRAILING_FRAMES) { v->in_speech = 0; v->speech_frames = 0; }
    return v->in_speech;
}
