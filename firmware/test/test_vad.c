#include <assert.h>
#include <stdio.h>
#include "vad.h"
#include "gen/features_config.h"

static void fill(int16_t *f, int n, int amp) { for (int i = 0; i < n; i++) f[i] = (i & 1) ? amp : -amp; }

/* Pushes `frames` frames of the given amplitude, returns the number of 0->1
   (rising) transitions of vad_push()'s return value seen along the way. */
static int push_run(vad_t *v, int16_t *f, int frames, int amp)
{
    int rises = 0, prev = v->in_speech;
    fill(f, KWS_HOP, amp);
    for (int i = 0; i < frames; i++) {
        int active = vad_push(v, f, KWS_HOP);
        if (active && !prev) rises++;
        prev = active;
    }
    return rises;
}

/* burst(200ms) + silence(800ms) + burst(200ms) + silence(long enough to close
   under either hangover): one continuous segment (1 rising edge) at the 1200 ms
   sentence/negs/wake hangover, since the 800 ms gap is under it; two segments
   (2 rising edges) at the 500 ms word hangover, since the gap exceeds it. */
static int rising_edges(int trailing_frames)
{
    vad_t v; vad_reset(&v, trailing_frames);
    int16_t f[KWS_HOP];
    int rises = 0;
    rises += push_run(&v, f, 10, 4000);  /* 200 ms burst */
    rises += push_run(&v, f, 40, 50);    /* 800 ms silence */
    rises += push_run(&v, f, 10, 4000);  /* 200 ms burst */
    rises += push_run(&v, f, trailing_frames + 5, 50);  /* trailing silence, closes for sure */
    return rises;
}

int main(void)
{
    vad_t v; vad_reset(&v, VAD_TRAILING_FRAMES);
    int16_t f[KWS_HOP];
    fill(f, KWS_HOP, 50);                      /* quiet room */
    for (int i = 0; i < 50; i++) assert(vad_push(&v, f, KWS_HOP) == 0);
    fill(f, KWS_HOP, 4000);                    /* speech */
    assert(vad_push(&v, f, KWS_HOP) == 0);     /* needs 2 consecutive frames */
    assert(vad_push(&v, f, KWS_HOP) == 1);
    for (int i = 0; i < 20; i++) assert(vad_push(&v, f, KWS_HOP) == 1);
    fill(f, KWS_HOP, 50);
    for (int i = 0; i < VAD_TRAILING_FRAMES - 1; i++) assert(vad_push(&v, f, KWS_HOP) == 1);
    assert(vad_push(&v, f, KWS_HOP) == 0);     /* closes exactly after 500 ms of silence */

    /* Per-set hangover: a natural mid-sentence pause (800 ms) stays inside one
       take at 1200 ms hangover, but splits the take at the 500 ms word hangover. */
    assert(rising_edges(1200 / 20) == 1);
    assert(rising_edges(500 / 20) == 2);

    /* False-start filter: a 40 ms transient (2 frames) followed by silence
       leaves under 200 ms of total above-threshold time — record.c's
       MIN_SPEECH_MS check discards such a take instead of saving it. */
    vad_t fs; vad_reset(&fs, VAD_TRAILING_FRAMES);
    push_run(&fs, f, 2, 4000);
    push_run(&fs, f, VAD_TRAILING_FRAMES + 5, 50);
    assert(fs.in_speech == 0);
    assert(fs.speech_total * 20 < 200);

    puts("test_vad OK");
    return 0;
}
