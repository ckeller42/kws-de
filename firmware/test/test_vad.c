#include <assert.h>
#include <stdio.h>
#include "vad.h"
#include "gen/features_config.h"

static void fill(int16_t *f, int n, int amp) { for (int i = 0; i < n; i++) f[i] = (i & 1) ? amp : -amp; }

int main(void)
{
    vad_t v; vad_reset(&v);
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
    puts("test_vad OK");
    return 0;
}
