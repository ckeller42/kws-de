#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "mfcc.h"
#include "gen/test_vectors.h"

int main(void)
{
    static float out[KWS_N_FRAMES][KWS_N_MFCC];
    mfcc_compute(TV_PCM, out);
    float worst = 0.f;
    for (int t = 0; t < KWS_N_FRAMES; t++)
        for (int c = 0; c < KWS_N_MFCC; c++) {
            float d = fabsf(out[t][c] - TV_MFCC[t][c]);
            if (d > worst) worst = d;
        }
    printf("mfcc max abs err %g\n", worst);
    assert(worst < 1e-2f);           /* float vs float64 numpy; int8 step is ~3.0 */

    /* Streaming path must equal the one-shot path. */
    mfcc_state_t s;
    mfcc_init(&s);
    for (int t = 0; t < KWS_N_FRAMES; t++) mfcc_push_frame(&s, TV_PCM + t * KWS_HOP);
    static float out2[KWS_N_FRAMES][KWS_N_MFCC];
    mfcc_finish(&s, out2);
    assert(memcmp(out, out2, sizeof out) == 0);

    /* Quantisation: scale 3.0, zp 80 -> clamps and rounds. */
    float q_in[KWS_N_FRAMES][KWS_N_MFCC] = {{0}};
    q_in[0][0] = 4.4f; q_in[0][1] = -1000.f; q_in[0][2] = 1000.f;
    int8_t q[KWS_N_FRAMES * KWS_N_MFCC];
    mfcc_quantize(q_in, q, 3.0f, 80);
    assert(q[0] == 81 && q[1] == -128 && q[2] == 127 && q[3] == 80);
    puts("test_mfcc OK");
    return 0;
}
