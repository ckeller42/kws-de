/* Front-end parity: the vendored TFLite-Micro microfrontend driven by
   wakefront.c must reproduce, bit-for-bit, the int8 rows pymicro-features
   produced for the same PCM (gen/wake_test_vectors.h). Any drift here means
   the device is feeding the wake model features it was never trained on. */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "wakefront.h"
#include "gen/wake_test_vectors.h"

int main(void)
{
    wakefront_init();
    wakefront_reset();

    int row = 0, worst = 0;
    int8_t got[WAKEFRONT_FEATURES];
    for (int i = 0; i + WAKEFRONT_STRIDE <= (int)(sizeof WT_PCM / sizeof WT_PCM[0]);
         i += WAKEFRONT_STRIDE) {
        if (wakefront_push(WT_PCM + i, WAKEFRONT_STRIDE) == 0) continue;
        assert(row < WT_ROWS);
        assert(wakefront_ready(1));
        wakefront_take(1, got);
        for (int c = 0; c < WAKEFRONT_FEATURES; c++) {
            int d = abs((int)got[c] - (int)WT_FEATURES[row][c]);
            if (d > worst) worst = d;
        }
        row++;
    }
    printf("wakefront rows %d/%d, max int8 deviation %d\n", row, WT_ROWS, worst);
    assert(row == WT_ROWS);
    assert(worst == 0);          /* same C, same config: exact or something moved */

    /* The 3-row take the model consumes must hand back the last three rows in
       oldest-first order. Re-run the tail and compare against the golden. */
    wakefront_reset();
    int rows = 0;
    int8_t block[3 * WAKEFRONT_FEATURES];
    for (int i = 0; i + WAKEFRONT_STRIDE <= (int)(sizeof WT_PCM / sizeof WT_PCM[0]);
         i += WAKEFRONT_STRIDE) {
        rows += wakefront_push(WT_PCM + i, WAKEFRONT_STRIDE);
        if (!wakefront_ready(3)) continue;
        wakefront_take(3, block);
        for (int f = 0; f < 3; f++)
            for (int c = 0; c < WAKEFRONT_FEATURES; c++)
                assert(block[f * WAKEFRONT_FEATURES + c] == WT_FEATURES[rows - 3 + f][c]);
    }
    assert(rows == WT_ROWS);

    puts("test_wakefront OK");
    return 0;
}
