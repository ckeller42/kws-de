/* Whole-model host parity for the stateless command model: MEAN and SOFTMAX
   are the only arithmetic the wake model does not exercise, and both carry
   their own requantisation, so they get their own byte-for-byte check against
   the TFLite interpreter. See test_wake_parity.c for the ANSI-vs-S3 note. */
#include <assert.h>
#include <stdio.h>
#include "gen/command_infer.h"
#include "gen/command_infer_vectors.h"

int main(void)
{
    static int8_t out[COMMAND_INFER_OUTPUT_LEN];
    int bad = 0;
    command_infer_init();
    for (int step = 0; step < COMMAND_STEPS; step++) {
        command_infer(COMMAND_IN[step], out);
        for (int i = 0; i < COMMAND_INFER_OUTPUT_LEN; i++)
            if (out[i] != COMMAND_EXPECT[step][i]) {
                if (bad < 8)
                    printf("command step %d byte %d: got %d want %d\n",
                           step, i, out[i], COMMAND_EXPECT[step][i]);
                bad++;
            }
    }
    printf("command parity: %d/%d bytes differ (arena %u B)\n",
           bad, COMMAND_STEPS * COMMAND_INFER_OUTPUT_LEN,
           (unsigned) command_infer_arena_bytes());
    assert(bad == 0);
    puts("test_command_parity OK");
    return 0;
}
