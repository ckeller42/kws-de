/* Whole-model host parity for the stateless command model: MEAN and SOFTMAX
   are the only arithmetic the wake model does not exercise, and both carry
   their own requantisation, so they get their own byte-for-byte check against
   the TFLite interpreter. See test_wake_parity.c for the ANSI-vs-S3 note.

   command_infer's `in`/`out` are caller-owned (unlike the wake model's `in`,
   which is copied into an aligned ring before esp-nn ever sees it), so the
   generated code asserts `in`'s 16-byte alignment in debug builds (S-7) --
   this harness allocates both buffers aligned. COMMAND_IN's row stride
   (COMMAND_INFER_INPUT_LEN bytes) is not itself a multiple of 16, so each row
   is copied into the aligned scratch below before the call. */
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "command_infer.h"
#include "command_infer_vectors.h"

int main(void)
{
    static int8_t in_buf[COMMAND_INFER_INPUT_LEN] __attribute__((aligned(16)));
    static int8_t out[COMMAND_INFER_OUTPUT_LEN] __attribute__((aligned(16)));
    int bad = 0;
    command_infer_init();
    for (int step = 0; step < COMMAND_STEPS; step++) {
        memcpy(in_buf, COMMAND_IN[step], sizeof in_buf);
        command_infer(in_buf, out);
        for (int i = 0; i < COMMAND_INFER_OUTPUT_LEN; i++)
            if (out[i] != COMMAND_EXPECT[step][i]) {
                if (bad < 8)
                    printf("command step %d byte %d: got %d want %d\n",
                           step, i, out[i], COMMAND_EXPECT[step][i]);
                bad++;
            }
    }
    printf("command parity: %d/%d bytes differ (%d clips, arena %u B)\n",
           bad, COMMAND_STEPS * COMMAND_INFER_OUTPUT_LEN, COMMAND_CLIPS,
           (unsigned) command_infer_arena_bytes());
    assert(bad == 0);
    puts("test_command_parity OK");
    return 0;
}
