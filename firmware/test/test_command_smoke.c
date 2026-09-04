/* Model-free, data-free command parity smoke test: 16 fixed-seed synthetic
   input windows, run once through the command model's own reference
   interpreter at generation time (kws_de.codegen.smoke_vectors_text) and
   frozen as gen/command_smoke_vectors.h. No .tflite model and no
   KWS_DATA_ROOT are needed to build or run this -- only to regenerate the
   frozen vectors, exactly like gen/command_infer.c itself. In $(TESTS), so
   host-test's default `make -C firmware/test` proves the committed generated
   C still matches a known-good answer on every CI run, with no data present
   at all -- unlike test_command_parity (real approved recordings, needs
   KWS_DATA_ROOT, kept out of $(TESTS)).

   The model is stateless, so each window is an independent call rather than
   one continuous clip (see test_wake_smoke.c for the streaming counterpart).
   COMMAND_SMOKE_IN's row stride is not a multiple of 16 and command_infer
   asserts its `in` is 16-byte aligned, so each row is copied into the aligned
   buffer below first -- same as test_command_parity.c. */
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "command_infer.h"
#include "command_smoke_vectors.h"

int main(void)
{
    static int8_t in_buf[COMMAND_INFER_INPUT_LEN] __attribute__((aligned(16)));
    static int8_t out[COMMAND_INFER_OUTPUT_LEN] __attribute__((aligned(16)));
    int bad = 0;
    /* The firmware refuses to run the generated path when the chip's own
       esp_nn_get_depthwise_conv_scratch_size_esp32s3 answers more than this
       reserve, so the reserve has to exist and to fit inside the arena it is
       carved from. Its actual size can only be checked against esp-nn on the
       device: the ANSI kernels this test links against ask for no scratch. */
    assert(COMMAND_INFER_SCRATCH_BYTES > 0
           && COMMAND_INFER_SCRATCH_BYTES <= COMMAND_INFER_ARENA_BYTES);
    assert(COMMAND_INFER_STATE_BYTES == 0);          /* stateless: nothing to reset between calls */
    command_infer_init();
    for (int step = 0; step < COMMAND_SMOKE_STEPS; step++) {
        memcpy(in_buf, COMMAND_SMOKE_IN[step], sizeof in_buf);
        command_infer(in_buf, out);
        for (int i = 0; i < COMMAND_INFER_OUTPUT_LEN; i++)
            if (out[i] != COMMAND_SMOKE_EXPECT[step][i]) {
                if (bad < 8)
                    printf("command smoke step %d byte %d: got %d want %d\n",
                           step, i, out[i], COMMAND_SMOKE_EXPECT[step][i]);
                bad++;
            }
    }
    printf("command smoke: %d/%d bytes differ\n",
           bad, COMMAND_SMOKE_STEPS * COMMAND_INFER_OUTPUT_LEN);
    assert(bad == 0);
    puts("test_command_smoke OK");
    return 0;
}
