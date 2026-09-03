/* Model-free, data-free wake parity smoke test: a fixed-seed synthetic
   streaming clip, run once through the wake model's own reference
   interpreter at generation time (kws_de.codegen.smoke_vectors_text) and
   frozen as gen/wake_smoke_vectors.h. No .tflite model and no KWS_DATA_ROOT
   are needed to build or run this -- only to regenerate the frozen vectors,
   exactly like gen/wake_infer.c itself. In $(TESTS), so host-test's default
   `make -C firmware/test` proves the committed generated C still matches a
   known-good answer on every CI run, with no data present at all -- unlike
   test_wake_parity (real approved recordings, needs KWS_DATA_ROOT, kept out
   of $(TESTS)). */
#include <assert.h>
#include <stdio.h>
#include "wake_infer.h"
#include "wake_smoke_vectors.h"

int main(void)
{
    int bad = 0;
    wake_infer_init();
    wake_infer_reset();
    for (int step = 0; step < WAKE_SMOKE_STEPS; step++) {
        uint8_t prob = 0;
        wake_infer_step(WAKE_SMOKE_IN[step], &prob);
        if (prob != WAKE_SMOKE_EXPECT[step][0]) {
            if (bad < 8)
                printf("wake smoke step %d: got %u want %u\n", step, prob, WAKE_SMOKE_EXPECT[step][0]);
            bad++;
        }
    }
    printf("wake smoke: %d/%d steps differ\n", bad, WAKE_SMOKE_STEPS);
    assert(bad == 0);
    puts("test_wake_smoke OK");
    return 0;
}
