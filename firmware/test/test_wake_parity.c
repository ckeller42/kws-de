/* Whole-model host parity: the generated wake inference compiled against
   esp-nn's ANSI-C reference kernels, run over complete streaming sequences and
   checked against the TFLite interpreter's answer for *every* step, not just
   the last one (a state bug shows up mid-sequence and is gone by the end).
   State is reset between clips, as it is between utterances on the device.

   kws_de.codegen writes both gen/ files. The device runs esp-nn's ESP32-S3
   kernels rather than these ANSI ones; esp-nn's contract is that the two agree
   bit-for-bit as long as SKIP_NUDGE / CONFIG_NN_SKIP_NUDGE is not defined,
   which the generated source itself #errors on. */
#include <assert.h>
#include <stdio.h>
#include "wake_infer.h"
#include "wake_infer_vectors.h"

int main(void)
{
    int bad = 0, step = 0;
    wake_infer_init();
    for (int clip = 0; clip < WAKE_CLIPS; clip++) {
        wake_infer_reset();
        for (int i = 0; i < WAKE_CLIP_STEPS[clip]; i++, step++) {
            uint8_t prob = 0;
            wake_infer_step(WAKE_IN[step], &prob);
            if (prob != WAKE_EXPECT[step][0]) {
                if (bad < 8)
                    printf("wake clip %d step %d: got %u want %u\n",
                           clip, i, prob, WAKE_EXPECT[step][0]);
                bad++;
            }
        }
    }
    printf("wake parity: %d/%d steps differ (%d clips, %u B state)\n",
           bad, WAKE_STEPS, WAKE_CLIPS, (unsigned) wake_infer_state_bytes());
    assert(bad == 0);
    puts("test_wake_parity OK");
    return 0;
}
