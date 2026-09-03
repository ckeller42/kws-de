/* Host parity harness: the generated inference C compiled against esp-nn's
   ANSI-C reference kernels, checked byte-for-byte against expectations the
   TFLite interpreter produced (kws_de.codegen writes the gen/ vectors header).

   The device runs esp-nn's ESP32-S3 kernels, not these ANSI ones; esp-nn's own
   contract is that the two agree bit-for-bit as long as CONFIG_NN_SKIP_NUDGE /
   SKIP_NUDGE is NOT defined (that macro selects a faster, non-bit-exact
   requantisation). The device parity log line closes the remaining gap. */
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "conv_probe_vectors.h"

#if defined(SKIP_NUDGE) || defined(CONFIG_NN_SKIP_NUDGE)
#error "esp-nn's SKIP_NUDGE requantisation is not bit-exact; parity is meaningless with it"
#endif

void conv_probe(const int8_t *in, int8_t *out, void *scratch);

int main(void)
{
    static int8_t out[CONV_PROBE_OUT_LEN];
    static int8_t scratch[CONV_PROBE_SCRATCH];
    memset(out, 0, sizeof out);
    conv_probe(CONV_PROBE_IN, out, scratch);
    int bad = 0;
    for (int i = 0; i < CONV_PROBE_OUT_LEN; i++)
        if (out[i] != CONV_PROBE_EXPECT[i]) {
            if (bad < 8)
                printf("conv byte %d: got %d want %d\n", i, out[i], CONV_PROBE_EXPECT[i]);
            bad++;
        }
    printf("conv parity: %d/%d bytes differ\n", bad, CONV_PROBE_OUT_LEN);
    assert(bad == 0);
    puts("test_infer_parity OK");
    return 0;
}
