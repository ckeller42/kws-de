/* CONFIG_KWS_INFER_PROFILE wiring test. firmware/test/Makefile's test_profile
   target compiles this and both generated *_infer.c with
   -DCONFIG_KWS_INFER_PROFILE=1 (every other target in $(TESTS) builds with
   the flag off, so the default `make -C firmware/test` proves flag-off
   parity is unchanged -- see test_command_smoke.c / test_wake_smoke.c).

   This does not pin timing numbers: kws_infer_ticks() is a host
   clock_gettime() counter here, not the device's esp_cpu_get_cycle_count(),
   so the microseconds it reports are host-CPU noise, not the device's. What
   it does pin is the wiring -- every kernel call in *_infer()/_infer_step()
   accumulates exactly once per invocation into <name>_infer_profile[], and
   profile_dump() resets those counts. The real per-layer numbers are
   captured separately, on the CoreS3 (docs/paper-notes.md, firmware.rst's
   Profiling section). */
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "command_infer.h"
#include "command_smoke_vectors.h"
#include "wake_infer.h"
#include "wake_smoke_vectors.h"

#define COUNT(arr) (sizeof(arr) / sizeof((arr)[0]))

int main(void)
{
    static int8_t cmd_in[COMMAND_INFER_INPUT_LEN] __attribute__((aligned(16)));
    static int8_t cmd_out[COMMAND_INFER_OUTPUT_LEN] __attribute__((aligned(16)));
    command_infer_init();
    for (int step = 0; step < COMMAND_SMOKE_STEPS; step++) {
        memcpy(cmd_in, COMMAND_SMOKE_IN[step], sizeof cmd_in);
        command_infer(cmd_in, cmd_out);
    }
    for (size_t i = 0; i < COUNT(command_infer_profile); i++)
        assert(command_infer_profile[i].calls == (uint32_t)COMMAND_SMOKE_STEPS);
    uint32_t cmd_us = command_infer_profile_dump();
    printf("test_profile: command profiled %u us (host clock) over %d steps\n",
           (unsigned)cmd_us, COMMAND_SMOKE_STEPS);
    for (size_t i = 0; i < COUNT(command_infer_profile); i++) {
        assert(command_infer_profile[i].calls == 0);   /* dump() resets both fields */
        assert(command_infer_profile[i].cycles == 0);
    }

    wake_infer_init();
    wake_infer_reset();
    for (int step = 0; step < WAKE_SMOKE_STEPS; step++) {
        uint8_t prob = 0;
        wake_infer_step(WAKE_SMOKE_IN[step], &prob);
    }
    for (size_t i = 0; i < COUNT(wake_infer_profile); i++)
        assert(wake_infer_profile[i].calls == (uint32_t)WAKE_SMOKE_STEPS);
    uint32_t wake_us = wake_infer_profile_dump();
    printf("test_profile: wake profiled %u us (host clock) over %d steps\n",
           (unsigned)wake_us, WAKE_SMOKE_STEPS);
    for (size_t i = 0; i < COUNT(wake_infer_profile); i++) {
        assert(wake_infer_profile[i].calls == 0);
        assert(wake_infer_profile[i].cycles == 0);
    }

    puts("test_profile OK");
    return 0;
}
