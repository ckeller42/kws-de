/**
 * @file nn_timers.h
 * @brief Per-kernel esp-nn timers: how much of an Invoke is assembly kernels.
 *
 * esp-tflite-micro's esp-nn wrappers accumulate microseconds into one global
 * per replaced op (conv, depthwise_conv, fully_connected, softmax, pooling).
 * Zero them right before an Invoke and read them right after and their sum is
 * that Invoke's assembly-kernel time; `invoke_us - sum` is the residual — the
 * reference-C ops plus interpreter dispatch. The residual is the number that
 * says whether replacing a reference op is worth an export change, so it is
 * logged next to the total rather than left to be inferred.
 *
 * Cost is five stores plus five loads per Invoke.
 */
#pragma once

extern long long conv_total_time, dc_total_time, fc_total_time;
extern long long softmax_total_time, pooling_total_time;

/** Zero all five; call immediately before Invoke(). */
#define NN_TIMERS_RESET()                                                    \
    (conv_total_time = dc_total_time = fc_total_time = softmax_total_time =  \
         pooling_total_time = 0)

#define NN_TIMERS_SUM \
    (conv_total_time + dc_total_time + fc_total_time + softmax_total_time + pooling_total_time)

/** Format string + arguments for one log line; `invoke_us` is the measured total. */
#define NN_TIMERS_FMT "conv %lld, dw %lld, fc %lld, sm %lld, pool %lld, rest %lld us"
#define NN_TIMERS_ARGS(invoke_us)                                            \
    conv_total_time, dc_total_time, fc_total_time, softmax_total_time,       \
        pooling_total_time, (long long)(invoke_us) - NN_TIMERS_SUM
