#include "wake.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include "arena.h"
#include "assist_gate.h"
#include "audio.h"
#include "beep.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_nn.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "gen/wake_infer.h"
#include "gen/wake_model_config.h"
#include "gen/wake_model_data.h"
#include "nn_timers.h"
#include "recognise.h"
#include "storage.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_resource_variable.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "ui/ui.h"
#include "wakefront.h"

static_assert(WAKEFRONT_FEATURES == KWS_WAKE_FEATURES,
              "front-end width must match the wake model's input width");
static_assert(WAKEFRONT_MAX_ROWS == KWS_WAKE_FRAMES,
              "front-end row buffer must match the wake model's per-Invoke stride");

/* The streaming graph keeps its ring state in TFLM resource variables, which
   live in their own small arena (mirrors ESPHome's micro_wake_word). 1 KB fits
   the ~20 handles these models declare. */
#define WAKE_VAR_ARENA_BYTES 1024

/* Loop cadence. The front-end consumes 10 ms per row and the model 3 rows per
   Invoke, so 10 ms of sleep keeps us a fraction of a step behind live audio
   while leaving the LVGL task (priority 4, same core) plenty of slack. */
#define WAKE_POLL_MS 10

static const char *TAG = "wake";

static SemaphoreHandle_t s_lock;
static wake_status_t s_st;
static volatile bool s_active;
static volatile bool s_restart;      /* set on activation: reset model + front-end state */
#if CONFIG_KWS_INFER_GENERATED
static volatile bool s_parity_pending; /* generated path: log both paths' output on the next step */
#endif
static uint8_t *s_arena;
static uint8_t s_var_arena[WAKE_VAR_ARENA_BYTES];
static FILE *s_log;
static assist_gate_t s_gate;         /* assist mode only: when the recogniser may run */
static bool s_listening;             /* last state pushed to recognise_set_active() */
static volatile bool s_inject;       /* console-injected fire, see wake_inject_fire() */

static void log_fire(uint32_t ms, float prob)
{
    if (!s_active) return;
    if (!s_log) {
        char p[32];
        snprintf(p, sizeof p, "%s/wake.log", storage_root());
        s_log = fopen(p, "a");
    }
    if (!s_log) return;
    fprintf(s_log, "[Wake] %lu %.3f\n", (unsigned long)ms, (double)prob);
    fflush(s_log);
}

static void wake_task(void *)
{
    /* 13 ops, exactly what models/hey_bus.tflite declares. */
    static tflite::MicroMutableOpResolver<13> resolver;
    resolver.AddConv2D(); resolver.AddDepthwiseConv2D(); resolver.AddFullyConnected();
    resolver.AddReshape(); resolver.AddConcatenation(); resolver.AddSplitV();
    resolver.AddStridedSlice(); resolver.AddLogistic(); resolver.AddQuantize();
    resolver.AddVarHandle(); resolver.AddReadVariable(); resolver.AddAssignVariable();
    resolver.AddCallOnce();

    const tflite::Model *model = tflite::GetModel(g_wake_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) { ESP_LOGE(TAG, "bad wake model schema"); vTaskDelete(nullptr); return; }
    tflite::MicroAllocator *ma = tflite::MicroAllocator::Create(s_var_arena, sizeof s_var_arena);
    tflite::MicroResourceVariables *mrv = tflite::MicroResourceVariables::Create(ma, 20);
    if (!mrv) { ESP_LOGE(TAG, "resource variables failed"); vTaskDelete(nullptr); return; }
    static tflite::MicroInterpreter interp(model, resolver, s_arena, KWS_WAKE_ARENA_BYTES, mrv);
    if (interp.AllocateTensors() != kTfLiteOk) { ESP_LOGE(TAG, "AllocateTensors failed"); vTaskDelete(nullptr); return; }
    ESP_LOGI(TAG, "arena used %u / %u", (unsigned)interp.arena_used_bytes(), (unsigned)KWS_WAKE_ARENA_BYTES);
    TfLiteTensor *in = interp.input(0), *out = interp.output(0);
#if CONFIG_KWS_INFER_GENERATED
    wake_infer_init();
    /* gen/wake_infer.c reserves its esp-nn scratch block from a Python port of
       esp_nn_get_conv_scratch_size_esp32s3() (kws_de/codegen.py). Ask the real
       one, on the real chip, for this model's widest conv — the first CONV_2D,
       with the dims gen/wake_infer.c itself passes — so a port that drifts from
       esp-nn is a boot-log line rather than a silent scratch overrun. */
    const data_dims_t sc_in = { .width = 1, .height = 5, .channels = 40, .extra = 1 };
    const data_dims_t sc_flt = { .width = 1, .height = 5, .channels = 40, .extra = 0 };
    const data_dims_t sc_out = { .width = 1, .height = 1, .channels = 32, .extra = 1 };
    const conv_params_t sc_p = { .in_offset = 128, .out_offset = -128, .stride = { 1, 3 },
                                 .padding = { 0, 0 }, .dilation = { 0, 0 }, .activation = { -128, 127 } };
    ESP_LOGI(TAG, "inference: generated (esp-nn), %u B arena + %u B state, esp-nn conv scratch %d B queried; TFLM arena %u B kept as fallback",
             (unsigned)wake_infer_arena_bytes(), (unsigned)wake_infer_state_bytes(),
             esp_nn_get_conv_scratch_size(&sc_in, &sc_flt, &sc_out, &sc_p),
             (unsigned)KWS_WAKE_ARENA_BYTES);
#if !CONFIG_KWS_INFER_PARITY_LOG
    (void)out;   /* only the parity log reads the interpreter's output tensor */
#endif
#else
    ESP_LOGI(TAG, "inference: TFLite Micro interpreter");
#endif

    wakefront_init();
    beep_init();

    uint32_t pos = 0;            /* absolute ring position we have consumed up to */
    int consecutive = 0;
    int64_t deaf_until_us = 0;
    int16_t chunk[WAKEFRONT_STRIDE];

    for (;;) {
        if (!s_active) { vTaskDelay(pdMS_TO_TICKS(50)); continue; }
        if (s_restart) {
            /* Fresh session: drop the streaming ring state and the front-end's
               noise/PCAN estimates, and start from live audio rather than
               replaying whatever sat in the ring. */
            interp.Reset();
            mrv->ResetAll();
#if CONFIG_KWS_INFER_GENERATED
            wake_infer_reset();
            s_parity_pending = true;
#endif
            wakefront_reset();
            pos = audio_write_pos();
            consecutive = 0;
            deaf_until_us = 0;
            assist_gate_reset(&s_gate);
            if (s_listening) { s_listening = false; recognise_set_active(false); }
            s_restart = false;
        }
        vTaskDelay(pdMS_TO_TICKS(WAKE_POLL_MS));

        uint32_t end = audio_write_pos();
        /* If we ever fall further behind than the ring holds, the old samples
           are already overwritten; skip forward rather than read garbage. */
        if (end - pos > AUDIO_RING_SAMPLES - KWS_SAMPLE_RATE) {
            ESP_LOGW(TAG, "ring overrun, skipping %lu samples", (unsigned long)(end - pos));
            pos = end;
            wakefront_reset();
        }

        while (end - pos >= WAKEFRONT_STRIDE) {
            pos += WAKEFRONT_STRIDE;
            audio_read(pos, chunk, WAKEFRONT_STRIDE);
            wakefront_push(chunk, WAKEFRONT_STRIDE);
            if (!wakefront_ready(KWS_WAKE_FRAMES)) continue;

            int64_t t0 = esp_timer_get_time();
            wakefront_take(KWS_WAKE_FRAMES, in->data.int8);
            NN_TIMERS_RESET();
            uint8_t prob_q;
            int64_t t_invoke = esp_timer_get_time();
#if CONFIG_KWS_INFER_GENERATED
            wake_infer_step(in->data.int8, &prob_q);
            int64_t invoke_us = esp_timer_get_time() - t_invoke;
#if CONFIG_KWS_INFER_PARITY_LOG
            if (s_parity_pending) {
                /* Same input through both paths, once per mode entry: the
                   generated ring state has already advanced, so this compares
                   the interpreter's answer for THIS step only — it is
                   meaningful precisely because both paths were reset together
                   a moment ago. A mismatch is a real regression; the host
                   parity tests should have caught it. Timed outside invoke_us. */
                if (interp.Invoke() == kTfLiteOk)
                    ESP_LOGI(TAG, "parity: generated %u, interpreter %u",
                             (unsigned)prob_q, (unsigned)out->data.uint8[0]);
                s_parity_pending = false;
            }
#endif
#else
            if (interp.Invoke() != kTfLiteOk) { ESP_LOGE(TAG, "Invoke failed"); continue; }
            int64_t invoke_us = esp_timer_get_time() - t_invoke;
            prob_q = out->data.uint8[0];
#endif
            /* uint8 output: prob = (q - zero_point) * scale, i.e. q/256. */
            float prob = (prob_q - KWS_WAKE_OUTPUT_ZERO_POINT) * KWS_WAKE_OUTPUT_SCALE;
            int64_t step_us = esp_timer_get_time() - t0;
            uint32_t ms = (uint32_t)(step_us / 1000);

            consecutive = (prob >= WAKE_THRESHOLD) ? consecutive + 1 : 0;
            bool fired = false;
            if (consecutive >= WAKE_MIN_CONSECUTIVE && esp_timer_get_time() >= deaf_until_us) {
                deaf_until_us = esp_timer_get_time() + (int64_t)WAKE_REFRACTORY_MS * 1000;
                consecutive = 0;
                fired = true;
            }
            if (s_inject) { s_inject = false; fired = true; }

            uint32_t now_ms = (uint32_t)(esp_timer_get_time() / 1000);
            /* Tuning trace: the peak probability of the last 2 s, so the serial
               log alone shows whether the model *hears* the phrase (peak near 1)
               or the threshold/consecutive gate is what stops it firing.
               Step cost is reported as mean +/- sd in microseconds over the same
               window: preemption by the LVGL task shows up as spread, not as a
               higher mean, so a single sample cannot tell the two apart. */
            static float peak = 0; static uint32_t nsteps = 0, last_trace = 0;
            static int64_t sum_us = 0, sumsq_us = 0;
            if (prob > peak) peak = prob;
            nsteps++;
            sum_us += step_us; sumsq_us += step_us * step_us;
            if (now_ms - last_trace >= 2000) {
                int64_t mean = sum_us / (int64_t)nsteps;
                int64_t var = sumsq_us / (int64_t)nsteps - mean * mean;
                ESP_LOGI(TAG, "peak %.3f over %lu steps, step %lld +/- %lld us (invoke %lld us: " NN_TIMERS_FMT ")",
                         (double)peak, (unsigned long)nsteps, mean, (int64_t)std::sqrt((double)(var > 0 ? var : 0)),
                         invoke_us, NN_TIMERS_ARGS(invoke_us));
                peak = 0; nsteps = 0; last_trace = now_ms; sum_us = sumsq_us = 0;
            }
            xSemaphoreTake(s_lock, portMAX_DELAY);
            s_st.prob = prob;
            s_st.infer_ms = ms;
            s_st.arena_used = interp.arena_used_bytes();
            if (fired) { s_st.fired_count++; s_st.fired_at_ms = now_ms; }
            wake_status_t copy = s_st;
            xSemaphoreGive(s_lock);

            /* Assist mode: a fire opens a short window in which the command
               recogniser runs, and it is switched off again when the window
               closes. The gate is pure logic (assist_gate.c, host-tested); all
               that happens here is turning the recogniser on and off at its
               edges, so the expensive model runs for ~2.5 s per interaction
               instead of continuously. */
            bool assist = app_get_mode() == UI_MODE_ASSIST;
            if (assist) {
                if (fired) assist_gate_on_wake(&s_gate, now_ms);
                bool listen = assist_gate_tick(&s_gate, now_ms);
                if (listen != s_listening) {
                    s_listening = listen;
                    /* Opening the window hands the recogniser its own deadline
                       so it stops even if this task stops being scheduled. */
                    if (listen) recognise_listen_for(ASSIST_WINDOW_MS);
                    else recognise_set_active(false);
                    ESP_LOGI(TAG, "assist: recogniser %s (window %lu)", listen ? "on" : "off",
                             (unsigned long)s_gate.windows);
                }
            }

            if (fired) {
                ESP_LOGI(TAG, "wake! prob %.3f (%lu ms)", (double)prob, (unsigned long)ms);
                log_fire(now_ms, prob);
            }
            if (assist) {
                recognise_status_t rst;
                recognise_get_status(&rst);
                ui_assist_refresh(&copy, &rst, s_listening);
            } else {
                ui_wake_refresh(&copy);       /* paint green before the tone blocks */
            }
            if (fired) beep_play();
            /* Yield inside the catch-up loop too: a backlog must never starve
               the LVGL task, or the Record button stops responding. */
            vTaskDelay(1);
            end = audio_write_pos();
        }
    }
}

extern "C" void wake_start(void)
{
    s_lock = xSemaphoreCreateMutex();
    s_arena = arena_alloc(TAG, "wake", KWS_WAKE_ARENA_BYTES);
    assert(s_arena);
    /* Core 0, priority 3 — same slot as the recogniser, off LVGL's core.
       Only one of the two is ever active. */
    xTaskCreatePinnedToCore(wake_task, "wake", 16384, nullptr, 3, nullptr, 0);
}

extern "C" void wake_set_active(bool on)
{
    if (on) s_restart = true;
    s_active = on;
    if (!on && s_log) { fclose(s_log); s_log = nullptr; }
}

extern "C" void wake_inject_fire(void) { s_inject = true; }

extern "C" void wake_get_status(wake_status_t *out)
{
    xSemaphoreTake(s_lock, portMAX_DELAY);
    *out = s_st;
    xSemaphoreGive(s_lock);
}
