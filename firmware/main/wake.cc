#include "wake.h"
#include <cassert>
#include <cstdio>
#include "arena.h"
#include "audio.h"
#include "beep.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "gen/wake_model_config.h"
#include "gen/wake_model_data.h"
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
static uint8_t *s_arena;
static uint8_t s_var_arena[WAKE_VAR_ARENA_BYTES];
static FILE *s_log;

static void log_fire(uint32_t ms, float prob)
{
    if (!s_active) return;
    if (!s_log) s_log = fopen("/rec/wake.log", "a");
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
            wakefront_reset();
            pos = audio_write_pos();
            consecutive = 0;
            deaf_until_us = 0;
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
            if (interp.Invoke() != kTfLiteOk) { ESP_LOGE(TAG, "Invoke failed"); continue; }
            /* uint8 output: prob = (q - zero_point) * scale, i.e. q/256. */
            float prob = (out->data.uint8[0] - KWS_WAKE_OUTPUT_ZERO_POINT) * KWS_WAKE_OUTPUT_SCALE;
            uint32_t ms = (uint32_t)((esp_timer_get_time() - t0) / 1000);

            consecutive = (prob >= WAKE_THRESHOLD) ? consecutive + 1 : 0;
            bool fired = false;
            if (consecutive >= WAKE_MIN_CONSECUTIVE && esp_timer_get_time() >= deaf_until_us) {
                deaf_until_us = esp_timer_get_time() + (int64_t)WAKE_REFRACTORY_MS * 1000;
                consecutive = 0;
                fired = true;
            }

            uint32_t now_ms = (uint32_t)(esp_timer_get_time() / 1000);
            /* Tuning trace: the peak probability of the last 2 s, so the serial
               log alone shows whether the model *hears* the phrase (peak near 1)
               or the threshold/consecutive gate is what stops it firing. */
            static float peak = 0; static uint32_t nsteps = 0, last_trace = 0;
            if (prob > peak) peak = prob;
            nsteps++;
            if (now_ms - last_trace >= 2000) {
                ESP_LOGI(TAG, "peak %.3f over %lu steps, %lu ms/step", (double)peak, (unsigned long)nsteps, (unsigned long)ms);
                peak = 0; nsteps = 0; last_trace = now_ms;
            }
            xSemaphoreTake(s_lock, portMAX_DELAY);
            s_st.prob = prob;
            s_st.infer_ms = ms;
            s_st.arena_used = interp.arena_used_bytes();
            if (fired) { s_st.fired_count++; s_st.fired_at_ms = now_ms; }
            wake_status_t copy = s_st;
            xSemaphoreGive(s_lock);

            if (fired) {
                ESP_LOGI(TAG, "wake! prob %.3f (%lu ms)", (double)prob, (unsigned long)ms);
                log_fire(now_ms, prob);
                ui_wake_refresh(&copy);       /* paint green before the tone blocks */
                beep_play();
            } else {
                ui_wake_refresh(&copy);
            }
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
    /* Priority 3, core 1 — same slot as the recogniser: below the LVGL task (4)
       so touch stays responsive. Only one of the two is ever active. */
    xTaskCreatePinnedToCore(wake_task, "wake", 16384, nullptr, 3, nullptr, 1);
}

extern "C" void wake_set_active(bool on)
{
    if (on) s_restart = true;
    s_active = on;
    if (!on && s_log) { fclose(s_log); s_log = nullptr; }
}

extern "C" void wake_get_status(wake_status_t *out)
{
    xSemaphoreTake(s_lock, portMAX_DELAY);
    *out = s_st;
    xSemaphoreGive(s_lock);
}
