#include "recognise.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include "arena.h"
#include "audio.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "gen/features_config.h"
#include "gen/labels.h"
#include "gen/model_config.h"
#include "gen/model_data.h"
#include "gen/test_vectors.h"
#include "mfcc.h"
#include "nn_timers.h"
#include "storage.h"
#include "stream.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "ui/ui.h"

_Static_assert(KWS_NUM_LABELS == KWS_MODEL_NUM_CLASSES,
               "label count (fwgen) must match model output classes (export)");

static const char *TAG = "recognise";

static SemaphoreHandle_t s_lock;
static recognise_status_t s_st;
static volatile bool s_active;
static uint8_t *s_arena;
static FILE *s_log;
static volatile int64_t s_off_at_us;   /* assist window deadline, 0 = run until told otherwise */

static void log_fire(const char *word, float conf)
{
    if (!s_active) return;
    if (!s_log) {
        char p[32];
        snprintf(p, sizeof p, "%s/recognise.log", storage_root());
        s_log = fopen(p, "a");
    }
    if (!s_log) return;
    fprintf(s_log, "[Log] %lld %s %.2f\n", esp_timer_get_time() / 1000, word, conf);
    fflush(s_log);
}

/* Duty accounting, logged once per 10 s of wall time.
 *
 * The always-on recognise mode is a measurement baseline, not a deployment:
 * the recogniser costs ~46 ms of CPU per 100 ms step, so running it
 * continuously is ~460 ms of inference per wall second. Assist mode gates it
 * behind a wake fire, and the gap between the two lines this prints is exactly
 * what the wake-gated design buys. Both modes emit the same line so they can be
 * compared straight out of the log.
 *
 * `busy_us` is the measured step cost, so this reports real CPU rather than an
 * estimate from a step count.
 */
static void duty_log(bool was_active, int64_t interval_us, int64_t busy_us)
{
    static int64_t win_us, act_us, cpu_us;
    win_us += interval_us;
    if (was_active) act_us += interval_us;
    cpu_us += busy_us;
    if (win_us < 10 * 1000 * 1000) return;
    ESP_LOGI(TAG, "KWS_DUTY mode %s: recogniser active %lu/1000 of wall, inference %lu ms per wall second",
             app_get_mode() == UI_MODE_ASSIST ? "assist" : "recognise",
             (unsigned long)(act_us * 1000 / win_us),
             (unsigned long)(cpu_us * 1000 / win_us));
    win_us = act_us = cpu_us = 0;
}

static void recognise_task(void *)
{
    static tflite::MicroMutableOpResolver<7> resolver;
    resolver.AddConv2D(); resolver.AddDepthwiseConv2D(); resolver.AddFullyConnected();
    resolver.AddMean(); resolver.AddSoftmax(); resolver.AddReshape(); resolver.AddAdd();
    const tflite::Model *model = tflite::GetModel(g_model);
    /* Explicit checks, not assert(): the side effects (AllocateTensors/Invoke)
       must run even if assertions are ever compiled out (e.g. a -O2/NDEBUG build). */
    if (model->version() != TFLITE_SCHEMA_VERSION) { ESP_LOGE(TAG, "bad model schema"); vTaskDelete(nullptr); return; }
    static tflite::MicroInterpreter interp(model, resolver, s_arena, KWS_MODEL_ARENA_BYTES);
    if (interp.AllocateTensors() != kTfLiteOk) { ESP_LOGE(TAG, "AllocateTensors failed"); vTaskDelete(nullptr); return; }
    ESP_LOGI(TAG, "arena used %u / %u", (unsigned)interp.arena_used_bytes(), (unsigned)KWS_MODEL_ARENA_BYTES);
    TfLiteTensor *in = interp.input(0), *out = interp.output(0);

    /* Numeric fingerprint of the inference path, once per boot: the golden MFCC
       vector through the real interpreter, printed as its 23 int8 outputs.
       Kernel-level build options (esp-nn's requantise rounding, for one) change
       device arithmetic that no host test can observe — the host runs neither
       esp-nn nor TFLM — so this line is the only way to tell a change that is
       bit-exact from one that quietly moved the model's outputs. Compare it
       across two boot logs. */
    mfcc_quantize(TV_MFCC, in->data.int8, KWS_MODEL_INPUT_SCALE, KWS_MODEL_INPUT_ZERO_POINT);
    if (interp.Invoke() == kTfLiteOk) {
        char line[6 * KWS_NUM_LABELS + 1];      /* "-128," is 5 chars plus the terminator */
        int n = 0;
        for (int i = 0; i < KWS_NUM_LABELS && n < (int)sizeof line - 1; i++)
            n += snprintf(line + n, sizeof line - (size_t)n, "%d,", out->data.int8[i]);
        ESP_LOGI(TAG, "selftest int8 out: %s", line);
    }

    static stream_t stream;
    static mfcc_state_t mstate;                            /* persistent ring of the last 49 log-mel frames */
    static int16_t frame[KWS_WIN];
    static float feats[KWS_N_FRAMES][KWS_N_MFCC];
    static float probs[KWS_NUM_LABELS];
    static uint32_t frame_start = 0;                       /* absolute sample index of the next frame to push */
    static bool primed = false;
    static uint32_t steps = 0;

    /* Duty accounting closes the *previous* iteration at the top of this one, so
       the several `continue`s below cannot skip it. */
    int64_t prev_us = 0, step_us = 0;
    bool prev_active = false;

    for (;;) {
        int64_t now_us = esp_timer_get_time();
        if (prev_us) duty_log(prev_active, now_us - prev_us, step_us);
        prev_us = now_us;
        prev_active = s_active;
        step_us = 0;
        /* Self-imposed window deadline: see recognise_listen_for(). */
        if (s_active && s_off_at_us && now_us >= s_off_at_us) recognise_set_active(false);

        if (!s_active) { vTaskDelay(pdMS_TO_TICKS(50)); stream_reset(&stream); primed = false; continue; }
        if (!primed) {                                     /* (re)entering: window = the last second up to now */
            mfcc_init(&mstate);
            uint32_t now = audio_write_pos();
            frame_start = now > KWS_SAMPLE_RATE ? now - KWS_SAMPLE_RATE : 0;
            primed = true;
        }
        vTaskDelay(pdMS_TO_TICKS(100));                    /* ~10 Hz cadence */
        int64_t t0 = esp_timer_get_time();
        /* Streaming front-end: push only the frames that arrived since the last
           step (~5 per 100 ms) instead of recomputing all 49 from a 1 s buffer —
           a ~10x cut in front-end work. Frame t covers [start, start+KWS_WIN)
           with start advancing by KWS_HOP, the same layout as mfcc_compute(), so
           the features are bit-identical (the host test checks streaming == one-shot). */
        int pushed = 0;
        int64_t t_fe = esp_timer_get_time();
        while (audio_write_pos() >= frame_start + KWS_WIN && pushed < KWS_N_FRAMES) {
            audio_read(frame_start + KWS_WIN, frame, KWS_WIN);
            mfcc_push_frame(&mstate, frame);
            frame_start += KWS_HOP;
            pushed++;
        }
        int64_t fe_us = esp_timer_get_time() - t_fe;
        if (audio_write_pos() > frame_start + 2 * KWS_SAMPLE_RATE) { primed = false; continue; }  /* stalled: resync */
        if (mstate.count < KWS_N_FRAMES) continue;         /* not a full 1 s of frames yet */
        mfcc_finish(&mstate, feats);
        mfcc_quantize(feats, in->data.int8, KWS_MODEL_INPUT_SCALE, KWS_MODEL_INPUT_ZERO_POINT);
        NN_TIMERS_RESET();
        int64_t t_invoke = esp_timer_get_time();
        if (interp.Invoke() != kTfLiteOk) { ESP_LOGE(TAG, "Invoke failed"); continue; }
        int64_t invoke_us = esp_timer_get_time() - t_invoke;
        int best = 0;
        for (int i = 0; i < KWS_NUM_LABELS; i++) {
            probs[i] = (out->data.int8[i] - KWS_MODEL_OUTPUT_ZERO_POINT) * KWS_MODEL_OUTPUT_SCALE;
            if (probs[i] > probs[best]) best = i;
        }
        int fired = stream_push(&stream, probs);
        step_us = esp_timer_get_time() - t0;
        uint32_t ms = (uint32_t)(step_us / 1000);
        if ((++steps % 50) == 0)                         /* ~every 5 s: front-end + inference cost */
            ESP_LOGI(TAG, "step %lu ms (front-end %lld us over %d new frames, invoke %lld us: " NN_TIMERS_FMT ")",
                     (unsigned long)ms, pushed ? fe_us / pushed : (int64_t)0, pushed,
                     invoke_us, NN_TIMERS_ARGS(invoke_us));

        xSemaphoreTake(s_lock, portMAX_DELAY);
        s_st.infer_ms = ms; s_st.arena_used = interp.arena_used_bytes();
        /* Test view: show the live top-1 prediction + its confidence every step, so
           speaking a word immediately shows what the model hears (not only threshold
           fires). fired_count still tracks how often the detector actually triggered. */
        strlcpy(s_st.word, KWS_LABELS[best], sizeof s_st.word);
        s_st.conf = probs[best];
        if (fired >= 0) s_st.fired_count++;
        recognise_status_t copy = s_st;
        xSemaphoreGive(s_lock);
        if (fired >= 0) { ESP_LOGI(TAG, "fired %s %.2f (%lu ms)", KWS_LABELS[fired], (double)probs[fired], (unsigned long)ms); log_fire(KWS_LABELS[fired], probs[fired]); }
        /* Only when the recognise screen is actually on display. In assist mode
           the wake task owns the screen (ui_assist_refresh) and this screen's
           LVGL objects were never created — painting them took the display lock
           and never gave it back, which stalled both model tasks. */
        if (app_get_mode() == UI_MODE_RECOGNISE) ui_recognise_refresh(&copy);
    }
}

extern "C" void recognise_start(void)
{
    s_lock = xSemaphoreCreateMutex();
    s_arena = arena_alloc(TAG, "command", KWS_MODEL_ARENA_BYTES);
    assert(s_arena);
    /* Core 0, priority 2. LVGL (priority 4, 5 ms tick) and the audio task own
       core 1, and LVGL preempted a 40 ms Invoke several times over.
       Priority 2 puts this BELOW the wake task (3), which shares core 0 in
       assist mode: the wake model is always on and costs 1.7 ms per 30 ms,
       while this one is a best-effort burst costing 46 ms per step. At equal
       priority the burst starved the detector. */
    xTaskCreatePinnedToCore(recognise_task, "recognise", 16384, nullptr, 2, nullptr, 0);
}
extern "C" void recognise_set_active(bool on)
{
    s_active = on;
    if (!on) { s_off_at_us = 0; if (s_log) { fclose(s_log); s_log = nullptr; } }
}
extern "C" void recognise_listen_for(uint32_t ms)
{
    s_off_at_us = esp_timer_get_time() + (int64_t)ms * 1000;
    s_active = true;
}
extern "C" void recognise_get_status(recognise_status_t *out) { xSemaphoreTake(s_lock, portMAX_DELAY); *out = s_st; xSemaphoreGive(s_lock); }
