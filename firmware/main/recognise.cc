#include "recognise.h"
#include <cassert>
#include <cstdio>
#include <cstring>
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
#include "mfcc.h"
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

static void log_fire(const char *word, float conf)
{
    if (!s_active) return;
    if (!s_log) s_log = fopen("/rec/recognise.log", "a");
    if (!s_log) return;
    fprintf(s_log, "[Log] %lld %s %.2f\n", esp_timer_get_time() / 1000, word, conf);
    fflush(s_log);
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

    static stream_t stream;
    static int16_t pcm[KWS_SAMPLE_RATE];
    static float feats[KWS_N_FRAMES][KWS_N_MFCC];
    static float probs[KWS_NUM_LABELS];

    for (;;) {
        if (!s_active) { vTaskDelay(pdMS_TO_TICKS(50)); stream_reset(&stream); continue; }
        vTaskDelay(pdMS_TO_TICKS(100));                    /* ~10 Hz cadence */
        uint32_t end = audio_write_pos();
        if (end < KWS_SAMPLE_RATE) continue;               /* need a full 1 s window (avoids ring underflow) */
        int64_t t0 = esp_timer_get_time();
        audio_read(end, pcm, KWS_SAMPLE_RATE);             /* always the freshest trailing 1 s */
        mfcc_compute(pcm, feats);                          /* one-shot; ponytail: streaming ring is a later optimisation */
        mfcc_quantize(feats, in->data.int8, KWS_MODEL_INPUT_SCALE, KWS_MODEL_INPUT_ZERO_POINT);
        if (interp.Invoke() != kTfLiteOk) { ESP_LOGE(TAG, "Invoke failed"); continue; }
        int best = 0;
        for (int i = 0; i < KWS_NUM_LABELS; i++) {
            probs[i] = (out->data.int8[i] - KWS_MODEL_OUTPUT_ZERO_POINT) * KWS_MODEL_OUTPUT_SCALE;
            if (probs[i] > probs[best]) best = i;
        }
        int fired = stream_push(&stream, probs);
        uint32_t ms = (uint32_t)((esp_timer_get_time() - t0) / 1000);

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
        ui_recognise_refresh(&copy);
    }
}

extern "C" void recognise_start(void)
{
    s_lock = xSemaphoreCreateMutex();
    s_arena = (uint8_t *)heap_caps_malloc(KWS_MODEL_ARENA_BYTES, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    assert(s_arena);
    /* Priority 3: BELOW the LVGL task (4, same core). Inference is heavy and
       best-effort; if it outranked LVGL it starved touch handling, so the
       Record button on the recognise screen never registered. */
    xTaskCreatePinnedToCore(recognise_task, "recognise", 16384, nullptr, 3, nullptr, 1);
}
extern "C" void recognise_set_active(bool on) { s_active = on; if (!on && s_log) { fclose(s_log); s_log = nullptr; } }
extern "C" void recognise_get_status(recognise_status_t *out) { xSemaphoreTake(s_lock, portMAX_DELAY); *out = s_st; xSemaphoreGive(s_lock); }
