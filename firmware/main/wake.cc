#include "wake.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include "arena.h"
#include "assist_gate.h"
#include "audio.h"
#include "beep.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "field.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "gen/wake_infer.h"
#include "gen/wake_model_config.h"
#include "gen/wake_model_data.h"
#include "infer_lock.h"
#include "nn_timers.h"
#include "nvs.h"
#include "recognise.h"
#include "record.h"
#include "storage.h"
#include "task.h"
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
static_assert(WAKE_INFER_INPUT_LEN == KWS_WAKE_FRAMES * KWS_WAKE_FEATURES,
              "generated wake input length must match the front-end block");

/* Is the interpreter in this build at all? It is the inference path when the
   generated one is switched off, and the reference the parity log needs when it
   is on. With the generated path alone — the shipped default — nothing here
   instantiates it and its 40 KB arena is never allocated, which is the point of
   generating the inference in the first place. */
#if !CONFIG_KWS_INFER_GENERATED || CONFIG_KWS_INFER_PARITY_LOG
#define KWS_WAKE_TFLM 1
#else
#define KWS_WAKE_TFLM 0
#endif

#if KWS_WAKE_TFLM
/* The streaming graph keeps its ring state in TFLM resource variables, which
   live in their own small arena (mirrors ESPHome's micro_wake_word). 1 KB fits
   the ~20 handles these models declare. */
#define WAKE_VAR_ARENA_BYTES 1024
#endif

/* Loop cadence. The front-end consumes 10 ms per row and the model 3 rows per
   Invoke, so 10 ms of sleep keeps us a fraction of a step behind live audio
   while leaving the LVGL task (priority 4, same core) plenty of slack. */
#define WAKE_POLL_MS 10

static const char *TAG = "wake";

static SemaphoreHandle_t s_lock;
static wake_status_t s_st;
static volatile bool s_active;
static volatile bool s_restart;      /* set on activation: reset model + front-end state */
#if CONFIG_KWS_INFER_GENERATED && CONFIG_KWS_INFER_PARITY_LOG
static bool s_parity_pending;        /* log both paths' output on the next step (wake_task only) */
#endif
#if KWS_WAKE_TFLM
static uint8_t *s_arena;
static uint8_t s_var_arena[WAKE_VAR_ARENA_BYTES];
#endif
static FILE *s_log;
static assist_gate_t s_gate;         /* assist mode only: when the recogniser may run */
static volatile bool s_listening;    /* last state pushed to recognise_set_active(); read by record.c */
static volatile bool s_inject;       /* console-injected fire, see wake_inject_fire() */
/* ponytail: s_field is written by the wake task (arm/disarm) and by
   wake_field_set() from the UI/console task, without a mutex — the only shared
   words are two bools that each task writes independently, so the worst race
   costs one take, never a torn read. Give it s_lock if it ever grows a field
   the two tasks must agree on. */
static field_state_t s_field;        /* assist mode only: opt-in capture of real interactions */
static uint32_t s_fire_ms;           /* ms-since-boot of the fire that armed s_field */
static float s_fire_prob;            /* wake probability at that fire */

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

/* The window has closed: hand the recorder the span to copy. NOTHING is written
   here — the copy and the FAT write happen on the record task, with the
   recogniser already off, so no I/O can lengthen a recognise step. */
static void post_field_take(uint32_t close_ms)
{
    field_take_t t = {};
    bool truncated = false;
    /* The gate pushes its deadline out on every fire, so the window is as long
       as it really stayed open — not ASSIST_WINDOW_MS. */
    t.window_ms = close_ms - s_fire_ms;
    if (!field_take_span(&s_field, t.window_ms, &t.start, &t.len, &truncated)) return;
    field_disarm(&s_field);
    t.fire_ms = s_fire_ms;
    t.wake_prob = s_fire_prob;
    if (!truncated) {
        /* The prediction may only name fires whose audio is in the WAV. A cut
           take cannot say which those are, so it carries none and travels as an
           unknown-prediction take — a case the QC pipeline already handles. */
        recognise_status_t rst;
        recognise_get_status(&rst);
        strlcpy(t.intent, rst.window_intent, sizeof t.intent);
        strlcpy(t.words, rst.window_words, sizeof t.words);
    }
    record_post_field_take(&t);
}

static void wake_task(void *)
{
#if KWS_WAKE_TFLM
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
    int8_t *feat = in->data.int8;      /* the front-end writes here, both paths read it */
#else
    static int8_t s_feat[WAKE_INFER_INPUT_LEN] __attribute__((aligned(16)));
    int8_t *feat = s_feat;
#endif

#if CONFIG_KWS_INFER_GENERATED
    bool use_generated = true;
    wake_infer_init();
    /* WAKE_INFER_SCRATCH_BYTES is this model's share of kws_infer_scratch,
       sized by a Python port of the esp_nn_get_*_scratch_size_esp32s3() family
       (kws_de/codegen.py). Ask the real ones, on the real chip: the query is
       generated from the very dims gen/wake_infer.c passes its kernels, so it
       cannot go stale behind a regenerated model the way the hand-copied dims
       it replaced could. If the port under-reserved, the kernels would scribble
       past the shared region into whatever the linker put above it, so this
       refuses to run rather than logging a number nobody diffs. */
    int scratch = wake_infer_scratch_query();
    if (scratch > WAKE_INFER_SCRATCH_BYTES) {
        ESP_LOGE(TAG, "esp-nn scratch %d B > the %u B gen/wake_infer.c reserved — regenerate with kws-codegen",
                 scratch, (unsigned)WAKE_INFER_SCRATCH_BYTES);
        use_generated = false;
#if !KWS_WAKE_TFLM
        ESP_LOGE(TAG, "no interpreter in this build (CONFIG_KWS_INFER_PARITY_LOG=n) — wake inference disabled");
        vTaskDelete(nullptr); return;
#endif
        ESP_LOGE(TAG, "falling back to the TFLite Micro interpreter");
    }
    ESP_LOGI(TAG, "inference: %s, %u B arena + %u B state + %u B shared scratch, "
                  "esp-nn scratch %d B queried / %u B reserved; TFLM %s; free internal %u",
             use_generated ? "generated (esp-nn)" : "TFLite Micro interpreter (generated path refused)",
             (unsigned)wake_infer_arena_bytes(), (unsigned)wake_infer_state_bytes(),
             (unsigned)KWS_INFER_SCRATCH_BYTES,
             scratch, (unsigned)WAKE_INFER_SCRATCH_BYTES,
             KWS_WAKE_TFLM ? "arena kept as the parity reference and fallback" : "not built in",
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
#else
    const bool use_generated = false;
    ESP_LOGI(TAG, "inference: TFLite Micro interpreter; free internal %u",
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
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
#if KWS_WAKE_TFLM
            interp.Reset();
            mrv->ResetAll();
#endif
#if CONFIG_KWS_INFER_GENERATED
            wake_infer_reset();
#if CONFIG_KWS_INFER_PARITY_LOG
            s_parity_pending = true;
#endif
#endif
            wakefront_reset();
            pos = audio_write_pos();
            consecutive = 0;
            deaf_until_us = 0;
            assist_gate_reset(&s_gate);
            if (s_listening) { s_listening = false; recognise_set_active(false); }
            field_disarm(&s_field);     /* a new session never inherits a pending take */
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
            wakefront_take(KWS_WAKE_FRAMES, feat);
            NN_TIMERS_RESET();
            uint8_t prob_q = 0;
            int64_t t_invoke = esp_timer_get_time();
#if CONFIG_KWS_INFER_GENERATED
            /* Held across the whole evaluation, the parity Invoke below
               included: both paths' kernels work in one esp-nn scratch region
               (infer_lock.h says why there is only one), and this task can
               preempt the recogniser mid-inference in assist mode. */
            kws_infer_lock();
#endif
            bool inferred = true;    /* checked after the unlock below, not here */
            if (use_generated) {
#if CONFIG_KWS_INFER_GENERATED
                wake_infer_step(feat, &prob_q);
#endif
            } else {
#if KWS_WAKE_TFLM
                inferred = interp.Invoke() == kTfLiteOk;
                if (inferred) prob_q = out->data.uint8[0];
#endif
            }
            int64_t invoke_us = esp_timer_get_time() - t_invoke;
#if CONFIG_KWS_INFER_GENERATED && CONFIG_KWS_INFER_PARITY_LOG
            if (use_generated && s_parity_pending) {
                /* Same input through both paths, once per mode entry: the
                   generated ring state has already advanced, so this compares
                   the interpreter's answer for THIS step only — it is
                   meaningful precisely because both paths were reset together
                   a moment ago. A mismatch is a real regression; the host
                   parity tests should have caught it. Outside invoke_us, but
                   inside this step's step_us, so the first trace window after a
                   mode entry reads a few ms high. */
                if (interp.Invoke() == kTfLiteOk)
                    ESP_LOGI(TAG, "parity: out byte generated %u, interpreter %u",
                             (unsigned)prob_q, (unsigned)out->data.uint8[0]);
                else
                    ESP_LOGE(TAG, "parity: interpreter Invoke failed");
                s_parity_pending = false;
            }
#endif
#if CONFIG_KWS_INFER_GENERATED
            kws_infer_unlock();
#endif
            if (!inferred) { ESP_LOGE(TAG, "inference failed"); continue; }
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
#if KWS_WAKE_TFLM
            s_st.arena_used = interp.arena_used_bytes();
#else
            /* No interpreter in this build: report what the generated path
               actually occupies instead of a field that would read 0. */
            s_st.arena_used = wake_infer_arena_bytes() + wake_infer_state_bytes();
#endif
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
                if (fired) {
                    assist_gate_on_wake(&s_gate, now_ms);
                    /* Latch the ARMING fire only, matching field_on_wake()'s own
                       "first fire wins": a later fire extends the window, but the
                       audio — and with it the file name, fire_ms and wake_prob —
                       stays anchored to the phrase the take begins with. */
                    if (!s_field.armed) { s_fire_ms = now_ms; s_fire_prob = prob; }
                    field_on_wake(&s_field, audio_write_pos());
                }
                bool listen = assist_gate_tick(&s_gate, now_ms);
                if (listen != s_listening) {
                    s_listening = listen;
                    /* Opening the window hands the recogniser its own deadline
                       so it stops even if this task stops being scheduled. */
                    if (listen) {
                        recognise_listen_for(ASSIST_WINDOW_MS);
                    } else {
                        recognise_set_active(false);
                        post_field_take(now_ms);
                    }
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
#if CONFIG_KWS_INFER_GENERATED
    kws_infer_lock_init();     /* before the task exists, so nothing races to create it */
#endif
#if KWS_WAKE_TFLM
    s_arena = arena_alloc(TAG, "wake", KWS_WAKE_ARENA_BYTES);
    assert(s_arena);
#endif
    /* Restore the opt-in capture toggle before the task exists, so it never
       observes an unrestored one. */
    nvs_handle_t h;
    if (nvs_open("kws", NVS_READWRITE, &h) == ESP_OK) {
        uint8_t on = 0;
        if (nvs_get_u8(h, "field", &on) != ESP_OK) on = 0;   /* off until turned on once */
        field_set_enabled(&s_field, on != 0);
        nvs_close(h);
    }
    /* Core 0, priority 3 — above the recogniser, which shares this core. In
       recognise mode only that one runs and in wake mode only this one, but
       assist mode has both live, which is why the inference lock exists. */
    task_spawn(TAG, wake_task, "wake", 16384, nullptr, 3, 0);
}

extern "C" void wake_set_active(bool on)
{
    if (on) s_restart = true;
    s_active = on;
    /* Leaving the mode closes the window too, so nothing is left waiting on a
       gate whose task has stopped ticking (see wake_window_open()). */
    if (!on) s_listening = false;
    if (!on && s_log) { fclose(s_log); s_log = nullptr; }
}

extern "C" bool wake_window_open(void) { return s_listening; }

extern "C" void wake_inject_fire(void) { s_inject = true; }

extern "C" void wake_get_status(wake_status_t *out)
{
    xSemaphoreTake(s_lock, portMAX_DELAY);
    *out = s_st;
    xSemaphoreGive(s_lock);
}

extern "C" bool wake_field_get(void) { return s_field.enabled; }

extern "C" void wake_field_set(bool on)
{
    field_set_enabled(&s_field, on);
    nvs_handle_t h;
    if (nvs_open("kws", NVS_READWRITE, &h) != ESP_OK) return;
    nvs_set_u8(h, "field", on ? 1 : 0);
    nvs_commit(h);
    nvs_close(h);
}
