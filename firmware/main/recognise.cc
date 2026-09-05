#include "recognise.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include "arena.h"
#include "assist_gate.h"
#include "audio.h"
#include "beep.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "gen/command_infer.h"
#include "gen/features_config.h"
#include "gen/labels.h"
#include "gen/model_config.h"
#include "gen/model_data.h"
#include "gen/test_vectors.h"
#include "infer_lock.h"
#include "mfcc.h"
#include "nn_timers.h"
#include "storage.h"
#include "stream.h"
#include "task.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "ui/ui.h"
#include "wake.h"

_Static_assert(KWS_NUM_LABELS == KWS_MODEL_NUM_CLASSES,
               "label count (fwgen) must match model output classes (export)");
static_assert(COMMAND_INFER_INPUT_LEN == KWS_N_FRAMES * KWS_N_MFCC,
              "generated command input length must match the feature window");
static_assert(COMMAND_INFER_OUTPUT_LEN == KWS_NUM_LABELS,
              "generated command output length must match the label count");
static_assert(COMMAND_INFER_STATE_BYTES == 0,
              "the command model is stateless; nothing to carry between steps");

/* Is the interpreter in this build at all? It is the inference path when the
   generated one is switched off, and the reference the parity log needs when it
   is on. With the generated path alone — the shipped default — nothing here
   instantiates it and its arena is never allocated. Same switch wake.cc uses. */
#if !CONFIG_KWS_INFER_GENERATED || CONFIG_KWS_INFER_PARITY_LOG
#define KWS_CMD_TFLM 1
#else
#define KWS_CMD_TFLM 0
#endif

/* Where gen/command_infer.c's 51 KB arena was linked (Kconfig choice
   KWS_INFER_COMMAND_ARENA; see CMakeLists.txt, which is what actually defines
   COMMAND_INFER_ARENA_ATTR). PSRAM by default: internal SRAM is what the task
   stacks need and cannot be moved out of. */
#ifdef CONFIG_KWS_INFER_COMMAND_ARENA_PSRAM
#define KWS_CMD_ARENA_WHERE "static, PSRAM"
#else
#define KWS_CMD_ARENA_WHERE "static, internal RAM"
#endif

static const char *TAG = "recognise";

static SemaphoreHandle_t s_lock;
static recognise_status_t s_st;
static volatile bool s_active;
#if KWS_CMD_TFLM
static uint8_t *s_arena;
#endif
#if CONFIG_KWS_INFER_GENERATED && CONFIG_KWS_INFER_PARITY_LOG
static bool s_parity_pending;        /* log both paths' output on the next step (recognise_task only) */
#endif
static FILE *s_log;
static volatile int64_t s_off_at_us;   /* assist window deadline, 0 = run until told otherwise */
static volatile int64_t s_win_open_us; /* when the current window opened (recognise_listen_for); ASSIST_WAKE_TAIL_MS anchor */
static volatile bool s_cmd_fired;      /* assist mode: a command fired in this window, tone still owed */

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
    /* The model's 23 int8 outputs, whichever path produced them. 16-byte
       aligned because esp-nn's S3 kernels write the softmax result straight
       into it (gen/command_infer.h says so). */
    static int8_t logits[COMMAND_INFER_OUTPUT_LEN] __attribute__((aligned(16)));
#if KWS_CMD_TFLM
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
    int8_t *feat = in->data.int8;      /* the front-end writes here, both paths read it */
#else
    static int8_t s_feat[COMMAND_INFER_INPUT_LEN] __attribute__((aligned(16)));
    int8_t *feat = s_feat;
#endif

#if CONFIG_KWS_INFER_GENERATED
    bool use_generated = true;
    command_infer_init();
    /* COMMAND_INFER_SCRATCH_BYTES is this model's share of kws_infer_scratch —
       the maximum over every op, sized by a Python port of the
       esp_nn_get_*_scratch_size_esp32s3() family (kws_de/codegen.py). Ask the
       real functions, on the real chip: the query is generated alongside the
       kernels from the same dims, so it covers every op that takes scratch
       (both the 3x3 depthwise convs and the convolutions — querying only the
       widest of today's would let a future esp-nn whose other formula grew slip
       past the guard) and cannot describe a model that is no longer here. If
       the port under-reserved, the kernels would scribble past the shared
       region, so this refuses to run rather than logging a number nobody
       diffs. */
    int scratch = command_infer_scratch_query();
    if (scratch > COMMAND_INFER_SCRATCH_BYTES) {
        ESP_LOGE(TAG, "esp-nn scratch %d B > the %u B gen/command_infer.c reserved — regenerate with kws-codegen",
                 scratch, (unsigned)COMMAND_INFER_SCRATCH_BYTES);
        use_generated = false;
#if !KWS_CMD_TFLM
        ESP_LOGE(TAG, "no interpreter in this build (CONFIG_KWS_INFER_PARITY_LOG=n) — recognition disabled");
        vTaskDelete(nullptr); return;
#endif
        ESP_LOGE(TAG, "falling back to the TFLite Micro interpreter");
    }
    /* The two models keep separate arenas but share one esp-nn scratch region,
       because esp-nn's scratch pointers are file-static globals and separate
       regions would be handed to the wrong model's kernels. In assist mode the
       wake task (priority 3) can preempt this one mid-inference, so the two
       evaluations are serialised on kws_infer_lock() — see infer_lock.h. */
    ESP_LOGI(TAG, "inference: %s, %u B arena (%s) + %u B state + %u B shared scratch, "
                  "esp-nn scratch %d B queried / %u B reserved; TFLM %s; free internal %u",
             use_generated ? "generated (esp-nn)" : "TFLite Micro interpreter (generated path refused)",
             (unsigned)command_infer_arena_bytes(), KWS_CMD_ARENA_WHERE,
             (unsigned)command_infer_state_bytes(), (unsigned)KWS_INFER_SCRATCH_BYTES,
             scratch, (unsigned)COMMAND_INFER_SCRATCH_BYTES,
             KWS_CMD_TFLM ? "arena kept as the parity reference and fallback" : "not built in",
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
#else
    const bool use_generated = false;
    ESP_LOGI(TAG, "inference: TFLite Micro interpreter; free internal %u",
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
#endif

    /* One model evaluation on whichever path is active, so the selftest below
       and the step loop cannot drift apart. `feat` is the input buffer both
       paths read; `logits` receives the 23 output bytes. */
    auto evaluate = [&]() -> bool {
        bool ok = false;
#if CONFIG_KWS_INFER_GENERATED
        /* Both models' kernels work in one esp-nn scratch region (infer_lock.h
           says why there is only one), and the wake task — priority 3, this
           core — can preempt this one mid-inference in assist mode. */
        kws_infer_lock();
        if (use_generated) { command_infer(feat, logits); ok = true; }
#endif
#if KWS_CMD_TFLM
        if (!ok && interp.Invoke() == kTfLiteOk) {
            memcpy(logits, out->data.int8, sizeof logits);
            ok = true;
        }
#endif
#if CONFIG_KWS_INFER_GENERATED
        kws_infer_unlock();
#endif
        return ok;
    };

    /* Numeric fingerprint of the inference path, once per boot: the golden MFCC
       vector through the active path, printed as its 23 int8 outputs.
       Kernel-level build options (esp-nn's requantise rounding, for one) change
       device arithmetic that no host test can observe — the host runs neither
       esp-nn's S3 kernels nor TFLM — so this line is the only way to tell a
       change that is bit-exact from one that quietly moved the model's outputs.
       Compare it across two boot logs. */
    mfcc_quantize(TV_MFCC, feat, KWS_MODEL_INPUT_SCALE, KWS_MODEL_INPUT_ZERO_POINT);
    if (evaluate()) {
        char line[6 * KWS_NUM_LABELS + 1];      /* "-128," is 5 chars plus the terminator */
        int n = 0;
        for (int i = 0; i < KWS_NUM_LABELS && n < (int)sizeof line - 1; i++)
            n += snprintf(line + n, sizeof line - (size_t)n, "%d,", logits[i]);
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
#if CONFIG_KWS_INFER_GENERATED && CONFIG_KWS_INFER_PARITY_LOG
            s_parity_pending = true;                       /* one parity line per mode entry */
#endif
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
        mfcc_quantize(feats, feat, KWS_MODEL_INPUT_SCALE, KWS_MODEL_INPUT_ZERO_POINT);
        NN_TIMERS_RESET();
        int64_t t_invoke = esp_timer_get_time();
        if (!evaluate()) { ESP_LOGE(TAG, "inference failed"); continue; }
        int64_t invoke_us = esp_timer_get_time() - t_invoke;
#if CONFIG_KWS_INFER_GENERATED && CONFIG_KWS_INFER_PARITY_LOG
        /* Once per mode entry, on the same live features: run the interpreter
           too and say whether the two paths still agree byte for byte. Costs
           one extra Invoke on that one step, so the trace window it lands in
           reads high. */
        if (s_parity_pending && use_generated) {
            s_parity_pending = false;
            kws_infer_lock();      /* TFLM's kernels move the same esp-nn globals */
            TfLiteStatus st = interp.Invoke();
            kws_infer_unlock();
            if (st != kTfLiteOk) {
                ESP_LOGE(TAG, "parity: interpreter Invoke failed");
            } else {
                int diff = 0;
                for (int i = 0; i < KWS_NUM_LABELS; i++)
                    if (logits[i] != out->data.int8[i]) diff++;
                ESP_LOGI(TAG, "parity: %d/%d output bytes differ", diff, KWS_NUM_LABELS);
            }
        }
#endif
        int best = 0;
        for (int i = 0; i < KWS_NUM_LABELS; i++) {
            probs[i] = (logits[i] - KWS_MODEL_OUTPUT_ZERO_POINT) * KWS_MODEL_OUTPUT_SCALE;
            if (probs[i] > probs[best]) best = i;
        }
        int fired = stream_push(&stream, probs);
        if (fired >= 0 && s_off_at_us) {
            /* Windowed (assist) session only — s_off_at_us is 0 outside one.
               Dropped as if nothing fired: it must not reach window_intent/
               window_words (field capture's device prediction) or the
               confirmation tone below, and the run it belonged to stays
               marked fired (see assist_gate.h) so a genuine later command on
               a different label still gets its own chance to fire. */
            int64_t since_open_ms = (esp_timer_get_time() - s_win_open_us) / 1000;
            if (assist_gate_in_wake_tail(since_open_ms)) {
                ESP_LOGI(TAG, "assist: dropped tail fire %s %.2f, %lld ms into the window (< %d)",
                         KWS_LABELS[fired], (double)probs[fired], (long long)since_open_ms,
                         ASSIST_WAKE_TAIL_MS);
                fired = -1;
            }
        }
        step_us = esp_timer_get_time() - t0;
        uint32_t ms = (uint32_t)(step_us / 1000);
        if ((++steps % 50) == 0)                         /* ~every 5 s: front-end + inference cost */
            /* The stack headroom is in the trace because RECOGNISE_STACK was
               cut to fit internal RAM (see below): the number stays checkable
               on any build instead of being a one-off measurement. */
            ESP_LOGI(TAG, "step %lu ms (front-end %lld us over %d new frames, invoke %lld us: " NN_TIMERS_FMT
                          ", stack %u B free)",
                     (unsigned long)ms, pushed ? fe_us / pushed : (int64_t)0, pushed,
                     invoke_us, NN_TIMERS_ARGS(invoke_us),
                     (unsigned)uxTaskGetStackHighWaterMark(nullptr));

        xSemaphoreTake(s_lock, portMAX_DELAY);
        s_st.infer_ms = ms;
#if KWS_CMD_TFLM
        s_st.arena_used = interp.arena_used_bytes();
#else
        s_st.arena_used = command_infer_arena_bytes();
#endif
        /* Test view: show the live top-1 prediction + its confidence every step, so
           speaking a word immediately shows what the model hears (not only threshold
           fires). fired_count still tracks how often the detector actually triggered. */
        strlcpy(s_st.word, KWS_LABELS[best], sizeof s_st.word);
        s_st.conf = probs[best];
        if (fired >= 0) {
            s_st.fired_count++;
            /* Field capture's device prediction: what THIS window fired, in
               order, kept next to the audio so the workstation can score the
               deployed model against Whisper's label. Never a label itself.

               Whole entries only: snprintf would truncate safely but mid-value,
               and a tail like "|an:0.8" reads as a confidence that was never
               measured. A dropped entry is honest, a mangled one is not. */
            const char *w = KWS_LABELS[fired];
            char e[48];
            size_t n = strlen(s_st.window_intent);
            int k = snprintf(e, sizeof e, "%s%s", n ? " " : "", w);
            if (k > 0 && (size_t)k < sizeof e && n + (size_t)k < sizeof s_st.window_intent)
                memcpy(s_st.window_intent + n, e, (size_t)k + 1);
            n = strlen(s_st.window_words);
            k = snprintf(e, sizeof e, "%s%s:%.2f", n ? "|" : "", w, (double)probs[fired]);
            if (k > 0 && (size_t)k < sizeof e && n + (size_t)k < sizeof s_st.window_words)
                memcpy(s_st.window_words + n, e, (size_t)k + 1);
        }
        recognise_status_t copy = s_st;
        xSemaphoreGive(s_lock);
        if (fired >= 0) { ESP_LOGI(TAG, "fired %s %.2f (%lu ms)", KWS_LABELS[fired], (double)probs[fired], (unsigned long)ms); log_fire(KWS_LABELS[fired], probs[fired]); }
        /* Confirmation tone. A fire on _unknown_ is the model saying it heard
           speech it has no command for, so it stays silent (stream_is_command).
           In assist mode the tone is only *owed* here and played by the wake
           task when the window closes — the recogniser's own deadline can
           expire while the window is still open (a second wake fire extends the
           gate without re-arming the deadline), and a tone inside the window
           would be captured by the microphone and end up in a field-capture
           take. Outside assist there is no window, so it plays straight away —
           unless field capture is toggled on anyway (left on from testing in
           this very mode): the toggle outlives a mode switch, and this tone's
           tail sits in the ring right where the pre-roll of the next Assistent
           take reaches back to, same leak as wake.cc's own tone, muted the same
           way. beep_double() only posts to the beep task either way. */
        if (stream_is_command(fired)) {
            if (app_get_mode() == UI_MODE_ASSIST) s_cmd_fired = true;
            else if (!wake_field_get()) beep_double();
        }
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
#if CONFIG_KWS_INFER_GENERATED
    kws_infer_lock_init();     /* before the task exists, so nothing races to create it */
#endif
#if KWS_CMD_TFLM
    s_arena = arena_alloc(TAG, "command", KWS_MODEL_ARENA_BYTES);
    assert(s_arena);
#endif
    /* Core 0, priority 2. LVGL (priority 4, 5 ms tick) and the audio task own
       core 1, and LVGL preempted a 40 ms Invoke several times over.
       Priority 2 puts this BELOW the wake task (3), which shares core 0 in
       assist mode: the wake model is always on and costs 1.7 ms per 30 ms,
       while this one is a best-effort burst costing 46 ms per step. At equal
       priority the burst starved the detector.

       10 KB of stack, down from the interpreter era's 16 KB, for both paths.
       A task stack must come out of internal RAM and cannot be moved to PSRAM
       (CONFIG_SPIRAM_ALLOW_STACK_EXTERNAL_MEMORY is off), and this task is only
       the first of four app_main creates — record (8 KB) and console (4 KB)
       follow it and were the ones that silently failed when the generated
       arena took 51 KB of internal SRAM. The arena lives in PSRAM by default
       now, which is the real fix, but the stack stays at its measured size
       rather than its historical one: the high-water mark is 6,436 B on the
       generated path and 6,368 B on the interpreter path (both with the
       recognise screen live), so ~3.8 KB is left for what that measurement
       does not reach — a fire opening recognise.log through FATFS. The
       "stack N B free" field in the step trace above is that high-water mark,
       so the size stays checkable on any build. */
#define RECOGNISE_STACK 10240
    task_spawn(TAG, recognise_task, "recognise", RECOGNISE_STACK, nullptr, 2, 0);
}
extern "C" void recognise_set_active(bool on)
{
    s_active = on;
    if (!on) { s_off_at_us = 0; if (s_log) { fclose(s_log); s_log = nullptr; } }
}
extern "C" void recognise_listen_for(uint32_t ms)
{
    /* A new window starts with an empty prediction: field capture reads these
       back when the window closes and must not see the previous one's fires. */
    xSemaphoreTake(s_lock, portMAX_DELAY);
    s_st.window_intent[0] = 0;
    s_st.window_words[0] = 0;
    xSemaphoreGive(s_lock);
    s_cmd_fired = false;      /* a new window never inherits the last one's owed tone */
    int64_t now_us = esp_timer_get_time();
    s_win_open_us = now_us;   /* ASSIST_WAKE_TAIL_MS is measured from here, not from s_off_at_us */
    s_off_at_us = now_us + (int64_t)ms * 1000;
    s_active = true;
}
extern "C" bool recognise_take_command_fired(void)
{
    bool v = s_cmd_fired;
    s_cmd_fired = false;
    return v;
}
extern "C" void recognise_get_status(recognise_status_t *out) { xSemaphoreTake(s_lock, portMAX_DELAY); *out = s_st; xSemaphoreGive(s_lock); }
