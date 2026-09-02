#include "record.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include "audio.h"
#include "storage.h"
#include "vad.h"
#include "wav.h"
#include "prompts.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "nvs.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "ui/ui.h"

static const char *TAG = "record";
#define PREROLL_SAMPLES (KWS_SAMPLE_RATE * 300 / 1000)
#define NO_SPEECH_MS 8000
#define MIN_SPEECH_MS 200     /* below this much total time above threshold, a closed take is a
                                  false start (breath/click) — discard and keep listening. */
#define HOLD_MS 700
/* Reads per prompt before advancing: prompt_takes_per_prompt() (2 normally, for
   wrong-read review; 1 for PROMPT_WAKE — real "Hey Bus" positives, not doubled reads). */
#define GETREADY_MS 800              /* "get ready" beat before the first take of a word */
#define BETWEEN_TAKES_MS 500         /* short pause before the second read */

static int s_take_idx;               /* 0-based take of the current prompt */
static int s_saved_takes;            /* takes saved since the last REC_CMD_START_SESSION */

static QueueHandle_t s_cmds;
static SemaphoreHandle_t s_lock;
static record_status_t s_st;
static prompt_session_t s_prompts;
static uint32_t s_speaker;
static int s_paused = 1;
static int16_t *s_take;                       /* PSRAM, 6 s + pre-roll */
#define TAKE_MAX (KWS_SAMPLE_RATE * 6 + PREROLL_SAMPLES)

static void status_set(record_phase_t ph)
{
    xSemaphoreTake(s_lock, portMAX_DELAY);
    s_st.phase = ph;
    s_st.set = s_prompts.set; s_st.seed = s_prompts.seed;
    s_st.index = s_prompts.index; s_st.count = s_prompts.count;
    s_st.take = s_take_idx + 1; s_st.takes = prompt_takes_per_prompt(s_prompts.set);
    s_st.saved_takes = s_saved_takes;
    strlcpy(s_st.prompt, prompt_text(&s_prompts), sizeof s_st.prompt);
    snprintf(s_st.speaker, sizeof s_st.speaker, "spk%02lu", (unsigned long)s_speaker);
    record_status_t copy = s_st;
    xSemaphoreGive(s_lock);
    ui_record_refresh(&copy);
}

static void nvs_load(void)
{
    nvs_handle_t h;
    ESP_ERROR_CHECK(nvs_open("kws", NVS_READWRITE, &h));
    if (nvs_get_u32(h, "speaker", &s_speaker) != ESP_OK) { s_speaker = 1; nvs_set_u32(h, "speaker", 1); }
    nvs_close(h);
}

static void nvs_bump_speaker(void)
{
    nvs_handle_t h;
    ESP_ERROR_CHECK(nvs_open("kws", NVS_READWRITE, &h));
    s_speaker++;
    nvs_set_u32(h, "speaker", s_speaker);
    nvs_commit(h);
    nvs_close(h);
}

/* /rec/spk03/licht/001.wav | /rec/spk03/hey-bus/001.wav | /rec/spk03/_phrase_/licht-hinten-an_001.wav | /rec/spk03/_neg_/... */
static int next_path(char *out, size_t n)
{
    char dir[64];
    int slugdir = s_prompts.set == PROMPT_WORDS || s_prompts.set == PROMPT_WAKE;
    const char *sub = slugdir ? prompt_slug(&s_prompts)
                    : s_prompts.set == PROMPT_SENTENCES ? "_phrase_" : "_neg_";
    snprintf(dir, sizeof dir, "/rec/%s", s_st.speaker);            mkdir(dir, 0777);
    snprintf(dir, sizeof dir, "/rec/%s/%s", s_st.speaker, sub);    mkdir(dir, 0777);
    for (int i = 1; i < 1000; i++) {
        struct stat st;
        if (slugdir) snprintf(out, n, "%s/%03d.wav", dir, i);
        else snprintf(out, n, "%s/%s_%03d.wav", dir, prompt_slug(&s_prompts), i);
        if (stat(out, &st) != 0) return 0;
    }
    return -1;
}

static void append_session_csv(const char *path, uint32_t ms, float peak_dbfs)
{
    char csv[64];
    snprintf(csv, sizeof csv, "/rec/%s/session.csv", s_st.speaker);
    struct stat st; int fresh = stat(csv, &st) != 0;
    FILE *f = fopen(csv, "a");
    if (!f) { ESP_LOGE(TAG, "csv open failed"); return; }
    if (fresh) fputs("prompt,file,ms,peak_dbfs,set,seed,ts\n", f);
    fprintf(f, "\"%s\",%s,%lu,%.1f,%s,%lu,%lld\n", prompt_text(&s_prompts), path + 5 /* strip /rec/ */,
            (unsigned long)ms, peak_dbfs, prompt_set_name(s_prompts.set), (unsigned long)s_prompts.seed,
            esp_timer_get_time() / 1000);
    fclose(f);
}

static int save_take(uint32_t n_samples, float peak_dbfs)
{
    char path[128];
    if (next_path(path, sizeof path) != 0) return -1;
    FILE *f = fopen(path, "wb");
    if (!f) { ESP_LOGE(TAG, "open %s failed", path); return -1; }
    uint8_t hdr[WAV_HEADER_BYTES];
    wav_write_header(hdr, n_samples, KWS_SAMPLE_RATE);
    fwrite(hdr, 1, sizeof hdr, f);
    fwrite(s_take, sizeof(int16_t), n_samples, f);
    fclose(f);
    append_session_csv(path, n_samples * 1000 / KWS_SAMPLE_RATE, peak_dbfs);
    ESP_LOGI(TAG, "saved %s (%lu samples)", path, (unsigned long)n_samples);
    return 0;
}

/* Returns: 0 saved, 1 redo (clipped/timeout/full), -1 command interrupted (cmd in *cmd) */
static int capture_one(record_cmd_t *cmd)
{
    if (storage_free_bytes() < STORAGE_MIN_FREE_BYTES) { status_set(REC_FULL); return 1; }
    vad_t vad; vad_reset(&vad, (int)(prompt_hangover_ms(s_prompts.set) / 20));
    int16_t frame[KWS_HOP];
    uint32_t cap = prompt_cap_ms(s_prompts.set) * (KWS_SAMPLE_RATE / 1000);
    uint32_t cursor = audio_write_pos();
    uint32_t speech_start = 0, n = 0, idle_frames = 0;
    int peak = 0, capturing = 0;
    status_set(REC_LISTENING);
    for (;;) {
        if (xQueueReceive(s_cmds, cmd, 0) == pdTRUE) return -1;
        while (audio_write_pos() < cursor + KWS_HOP) vTaskDelay(pdMS_TO_TICKS(5));
        audio_read(cursor + KWS_HOP, frame, KWS_HOP);
        cursor += KWS_HOP;
        int active = vad_push(&vad, frame, KWS_HOP);
        for (int i = 0; i < KWS_HOP; i++) { int a = frame[i] < 0 ? -frame[i] : frame[i]; if (a > peak) peak = a; }
        if ((cursor / KWS_HOP) % 5 == 0) {        /* level bar every 100 ms */
            xSemaphoreTake(s_lock, portMAX_DELAY);
            s_st.level_dbfs = 20.f * log10f((vad.noise > 1 ? vad.noise : 1) / 32768.f);
            xSemaphoreGive(s_lock);
            ui_record_refresh(&s_st);
        }
        if (!capturing) {
            if (active) {
                capturing = 1; peak = 0;
                speech_start = cursor - KWS_HOP - PREROLL_SAMPLES;
                n = PREROLL_SAMPLES + KWS_HOP;
                audio_read(cursor, s_take, n);
                status_set(REC_CAPTURING);
            } else if (++idle_frames * 20 >= NO_SPEECH_MS) { status_set(REC_TIMEOUT); return 1; }
            continue;
        }
        memcpy(s_take + n, frame, sizeof frame); n += KWS_HOP;
        if (peak >= 32767) { status_set(REC_CLIPPED); return 1; }
        if (!active || n >= cap + PREROLL_SAMPLES) {
            if (vad.speech_total * 20 < MIN_SPEECH_MS) {
                /* False start (breath/click before speech): discard and keep listening
                   in this same call, so idle_frames — and the NO_SPEECH_MS timeout —
                   keeps running from the original start instead of restarting. */
                capturing = 0; n = 0; peak = 0;
                vad_reset(&vad, (int)(prompt_hangover_ms(s_prompts.set) / 20));
                status_set(REC_LISTENING);
                continue;
            }
            break;
        }
    }
    (void)speech_start;
    float peak_dbfs = 20.f * log10f((peak > 0 ? peak : 1) / 32768.f);
    if (save_take(n, peak_dbfs) != 0) { status_set(REC_FULL); return 1; }
    s_saved_takes++;
    status_set(REC_SAVED);
    vTaskDelay(pdMS_TO_TICKS(HOLD_MS));
    return 0;
}

static void record_task(void *arg)
{
    (void)arg;
    record_cmd_t cmd;
    for (;;) {
        if (s_paused) { xQueueReceive(s_cmds, &cmd, portMAX_DELAY); }
        else {
            /* "get ready" beat before each read — a longer one for the first take
               of a word, a short one before the second read. Paces the session so
               it no longer flies past; the prompt label stays on screen throughout. */
            status_set(REC_GETREADY);
            vTaskDelay(pdMS_TO_TICKS(s_take_idx == 0 ? GETREADY_MS : BETWEEN_TAKES_MS));
            int r = capture_one(&cmd);
            if (r == 0) {                                 /* take saved */
                if (++s_take_idx >= prompt_takes_per_prompt(s_prompts.set)) {  /* all reads done → next prompt */
                    s_take_idx = 0;
                    if (!prompt_advance(&s_prompts)) {
                        /* Sentences done → auto-chain into negatives; negatives/wake done → session over. */
                        if (s_prompts.set == PROMPT_SENTENCES) {
                            prompt_session_init(&s_prompts, PROMPT_NEGS, (uint32_t)esp_timer_get_time());
                        } else {
                            status_set(REC_SESSION_DONE);
                            s_paused = 1;
                        }
                    }
                }
                continue;
            }
            if (r == 1) { vTaskDelay(pdMS_TO_TICKS(HOLD_MS)); continue; }  /* redo this take */
        }
        switch (cmd) {                                    /* r == -1 or woken while paused */
        case REC_CMD_PAUSE: s_paused = 1; status_set(REC_IDLE); break;
        case REC_CMD_START_SESSION:
            s_take_idx = 0; s_saved_takes = 0;
            nvs_bump_speaker();
            prompt_session_init(&s_prompts, PROMPT_SENTENCES, (uint32_t)esp_timer_get_time());
            s_paused = 0;
            break;
        case REC_CMD_START_WAKE_SESSION:
            s_take_idx = 0; s_saved_takes = 0;
            nvs_bump_speaker();
            prompt_session_init(&s_prompts, PROMPT_WAKE, (uint32_t)esp_timer_get_time());
            s_paused = 0;
            break;
        }
    }
}

void record_start(void)
{
    s_cmds = xQueueCreate(8, sizeof(record_cmd_t));
    s_lock = xSemaphoreCreateMutex();
    s_take = heap_caps_malloc(TAKE_MAX * sizeof(int16_t), MALLOC_CAP_SPIRAM);
    assert(s_take);
    nvs_load();
    /* Idle-state placeholder only; REC_CMD_START_SESSION re-seeds this for real
       when the menu's Record button is tapped. PROMPT_WORDS stays unused here
       (isolated words are not part of the guided session). */
    prompt_session_init(&s_prompts, PROMPT_SENTENCES, (uint32_t)esp_timer_get_time() | 1);
    snprintf(s_st.speaker, sizeof s_st.speaker, "spk%02lu", (unsigned long)s_speaker);
    xTaskCreatePinnedToCore(record_task, "record", 8192, NULL, 5, NULL, 0);
    status_set(REC_IDLE);
}

void record_post(record_cmd_t cmd) { xQueueSend(s_cmds, &cmd, 0); }

void record_get_status(record_status_t *out)
{
    xSemaphoreTake(s_lock, portMAX_DELAY); *out = s_st; xSemaphoreGive(s_lock);
}
