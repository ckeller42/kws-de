#include "beep.h"
#include <math.h>
#include "bsp/esp-bsp.h"
#include "esp_codec_dev.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "gen/features_config.h"
#include "task.h"

/* Tunables. The wake tone only has to be audible and short enough to stay
   inside the wake detector's refractory window (WAKE_REFRACTORY_MS). */
#define BEEP_HZ 1000        /* tone frequency */
#define BEEP_MS 150         /* tone length */
#define BEEP_VOLUME 60      /* esp_codec_dev output volume, 0..100 */
#define BEEP_AMPLITUDE 0.35f /* fraction of full scale; loud enough, no clipping */

/* Command-confirmation tone: two short pips a little higher than the wake tone,
   so "I heard the wake word" and "I understood the command" never sound alike
   from across the van. Both frequencies divide 16 kHz into a whole number of
   samples per tone (150 and 120 periods), so each buffer starts and ends at
   zero crossings and neither tone clicks. */
#define BEEP2_HZ 1500
#define BEEP2_MS 80
#define BEEP2_GAP_MS 60

#define BEEP_FRAMES (KWS_SAMPLE_RATE / 1000 * BEEP_MS)
#define BEEP2_FRAMES (KWS_SAMPLE_RATE / 1000 * BEEP2_MS)

static const char *TAG = "beep";
static esp_codec_dev_handle_t s_spk;
/* Stereo interleaved: the shared I2S runs 2-channel because the mic does. */
static int16_t s_tone[BEEP_FRAMES * 2];
static int16_t s_tone2[BEEP2_FRAMES * 2];
static TaskHandle_t s_task;

static void render(int16_t *dst, int frames, int hz)
{
    for (int i = 0; i < frames; i++) {
        int16_t v = (int16_t)(BEEP_AMPLITUDE * 32767.0f *
                              sinf(2.0f * (float)M_PI * hz * i / KWS_SAMPLE_RATE));
        dst[2 * i] = v;
        dst[2 * i + 1] = v;
    }
}

/* The whole point of the task: beep_double() is called from the model tasks and
   must not block them, and the gap between the two pips is a delay, not work. */
static void beep_task(void *arg)
{
    (void)arg;
    s_task = xTaskGetCurrentTaskHandle();   /* published here, so no trace facility is needed */
    for (;;) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        for (int i = 1; i <= 2; i++) {
            ESP_LOGI(TAG, "confirm tone %d/2 (%d Hz, %d ms)", i, BEEP2_HZ, BEEP2_MS);
            esp_codec_dev_write(s_spk, s_tone2, sizeof s_tone2);
            if (i == 1) vTaskDelay(pdMS_TO_TICKS(BEEP2_GAP_MS));
        }
    }
}

void beep_init(void)
{
    if (s_spk) return;
    esp_codec_dev_handle_t spk = bsp_audio_codec_speaker_init();
    if (!spk) { ESP_LOGE(TAG, "speaker init failed — running without a beep"); return; }
    /* Must match audio.c's mic format exactly, see beep.h. */
    esp_codec_dev_sample_info_t fs = {.bits_per_sample = 16, .channel = 2, .sample_rate = KWS_SAMPLE_RATE};
    esp_err_t err = esp_codec_dev_open(spk, &fs);
    if (err != ESP_OK) { ESP_LOGE(TAG, "speaker open failed (%s)", esp_err_to_name(err)); return; }
    esp_codec_dev_set_out_vol(spk, BEEP_VOLUME);
    render(s_tone, BEEP_FRAMES, BEEP_HZ);
    render(s_tone2, BEEP2_FRAMES, BEEP2_HZ);
    s_spk = spk;
    /* Priority 1, core 0: below both model tasks, so a tone can never delay an
       inference step. Underrun is not a risk at that priority — the BSP's I2S
       channel carries 6 x 240 frames of DMA (90 ms at 16 kHz), more than one
       pip, and each pip is a single write. */
    task_spawn(TAG, beep_task, "beep", 3072, NULL, 1, 0);
    ESP_LOGI(TAG, "speaker ready (wake %d Hz/%d ms, confirm 2 x %d Hz/%d ms)",
             BEEP_HZ, BEEP_MS, BEEP2_HZ, BEEP2_MS);
}

void beep_play(void)
{
    if (!s_spk) return;
    esp_codec_dev_write(s_spk, s_tone, sizeof s_tone);
}

void beep_double(void)
{
    if (s_task) xTaskNotifyGive(s_task);
}
