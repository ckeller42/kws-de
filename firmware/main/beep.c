#include "beep.h"
#include <math.h>
#include "bsp/esp-bsp.h"
#include "esp_codec_dev.h"
#include "esp_log.h"
#include "gen/features_config.h"

/* Tunables. The tone only has to be audible and short enough to stay inside
   the wake detector's refractory window (WAKE_REFRACTORY_MS). */
#define BEEP_HZ 1000        /* tone frequency */
#define BEEP_MS 150         /* tone length */
#define BEEP_VOLUME 60      /* esp_codec_dev output volume, 0..100 */
#define BEEP_AMPLITUDE 0.35f /* fraction of full scale; loud enough, no clipping */

#define BEEP_FRAMES (KWS_SAMPLE_RATE / 1000 * BEEP_MS)

static const char *TAG = "beep";
static esp_codec_dev_handle_t s_spk;
/* Stereo interleaved: the shared I2S runs 2-channel because the mic does. */
static int16_t s_tone[BEEP_FRAMES * 2];

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
    for (int i = 0; i < BEEP_FRAMES; i++) {
        int16_t v = (int16_t)(BEEP_AMPLITUDE * 32767.0f *
                              sinf(2.0f * (float)M_PI * BEEP_HZ * i / KWS_SAMPLE_RATE));
        s_tone[2 * i] = v;
        s_tone[2 * i + 1] = v;
    }
    s_spk = spk;
    ESP_LOGI(TAG, "speaker ready (%d Hz, %d ms)", BEEP_HZ, BEEP_MS);
}

void beep_play(void)
{
    if (!s_spk) return;
    esp_codec_dev_write(s_spk, s_tone, sizeof s_tone);
}
