#include "audio.h"
#include <assert.h>
#include <string.h>
#include "bsp/esp-bsp.h"
#include "esp_codec_dev.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "audio";
static int16_t *s_ring;                 /* PSRAM, AUDIO_RING_SAMPLES */
static volatile uint32_t s_pos;         /* absolute sample count written */
static esp_codec_dev_handle_t s_mic;

static void audio_task(void *arg)
{
    (void)arg;
    int16_t chunk[KWS_HOP * 2];         /* stereo read, 20 ms */
    for (;;) {
        ESP_ERROR_CHECK(esp_codec_dev_read(s_mic, chunk, sizeof chunk));
        uint32_t pos = s_pos;
        for (int i = 0; i < KWS_HOP; i++) s_ring[(pos + i) % AUDIO_RING_SAMPLES] = chunk[2 * i];  /* left mic */
        s_pos = pos + KWS_HOP;
    }
}

void audio_start(void)
{
    s_ring = heap_caps_calloc(AUDIO_RING_SAMPLES, sizeof(int16_t), MALLOC_CAP_SPIRAM);
    assert(s_ring);
    s_mic = bsp_audio_codec_microphone_init();
    assert(s_mic);
    esp_codec_dev_sample_info_t fs = {.bits_per_sample = 16, .channel = 2, .sample_rate = KWS_SAMPLE_RATE};
    ESP_ERROR_CHECK(esp_codec_dev_open(s_mic, &fs));
    /* ponytail: mic gain is a calibration knob. First real recordings peaked at
       ~-18 dBFS (RMS ~-40) at 30 dB — usable but quiet; +6 dB lifts SNR. Retune
       here if reads clip (the recorder flags clipping and asks for a redo). */
    ESP_ERROR_CHECK(esp_codec_dev_set_in_gain(s_mic, 36.0));
    xTaskCreatePinnedToCore(audio_task, "audio", 4096, NULL, 10, NULL, 1);
    ESP_LOGI(TAG, "capture running");
}

uint32_t audio_write_pos(void) { return s_pos; }

void audio_read(uint32_t end, int16_t *dst, uint32_t n)
{
    uint32_t start = end - n;
    for (uint32_t i = 0; i < n; i++) dst[i] = s_ring[(start + i) % AUDIO_RING_SAMPLES];
}
