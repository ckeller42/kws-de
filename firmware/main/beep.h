/**
 * @file beep.h
 * @brief Short confirmation tone on the built-in speaker.
 *
 * The CoreS3 codec sits on ONE full-duplex I2S channel pair shared by the
 * ES7210 microphone and the AW88298 amplifier (BSP `bsp_audio_init()` creates
 * both directions from a single `i2s_std_config_t`). `esp_codec_dev` allows the
 * mic and the speaker to be open at the same time only while their sample
 * rates agree, so the speaker is opened with exactly the format `audio.c`
 * already opened the mic with (16 kHz, 16-bit, 2 channels). Any other rate
 * would be rejected with "conflict sample_rate" and would take the mic down.
 */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Open the speaker codec and pre-render the tone.
 *
 * Safe to call when the microphone is already capturing. Logs and gives up if
 * the codec cannot be opened — a missing beep must never stop wake detection.
 */
void beep_init(void);

/**
 * @brief Play the confirmation tone (blocking, ~BEEP_MS).
 *
 * No-op if beep_init() failed. Called from the wake task right after a
 * detection, inside the detector's refractory window, so the stall cannot
 * cost a second detection. Audio capture is unaffected: a separate always-on
 * task keeps filling the 10 s ring while this blocks.
 */
void beep_play(void);

#ifdef __cplusplus
}
#endif
