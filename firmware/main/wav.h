#pragma once
/* wav.h */
#include <stdint.h>

#define WAV_HEADER_BYTES 44
void wav_write_header(uint8_t out[WAV_HEADER_BYTES], uint32_t n_samples, uint32_t sample_rate);
