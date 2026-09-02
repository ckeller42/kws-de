/**
 * @file wav.h
 * @brief Minimal canonical PCM WAV header writer (mono, 16-bit).
 */
#pragma once
#include <stdint.h>

#define WAV_HEADER_BYTES 44  /**< Size of a canonical WAV header. */
/**
 * @brief Write a canonical 44-byte mono 16-bit PCM WAV header.
 * @param out         Destination buffer, WAV_HEADER_BYTES bytes.
 * @param n_samples   Number of mono samples that will follow the header.
 * @param sample_rate Sample rate in Hz.
 */
void wav_write_header(uint8_t out[WAV_HEADER_BYTES], uint32_t n_samples, uint32_t sample_rate);
