#include "wav.h"
#include <string.h>

static void put32(uint8_t *p, uint32_t v) { p[0] = v; p[1] = v >> 8; p[2] = v >> 16; p[3] = v >> 24; }
static void put16(uint8_t *p, uint16_t v) { p[0] = v; p[1] = v >> 8; }

void wav_write_header(uint8_t out[WAV_HEADER_BYTES], uint32_t n_samples, uint32_t sample_rate)
{
    uint32_t data_bytes = n_samples * 2;   /* mono int16 */
    memcpy(out, "RIFF", 4);      put32(out + 4, 36 + data_bytes);
    memcpy(out + 8, "WAVEfmt ", 8);
    put32(out + 16, 16);         put16(out + 20, 1); put16(out + 22, 1);
    put32(out + 24, sample_rate); put32(out + 28, sample_rate * 2);
    put16(out + 32, 2);          put16(out + 34, 16);
    memcpy(out + 36, "data", 4); put32(out + 40, data_bytes);
}
