#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "wav.h"

static uint32_t le32(const uint8_t *p) { return p[0] | p[1] << 8 | p[2] << 16 | (uint32_t)p[3] << 24; }
static uint16_t le16(const uint8_t *p) { return p[0] | p[1] << 8; }

int main(void)
{
    uint8_t h[WAV_HEADER_BYTES];
    wav_write_header(h, 16000, 16000);
    assert(!memcmp(h, "RIFF", 4) && !memcmp(h + 8, "WAVEfmt ", 8) && !memcmp(h + 36, "data", 4));
    assert(le32(h + 4) == 36 + 32000);
    assert(le32(h + 16) == 16 && le16(h + 20) == 1 && le16(h + 22) == 1);
    assert(le32(h + 24) == 16000 && le32(h + 28) == 32000 && le16(h + 32) == 2 && le16(h + 34) == 16);
    assert(le32(h + 40) == 32000);
    wav_write_header(h, 0, 16000);   assert(le32(h + 40) == 0 && le32(h + 4) == 36);
    wav_write_header(h, 96000, 16000); assert(le32(h + 40) == 192000);
    puts("test_wav OK");
    return 0;
}
