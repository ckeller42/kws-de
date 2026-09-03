/**
 * @file mfcc_fft.cc
 * @brief 480-point real FFT power spectrum for the MFCC front end.
 *
 * The vendored kissfft (``microfrontend/kissfft``) is C++ with ``extern "C++"``
 * headers, so this one translation unit is the C-linkage wrapper the C front
 * end in mfcc.c calls. It is the only reason mfcc.c itself stays plain C.
 *
 * 480 = 2^5*3*5, so ``kiss_fftr`` runs a 240-point complex transform that
 * kissfft factors exactly as 4,4,3,5 — every stage has a dedicated butterfly
 * and no zero-padding to 512 is involved, so the mel energies are the ones the
 * models were trained on, not a rescaled approximation of them.
 *
 * Memory: the vendored kissfft has ``KISS_FFT_MALLOC`` patched out (returns
 * NULL, TFLM style), so the config must be handed its own buffer through the
 * ``(mem, lenmem)`` protocol. s_mem below is that buffer, in .bss — the exact
 * need for nfft=480 is ~5.1 kB (state + 240-pt sub-state twiddles + tmpbuf +
 * super-twiddles); kiss_fftr_alloc returns NULL if it ever does not fit, which
 * the host test catches.
 */
#include <cassert>
#include <cstddef>
#include "gen/features_config.h"
#include "mfcc.h"
#include "tools/kiss_fftr.h"

static char s_mem[6144];
static kiss_fftr_cfg s_cfg;

extern "C" void mfcc_fft_init(void)
{
    if (s_cfg) return;
    size_t len = sizeof s_mem;
    s_cfg = kiss_fftr_alloc(KWS_WIN, 0, s_mem, &len);
    assert(s_cfg); /* too small: len now holds the size kissfft wants */
}

extern "C" void mfcc_fft_power(const float in[KWS_WIN], float out[KWS_N_BINS])
{
    kiss_fft_cpx spec[KWS_N_BINS];
    kiss_fftr(s_cfg, in, spec);
    for (int k = 0; k < KWS_N_BINS; k++) out[k] = spec[k].r * spec[k].r + spec[k].i * spec[k].i;
}
