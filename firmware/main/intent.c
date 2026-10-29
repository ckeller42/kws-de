#include "intent.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "assist_gate.h"
#include "gen/grammar.h"
#include "gen/labels.h"

/* The grammar's tables -- device/zone/action counts, which devices take a zone,
   each device's valid actions, the light compounds and the scene triggers -- are
   generated from kws_de/config.py into gen/grammar.h by kws-fwgen, in the same
   order as KWS_LABELS (DEVICES, ZONES, ACTIONS, then "_unknown_"/"_silence_").
   kws-fwgen asserts that layout; gen-fresh CI checks the committed header is
   current; test_intent.c checks every case against kws_de.grammar.parse().
   The compounds and scene triggers are pending vocabulary: not a trained class
   until the next retrain (docs/paper-notes.md E50/E54), matched here as literal
   strings so the C port is already in lockstep with grammar.py. */
#define N_DEVICES KWS_N_DEVICES
#define N_ZONES KWS_N_ZONES
#define N_ACTIONS KWS_N_ACTIONS
#define N_LEVELS KWS_N_LEVELS /* the last N_LEVELS actions */

_Static_assert(N_DEVICES + N_ZONES + N_ACTIONS + 2 == KWS_NUM_LABELS,
               "gen/grammar.h and gen/labels.h disagree -- rerun kws-fwgen");

static bool device_takes_zone(int device_idx) { return (KWS_ZONED_DEVICE_MASK >> device_idx) & 1u; }

/* Bitmask over the action-local index (KWS_LABELS index minus N_DEVICES+N_ZONES). */
#define kDeviceActions KWS_DEVICE_ACTIONS
#define kLightCompounds KWS_LIGHT_COMPOUNDS
#define N_LIGHT_COMPOUNDS KWS_NUM_LIGHT_COMPOUNDS
#define kSceneTriggers KWS_SCENE_TRIGGERS
#define N_SCENE_TRIGGERS KWS_NUM_SCENE_TRIGGERS

static int label_index(const char *tok)
{
    for (int i = 0; i < KWS_NUM_LABELS; i++)
        if (strcmp(tok, KWS_LABELS[i]) == 0) return i;
    return -1;
}

static const intent_t kInvalid; /* all-zero: valid=false, every pointer NULL */

intent_t intent_parse(const char *words)
{
    if (!words || !*words) return kInvalid;
    char buf[64];
    strncpy(buf, words, sizeof buf - 1);
    buf[sizeof buf - 1] = 0;

    int device_idx = -1, action_idx = -1;
    const char *zone = NULL;
    for (char *tok = strtok(buf, " "); tok; tok = strtok(NULL, " ")) {
        const kws_scene_trigger_t *trig = NULL;
        for (size_t ti = 0; ti < N_SCENE_TRIGGERS; ti++) {
            if (strcmp(tok, kSceneTriggers[ti].word) == 0) {
                trig = &kSceneTriggers[ti];
                break;
            }
        }
        if (trig) {
            /* A scene trigger sets device+zone+action all at once; any slot
               already set from an earlier token is a conflict (grammar.py
               mirrors this with the same "duplicate device" rejection). */
            if (device_idx >= 0 || zone || action_idx >= 0) return kInvalid;
            device_idx = trig->device_idx;
            zone = trig->zone;
            action_idx = N_DEVICES + N_ZONES + trig->action_idx;
            continue;
        }
        const char *compound_zone = NULL;
        for (size_t ci = 0; ci < N_LIGHT_COMPOUNDS; ci++) {
            if (strcmp(tok, kLightCompounds[ci].word) == 0) {
                compound_zone = kLightCompounds[ci].zone;
                break;
            }
        }
        if (compound_zone) {
            if (device_idx >= 0) return kInvalid;         /* duplicate device */
            if (zone) return kInvalid;                    /* duplicate zone */
            if (action_idx >= 0) return kInvalid;         /* device out of order */
            device_idx = 0;                               /* Licht -- see device_takes_zone() */
            zone = compound_zone;
            continue;
        }
        int idx = label_index(tok);
        if (idx < 0 || idx == KWS_UNKNOWN_INDEX || idx == KWS_SILENCE_INDEX) {
            if (idx < 0) return kInvalid; /* unknown token: reject */
            continue;                     /* _unknown_/_silence_: dropped, like the Python filter */
        }
        if (idx < N_DEVICES) {
            if (device_idx >= 0) return kInvalid;        /* duplicate device */
            if (zone || action_idx >= 0) return kInvalid; /* device out of order */
            device_idx = idx;
        } else if (idx < N_DEVICES + N_ZONES) {
            if (zone) return kInvalid;                      /* duplicate zone */
            if (device_idx < 0 || action_idx >= 0) return kInvalid; /* zone out of order */
            zone = KWS_LABELS[idx];
        } else { /* action */
            if (action_idx >= 0) return kInvalid; /* duplicate action */
            action_idx = idx;
        }
    }
    if (device_idx < 0) return kInvalid;                      /* missing device */
    if (action_idx < 0) return kInvalid;                      /* missing action */
    if (zone && !device_takes_zone(device_idx)) return kInvalid; /* device takes no zone */
    unsigned action_bit = 1u << (action_idx - (N_DEVICES + N_ZONES));
    if (!(kDeviceActions[device_idx] & action_bit)) return kInvalid; /* action invalid for device */

    intent_t r = {
        .valid = true,
        .device = KWS_LABELS[device_idx],
        .zone = zone,
        .action = KWS_LABELS[action_idx],
        .level = (action_idx - (N_DEVICES + N_ZONES)) >= (N_ACTIONS - N_LEVELS),
    };
    return r;
}

intent_t intent_rescore(const char *words, const char *seconds, float floor,
                        const char **from, const char **to)
{
    intent_t base = intent_parse(words);
    if (base.valid || !words || !seconds) return base;

    char wbuf[64], sbuf[96];
    strncpy(wbuf, words, sizeof wbuf - 1); wbuf[sizeof wbuf - 1] = 0;
    strncpy(sbuf, seconds, sizeof sbuf - 1); sbuf[sizeof sbuf - 1] = 0;

    char *wtok[16]; int nw = 0;
    for (char *t = strtok(wbuf, " "); t && nw < 16; t = strtok(NULL, " ")) wtok[nw++] = t;
    char *stok[16]; int ns = 0;
    for (char *t = strtok(sbuf, "|"); t && ns < 16; t = strtok(NULL, "|")) stok[ns++] = t;
    if (nw != ns) return base; /* seconds not aligned 1:1 with words: nothing safe to substitute */

    int sub = -1;
    int sub_idx = -1;
    for (int i = 0; i < nw; i++) {
        if (strcmp(wtok[i], "_unknown_") != 0) continue;
        char *colon = strchr(stok[i], ':');
        if (!colon) continue;
        float p = strtof(colon + 1, NULL);
        if (p < floor) continue;
        if (sub >= 0) return base; /* a second fixable slot: one substitution cannot cover both */
        *colon = 0;
        sub_idx = label_index(stok[i]);
        if (sub_idx < 0 || sub_idx == KWS_UNKNOWN_INDEX || sub_idx == KWS_SILENCE_INDEX)
            continue; /* not a real command word: no usable candidate */
        sub = i;
    }
    if (sub < 0) return base;

    /* Bounded rebuild: the substitute can be longer than "_unknown_" (up to
       "fünfundsiebzig", 15 bytes), so a near-full window no longer fits the
       63-byte line intent_parse() reads -- give up rather than truncate. */
    char merged[64];
    size_t len = 0;
    for (int i = 0; i < nw; i++) {
        int k = snprintf(merged + len, sizeof merged - len, "%s%s", i ? " " : "",
                         i == sub ? KWS_LABELS[sub_idx] : wtok[i]);
        if (k < 0 || (size_t)k >= sizeof merged - len) return base;
        len += (size_t)k;
    }
    intent_t r = intent_parse(merged);
    if (!r.valid) return base;
    *from = "_unknown_";
    *to = KWS_LABELS[sub_idx];
    return r;
}

/* Viterbi over bg w1 bg ... wL bg (kws_de.window_intent._path_score). Returns
   the best path's total log-probability (-inf if no path) and its token
   confidence. Log-posteriors are taken on the fly rather than tabulated: a
   32 x 23 float table would be 2.9 kB of stack on the wake task, which
   intent_rescore() already overflowed once (E56). */
#define ALIGN_MAX_TOKENS 3
#define ALIGN_MAX_STATES (2 * ALIGN_MAX_TOKENS + 1)

static float align_emit(const float *post, int t, int k, float floor, int aus_idx, bool in_tail)
{
    float p = post[t * KWS_NUM_LABELS + k];
    if (p < floor || (in_tail && k == aus_idx)) return -INFINITY;
    return logf(p);
}

static float align_path(const float *post, int T, const float *bg, const int *idx, int L,
                        float floor, int aus_idx, int first_ms, int step_ms, float *conf)
{
    int S = 2 * L + 1;
    float score[ALIGN_MAX_STATES], ntok[ALIGN_MAX_STATES], tsum[ALIGN_MAX_STATES];
    float nscore[ALIGN_MAX_STATES], nn[ALIGN_MAX_STATES], ns[ALIGN_MAX_STATES];
    for (int s = 0; s < S; s++) { score[s] = -INFINITY; ntok[s] = tsum[s] = 0.f; }
    bool tail0 = assist_gate_in_wake_tail(first_ms, "aus");
    score[0] = bg[0];
    score[1] = align_emit(post, 0, idx[0], floor, aus_idx, tail0);
    ntok[1] = 1.f;
    tsum[1] = score[1];
    for (int t = 1; t < T; t++) {
        bool tail = assist_gate_in_wake_tail(first_ms + (int64_t)t * step_ms, "aus");
        for (int s = 0; s < S; s++) {
            bool tok = (s & 1) != 0;
            float emit = tok ? align_emit(post, t, idx[s / 2], floor, aus_idx, tail) : bg[t];
            /* stay, advance from s-1, or (token state) skip the background from s-2;
               ties keep the earlier source, as the Python reference's max() does */
            int u = s;
            if (s >= 1 && score[s - 1] > score[u]) u = s - 1;
            if (tok && s >= 2 && score[s - 2] > score[u]) u = s - 2;
            if (score[u] == -INFINITY) { nscore[s] = -INFINITY; nn[s] = ns[s] = 0.f; continue; }
            nscore[s] = score[u] + emit;
            nn[s] = ntok[u] + (tok ? 1.f : 0.f);
            ns[s] = tsum[u] + (tok ? emit : 0.f);
        }
        memcpy(score, nscore, sizeof(float) * (size_t)S);
        memcpy(ntok, nn, sizeof(float) * (size_t)S);
        memcpy(tsum, ns, sizeof(float) * (size_t)S);
    }
    int end = (score[S - 2] > score[S - 1]) ? S - 2 : S - 1;
    if (score[end] == -INFINITY) { *conf = 0.f; return -INFINITY; }
    *conf = expf(tsum[end] / ntok[end]);
    return score[end];
}

intent_t intent_align(const float *post, int n_steps, int first_ms, int step_ms, float floor,
                      float tau, float *conf)
{
    if (conf) *conf = 0.f;
    if (!post || n_steps <= 0) return kInvalid;
    int T = n_steps > INTENT_ALIGN_MAX_STEPS ? INTENT_ALIGN_MAX_STEPS : n_steps;
    int aus_idx = label_index("aus");
    float bg[INTENT_ALIGN_MAX_STEPS];
    for (int t = 0; t < T; t++) {
        float b = post[t * KWS_NUM_LABELS + KWS_SILENCE_INDEX] + post[t * KWS_NUM_LABELS + KWS_UNKNOWN_INDEX];
        bg[t] = b > 0.f ? logf(b) : -INFINITY;
    }
    /* Candidates in kws_de.window_intent.candidates() order: device, then no
       zone / each zone (zoned devices only), then the device's actions in
       KWS_LABELS order. The first strictly-best path wins, as the reference's
       stable sort does. */
    float best = -INFINITY, best_conf = 0.f;
    int best_d = -1, best_z = -1, best_a = -1;
    for (int d = 0; d < N_DEVICES; d++) {
        int nz = device_takes_zone(d) ? N_ZONES : 0;
        for (int z = -1; z < nz; z++) {
            for (int a = 0; a < N_ACTIONS; a++) {
                if (!((kDeviceActions[d] >> a) & 1u)) continue;
                int idx[ALIGN_MAX_TOKENS], L = 0;
                idx[L++] = d;
                if (z >= 0) idx[L++] = N_DEVICES + z;
                idx[L++] = N_DEVICES + N_ZONES + a;
                float c;
                float s = align_path(post, T, bg, idx, L, floor, aus_idx, first_ms, step_ms, &c);
                if (s > best) { best = s; best_conf = c; best_d = d; best_z = z; best_a = a; }
            }
        }
    }
    if (best_d < 0) return kInvalid;
    if (conf) *conf = best_conf;
    if (best_conf < tau) return kInvalid;
    intent_t r = {
        .valid = true,
        .device = KWS_LABELS[best_d],
        .zone = best_z >= 0 ? KWS_LABELS[N_DEVICES + best_z] : NULL,
        .action = KWS_LABELS[N_DEVICES + N_ZONES + best_a],
        .level = best_a >= (N_ACTIONS - N_LEVELS),
    };
    return r;
}

int intent_format(const intent_t *in, char *buf, int n)
{
    if (!in->valid) {
        if (n > 0) buf[0] = 0;
        return 0;
    }
    int k = in->zone
        ? snprintf(buf, (size_t)n, "%s %s -> %s%s", in->device, in->zone, in->action,
                   in->level ? " Prozent" : "")
        : snprintf(buf, (size_t)n, "%s -> %s%s", in->device, in->action, in->level ? " Prozent" : "");
    return k < 0 ? 0 : k;
}
