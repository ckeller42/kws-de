#include "intent.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
