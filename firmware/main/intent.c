#include "intent.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gen/labels.h"

/* KWS_LABELS (gen/labels.h) is generated from kws_de.config.COMMAND_LABELS in
   exactly this order: DEVICES, ZONES, ACTIONS, then "_unknown_"/"_silence_"
   (see kws_de/config.py). These counts mirror that layout so the label array
   itself never has to be duplicated here -- only its structure does. A count
   drift is caught at compile time (the _Static_assert below) and any drift in
   WHICH label lands where is caught by test_intent.c comparing every case
   against kws_de.grammar.parse(). */
#define N_DEVICES 4  /* Licht, Kühlschrank, Heizung, Aufstelldach */
#define N_ZONES 4    /* Küche, Dach, Außen, Lesen */
#define N_ACTIONS 13 /* an, aus, auf, zu, heller, dunkler, wärmer, kälter, leise, + 4 levels */
#define N_LEVELS 4   /* fünfundzwanzig, fünfzig, fünfundsiebzig, hundert -- the last N_LEVELS actions */

_Static_assert(N_DEVICES + N_ZONES + N_ACTIONS + 2 == KWS_NUM_LABELS,
               "intent.c's device/zone/action layout must match gen/labels.h");

/* Only Licht (label index 0) takes a zone: kws_de.config.ZONED_DEVICES. */
static bool device_takes_zone(int device_idx) { return device_idx == 0; }

/* kws_de.config.DEVICE_ACTIONS, as a bitmask over the action-local index
   (0..N_ACTIONS-1, i.e. KWS_LABELS index minus N_DEVICES+N_ZONES):
   an=0 aus=1 auf=2 zu=3 heller=4 dunkler=5 wärmer=6 kälter=7 leise=8, levels=9..12. */
static const unsigned kDeviceActions[N_DEVICES] = {
    (1u << 0) | (1u << 1) | (1u << 4) | (1u << 5) | (1u << 9) | (1u << 10) | (1u << 11) | (1u << 12), /* Licht */
    (1u << 0) | (1u << 1) | (1u << 8),                                                                /* Kühlschrank */
    (1u << 0) | (1u << 1) | (1u << 6) | (1u << 7),                                                    /* Heizung */
    (1u << 2) | (1u << 3),                                                                            /* Aufstelldach */
};

/* kws_de.config.LIGHT_COMPOUNDS: fused Licht+zone words ("Küchenlicht" ->
   "Küche"), matched here as literal strings rather than via KWS_LABELS --
   they are not yet a trained class (KWS_MODEL_NUM_CLASSES is still 23, see
   the _Static_assert in recognise.cc), so the device can never actually emit
   one until the next retrain adds them (docs/paper-notes.md E50). This table
   keeps the C port in lockstep with kws_de.grammar.parse() for that day; it
   is unreachable in production until then. "Dach" has no natural compound --
   see config.LIGHT_COMPOUNDS' comment -- so it is not in this table. */
static const struct { const char *word; const char *zone; } kLightCompounds[] = {
    {"Küchenlicht", "Küche"},
    {"Außenlicht", "Außen"},
    {"Leselicht", "Lesen"},
};
#define N_LIGHT_COMPOUNDS (sizeof kLightCompounds / sizeof kLightCompounds[0])

/* kws_de.config.SCENE_TRIGGERS: a single token standing for a COMPLETE intent
   (device+zone+action all at once), unlike kLightCompounds which still needs a
   following action word. Same pending status: not a trained class yet, matched
   here as a literal string for the day a retrain adds it (docs/paper-notes.md
   E50/E54). zone==NULL means "no zone" (device_takes_zone() still governs
   whether a real zone may be attached downstream, matching grammar.py). */
struct scene_trigger { const char *word; int device_idx; const char *zone; int action_idx; };
static const struct scene_trigger kSceneTriggers[] = {
    {"GuteNacht", 0, NULL, 1},     /* Licht, no zone, aus */
    {"GutenMorgen", 0, NULL, 0},   /* Licht, no zone, an */
    {"Leseratte", 0, "Lesen", 0},  /* Licht, Lesen, an */
    {"Nachtlicht", 0, NULL, 9},    /* Licht, no zone, fuenfundzwanzig (25%) */
};
#define N_SCENE_TRIGGERS (sizeof kSceneTriggers / sizeof kSceneTriggers[0])

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
        const struct scene_trigger *trig = NULL;
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

    char merged[64] = {0};
    for (int i = 0; i < nw; i++) {
        strcat(merged, i == sub ? KWS_LABELS[sub_idx] : wtok[i]);
        if (i + 1 < nw) strcat(merged, " ");
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
