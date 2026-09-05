#include "intent.h"
#include <stdio.h>
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

int intent_format(const intent_t *in, char *buf, int n)
{
    if (!in->valid) {
        if (n > 0) buf[0] = 0;
        return 0;
    }
    int k = in->zone
        ? snprintf(buf, (size_t)n, "%s %s → %s%s", in->device, in->zone, in->action,
                   in->level ? " Prozent" : "")
        : snprintf(buf, (size_t)n, "%s → %s%s", in->device, in->action, in->level ? " Prozent" : "");
    return k < 0 ? 0 : k;
}
