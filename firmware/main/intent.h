/**
 * @file intent.h
 * @brief On-device port of kws_de.grammar.parse(): device/zone/action grammar
 * over a window's fired command words.
 *
 * Pure C, no FreeRTOS/ESP dependency, so it is host-testable exactly like
 * assist_gate.c and field.c (see firmware/test/test_intent.c). The vocabulary
 * itself lives in ONE place, gen/labels.h (generated from
 * kws_de.config.COMMAND_LABELS); this file only encodes which of those labels
 * are a device/zone/action and which actions each device accepts -- the same
 * structure kws_de.config.DEVICE_ACTIONS/ZONED_DEVICES describe on the Python
 * side. Drift between the two is caught by test_intent.c, which checks the C
 * port against kws_de.grammar.parse() case for case.
 */
#pragma once
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief A parsed device/zone/action command, or the invalid marker. */
typedef struct {
    bool valid;          /**< False on any grammar violation; every other field is NULL/0 then. */
    const char *device;  /**< Points into KWS_LABELS (gen/labels.h). NULL if invalid. */
    const char *zone;    /**< Points into KWS_LABELS. NULL if the command has no zone. */
    const char *action;  /**< Points into KWS_LABELS; a brightness level counts as an action. NULL if invalid. */
    bool level;          /**< True when `action` is a Licht brightness level ("fünfzig" etc.) -- intent_format() appends "Prozent". */
} intent_t;

/**
 * @brief Parse a window's fired command words into a device/zone/action intent.
 *
 * Mirrors kws_de.grammar.parse(): a device, at most one optional zone, and
 * exactly one action, in that order, no duplicates; "_unknown_"/"_silence_"
 * tokens are ignored (matching the Python filter), any other unrecognised
 * token invalidates the parse.
 *
 * @param words Space-joined fired labels in fire order, e.g. "Licht Küche an"
 *              -- the format of recognise_status_t.window_intent /
 *              field_take_t's pre-formatting raw fires.
 * @return .valid true with device/action (and optionally zone) set, or
 *         .valid false (all pointers NULL) on any grammar violation.
 */
intent_t intent_parse(const char *words);

/** @brief Minimum second-best probability intent_rescore() will act on (see below). */
#define INTENT_RESCORE_FLOOR 0.25f

/**
 * @brief intent_parse(), with one retry: if the plain parse fails, substitute
 * the stream decoder's second-best command word at each "_unknown_" slot in
 * @p words and re-parse, accepting the result only if EXACTLY ONE such
 * substitution (at or above @p floor) makes it valid.
 *
 * A window that never fired a real slot at all (e.g. no zone -- optional
 * anyway -- or truly missing action) has no second-best data to substitute
 * from, so it is left to intent_parse()'s existing verdict; there is nothing
 * here to rescore. Two or more "_unknown_" slots both clearing @p floor are
 * refused outright (one substitution cannot fix two slots), same as one
 * "_unknown_" whose second-best falls below @p floor.
 *
 * @param words   Same as intent_parse().
 * @param seconds One "<label>:<prob>" entry per space-joined token in @p words,
 *                same order, joined by '|' (recognise_status_t.window_seconds'
 *                format) -- the per-step runner-up command word the stream
 *                decoder saw and its smoothed probability. NULL/mismatched
 *                token count is treated as "no rescore data".
 * @param floor   Minimum runner-up probability to accept, e.g. INTENT_RESCORE_FLOOR.
 * @param from    On a successful rescore, set to "_unknown_"; untouched otherwise.
 * @param to      On a successful rescore, set to the substituted label (points
 *                into KWS_LABELS); untouched otherwise. Together these are the
 *                "X->Y" of the "intent: ... (rescored: X->Y)" log line.
 * @return The plain parse if it already succeeded or no rescore applies, else
 *         the rescored parse (valid or not).
 */
intent_t intent_rescore(const char *words, const char *seconds, float floor,
                         const char **from, const char **to);

/**
 * @brief Format a valid intent as a person would read it off the screen:
 * "Licht Küche → an", "Licht → fünfzig Prozent" -- device, optional zone, an
 * arrow, then the action (plus "Prozent" for a brightness level). Matches
 * kws_de.eval.intent_text's word order/level suffix; normalise() in
 * kws_de/qc.py strips the arrow, so this round-trips through
 * kws_de.grammar.parse() unchanged.
 *
 * @param in Intent to format.
 * @param buf Output buffer; left as an empty string when `in` is invalid.
 * @param n Size of `buf`.
 * @return Number of characters written (excluding the NUL), 0 if invalid.
 */
int intent_format(const intent_t *in, char *buf, int n);

#ifdef __cplusplus
}
#endif
