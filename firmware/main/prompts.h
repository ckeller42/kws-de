/**
 * @file prompts.h
 * @brief Randomised prompt-set session for the guided recorder (wraps gen/prompts.h).
 */
#pragma once
#include <stdint.h>
#include "gen/prompts.h"

/** @brief Which prompt table a session draws from. PROMPT_ELICIT ("Situationen")
 * shows a scene/question and expects the speaker's own phrasing, wake word
 * included; its display text (prompt_text()) is the scene, but the EXPECTED
 * INTENT text (prompt_intent()) is what gets written to session.csv's prompt
 * column, so QC compares what was said against what was meant. PROMPT_SCENE
 * ("Szenen") is the scene-trigger set: the speaker reads one fixed trigger
 * phrase ("Gute Nacht"), no wake word, exactly as in the words set — but the
 * phrase maps to a whole Intent (config.SCENE_TRIGGERS), a class pending until
 * the next retrain. Its takes are QC'd against the read phrase and filed under
 * approved/scene/<token>/. */
typedef enum {
    PROMPT_WORDS = 0, PROMPT_SENTENCES = 1, PROMPT_NEGS = 2, PROMPT_WAKE = 3, PROMPT_ELICIT = 4,
    PROMPT_SCENE = 5,
} prompt_set_t;
/** @brief A shuffled walk through one prompt set. */
typedef struct {
    prompt_set_t set;
    uint32_t seed;
    int order[64];  /**< Shuffled prompt indices; order[index] is the current prompt. */
    int count;      /**< Number of prompts in this set. */
    int index;      /**< Current position in `order`. */
} prompt_session_t;

/** @brief Init a session with a Fisher-Yates shuffle (32-bit xorshift seeded from `seed`; same seed -> same order). */
void        prompt_session_init(prompt_session_t *p, prompt_set_t set, uint32_t seed);
/** @brief Display text for the current prompt (order[index]). */
const char *prompt_text(const prompt_session_t *p);
/** @brief Filename-safe slug for the current prompt. */
const char *prompt_slug(const prompt_session_t *p);
/** @brief PROMPT_ELICIT only: the expected-intent text for the current prompt
 * (e.g. "Licht Küche an"), not the scene/question shown on screen. Used for
 * session.csv's prompt column instead of prompt_text() so QC can compare the
 * speaker's actual words against what the scene was meant to elicit. */
const char *prompt_intent(const prompt_session_t *p);
/** @brief Advance to the next prompt. @return 0 when the set is exhausted (index unchanged), 1 otherwise. */
int         prompt_advance(prompt_session_t *p);
/** @brief Recording time cap for a prompt set, in ms (4000 for words, 9800 for
 * elicit — an unscripted answer runs longer than a read sentence — 6000 otherwise,
 * scene included: a two-word trigger like "Guten Morgen" a speaker pauses between
 * gets the sentence cap's headroom, not the tighter 4000 ms word cap). */
uint32_t    prompt_cap_ms(prompt_set_t set);
/** @brief Trailing-silence hangover before a take closes, in ms: 500 for words, 1200
 * for sentences/negatives/wake/elicit/scene. A natural reading pause between the words
 * of a longer prompt (e.g. between "Guten" and "Morgen") exceeds 500 ms, so those sets
 * need the longer hangover or the take gets cut after the first word. */
uint32_t    prompt_hangover_ms(prompt_set_t set);
/** @brief Reads captured per prompt before advancing (2 normally, for wrong-read review;
 * 1 for PROMPT_WAKE — a "Hey Bus" session wants exactly config.WAKE_PROMPT_REPEATS real
 * positives, not doubled reads; 1 for PROMPT_ELICIT — an elicited answer is one natural
 * take, not a read to redo; 3 for PROMPT_SCENE — a bootstrap class starts with zero real
 * clips, so grab several takes of each of the four triggers per speaker). */
int         prompt_takes_per_prompt(prompt_set_t set);
/** @brief Set name as used in session.csv and the UI progress line
 * ("words"|"sentences"|"negatives"|"wake"|"elicit"|"scene"). */
const char *prompt_set_name(prompt_set_t set);
