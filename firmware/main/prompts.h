/**
 * @file prompts.h
 * @brief Randomised prompt-set session for the guided recorder (wraps gen/prompts.h).
 */
#pragma once
#include <stdint.h>
#include "gen/prompts.h"

/** @brief Which prompt table a session draws from. */
typedef enum { PROMPT_WORDS = 0, PROMPT_SENTENCES = 1, PROMPT_NEGS = 2, PROMPT_WAKE = 3 } prompt_set_t;
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
/** @brief Advance to the next prompt. @return 0 when the set is exhausted (index unchanged), 1 otherwise. */
int         prompt_advance(prompt_session_t *p);
/** @brief Recording time cap for a prompt set, in ms (4000 for words, 6000 otherwise). */
uint32_t    prompt_cap_ms(prompt_set_t set);
/** @brief Reads captured per prompt before advancing (2 normally, for wrong-read review;
 * 1 for PROMPT_WAKE — a "Hey Bus" session wants exactly config.WAKE_PROMPT_REPEATS real
 * positives, not doubled reads). */
int         prompt_takes_per_prompt(prompt_set_t set);
/** @brief Set name as used in session.csv and the UI progress line
 * ("words"|"sentences"|"negatives"|"wake"). */
const char *prompt_set_name(prompt_set_t set);
