/**
 * @file prompts.h
 * @brief Randomised prompt-set session for the guided recorder (wraps gen/prompts.h).
 */
#pragma once
#include <stdint.h>
#include "gen/prompts.h"

/** @brief Which prompt table a session draws from. */
typedef enum { PROMPT_WORDS = 0, PROMPT_SENTENCES = 1, PROMPT_NEGS = 2 } prompt_set_t;
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
