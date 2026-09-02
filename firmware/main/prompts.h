#pragma once
/* prompts.h (our module; includes gen/prompts.h) */
#include <stdint.h>
#include "gen/prompts.h"

typedef enum { PROMPT_WORDS = 0, PROMPT_SENTENCES = 1, PROMPT_NEGS = 2 } prompt_set_t;
typedef struct { prompt_set_t set; uint32_t seed; int order[64]; int count; int index; } prompt_session_t;

/* Fisher–Yates with a 32-bit xorshift seeded from `seed`; same seed → same order. */
void        prompt_session_init(prompt_session_t *p, prompt_set_t set, uint32_t seed);
const char *prompt_text(const prompt_session_t *p);   /* display text for order[index] */
const char *prompt_slug(const prompt_session_t *p);
int         prompt_advance(prompt_session_t *p);      /* returns 0 when the set is exhausted */
uint32_t    prompt_cap_ms(prompt_set_t set);          /* 4000 words, 6000 otherwise */
