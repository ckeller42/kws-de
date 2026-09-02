#include "prompts.h"
#include <string.h>

_Static_assert(KWS_NUM_SENTENCE_PROMPTS <= 64, "grow prompt_session_t.order");

static uint32_t xorshift(uint32_t *s) { *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5; return *s; }

static void set_tables(prompt_set_t set, const char *const **text, const char *const **slug, int *n)
{
    switch (set) {
    case PROMPT_SENTENCES: *text = KWS_SENTENCE_PROMPTS; *slug = KWS_SENTENCE_SLUGS; *n = KWS_NUM_SENTENCE_PROMPTS; break;
    case PROMPT_NEGS:      *text = KWS_NEG_PROMPTS;      *slug = KWS_NEG_SLUGS;      *n = KWS_NUM_NEG_PROMPTS;      break;
    case PROMPT_WAKE:      *text = KWS_WAKE_PROMPTS;     *slug = KWS_WAKE_SLUGS;     *n = KWS_NUM_WAKE_PROMPTS;     break;
    default:               *text = KWS_WORD_PROMPTS;     *slug = KWS_WORD_SLUGS;     *n = KWS_NUM_WORD_PROMPTS;     break;
    }
}

void prompt_session_init(prompt_session_t *p, prompt_set_t set, uint32_t seed)
{
    const char *const *t, *const *s; int n;
    set_tables(set, &t, &s, &n);
    p->set = set; p->seed = seed; p->count = n; p->index = 0;
    for (int i = 0; i < n; i++) p->order[i] = i;
    uint32_t rng = seed ? seed : 0x9E3779B9u;
    for (int i = n - 1; i > 0; i--) {       /* Fisher–Yates */
        int j = (int)(xorshift(&rng) % (uint32_t)(i + 1));
        int tmp = p->order[i]; p->order[i] = p->order[j]; p->order[j] = tmp;
    }
}

const char *prompt_text(const prompt_session_t *p)
{
    const char *const *t, *const *s; int n; set_tables(p->set, &t, &s, &n);
    return t[p->order[p->index]];
}

const char *prompt_slug(const prompt_session_t *p)
{
    const char *const *t, *const *s; int n; set_tables(p->set, &t, &s, &n);
    return s[p->order[p->index]];
}

int prompt_advance(prompt_session_t *p)
{
    if (p->index + 1 >= p->count) return 0;
    p->index++;
    return 1;
}

uint32_t prompt_cap_ms(prompt_set_t set) { return set == PROMPT_WORDS ? 4000 : 6000; }

int prompt_takes_per_prompt(prompt_set_t set) { return set == PROMPT_WAKE ? 1 : 2; }

const char *prompt_set_name(prompt_set_t set)
{
    static const char *names[] = {"words", "sentences", "negatives", "wake"};
    return names[set];
}
