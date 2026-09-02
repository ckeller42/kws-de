#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "prompts.h"

int main(void)
{
    prompt_session_t a, b;
    prompt_session_init(&a, PROMPT_WORDS, 17);
    prompt_session_init(&b, PROMPT_WORDS, 17);
    assert(a.count == KWS_NUM_WORD_PROMPTS);
    assert(!memcmp(a.order, b.order, sizeof(int) * a.count));
    prompt_session_init(&b, PROMPT_WORDS, 18);
    assert(memcmp(a.order, b.order, sizeof(int) * a.count) != 0);

    /* every index appears exactly once */
    int seen[64] = {0};
    for (int i = 0; i < a.count; i++) seen[a.order[i]]++;
    for (int i = 0; i < a.count; i++) assert(seen[i] == 1);

    /* walk to the end */
    int steps = 1;
    while (prompt_advance(&a)) steps++;
    assert(steps == a.count);

    prompt_session_init(&a, PROMPT_SENTENCES, 1); assert(a.count == KWS_NUM_SENTENCE_PROMPTS);
    prompt_session_init(&a, PROMPT_NEGS, 1);      assert(a.count == KWS_NUM_NEG_PROMPTS);
    assert(strlen(prompt_text(&a)) > 0 && strlen(prompt_slug(&a)) > 0);
    assert(prompt_cap_ms(PROMPT_WORDS) == 4000 && prompt_cap_ms(PROMPT_NEGS) == 6000);

    /* Trailing-silence hangover: words keep the short 500 ms hangover; sentences,
       negatives and wake need 1200 ms so a natural pause doesn't cut a take. */
    assert(prompt_hangover_ms(PROMPT_WORDS) == 500);
    assert(prompt_hangover_ms(PROMPT_SENTENCES) == 1200);
    assert(prompt_hangover_ms(PROMPT_NEGS) == 1200);
    assert(prompt_hangover_ms(PROMPT_WAKE) == 1200);

    /* wake set: KWS_NUM_WAKE_PROMPTS reads, all "Hey Bus", set name "wake",
       exactly one take per prompt (real positives, no doubled reads). */
    prompt_session_init(&a, PROMPT_WAKE, 1);
    assert(a.count == KWS_NUM_WAKE_PROMPTS && a.count == 5);
    for (int i = 0; i < a.count; i++) {
        assert(!strcmp(prompt_text(&a), "Hey Bus"));
        assert(!strcmp(prompt_slug(&a), "hey-bus"));
        prompt_advance(&a);
    }
    assert(!strcmp(prompt_set_name(PROMPT_WAKE), "wake"));
    assert(prompt_takes_per_prompt(PROMPT_WAKE) == 1);
    assert(prompt_takes_per_prompt(PROMPT_SENTENCES) == 2);

    puts("test_prompts OK");
    return 0;
}
