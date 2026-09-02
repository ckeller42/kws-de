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
    puts("test_prompts OK");
    return 0;
}
