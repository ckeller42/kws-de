/* #133: the light-only build (CONFIG_KWS_INTENTS_LIGHT_ONLY) lets only
 * "Licht -> an" and "Licht -> aus" out of intent_allowed() and intent_align(). */
#include <stdio.h>
#include <string.h>
#include "gen/labels.h"
#include "intent.h"

static int fails;
#define CHECK(c)                                                    \
    do {                                                            \
        if (!(c)) { printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); fails++; } \
    } while (0)

static int idx(const char *tok)
{
    for (int i = 0; i < KWS_NUM_LABELS; i++)
        if (strcmp(tok, KWS_LABELS[i]) == 0) return i;
    return -1;
}

/* T steps: silence, then `a` for 4 steps, silence, `b` for 4 steps, silence. */
static intent_t align_two(const char *a, const char *b)
{
    enum { T = 14 };
    static float post[T][KWS_NUM_LABELS];
    memset(post, 0, sizeof post);
    for (int t = 0; t < T; t++) {
        int k = (t >= 2 && t < 6) ? idx(a) : (t >= 7 && t < 11) ? idx(b) : KWS_SILENCE_INDEX;
        post[t][k] = 0.95f;
        post[t][KWS_UNKNOWN_INDEX] += 0.05f;
    }
    float conf;
    /* first_ms far past the wake tail, so "aus" is not masked */
    return intent_align(&post[0][0], T, 5000, 100, INTENT_ALIGN_FLOOR, 0.f, &conf);
}

int main(void)
{
    intent_t on = intent_parse("Licht an"), off = intent_parse("Licht aus");
    CHECK(intent_allowed(&on));
    CHECK(intent_allowed(&off));
    const char *blocked[] = {"Licht Küche an", "Heizung an", "Licht heller", "Licht fünfzig", "Aufstelldach auf"};
    for (size_t i = 0; i < sizeof blocked / sizeof *blocked; i++) {
        intent_t in = intent_parse(blocked[i]);
        CHECK(in.valid);
        CHECK(!intent_allowed(&in));
    }
    intent_t none = intent_parse("");
    CHECK(!intent_allowed(&none));
    CHECK(!intent_allowed(NULL));

    intent_t r = align_two("Licht", "an");
    CHECK(r.valid && strcmp(r.action, "an") == 0 && !r.zone);
    r = align_two("Licht", "aus");
    CHECK(r.valid && strcmp(r.action, "aus") == 0 && !r.zone);
    /* A window that says something else can only align to an allowed intent. */
    r = align_two("Heizung", "an");
    CHECK(!r.valid || intent_allowed(&r));

    if (fails) return 1;
    printf("test_intent_light OK\n");
    return 0;
}
