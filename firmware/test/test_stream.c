#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "stream.h"

static int LICHT, AN, SIL;

static void one_hot(float *v, int i, float p)
{
    for (int k = 0; k < KWS_NUM_LABELS; k++) v[k] = (1.f - p) / (KWS_NUM_LABELS - 1);
    v[i] = p;
}

/* push a sequence, collect fired labels */
static int run(stream_t *s, const int *seq, int n, int *events)
{
    int e = 0;
    float v[KWS_NUM_LABELS];
    for (int i = 0; i < n; i++) {
        one_hot(v, seq[i], 0.9f);
        int r = stream_push(s, v);
        if (r >= 0) events[e++] = r;
    }
    return e;
}

static int find(const char *name)
{
    for (int i = 0; i < KWS_NUM_LABELS; i++) if (!strcmp(KWS_LABELS[i], name)) return i;
    return -1;
}

int main(void)
{
    LICHT = find("Licht"); AN = find("an"); SIL = KWS_SILENCE_INDEX;
    assert(LICHT >= 0 && AN >= 0);
    stream_t s; int ev[16]; int n;

    /* fires once per sustained word (smooth_win=3 -> first candidate needs the mean to cross) */
    stream_reset(&s);
    int a[] = {LICHT, LICHT, LICHT, LICHT, LICHT, LICHT};
    n = run(&s, a, 6, ev); assert(n == 1 && ev[0] == LICHT);

    /* two words back to back, no swallowing */
    stream_reset(&s);
    int b[] = {LICHT, LICHT, LICHT, LICHT, AN, AN, AN, AN, AN};
    n = run(&s, b, 9, ev); assert(n == 2 && ev[0] == LICHT && ev[1] == AN);

    /* same word twice with a silence gap >= gap_steps between runs */
    stream_reset(&s);
    int c[] = {LICHT, LICHT, LICHT, LICHT, SIL, SIL, SIL, SIL, SIL, LICHT, LICHT, LICHT, LICHT};
    n = run(&s, c, 13, ev); assert(n == 2 && ev[0] == LICHT && ev[1] == LICHT);

    /* silence never fires */
    stream_reset(&s);
    int d[] = {SIL, SIL, SIL, SIL, SIL};
    n = run(&s, d, 5, ev); assert(n == 0);

    /* below threshold never fires */
    stream_reset(&s);
    float weak[KWS_NUM_LABELS]; n = 0;
    for (int i = 0; i < 6; i++) { one_hot(weak, LICHT, 0.3f); if (stream_push(&s, weak) >= 0) n++; }
    assert(n == 0);

    /* Confirmation-tone trigger: only a fire on a real command word counts.
       Nothing fired, _unknown_ and _silence_ all stay silent. */
    assert(stream_is_command(LICHT));
    assert(stream_is_command(AN));
    assert(!stream_is_command(-1));
    assert(!stream_is_command(KWS_UNKNOWN_INDEX));
    assert(!stream_is_command(KWS_SILENCE_INDEX));

    /* End to end: a sustained "Licht" is a command; a sustained _unknown_ does
       fire (stream_push suppresses only silence) but is not one. */
    stream_reset(&s);
    int e[] = {LICHT, LICHT, LICHT, LICHT};
    n = run(&s, e, 4, ev); assert(n == 1 && stream_is_command(ev[0]));
    stream_reset(&s);
    int f[] = {KWS_UNKNOWN_INDEX, KWS_UNKNOWN_INDEX, KWS_UNKNOWN_INDEX, KWS_UNKNOWN_INDEX};
    n = run(&s, f, 4, ev); assert(n == 1 && !stream_is_command(ev[0]));

    puts("test_stream OK");
    return 0;
}
