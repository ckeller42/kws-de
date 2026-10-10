#!/usr/bin/env python3
"""Generate firmware/test/intent_cases.h: a fixed list of fired-word sequences run
through kws_de.grammar.parse(), committed as the ground truth firmware/test/test_intent.c
checks its C port (main/intent.c) against.

Run from the repo root: uv run python scripts/gen-intent-cases.py
"""

import pathlib

import numpy as np

from kws_de import config, window_intent
from kws_de.grammar import Intent, parse
from kws_de.window_intent import rescore

OUT = pathlib.Path(__file__).resolve().parent.parent / "firmware/test/intent_cases.h"

# Word sequences a fired window can produce -- space-joined KWS_LABELS entries,
# in fire order. Covers: valid (no zone), valid with zone, valid brightness
# level, missing device, missing action, wrong order (zone/action before
# device), duplicate device, duplicate action, a zone on a device that takes
# none, an action the device does not support, a level without a device, two
# devices, an "_unknown_" word interleaved (dropped, not rejecting), and empty.
CASES = [
    "Licht an",
    "Licht Küche an",
    "Licht fünfzig",
    "Kühlschrank leise",
    "Heizung wärmer",
    "Aufstelldach auf",
    "an",
    "Licht",
    "an Licht",
    "Küche Licht an",
    "Licht Licht an",
    "Licht an aus",
    "Aufstelldach Küche auf",
    "Kühlschrank auf",
    "fünfzig",
    "Licht Heizung an",
    "Licht _unknown_ Küche an",
    "_unknown_",
    "",
    # Pending LIGHT_COMPOUNDS (kws_de.config): grammar-ready, not yet a
    # recognizable label until the next retrain -- see config's comment.
    "Küchenlicht an",
    "Außenlicht aus",
    "Leselicht an",
    "Küchenlicht heller",
    "Küchenlicht",
    "Licht Küchenlicht an",
    "Küchenlicht Küche an",
    # Pending SCENE_TRIGGERS (kws_de.config): a single token stands for a
    # COMPLETE intent (unlike LIGHT_COMPOUNDS, no following action needed).
    # Same pending status -- grammar-ready, not yet a recognizable label.
    "GuteNacht",
    "GutenMorgen",
    "Leseratte",
    "Nachtlicht",
    "Licht GuteNacht",
    "GuteNacht Licht",
    "Leseratte Lesen",
    "Nachtlicht fünfzig",
    # Embedded-word collisions: a sliding decoder also fires the class word a
    # compound contains -- its prefix ("Küche" in "Küchenlicht", "Lesen" in
    # "Leseratte") before it, or its "Licht" suffix after it. Both parsers
    # reject these today; they pin the C port to grammar.py on exactly the
    # sequences the pending words will produce once trained. Fixing the
    # rejection is decoder work (grammar-constrained decoding), not a table
    # change.
    "Küche Küchenlicht an",
    "Außen Außenlicht aus",
    "Lesen Leselicht an",
    "Küchenlicht Licht an",
    "Nachtlicht Licht",
    "Licht Nachtlicht",
    "Lesen Leseratte",
]


# intent_rescore() cases: (window_intent, window_seconds) as the device builds
# them, expected verdict from kws_de.window_intent.rescore() -- one per branch.
RESCORE_CASES = [
    ("Licht _unknown_", "Licht:0.90|an:0.83"),  # fills the missing action
    ("_unknown_ an", "Licht:0.90|an:0.83"),  # fills the missing device
    # base already valid ("Licht an"): a zone lost as _unknown_ is never recovered
    ("Licht _unknown_ an", "Licht:0.90|Küche:0.80|an:0.70"),
    ("Licht _unknown_", "Licht:0.90|an:0.10"),  # below the floor
    ("Licht _unknown_", "Licht:0.90|an:0.25"),  # exactly at the floor
    ("_unknown_ _unknown_", "Licht:0.90|an:0.83"),  # two fixable slots
    ("Licht _unknown_", "an:0.90"),  # seconds misaligned
    ("_unknown_ _unknown_ an", "_silence_:0.90|Licht:0.90|an:0.50"),  # bad candidate, then good
    ("Licht _unknown_", "Licht:0.90|Heizung:0.90"),  # substitution still invalid
    ("Licht an", "Licht:0.90|an:0.90"),  # already valid: untouched
    # near-full window + longest runner-up: would not fit intent_parse()'s line
    (
        "_unknown_ _unknown_ _unknown_ _unknown_ _unknown_ _unknown_ an",
        "fünfundsiebzig:0.90|Licht:0.10|Licht:0.10|Licht:0.10|Licht:0.10|Licht:0.10|an:0.10",
    ),
]


# intent_align() cases (E70): the window's smoothed posteriors as the device
# holds them (one row per step), expected verdict from
# kws_de.window_intent.align_scores()+decide() at the firmware constants.
# Built from raw per-step vectors and smoothed like stream.c (trailing mean
# over KWS_SMOOTH_WIN) so the C function sees exactly stream_t.last_smoothed.
def _post(label, p=0.9, second=None, p2=0.0):
    v = np.full(len(config.COMMAND_LABELS), 0.001)
    v[config.COMMAND_LABELS.index(label)] = p
    if second:
        v[config.COMMAND_LABELS.index(second)] = p2
    return v


_SIL = [_post("_silence_")]
ALIGN_CASES = [
    # (name, raw steps, first_ms)
    ("licht an", _SIL * 2 + [_post("Licht")] * 3 + [_post("an")] * 3, 100),
    (
        "licht kueche an",
        _SIL * 2 + [_post("Licht")] * 3 + [_post("Küche")] * 2 + [_post("an")] * 3,
        100,
    ),
    ("licht fuenfzig (level)", _SIL + [_post("Licht")] * 3 + [_post("fünfzig")] * 3, 100),
    ("aufstelldach auf", [_post("Aufstelldach")] * 3 + [_post("auf")] * 3, 100),
    # a one-step zone the fire path cannot emit (run 1 < KWS_MIN_CONSECUTIVE)
    (
        "one-step zone",
        _SIL * 3
        + [_post("Licht")] * 3
        + [_post("Küche", 0.6, "_unknown_", 0.39)]
        + [_post("_unknown_", 0.6, "Küche", 0.39)]
        + [_post("an")] * 3,
        100,
    ),
    # zone never above background: not invented
    (
        "zone unsupported",
        [_post("Licht")] * 3 + [_post("_unknown_", 0.5, "Küche", 0.45)] * 3 + [_post("an")] * 3,
        100,
    ),
    # weak everything: a candidate exists but its confidence is under tau
    (
        "below tau",
        [_post("Licht", 0.4, "_unknown_", 0.5)] * 3 + [_post("an", 0.4, "_unknown_", 0.5)] * 3,
        100,
    ),
    # nothing clears the floor: no candidate at all
    ("noise", [np.full(len(config.COMMAND_LABELS), 1 / len(config.COMMAND_LABELS))] * 8, 100),
    # action before device: no monotone path
    ("wrong order", [_post("an")] * 3 + _SIL * 3 + [_post("Licht")] * 3, 100),
    # a single step cannot carry two tokens
    ("one step", [_post("Licht", 0.6, "an", 0.3)], 100),
    # "aus" inside the wake tail is the "...Bus" artefact: masked, Licht + an remain
    ("tail aus", [_post("aus")] * 2 + [_post("Licht")] * 3 + [_post("an")] * 3, 100),
    # the same window opened late enough that "aus" is real: duplicate action, "an" wins the path
    ("late aus", [_post("aus")] * 2 + [_post("Licht")] * 3 + [_post("an")] * 3, 600),
    # exact tie between two actions: the first candidate in grammar order wins (an before aus)
    ("tie", [_post("Licht")] * 3 + [_post("an", 0.5, "aus", 0.5)] * 3, 100),
    # an action the device does not take: Kühlschrank + auf has no candidate; rejected or weak
    ("invalid action", [_post("Kühlschrank")] * 3 + [_post("auf")] * 3, 100),
    # longer than the device keeps: steps past INTENT_ALIGN_MAX_STEPS are dropped
    ("too long", _SIL * 30 + [_post("Licht")] * 3 + [_post("an")] * 3, 100),
]


def c_str(s: str | None) -> str:
    """A C string literal holding `s` verbatim (UTF-8 source, like gen/labels.h)."""
    if s is None:
        return "NULL"
    escaped = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def main() -> None:
    rows = []
    for words in CASES:
        got = parse([w for w in words.split() if w])
        if isinstance(got, Intent):
            rows.append((words, True, got.device, got.zone, got.action))
        else:
            rows.append((words, False, None, None, None))

    lines = [
        "/* generated by scripts/gen-intent-cases.py from kws_de.grammar.parse() -- do not edit */",
        "#pragma once",
        "",
        "typedef struct {",
        "    const char *words;",
        "    bool valid;",
        "    const char *device;",
        "    const char *zone;",
        "    const char *action;",
        "} intent_case_t;",
        "",
        f"#define INTENT_CASE_COUNT {len(rows)}",
        "static const intent_case_t INTENT_CASES[INTENT_CASE_COUNT] = {",
    ]
    for words, valid, device, zone, action in rows:
        lines.append(
            f"  {{{c_str(words)}, {'true' if valid else 'false'}, "
            f"{c_str(device)}, {c_str(zone)}, {c_str(action)}}},"
        )
    lines.append("};")
    lines += [
        "",
        "typedef struct {",
        "    const char *words;",
        "    const char *seconds;",
        "    bool valid;",
        "    const char *device;",
        "    const char *zone;",
        "    const char *action;",
        "    const char *to; /* substituted label, NULL if no substitution */",
        "} rescore_case_t;",
        "",
        f"#define RESCORE_CASE_COUNT {len(RESCORE_CASES)}",
        "static const rescore_case_t RESCORE_CASES[RESCORE_CASE_COUNT] = {",
    ]
    for words, seconds in RESCORE_CASES:
        got, sub = rescore(words.split(), seconds.split("|"))
        ok = isinstance(got, Intent)
        d, z, a = (got.device, got.zone, got.action) if ok else (None, None, None)
        lines.append(
            f"  {{{c_str(words)}, {c_str(seconds)}, {'true' if ok else 'false'}, "
            f"{c_str(d)}, {c_str(z)}, {c_str(a)}, {c_str(sub[1] if sub else None)}}},"
        )
    lines.append("};")
    lines += [
        "",
        "typedef struct {",
        "    const char *name;",
        "    int n_steps;",
        "    int first_ms;",
        "    const float *post; /* n_steps x KWS_NUM_LABELS smoothed posteriors, row-major */",
        "    bool valid;",
        "    const char *device;",
        "    const char *zone;",
        "    const char *action;",
        "} align_case_t;",
        "",
        f"#define ALIGN_CASE_COUNT {len(ALIGN_CASES)}",
    ]
    refs = []
    for i, (name, raw, first_ms) in enumerate(ALIGN_CASES):
        sm = window_intent.smoothed(raw, window_intent.ALIGN_SMOOTH_WIN)
        got, _, _ = window_intent.decide(
            window_intent.align_scores(
                sm[: window_intent.ALIGN_MAX_STEPS],
                config.COMMAND_LABELS,
                floor=window_intent.ALIGN_FLOOR,
                step_ms=100,
                first_ms=first_ms,
                smooth_win=1,  # already smoothed above
            ),
            window_intent.ALIGN_TAU,
            0.0,
        )
        ok = isinstance(got, Intent)
        d, z, a = (got.device, got.zone, got.action) if ok else (None, None, None)
        vals = ", ".join(f"{x:.6f}f" for x in sm.ravel())
        lines.append(f"static const float ALIGN_POST_{i}[{sm.size}] = {{{vals}}};")
        refs.append(
            f"  {{{c_str(name)}, {sm.shape[0]}, {first_ms}, ALIGN_POST_{i}, "
            f"{'true' if ok else 'false'}, {c_str(d)}, {c_str(z)}, {c_str(a)}}},"
        )
    lines.append("static const align_case_t ALIGN_CASES[ALIGN_CASE_COUNT] = {")
    lines += refs
    lines.append("};")
    lines.append("")
    OUT.write_text("\n".join(lines))
    print(
        f"wrote {OUT} ({len(rows)} parse + {len(RESCORE_CASES)} rescore + "
        f"{len(ALIGN_CASES)} align cases)"
    )


if __name__ == "__main__":
    main()
