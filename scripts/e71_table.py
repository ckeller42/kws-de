"""E71 summary: one row per arm x {streamed, gated} from the E69 harness JSONs in .e67/,
plus the same rows restricted to one recording session (default spk22, in no training set).

AC@0FA = accept-and-correct at the session-disjoint threshold (mean of the two folds),
FA = false accepts summed over both test folds; AC@<=1FA = operating point on all negatives.
Session subset: the threshold of the fold whose calibration excludes that session, applied
to its held-out clips and its own negatives; <=1 FA threshold as above.

Usage: uv run --no-sync python scripts/e71_table.py [--spk spk22]
"""

import argparse
import json
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parents[1] / ".e67"
ARMS = ["cnn", "dtw", "embed", "agree", "agree-scoredtw"]


def fmt(xs) -> str:
    return " | ".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in xs)


def subset(p: dict, spk: str) -> tuple:
    c = p["per_clip"]
    m = np.array(c["spk"]) == spk
    s, ok = np.array(c["score"])[m], np.array(c["correct"], dtype=bool)[m]
    lv = np.array(c["neg_level"])[np.array(c["neg_spk"]) == spk]
    thr0 = next(d for d in p["open_set"]["speaker_disjoint"] if spk in d["test"])["threshold"]
    thr1 = p["open_set"]["operating_points"]["1"]["threshold"]
    return (
        p["pooled"]["per_spk"][spk]["acc"],
        float(np.mean(ok & (s < thr0))),
        float(np.mean(ok & (s < thr1))),
        f"{int((lv < thr0).sum())}/{int((lv < thr1).sum())} of {len(lv)}",
    )


def cells(p: dict) -> tuple:
    o = p["open_set"]
    ac0 = np.mean([d["accept_correct"] for d in o["speaker_disjoint"]])
    return (
        p["pooled"]["acc"],
        ac0,
        o["operating_points"]["1"]["accept_correct"],
        o["disjoint_fa_sum"],
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spk", default="spk22")
    spk = ap.parse_args().spk

    def head(fa: str) -> str:
        cols = ["top-1", "AC@0FA", "AC@<=1FA", fa]
        return " | ".join(["| arm | mode"] + [f"{p} {c}" for p in "WP" for c in cols])

    rule = "|---" * 11 + "|"
    rows = [head("FA") + " | ms/match W / P |", rule]
    sub = [head("FA at 0/<=1") + " | |", rule]
    for arm in ARMS:
        for mode in ("streamed", "gated"):
            f = RES / f"results-{arm}-both-none{'-gated' if mode == 'gated' else ''}.json"
            if not f.exists():
                continue
            pools = json.loads(f.read_text())["pools"]
            w, p = pools["W"], pools["P"]
            ms = f"{w['ms_per_match']:.1f} / {p['ms_per_match']:.1f}"
            rows.append(f"| {arm} | {mode} | {fmt(cells(w))} | {fmt(cells(p))} | {ms} |")
            sub.append(f"| {arm} | {mode} | {fmt(subset(w, spk))} | {fmt(subset(p, spk))} | |")
    print("\n".join(rows + ["", f"Restricted to {spk}:", ""] + sub))


if __name__ == "__main__":
    main()
