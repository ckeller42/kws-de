"""E69 harness: score a few-shot enrollment matcher (an "arm", `scripts/e69_<arm>.py`
exposing `make_matcher(templates) -> match(sig) -> [(label, distance), ...]`
ascending) by leave-one-take-out on the approved recordings, next to the deployed
command model on the same held-out clips.

Pools: W = approved/words + approved/context (label = dir), P = approved/phrases
(label = the grammar intent of the prompt, via `prompt_intent`; prompts with no
intent dropped). Every clip whose (speaker, label) has >=2 takes is held out once;
templates = every other clip of the pool, capped at --max-takes per (speaker,
label) (lowest takes, guided words before context), pooled over all speakers (the
device does not know who speaks) or same-speaker only.

Open set: approved/negatives, W = trailing 1 s window every 100 ms (the device
stream), false accept = score < threshold on 2 consecutive steps; P = whole clip +
2.5 s windows every 100 ms, false accept on any. --gated (E71): every negative gets ONE
match on the whole clip, like the positives (one match right after the wake word).
Score (--norm): none = best
distance d1, margin = d1/d2 (best over second-best label; lower = more confident).
Accept-and-correct = held-out clip's top label right AND its score < threshold.
Threshold calibration is speaker-disjoint: negatives split by speaker into 2 folds
(greedy by clip count), threshold = largest with 0 FA on one fold, FA counted on the
other, then swapped. The old number (calibrated and counted on all negatives) is kept
as a labelled optimistic row. Plus a 30-quantile threshold sweep (FA vs
accept-and-correct) and the operating points at <=1 and <=2 FA on all negatives.

Usage:
  uv run --no-sync python scripts/e69_enroll_eval.py --arm dtw --pool both --norm margin
"""

import argparse
import csv
import importlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

from kws_de import config
from kws_de.eval import _stream_posteriors, intent_text, prompt_intent

SCRIPTS = Path(__file__).resolve().parent
STEP = config.SAMPLE_RATE // 10  # 100 ms
P_WIN = config.SAMPLE_RATE * 5 // 2  # 2.5 s
N_QUANTILES = 30
OP_FA = (1, 2)

# Worker state, set before the fork (read-only in workers).
ARM = None
CLIPS: dict[str, list[dict]] = {}
NEG: list[np.ndarray] = []
NEG_SPK: list[str] = []
MAX_TAKES = 5
NORM = "none"
GATED = False
_FULL: dict = {}


def score(ranked: list[tuple[str, float]]) -> float:
    """Rejection score of a match result, lower = more confident."""
    if NORM == "margin":
        return ranked[0][1] / max(ranked[1][1], 1e-12) if len(ranked) > 1 else 1.0
    return ranked[0][1]


def speaker_folds(spks: list[str]) -> np.ndarray:
    """Fold (0/1) per clip: speakers, largest first, go to the lighter fold."""
    load, fold = [0, 0], {}
    for s, n in sorted(Counter(spks).items(), key=lambda x: (-x[1], x[0])):
        fold[s] = load.index(min(load))
        load[fold[s]] += n
    return np.array([fold[s] for s in spks])


def load_arm(name: str):
    sys.path.insert(0, str(SCRIPTS))
    return importlib.import_module(f"e69_{name}")


def _read(path: Path) -> np.ndarray:
    return sf.read(path, dtype="float32", always_2d=True)[0][:, 0]


def load_pool(approved: Path, pool: str) -> list[dict]:
    clips = []
    if pool == "W":
        for order, src in enumerate(("words", "context")):
            for f in sorted((approved / src).glob("*/*.wav")):
                spk, take = f.stem.split("_")[0], int(f.stem.rsplit("_", 1)[1])
                clips.append({"file": f, "label": f.parent.name, "spk": spk, "key": (order, take)})
    else:
        with (approved / "phrases" / "index.csv").open() as fh:
            for r in csv.DictReader(fh):
                intent = prompt_intent(r["prompt"])
                if type(intent).__name__ != "Intent":
                    continue
                f = approved / r["file"]
                take = int(f.stem.rsplit("_", 1)[1])
                clips.append(
                    {"file": f, "label": intent_text(intent), "spk": r["speaker"],
                     "key": (take, f.name), "intent": intent}
                )  # fmt: skip
    for c in clips:
        c["sig"] = _read(c["file"])
    return clips


def templates(clips: list[dict], held: int | None, same_spk: bool) -> dict:
    groups = defaultdict(list)
    for i, c in enumerate(clips):
        if i == held or (same_spk and c["spk"] != clips[held]["spk"]):
            continue
        groups[c["spk"], c["label"]].append(c)
    out = defaultdict(list)
    for (_, lab), cs in groups.items():
        out[lab] += [c["sig"] for c in sorted(cs, key=lambda c: c["key"])[:MAX_TAKES]]
    return out


def _loto(task):
    pool, h, same_spk = task
    match = ARM.make_matcher(templates(CLIPS[pool], h, same_spk))
    t = time.perf_counter()
    ranked = match(CLIPS[pool][h]["sig"])
    return ranked[0][0], score(ranked), (time.perf_counter() - t) * 1e3


def _full_matcher(pool):
    # Built lazily per worker (an arm may hold a framework that is not fork-safe).
    if pool not in _FULL:
        _FULL[pool] = ARM.make_matcher(templates(CLIPS[pool], None, False))
    return _FULL[pool]


def _neg_fire_level(task) -> float:
    """Smallest threshold above which this negative clip fires (inf: never)."""
    pool, j = task
    match, sig = _full_matcher(pool), NEG[j]
    best = lambda w: score(match(w))  # noqa: E731
    if GATED:
        return best(sig)
    if pool == "W":
        d = [float(x) for x in _stream_posteriors(best, sig, STEP)]
        return min((max(a, b) for a, b in zip(d, d[1:], strict=False)), default=math.inf)
    starts = range(0, max(len(sig) - P_WIN, 0) + 1, STEP) if len(sig) > P_WIN else []
    return min([best(sig)] + [best(sig[s : s + P_WIN]) for s in starts])


def acc_rows(clips, held, correct) -> dict:
    per = defaultdict(lambda: [0, 0])
    for h, ok in zip(held, correct, strict=True):
        per[clips[h]["spk"]][0] += bool(ok)
        per[clips[h]["spk"]][1] += 1
    n = len(held)
    return {
        "n": n,
        "acc": sum(map(bool, correct)) / n if n else 0.0,
        "per_spk": {s: {"n": t, "acc": k / t} for s, (k, t) in sorted(per.items())},
    }


def baseline(pool, clips, held) -> list[bool]:
    from kws_de.codegen import model_bytes
    from kws_de.eval import make_command_predict_fn
    from kws_de.window_intent import decode_window

    root = SCRIPTS.parent
    fn = make_command_predict_fn(model_bytes(root / "firmware/main/gen/model_data.h"))
    labels = config.COMMAND_LABELS
    if pool == "W":
        return [labels[int(np.argmax(fn(clips[h]["sig"])))] == clips[h]["label"] for h in held]
    out = []
    for h in held:  # the eval_recordings phrase path, on the held-out clips only
        steps = _stream_posteriors(fn, clips[h]["sig"], STEP)
        out.append(decode_window(steps, labels, 100, first_ms=config.CLIP_MS) == clips[h]["intent"])
    return out


def run_pool(pool: str, args, pmap) -> dict:
    clips = CLIPS[pool]
    count = defaultdict(int)
    for c in clips:
        count[c["spk"], c["label"]] += 1
    held = [i for i, c in enumerate(clips) if count[c["spk"], c["label"]] >= 2]
    negs = list(range(len(NEG)))
    if args.limit:
        held = held[:: max(1, len(held) // args.limit)][: args.limit]
        negs = negs[: args.limit]
    res = {"labels": len({c["label"] for c in clips}), "templates_pool": len(clips)}
    pooled = pmap(_loto, [(pool, h, False) for h in held])
    own = pmap(_loto, [(pool, h, True) for h in held])
    for name, r in (("pooled", pooled), ("same_speaker", own)):
        res[name] = acc_rows(
            clips, held, [lab == clips[h]["label"] for h, (lab, _, _) in zip(held, r, strict=True)]
        )
    res["ms_per_match_loaded"] = float(np.mean([ms for *_, ms in pooled]))
    match = ARM.make_matcher(templates(clips, held[0], False))  # unloaded CPU timing
    match(np.random.default_rng(0).standard_normal(config.CLIP_SAMPLES).astype(np.float32))
    ts = []  # after a warm-up: lazy model loads (cnn/embed) are not per-match cost
    for h in held[:20]:
        t = time.perf_counter()
        match(clips[h]["sig"])
        ts.append((time.perf_counter() - t) * 1e3)
    res["ms_per_match"] = float(np.mean(ts))
    levels = pmap(_neg_fire_level, [(pool, j) for j in negs])
    correct = [lab == clips[h]["label"] for h, (lab, _, _) in zip(held, pooled, strict=True)]
    res["open_set"] = open_set(
        levels, [NEG_SPK[j] for j in negs], [s for _, s, _ in pooled], correct
    )
    res["deployed"] = acc_rows(clips, held, baseline(pool, clips, held))
    res["per_clip"] = {  # for post-hoc subsets (scripts/e71_table.py)
        "spk": [clips[h]["spk"] for h in held],
        "score": [s for _, s, _ in pooled],
        "correct": correct,
        "neg_spk": [NEG_SPK[j] for j in negs],
        "neg_level": [float(x) for x in levels],
    }
    return res


def open_set(levels, neg_spk, scores, correct) -> dict:
    """levels: per negative, the threshold above which it fires; scores/correct: held-out."""
    lv, s, ok = np.array(levels), np.array(scores), np.array(correct, dtype=bool)

    def point(thr, test=lv) -> dict:
        return {
            "threshold": float(thr),
            "false_accepts": int((test < thr).sum()),
            "n_test": len(test),
            "accept_correct": float(np.mean(ok & (s < thr))),
        }

    fold = speaker_folds(neg_spk)
    spk = [sorted({x for x, f in zip(neg_spk, fold, strict=True) if f == k}) for k in (0, 1)]
    disjoint = [
        {"calibrate": spk[a], "test": spk[1 - a], **point(lv[fold == a].min(), lv[fold != a])}
        for a in (0, 1)
    ]
    pooled = np.concatenate([s[np.isfinite(s)], lv[np.isfinite(lv)]])
    grid = np.unique(np.quantile(pooled, np.linspace(0, 1, N_QUANTILES)))
    srt = np.sort(lv)
    return {
        "n_negatives": len(lv),
        "norm": NORM,
        "speaker_disjoint": disjoint,
        "disjoint_fa_sum": sum(d["false_accepts"] for d in disjoint),
        "optimistic": point(lv.min()),  # calibrated and counted on the same negatives
        "operating_points": {k: point(srt[k] if k < len(srt) else math.inf) for k in OP_FA},
        "curve": [point(t) for t in grid],
    }


def render(arm: str, results: dict) -> str:
    out = [f"# E69 enrollment eval, arm `{arm}`, max-takes {MAX_TAKES}", ""]
    for pool, r in results.items():
        spks = sorted(r["pooled"]["per_spk"])
        out += [
            f"## Pool {pool}: {r['pooled']['n']} held-out clips, {r['labels']} labels, "
            f"{r['ms_per_match']:.1f} ms/match single process "
            f"({r['ms_per_match_loaded']:.1f} under --jobs load), pooled templates",
            "",
            "| method | overall | " + " | ".join(spks) + " |",
            "|---|---|" + "---|" * len(spks),
        ]
        for name, row in (
            (f"{arm} pooled templates", r["pooled"]),
            (f"{arm} same-speaker templates", r["same_speaker"]),
            ("deployed model (86b7105e path)", r["deployed"]),
        ):
            cells = [f"{row['per_spk'][s]['acc']:.3f} ({row['per_spk'][s]['n']})" for s in spks]
            out.append(f"| {name} | {row['acc']:.3f} ({row['n']}) | " + " | ".join(cells) + " |")
        o = r["open_set"]
        n = o["n_negatives"]
        out += [
            "",
            f"Open set, score `{o['norm']}` ({n} negatives), speaker-disjoint calibration "
            "(threshold = largest with 0 FA on the calibration fold):",
            "",
            "| calibrate on | test on | threshold | FA on test fold | accept-and-correct |",
            "|---|---|---|---|---|",
        ]
        rows = [(",".join(d["calibrate"]), ",".join(d["test"]), d) for d in o["speaker_disjoint"]]
        rows.append(("all (optimistic)", "same negatives", o["optimistic"]))
        for cal, test, d in rows:
            out.append(
                f"| {cal} | {test} ({d['n_test']}) | {d['threshold']:.4f} | "
                f"{d['false_accepts']} | {d['accept_correct']:.3f} |"
            )
        acs = [d["accept_correct"] for d in o["speaker_disjoint"]]
        out.append(f"| sum / mean | | | {o['disjoint_fa_sum']} | {np.mean(acs):.3f} |")
        out += ["", f"Operating points on all {n} negatives:", ""]
        for k, d in o["operating_points"].items():
            out.append(
                f"- <={k} FA: threshold {d['threshold']:.4f}, {d['false_accepts']} FA, "
                f"accept-and-correct {d['accept_correct']:.3f}"
            )
        curve = o["curve"]
        k = next((i for i, c in enumerate(curve) if c["false_accepts"]), len(curve))
        out += [
            "",
            f"Threshold sweep ({len(curve)} quantiles of held-out scores + negative levels; "
            "rows around the first FA):",
            "",
            "| threshold | FA | accept-and-correct |",
            "|---|---|---|",
        ]
        for c in curve[max(0, k - 6) : k + 6]:
            out.append(
                f"| {c['threshold']:.4f} | {c['false_accepts']} | {c['accept_correct']:.3f} |"
            )
        out.append("")
    return "\n".join(out)


def main() -> None:
    global ARM, MAX_TAKES, NORM, GATED
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--norm", default="none", choices=["none", "margin"])
    ap.add_argument(
        "--arm", required=True, help="dtw | embed | cnn | agree -> scripts/e69_<arm>.py"
    )
    ap.add_argument("--gated", action="store_true", help="one whole-clip match per negative")
    ap.add_argument("--pool", default="both", choices=["W", "P", "both"])
    ap.add_argument("--max-takes", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="held-out clips / negatives per pool")
    ap.add_argument("--jobs", type=int, default=min(32, os.cpu_count() or 1))
    ap.add_argument("--approved", type=Path, default=config.DATA_DIR / "recordings" / "approved")
    args = ap.parse_args()

    ARM, MAX_TAKES, NORM, GATED = load_arm(args.arm), args.max_takes, args.norm, args.gated
    pools = ["W", "P"] if args.pool == "both" else [args.pool]
    for p in pools:
        CLIPS[p] = load_pool(args.approved, p)
    with (args.approved / "negatives" / "index.csv").open() as fh:
        for r in csv.DictReader(fh):
            NEG.append(_read(args.approved / r["file"]))
            NEG_SPK.append(r["speaker"])

    results = {}
    for p in pools:
        if args.jobs > 1:
            with Pool(args.jobs) as mp:
                results[p] = run_pool(p, args, lambda f, xs, mp=mp: mp.map(f, xs, chunksize=1))
        else:
            results[p] = run_pool(p, args, lambda f, xs: [f(x) for x in xs])

    mode = getattr(ARM, "MODE", None)  # e.g. E69_EMBED_MODE
    name = "-".join(x for x in (args.arm, mode, args.pool, NORM) if x and x != "auto")
    if GATED:
        name += "-gated"
    if args.limit:
        name += f"-limit{args.limit}"
    report = render(name, results)
    print(report)
    out = SCRIPTS.parent / ".e67" / f"results-{name}"
    out.parent.mkdir(exist_ok=True)
    out.with_suffix(".md").write_text(report)
    meta = {
        "arm": args.arm,
        "mode": mode,
        "norm": NORM,
        "max_takes": MAX_TAKES,
        "limit": args.limit,
        "gated": GATED,
    }
    out.with_suffix(".json").write_text(
        json.dumps({**meta, "pools": results}, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
