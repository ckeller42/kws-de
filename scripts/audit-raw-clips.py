"""Read-only quality audit of a raw clip cache plus the merged QC-approved recordings (E74).

Prints, per command label, the clip count by source with the distinct speaker/voice ids and
(for TTS) the distinct waveforms; the median level per source; the clip-shape findings; exact
duplicates; and how many val/test clips have an exact twin in train under the build's own
split. Nothing is written.

Usage:
  uv run --no-sync python scripts/audit-raw-clips.py [--cache raw_clips_v3.pkl] [--seed 0]
"""

import argparse
import collections
import hashlib
import pickle

import numpy as np

from kws_de import config
from kws_de.data import merge_recordings, split_three_way
from kws_de.dataset import force_rec_to_train

SRCS = ["mswc", "rec", "ctx", "say", "piper", "tts-other"]
N = config.CLIP_SAMPLES


def src(s: str) -> str:
    if s.startswith("tts:say"):
        return "say"
    if s.startswith("tts:piper"):
        return "piper"
    if s.startswith("tts:"):
        return "tts-other"
    if s.startswith(("rec:", "ctx:")):
        return s[:3]
    return "mswc"


def wave_key(x) -> str:
    return hashlib.sha1(np.asarray(x, np.float32).tobytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="raw_clips_v3.pkl")
    ap.add_argument("--seed", type=int, default=0, help="split seed for the leak check")
    args = ap.parse_args()

    path = config.DATA_DIR / args.cache
    blob = path.read_bytes()
    # pickle: our own gitignored local cache written by kws_de.data, never untrusted input.
    ws = pickle.loads(blob)["clips"]  # noqa: S301
    n_cache = sum(len(v) for v in ws.values())
    merge_recordings(ws)
    print(f"cache {args.cache}: sha256 {hashlib.sha256(blob).hexdigest()[:16]}, {n_cache} clips")
    print(f"with merged recordings: {sum(len(v) for v in ws.values())} clips\n")

    labels = [w for w in config.COMMAND_LABELS if w != "_silence_"]
    issues = collections.Counter()
    by_src = collections.Counter()  # (src, finding) -> n, word labels only
    level = collections.defaultdict(lambda: collections.defaultdict(list))
    groups = collections.defaultdict(list)  # waveform -> [(label, speaker id)]
    lengths = collections.defaultdict(list)
    rows = {}
    for w, items in ws.items():
        cnt = collections.Counter()
        ids = collections.defaultdict(set)
        waves = collections.defaultdict(set)
        for x, s in items:
            k = src(s)
            x = np.asarray(x, np.float32).ravel()
            h = wave_key(x)
            cnt[k] += 1
            ids[k].add(s)
            waves[k].add(h)
            groups[h].append((w, s))
            if not np.isfinite(x).all():
                issues[("nan", k)] += 1
                continue
            rms = float(np.sqrt(np.mean(x**2))) if x.size else 0.0
            db = 20 * np.log10(rms + 1e-9)
            level[k][w].append(db)
            lengths[k].append(len(x))
            issues[("length != 16,000", k)] += int(len(x) != N)
            issues[("very quiet (< -50 dBFS)", k)] += int(db < -50)
            issues[("clipped (> 10 samples at |x| > 0.99)", k)] += int(
                (np.abs(x) > 0.99).sum() > 10
            )
            issues[("DC offset (|mean| > 0.02)", k)] += int(abs(float(x.mean())) > 0.02)
            if w == "_unknown_":
                continue
            # The window the build feeds: pad at the end, keep the first second
            # (kws_de.data._random_shift), then shift by up to +/-200 ms.
            y = np.pad(x, (0, max(0, N - len(x))))[:N]
            fr = np.sqrt(np.mean(y.reshape(-1, 160) ** 2, axis=1))
            thr = max(fr.max() * 0.3, 1e-4)
            by_src[(k, "n")] += 1
            by_src[(k, "short (< 1 s, end-padded)")] += len(x) < N
            by_src[(k, "long (> 1 s, tail cut)")] += len(x) > N
            by_src[(k, "speech in first 50 ms")] += bool(fr[:5].max() > thr)
            by_src[(k, "speech in last 50 ms")] += bool(fr[-5:].max() > thr)
            if len(x) > N:
                e = x**2
                by_src[(k, "> 50 % energy after 1 s")] += bool(e[N:].sum() > 0.5 * e.sum())
        rows[w] = (cnt, ids, waves)

    print("== per label: clips[ids] by source; TTS also {distinct waveforms} ==")
    print(f"{'label':16}" + "".join(f"{k:>16}" for k in SRCS) + f"{'total':>7}{'TTS%':>6}")
    order = {w: i for i, w in enumerate(labels)}
    for w in sorted(rows, key=lambda w: order.get(w, 99)):
        cnt, ids, waves = rows[w]
        cells = ""
        for k in SRCS:
            cell = f"{cnt[k]}[{len(ids[k])}]" if cnt[k] else "-"
            if cnt[k] and k in ("say", "piper", "tts-other"):
                cell += f"{{{len(waves[k])}}}"
            cells += f"{cell:>16}"
        tot = sum(cnt.values())
        tts = cnt["say"] + cnt["piper"] + cnt["tts-other"]
        flag = "" if w in labels else "  (not a class)"
        print(f"{w:16}{cells}{tot:>7}{100 * tts / max(tot, 1):>5.0f}%{flag}")
    print("\nclass labels with no clips:", [w for w in labels if w not in ws])

    print("\n== median RMS level per source (dBFS; range of the per-label medians) ==")
    for k in SRCS:
        med = [float(np.median(v)) for v in level[k].values() if v]
        if med:
            print(f"  {k:6} {np.median(med):6.1f}  ({min(med):.1f} .. {max(med):.1f})")

    print("\n== issues, all clips, by source ==")
    for f in dict.fromkeys(f for f, _ in issues):
        per = {k: issues[(f, k)] for k in SRCS if issues[(f, k)]}
        print(f"  {f}: {sum(per.values())}  {per}")
    print("\n== clip length per source (samples; p10 / p50 / p90) ==")
    for k in SRCS:
        if lengths[k]:
            p10, p50, p90 = (int(v) for v in np.percentile(lengths[k], [10, 50, 90]))
            print(f"  {k:6} {p10:7} {p50:7} {p90:7}")
    print("\n== window findings, command-word clips, by source ==")
    for k in SRCS:
        n = by_src[(k, "n")]
        if not n:
            continue
        found = {f: v for (s, f), v in by_src.items() if s == k and f != "n" and v}
        print(f"  {k:6} n={n:5}  " + ", ".join(f"{f} {v}" for f, v in found.items()))

    print("\n== exact duplicates ==")
    dups = [g for g in groups.values() if len(g) > 1]
    print(f"  {len(dups)} groups, {sum(len(g) - 1 for g in dups)} extra copies")
    print(f"  across labels: {sum(len({w for w, _ in g}) > 1 for g in dups)}")
    nids = collections.Counter(len({s for _, s in g}) for g in dups)
    print(f"  distinct speaker ids per group: {sorted(nids.items())}")
    print(f"  sources: {collections.Counter(src(g[0][1]) for g in dups)}")

    print("\n== voices ==")
    for k in ("say", "piper"):
        voices = collections.Counter(
            s.split(":", 2)[2] for v in ws.values() for _, s in v if src(s) == k
        )
        n_waves = len({h for h, g in groups.items() if src(g[0][1]) == k})
        print(f"  {k}: {len(voices)} voice ids, {sum(voices.values())} clips, {n_waves} waveforms")
        print(f"    top: {voices.most_common(5)}")

    print(
        f"\n== exact twins across the build's split (seed {args.seed}, rec/ctx forced to train) =="
    )
    tr, va, te = split_three_way(ws, np.random.default_rng(args.seed), keep_speaker=True)
    force_rec_to_train(tr, va, te)
    train_keys = {wave_key(x) for v in tr.values() for x, _ in v}
    for name, split in (("val", va), ("test", te)):
        tw = collections.Counter()
        for v in split.values():
            for x, s in v:
                tw[(src(s), "n")] += 1
                tw[(src(s), "twin")] += wave_key(x) in train_keys
        print(
            f"  {name}: "
            + ", ".join(f"{k} {tw[(k, 'twin')]}/{tw[(k, 'n')]}" for k in SRCS if tw[(k, "n")])
        )


if __name__ == "__main__":
    main()
