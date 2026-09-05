"""Drop `say`-engine (and legacy pre-engine-field) TTS clips from a `kws-dataset build`
raw-clip cache, so the next build regenerates them through the synthetic-clip gate
(`kws_de.qc.tts_gate`) — see `docs/sphinx/pipeline.rst` "Synthetic-clip gate" and
`docs/paper-notes.md` E23. Real clips (MSWC, no `tts:` prefix) and device recordings
(`rec:...`) are never touched.

Usage:
  uv run --no-sync python scripts/drop-tts-clips.py <cache.pkl> [--dry-run]
"""

# NOTE: pickle here is this project's own local, gitignored raw-clip cache (see the
# same note at the top of kws_de/data.py) — written and read by this same codebase,
# never untrusted input, so unpickling's arbitrary-code-execution risk doesn't apply.
import argparse
import pickle
from pathlib import Path


def _is_droppable(speaker: str) -> bool:
    if not speaker.startswith("tts:"):
        return False
    parts = speaker.split(":")
    # Legacy format tts:<voice>:<rate> ends in a rate (digits, from tts.RATES) —
    # `say` was the only engine when this format was written, so it is say-engine
    # by construction. New format tts:<engine>:<voice> ends in a voice name
    # (never digits), and its engine is named explicitly in parts[1].
    if len(parts) == 3 and parts[2].isdigit():
        return True
    engine = parts[1] if len(parts) >= 2 else None
    return engine == "say"


def drop_tts_clips(clips: dict) -> dict[str, tuple[int, int]]:
    """Mutates `clips` in place. Returns {word: (before, after)} clip counts."""
    counts = {}
    for word, items in clips.items():
        before = len(items)
        kept = [(c, s) for c, s in items if not _is_droppable(s)]
        clips[word] = kept
        counts[word] = (before, len(kept))
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cache", type=Path, help="raw-clip cache, e.g. data/raw_clips_v3.pkl")
    ap.add_argument("--dry-run", action="store_true", help="report counts, write nothing back")
    args = ap.parse_args()

    with open(args.cache, "rb") as fh:
        cached = pickle.load(fh)
    counts = drop_tts_clips(cached["clips"])

    total_before = sum(b for b, _ in counts.values())
    total_after = sum(a for _, a in counts.values())
    for word in sorted(counts):
        before, after = counts[word]
        if before != after:
            print(f"  {word}: {before} -> {after} (-{before - after})")
    print(f"total: {total_before} -> {total_after} (-{total_before - total_after})")

    if args.dry_run:
        print("dry run: cache not written")
        return
    with open(args.cache, "wb") as fh:
        pickle.dump(cached, fh)
    print(f"wrote {args.cache}")


if __name__ == "__main__":
    main()
