"""Audit the approved recording tree: is every clip usable, and is it what its
path says it is?

QC writes `approved/` one session at a time, so nothing ever looks at the tree as
a whole. This does, and it is the check that would have caught #58 before a wake
round trained on the result:

* every clip readable, 16 kHz mono PCM_16, and inside its set's duration band
  (wake 0.4-2.0 s, words ~1 s, phrases 0.5-9.8 s, negatives up to 9.8 s);
* no phrase or negative that still contains the wake phrase — the field-derived
  ones are transcribed and matched against `qc._WAKE_RE`. Guided clips were
  already matched against their prompt at QC time, so they are not re-transcribed;
* `index.csv` rows and files one-to-one, in both directions;
* speaker directories named `spkNN`;
* counts per set, per speaker and per source (guided session vs field session).

A clip's source is its QC stamp: a session whose `qc.csv` holds any `set=field`
row is a field session, and every path in that stamp's `written.txt` is
field-derived. Clips written before stamps were kept report as `unknown`.

Usage:
  uv run --no-sync python scripts/audit-approved.py [--no-transcribe] [<approved>]

Exits 1 when anything failed, so it can gate a data pull.
"""

import collections
import csv
import re
import sys
from pathlib import Path

import soundfile as sf

from kws_de import config
from kws_de.qc import _WAKE_RE, normalise

SETS = ("words", "phrases", "negatives", "wake")
# (min, max) seconds. words: segmented clips are exactly config.CLIP_SAMPLES,
# bare word takes are raw ~1 s recordings, so the band is around 1 s rather than at it.
DURATION_S = {
    "words": (0.5, 2.0),
    "wake": (0.4, 2.0),
    "phrases": (0.5, 9.8),
    "negatives": (0.3, 9.8),
}
SPEAKER_RE = re.compile(r"^spk\d\d$")


def sources(recordings: Path) -> dict[str, str]:
    """approved-relative path -> "field" | "guided", read off the QC stamps."""
    out: dict[str, str] = {}
    qc_dir = recordings / "qc"
    for stamp in sorted(d for d in qc_dir.iterdir() if d.is_dir()) if qc_dir.is_dir() else []:
        written, qc_csv = stamp / "written.txt", stamp / "qc.csv"
        if not (written.exists() and qc_csv.exists()):
            continue
        with qc_csv.open() as fh:
            kind = "field" if any(r["set"] == "field" for r in csv.DictReader(fh)) else "guided"
        for line in written.read_text().splitlines():
            if line.strip():
                out[line.strip()] = kind
    return out


def transcriber_or_none(enabled: bool):
    if not enabled:
        return None
    from kws_de.qc import whisper_transcriber

    return whisper_transcriber()


def main() -> int:
    args = sys.argv[1:]
    transcribe = "--no-transcribe" not in args
    argv = [a for a in args if a != "--no-transcribe"]
    recordings = config.DATA_DIR / "recordings"
    approved = Path(argv[0]) if argv else recordings / "approved"

    problems: list[str] = []
    src = sources(recordings)
    per_set: dict[str, collections.Counter] = {s: collections.Counter() for s in SETS}
    per_source: collections.Counter = collections.Counter()
    field_speech: list[Path] = []  # field-derived phrases/negatives, to transcribe

    for name in SETS:
        root = approved / name
        if not root.is_dir():
            continue
        for wav in sorted(root.rglob("*.wav")):
            rel = str(wav.relative_to(approved))
            group = wav.parent.name  # spkNN for phrases/negatives/wake, a label for words
            if name != "words" and not SPEAKER_RE.match(group):
                problems.append(f"{rel}: speaker dir {group!r} is not spkNN")
            try:
                info = sf.info(wav)
            except Exception as e:  # noqa: BLE001 - an unreadable clip is a finding, not a crash
                problems.append(f"{rel}: unreadable ({type(e).__name__})")
                continue
            if (info.samplerate, info.channels, info.subtype) != (config.SAMPLE_RATE, 1, "PCM_16"):
                problems.append(
                    f"{rel}: {info.samplerate} Hz {info.channels}ch {info.subtype}, "
                    f"want {config.SAMPLE_RATE} Hz 1ch PCM_16"
                )
            dur = info.frames / info.samplerate
            lo, hi = DURATION_S[name]
            if not lo <= dur <= hi:
                problems.append(f"{rel}: {dur:.2f} s outside {lo}-{hi} s for {name}")
            per_set[name][group] += 1
            kind = src.get(rel, "unknown")
            per_source[(name, kind)] += 1
            if name in ("phrases", "negatives") and kind == "field":
                field_speech.append(wav)

        idx = root / "index.csv"
        if name == "words":
            continue  # the label directory is the index; there is no index.csv
        listed = set()
        if idx.exists():
            with idx.open() as fh:
                for row in csv.DictReader(fh):
                    if row["file"] in listed:
                        problems.append(f"{idx.name} ({name}): duplicate row {row['file']}")
                    listed.add(row["file"])
        on_disk = {str(w.relative_to(approved)) for w in root.rglob("*.wav")}
        for missing in sorted(on_disk - listed):
            problems.append(f"{missing}: on disk, no {name}/index.csv row")
        for orphan in sorted(listed - on_disk):
            problems.append(f"{orphan}: {name}/index.csv row, no file")

    tr = transcriber_or_none(transcribe and bool(field_speech))
    if tr is not None:
        print(f"transcribing {len(field_speech)} field-derived phrase/negative clips...")
        for wav in field_speech:
            text = tr(wav).get("text", "")
            toks = normalise(text)
            if any(
                _WAKE_RE.fullmatch("".join(toks[i : i + n]))
                for i in range(len(toks))
                for n in (1, 2)
            ):
                problems.append(
                    f"{wav.relative_to(approved)}: contains the wake phrase — heard {text!r}"
                )
    elif field_speech:
        print(f"(skipped transcribing {len(field_speech)} field-derived clips)")

    print("\n## Counts per set and speaker\n")
    for name in SETS:
        total = sum(per_set[name].values())
        by = ", ".join(f"{k} {v}" for k, v in sorted(per_set[name].items()))
        print(f"{name:10} {total:4}   {by}")
    print("\n## Counts per set and source\n")
    for (name, kind), n in sorted(per_source.items()):
        print(f"{name:10} {kind:8} {n:4}")

    print(f"\n{len(problems)} problems\n")
    for p in problems:
        print(f"- {p}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
