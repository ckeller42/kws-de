"""Audit the approved recording tree: is every clip usable, and is it what its
path says it is?

QC writes `approved/` one session at a time, so nothing ever looks at the tree as
a whole. This does, and it is the check that would have caught #58 before a wake
round trained on the result:

* every clip readable, 16 kHz mono PCM_16, and inside its set's duration band
  (wake 0.4-2.6 s, words/context ~1 s, phrases 0.5-9.8 s, negatives up to 9.8 s);
* no phrase or negative that still contains the wake phrase — the field-derived
  ones are transcribed and matched against `qc._WAKE_RE`. Guided clips were
  already matched against their prompt at QC time, so they are not re-transcribed;
* `index.csv` rows and files one-to-one, in both directions (words/context are
  the label directory itself, so there is no index.csv for either);
* speaker directories named `spkNN` (words/context are label directories instead);
* counts per set, per speaker and per source (guided session vs field session) -
  `approved/words/` and `approved/context/` (E48) are reported as separate sets
  throughout, never combined;
* (report-only, E47/E48) every word clip (both trees) re-transcribed and
  flagged when Whisper hears >= 2 vocabulary words in it, or the label word's
  own span isn't within +/-150 ms of the clip's centre — a cutter regression
  that clips readable by the other checks above would still miss, split guided
  vs context since context clips are expected to flag more often (E47's "not
  fixable by cutting" — a fast, compact sentence puts a real neighbour word in
  the fixed 1 s window regardless of centring).

A clip's source is its QC stamp: a session whose `qc.csv` holds any `set=field`
row is a field session, and every path in that stamp's `written.txt` is
field-derived. Clips written before stamps were kept report as `unknown`.
`approved/words/<label>/` holds only guided single-word takes (E48: `kws_de.qc`
writes a word clip there only when the take's own `set` was `words`); every
other word clip - cut out of a guided sentence, or a field/elicit take - is
context and lives in `approved/context/<label>/` instead.

Usage:
  uv run --no-sync python scripts/audit-approved.py [--no-transcribe] [<approved>]

Exits 1 when anything failed (the checks above the word-content one); the
word-content check is report-only and never changes the exit code — it is a
per-clip heuristic (a short command word can legitimately sit near another
one in a fast compound), not a hard pass/fail gate like format or duplication.
"""

import collections
import csv
import re
import sys
from pathlib import Path

import soundfile as sf

from kws_de import config
from kws_de.qc import _WAKE_RE, normalise, vocab

SETS = ("words", "context", "phrases", "negatives", "wake")
# words/context are the two word-clip buckets (E48): "words" is a guided
# single-word take, "context" is a word cut out of a sentence/field/elicit
# take (kws_de.qc.run_qc) - same clip shape (segment_word's CLIP_SAMPLES
# window, or a raw guided take), so they share a duration band and every
# other per-clip check below; they are only ever reported/counted separately.
WORD_SETS = ("words", "context")
# (min, max) seconds. words/context: segmented clips are exactly
# config.CLIP_SAMPLES, bare guided word takes are raw ~1 s recordings, so the
# band is around 1 s rather than at it.
DURATION_S = {
    "words": (0.5, 2.0),
    "context": (0.5, 2.0),
    "wake": (0.4, 2.6),  # a field-cut clip carries the 2.5 s pre-roll budget in front of the phrase
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


WORD_CENTRE_TOL_MS = 150
_V = vocab()


def word_content_flags(wav: Path, label: str, transcriber) -> tuple[bool, bool]:
    """(multi, offcentre) for one word clip: `multi` is Whisper hearing >= 2
    vocabulary words in it; `offcentre` is the label word (or edit-distance
    match) not found within WORD_CENTRE_TOL_MS of the clip's own centre — the
    label word missing entirely counts as offcentre too."""
    tr = transcriber(wav)
    info = sf.info(wav)
    centre_ms = 1000 * info.frames / info.samplerate / 2
    flat = [(t, w) for w in tr.get("words", []) for t in normalise(w["word"])]
    multi = sum(1 for t, _ in flat if t in _V) >= 2
    lab = normalise(label)[0]
    hit = next((w for t, w in flat if t == lab), None)
    offcentre = hit is None or abs(
        (float(hit["start"]) + float(hit["end"])) / 2 * 1000 - centre_ms
    ) > (WORD_CENTRE_TOL_MS)
    return multi, offcentre


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
    word_clips: list[tuple[Path, str, str]] = []  # (wav, label, bucket="words"/"context")

    for name in SETS:
        root = approved / name
        if not root.is_dir():
            continue
        for wav in sorted(root.rglob("*.wav")):
            rel = str(wav.relative_to(approved))
            group = wav.parent.name  # spkNN for phrases/negatives/wake, a label for words/context
            if name not in WORD_SETS and not SPEAKER_RE.match(group):
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
            if name in WORD_SETS:
                word_clips.append((wav, group, name))

        idx = root / "index.csv"
        if name in WORD_SETS:
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

    tr = transcriber_or_none(transcribe and bool(field_speech or word_clips))
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
        print(f"transcribing {len(word_clips)} word clips (content check, report-only)...")
        word_flags: collections.Counter = collections.Counter()
        word_totals: collections.Counter = collections.Counter()
        for wav, label, kind in word_clips:
            word_totals[kind] += 1
            multi, offcentre = word_content_flags(wav, label, tr)
            if multi or offcentre:
                word_flags[kind] += 1
    elif field_speech or word_clips:
        print(
            f"(skipped transcribing {len(field_speech)} field-derived and "
            f"{len(word_clips)} word clips)"
        )
        word_flags = word_totals = collections.Counter()
    else:
        word_flags = word_totals = collections.Counter()

    print("\n## Counts per set and speaker\n")
    for name in SETS:
        total = sum(per_set[name].values())
        by = ", ".join(f"{k} {v}" for k, v in sorted(per_set[name].items()))
        print(f"{name:10} {total:4}   {by}")
    print("\n## Counts per set and source\n")
    for (name, kind), n in sorted(per_source.items()):
        print(f"{name:10} {kind:8} {n:4}")

    print(
        "\n## Word content check (report-only, E47/E48 — never affects the exit code: "
        ">=2 vocabulary words heard, or the label word not within "
        f"+/-{WORD_CENTRE_TOL_MS} ms of the clip's centre; guided = approved/words/ "
        "(a dedicated single-word take), context = approved/context/ (cut from a "
        "sentence/field/elicit take))\n"
    )
    for name in WORD_SETS:
        n = word_totals.get(name, 0)
        if n:
            label = "guided" if name == "words" else name
            print(f"{label:10} {word_flags.get(name, 0):4} / {n:4} flagged")

    print(f"\n{len(problems)} problems\n")
    for p in problems:
        print(f"- {p}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
