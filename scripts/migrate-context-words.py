"""One-time migration (E48): split `approved/words/` into two buckets.

Every row `kws_de.qc.run_qc` has ever written to a stamp's `words.csv` comes
from the SAME branch — cutting word spans out of a sentence/field/elicit take
(`run_qc`'s "sentences" branch; see `docs/paper-notes.md` E47/E48) — so by
construction every one of those clips is context-origin, not a dedicated
single-word take. Before this fix they were filed under `approved/words/`
next to the ~51 genuinely guided single-word clips (no `words.csv` row at
all); this script moves them to `approved/context/<label>/` instead, to match
where `kws_de.qc` now writes them, and rewrites the `out_file` column of every
`words.csv` plus the matching lines of `written.txt` so both keep pointing at
a real file. Nothing is deleted or re-cut — files move byte-for-byte.

Run once, by hand, against a real `$KWS_DATA_ROOT`:
    uv run --no-sync python scripts/migrate-context-words.py [<recordings-dir>]
(default: config.DATA_DIR / "recordings")

Idempotent: a row whose `out_file` no longer starts with "words/" (already
migrated, by an earlier run of this script or by a fresh `kws-qc` run after
this fix) is left alone.
"""

import collections
import csv
import hashlib
import shutil
import sys
from pathlib import Path

from kws_de import config

WORDS_CSV_FIELDS = ["src", "word", "speaker", "start_ms", "end_ms", "out_file"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def migrate(recordings: Path) -> dict:
    approved = recordings / "approved"
    qc_dir = recordings / "qc"
    per_label: collections.Counter = collections.Counter()
    hashes: dict[Path, str] = {}  # moved file -> pre-move hash, for the integrity check

    before_total = len(list(approved.rglob("*.wav"))) if approved.is_dir() else 0

    stamps = sorted(d for d in qc_dir.iterdir() if d.is_dir()) if qc_dir.is_dir() else []
    for stamp in stamps:
        wcsv = stamp / "words.csv"
        if not wcsv.exists():
            continue
        with wcsv.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            continue

        renames: dict[str, str] = {}  # written.txt line (old rel) -> new rel, this stamp only
        changed = False
        for r in rows:
            # `out_file` is the ABSOLUTE path recorded at QC time - on the same host
            # for a real run, but a scratch dry-run copies the tree elsewhere, so
            # locate it by its "approved/..." suffix rather than trusting the
            # recorded prefix (also makes this robust to the data root moving).
            parts = Path(r["out_file"]).parts
            try:
                rel = Path(*parts[parts.index("approved") + 1 :])
            except ValueError:
                continue  # no "approved" segment at all: not this tree
            if rel.parts[0] != "words":
                continue  # already migrated
            out = approved / rel
            if not out.exists():
                continue  # moved by an earlier run of this script already
            new_rel = Path("context", *rel.parts[1:])
            new_out = approved / new_rel
            new_out.parent.mkdir(parents=True, exist_ok=True)
            hashes[new_out] = _sha256(out)
            shutil.move(str(out), str(new_out))
            r["out_file"] = str(new_out)
            renames[str(rel)] = str(new_rel)
            per_label[rel.parts[1]] += 1
            changed = True

        if not changed:
            continue
        with wcsv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=WORDS_CSV_FIELDS)
            w.writeheader()
            w.writerows(rows)

        written = stamp / "written.txt"
        if written.exists():
            lines = written.read_text().splitlines()
            new_lines = [renames.get(line, line) for line in lines]
            written.write_text("".join(line + "\n" for line in new_lines))

    after_total = len(list(approved.rglob("*.wav"))) if approved.is_dir() else 0
    bad_hashes = [str(p) for p, h in hashes.items() if _sha256(p) != h]

    return {
        "moved": sum(per_label.values()),
        "per_label": dict(sorted(per_label.items())),
        "before_total": before_total,
        "after_total": after_total,
        "integrity_ok": after_total == before_total and not bad_hashes,
        "bad_hashes": bad_hashes,
    }


def main() -> None:
    recordings = Path(sys.argv[1]) if len(sys.argv) > 1 else config.DATA_DIR / "recordings"
    result = migrate(recordings)
    print(f"moved {result['moved']} clips words/ -> context/:")
    for label, n in result["per_label"].items():
        print(f"  {label:16} {n:4}")
    print(
        f"\napproved/ .wav count: {result['before_total']} before, "
        f"{result['after_total']} after (equal: {result['before_total'] == result['after_total']})"
    )
    status = "OK" if result["integrity_ok"] else f"FAILED: {result['bad_hashes']}"
    print(f"content-hash check: {status}")
    if not result["integrity_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
