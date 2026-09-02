"""Quality control for CoreS3 recording sessions.

Pure core: every rule is a small function over strings/arrays with an injected
transcriber, so it is unit-tested without a model; `kws-qc` (Task 3) wires in
Whisper. Layout and rules: docs/superpowers/specs/2026-09-02-recording-pipeline-design.md.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from kws_de import config

Transcript = dict
Transcriber = Callable[[Path], Transcript]

CAP_MS = {"words": 4000, "sentences": 6000, "negatives": 6000}
MIN_MS = 300
MIN_RMS_DBFS = -45.0
CLIP_DBFS = -0.5
FILLER = {"prozent"}


@dataclass
class Take:
    file: Path
    set: str
    prompt: str
    speaker: str


@dataclass
class QcRow:
    file: str
    set: str
    prompt: str
    speaker: str
    verdict: str
    reason: str
    transcript: str
    match_score: float
    rms_dbfs: float
    peak_dbfs: float
    dur_ms: int


def normalise(text: str) -> list[str]:
    t = unicodedata.normalize("NFC", text).lower().replace("ß", "ss")
    t = re.sub(r"[^\w\s]", " ", t)
    return [w for w in t.split() if w not in FILLER]


def vocab() -> set[str]:
    words = config.DEVICES + config.ZONES + config.ACTIONS
    return {tok for w in words for tok in normalise(w)}


def required_tokens(prompt: str, set_name: str) -> list[str]:
    toks = normalise(prompt)
    if set_name == "words":
        return toks[:1]
    if set_name == "sentences":
        v = vocab()
        return [t for t in toks if t in v]
    return []


def label_for_token(token: str) -> str | None:
    """Map a normalised command token back to its canonical config label
    (original case/umlauts), e.g. "licht" -> "Licht", "aussen" -> "Außen".

    Used by Task 6's prompt_intent() to recover config.DEVICES/ZONES/ACTIONS
    labels from a normalised transcript.
    """
    for w in config.DEVICES + config.ZONES + config.ACTIONS:
        if normalise(w) == [token]:
            return w
    return None


def _edit1(a: str, b: str) -> bool:
    if abs(len(a) - len(b)) > 1:
        return False
    i = j = diff = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
            j += 1
            continue
        diff += 1
        if diff > 1:
            return False
        if len(a) > len(b):
            i += 1
        elif len(b) > len(a):
            j += 1
        else:
            i += 1
            j += 1
    return diff + (len(a) - i) + (len(b) - j) <= 1


def _matches(want: str, heard: str) -> bool:
    # brief's test requires "Licht" (5 letters) exact-only but "Kühlschrank" (11)
    # edit-distance-1 forgiving -> cutoff is > 5, not >= 5 as the plan summary says.
    return want == heard or (len(want) > 5 and _edit1(want, heard))


def content_gate(set_name: str, prompt: str, transcript_text: str) -> tuple[float, str | None]:
    heard = normalise(transcript_text)
    if set_name == "negatives":
        bad = [h for h in heard if h in vocab()]
        return (0.0, f"contains_command:{bad[0]}") if bad else (1.0, None)
    need = required_tokens(prompt, set_name)
    if set_name == "words":
        ok = any(_matches(need[0], h) for h in heard) if need else False
        return (1.0, None) if ok else (0.0, f"wrong_word:{' '.join(heard) or '-'}")
    pos, found = 0, 0
    for tok in need:
        while pos < len(heard) and not _matches(tok, heard[pos]):
            pos += 1
        if pos < len(heard):
            found += 1
            pos += 1
    score = found / len(need) if need else 1.0
    if found == len(need):
        return 1.0, None
    return score, f"missing:{' '.join(need)} (order)"


def audio_gate(path: Path, set_name: str) -> tuple[dict, str | None]:
    try:
        sig, sr = sf.read(path, dtype="float32", always_2d=True)
        info = sf.info(path)
    except Exception as e:  # corrupt/missing wav -> reject, don't abort the batch
        return {}, f"unreadable: {type(e).__name__}"
    ch = sig.shape[1]
    mono = sig[:, 0]
    dur_ms = int(1000 * len(mono) / sr)
    peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
    rms = float(np.sqrt(np.mean(mono**2))) if len(mono) else 0.0
    m = {
        "sr": sr,
        "channels": ch,
        "subtype": info.subtype,
        "dur_ms": dur_ms,
        "peak_dbfs": 20 * np.log10(max(peak, 1e-9)),
        "rms_dbfs": 20 * np.log10(max(rms, 1e-9)),
    }
    if sr != config.SAMPLE_RATE or ch != 1 or info.subtype != "PCM_16":
        return m, "format"
    if dur_ms < MIN_MS:
        return m, "too_short"
    if dur_ms > CAP_MS.get(set_name, 6000):
        return m, "too_long"
    if m["peak_dbfs"] >= CLIP_DBFS:
        return m, "clipped"
    if m["rms_dbfs"] < MIN_RMS_DBFS:
        return m, "too_quiet"
    return m, None


def judge(take: Take, transcriber: Transcriber) -> tuple[QcRow, Transcript]:
    m, reason = audio_gate(take.file, take.set)
    tr: Transcript = {"text": "", "words": []}
    score = 0.0
    if reason is None:
        tr = transcriber(take.file)
        score, reason = content_gate(take.set, take.prompt, tr.get("text", ""))
    row = QcRow(
        file=str(take.file),
        set=take.set,
        prompt=take.prompt,
        speaker=take.speaker,
        verdict="approve" if reason is None else "reject",
        reason=reason or "",
        transcript=tr.get("text", ""),
        match_score=round(score, 3),
        rms_dbfs=round(m.get("rms_dbfs", 0.0), 1),
        peak_dbfs=round(m.get("peak_dbfs", 0.0), 1),
        dur_ms=m.get("dur_ms", 0),
    )
    return row, tr


def read_sessions(incoming: Path) -> list[Take]:
    incoming = Path(incoming)
    takes = []
    with (incoming / "sessions.csv").open(newline="") as fh:
        for r in csv.DictReader(fh):
            takes.append(
                Take(
                    file=incoming / r["file"],
                    set=r["set"],
                    prompt=r["prompt"],
                    speaker=r["speaker"],
                )
            )
    return takes


def write_qc_csv(rows: list[QcRow], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(QcRow.__dataclass_fields__))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def segment_word(sig: np.ndarray, sr: int, start_s: float, end_s: float) -> np.ndarray:
    """1 s window (config.CLIP_SAMPLES) centred on the word span, zero-padded at edges."""
    n = config.CLIP_SAMPLES
    centre = int(round((start_s + end_s) / 2 * sr))
    lo = centre - n // 2
    out = np.zeros(n, dtype=np.float32)
    src_lo, src_hi = max(lo, 0), min(lo + n, len(sig))
    if src_hi > src_lo:
        out[src_lo - lo : src_hi - lo] = sig[src_lo:src_hi]
    return out


def _slug_of(path: Path) -> str:
    return re.sub(r"_\d{3}\.wav$", "", path.name)


def _clear_stamp(approved: Path, qc_dir: Path) -> None:
    """Undo exactly what THIS stamp (qc_dir) wrote last run, via its own
    written.txt manifest, so re-running one stamp never touches another
    stamp's or another speaker's approved output. No-op on a first run."""
    manifest = qc_dir / "written.txt"
    if not manifest.exists():
        return
    prev = {line.strip() for line in manifest.read_text().splitlines() if line.strip()}
    for rel in prev:
        f = approved / rel
        if f.exists():
            f.unlink()
    for sub in ("phrases", "negatives"):
        idx = approved / sub / "index.csv"
        if idx.exists():
            keep = [r for r in csv.DictReader(idx.open()) if r["file"] not in prev]
            idx.unlink()
            for r in keep:
                _append_index(idx, r)


def _next_no(d: Path, prefix: str) -> str:
    """Next-free <NNN> for '<prefix>_<NNN>.wav' inside dir d, scanning what's
    already there. Independent of the source take number, so different write
    sources (bare word vs. phrase-segmented word) or different sessions
    (stamps) for the same speaker/slug/label can never collide on one path.
    Used for approved/words/<label>/<speaker>_<NNN>.wav (prefix=speaker) and
    approved/{phrases,negatives}/<speaker>/<slug>_<NNN>.wav (prefix=slug).
    ponytail: rescans the dir on every call (O(files-in-dir) per write); fine
    at recording-pipeline volumes — cache per (d, prefix) within a run if this
    shows up in profiling."""
    pat = re.compile(rf"{re.escape(prefix)}_(\d+)\.wav$")
    nums = [int(m.group(1)) for f in d.glob(f"{prefix}_*.wav") if (m := pat.match(f.name))]
    return f"{(max(nums) + 1) if nums else 1:03d}"


def _append_index(path: Path, row: dict) -> None:
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "prompt", "speaker"])
        if new:
            w.writeheader()
        w.writerow(row)


def run_qc(incoming: Path, qc_dir: Path, approved: Path, transcriber: Transcriber) -> dict:
    incoming, qc_dir, approved = Path(incoming), Path(qc_dir), Path(approved)
    _clear_stamp(approved, qc_dir)

    takes = read_sessions(incoming)
    rows, words_rows, written, gap_files = [], [], [], []
    n_words = n_skipped = 0
    for t in takes:
        row, tr = judge(t, transcriber)
        rows.append(row)
        if row.verdict != "approve":
            continue
        if t.set == "words":
            tok = required_tokens(t.prompt, "words")[0]
            lab = label_for_token(tok)
            if lab is None:  # unmapped token: reject filing, don't mislabel
                n_skipped += 1
                continue
            d = approved / "words" / lab
            dst = d / f"{t.speaker}_{_next_no(d, t.speaker)}.wav"
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(t.file.read_bytes())
            written.append(str(dst.relative_to(approved)))
            n_words += 1
        elif t.set == "sentences":
            sig, sr = sf.read(t.file, dtype="float32", always_2d=True)
            sig = sig[:, 0]
            slug = _slug_of(t.file)
            d = approved / "phrases" / t.speaker
            dst = d / f"{slug}_{_next_no(d, slug)}.wav"
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(t.file.read_bytes())
            written.append(str(dst.relative_to(approved)))
            _append_index(
                approved / "phrases" / "index.csv",
                {
                    "file": str(dst.relative_to(approved)),
                    "prompt": t.prompt,
                    "speaker": t.speaker,
                },
            )
            need = required_tokens(t.prompt, "sentences")
            spans = [(normalise(w["word"]), w["start"], w["end"]) for w in tr.get("words", [])]
            pos = 0
            for i, tok in enumerate(need):
                while pos < len(spans) and not (spans[pos][0] and _matches(tok, spans[pos][0][0])):
                    pos += 1
                if pos >= len(spans):  # Whisper's word spans didn't cover this token
                    n_skipped += len(need) - i
                    gap_files.append(str(t.file.relative_to(incoming)))
                    break
                _, s, e = spans[pos]
                pos += 1
                lab = label_for_token(tok)
                if lab is None:  # unmapped token: skip this clip, don't mislabel
                    n_skipped += 1
                    continue
                wd = approved / "words" / lab
                out = wd / f"{t.speaker}_{_next_no(wd, t.speaker)}.wav"
                out.parent.mkdir(parents=True, exist_ok=True)
                sf.write(out, segment_word(sig, sr, s, e), sr, subtype="PCM_16")
                written.append(str(out.relative_to(approved)))
                words_rows.append(
                    {
                        "src": str(t.file),
                        "word": lab,
                        "speaker": t.speaker,
                        "start_ms": int(s * 1000),
                        "end_ms": int(e * 1000),
                        "out_file": str(out),
                    }
                )
                n_words += 1
        else:
            slug = _slug_of(t.file)
            d = approved / "negatives" / t.speaker
            dst = d / f"{slug}_{_next_no(d, slug)}.wav"
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(t.file.read_bytes())
            written.append(str(dst.relative_to(approved)))
            _append_index(
                approved / "negatives" / "index.csv",
                {
                    "file": str(dst.relative_to(approved)),
                    "prompt": t.prompt,
                    "speaker": t.speaker,
                },
            )

    qc_dir.mkdir(parents=True, exist_ok=True)
    write_qc_csv(rows, qc_dir / "qc.csv")
    with (qc_dir / "words.csv").open("w", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["src", "word", "speaker", "start_ms", "end_ms", "out_file"]
        )
        w.writeheader()
        w.writerows(words_rows)
    (qc_dir / "written.txt").write_text("".join(p + "\n" for p in written))

    approved_n = sum(r.verdict == "approve" for r in rows)
    rejects = [r for r in rows if r.verdict != "approve"]
    (qc_dir / "report.md").write_text(
        f"# QC {incoming.name}\n\n{len(rows)} takes, {approved_n} approved, "
        f"{len(rejects)} rejected, {n_words} word clips written, "
        f"{n_skipped} word clips skipped.\n\n## Rejects\n\n"
        + "".join(
            f"- `{Path(r.file).relative_to(incoming)}` — reject: {r.reason} "
            f'(heard: "{r.transcript}")\n'
            for r in rejects
        )
        + "\n## Segmentation gaps\n\n"
        + ("".join(f"- `{f}`\n" for f in gap_files) or "(none)\n")
    )
    return {
        "takes": len(rows),
        "approved": approved_n,
        "rejected": len(rejects),
        "words_written": n_words,
        "words_skipped": n_skipped,
    }


def whisper_transcriber(
    model_id: str = "mlx-community/whisper-large-v3-mlx",
) -> Transcriber:  # pragma: no cover - model
    import mlx_whisper

    def transcribe(path: Path) -> Transcript:
        r = mlx_whisper.transcribe(
            str(path),
            path_or_hf_repo=model_id,
            language="de",
            word_timestamps=True,
            temperature=0.0,
        )
        words = [
            {"word": w["word"].strip(), "start": float(w["start"]), "end": float(w["end"])}
            for seg in r.get("segments", [])
            for w in seg.get("words", [])
        ]
        return {"text": r.get("text", ""), "words": words}

    return transcribe


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        prog="kws-qc", description="quality-control a pulled recording session"
    )
    ap.add_argument("incoming")
    ap.add_argument("--model", default="mlx-community/whisper-large-v3-mlx")
    ap.add_argument(
        "--out", default=None, help="qc dir (default data/recordings/qc/<incoming name>)"
    )
    ap.add_argument(
        "--approved", default=None, help="approved tree (default data/recordings/approved)"
    )
    ap.add_argument("--dry-run", action="store_true", help="list takes; no model, no writes")
    a = ap.parse_args()
    inc = Path(a.incoming)
    if not (inc / "sessions.csv").exists():
        raise SystemExit(f"{inc}: no sessions.csv (exit 2)")
    takes = read_sessions(inc)
    if a.dry_run:
        print(f"{len(takes)} takes in {inc}")
        for t in takes:
            print(f"  {t.set:9s} {t.speaker} {t.file.relative_to(inc)}  '{t.prompt}'")
        return
    qc_dir = Path(a.out) if a.out else config.DATA_DIR / "recordings" / "qc" / inc.name
    approved = Path(a.approved) if a.approved else config.DATA_DIR / "recordings" / "approved"
    try:
        tr = whisper_transcriber(a.model)
    except Exception as e:  # noqa: BLE001 - model download/import failure is a user-facing exit
        raise SystemExit(f"could not load {a.model}: {e} (exit 4)") from e
    counts = run_qc(inc, qc_dir, approved, tr)
    (qc_dir / "report.md").open("a").write(f"\nModel: `{a.model}`\n")
    print(f"qc: {counts} -> {qc_dir}")
