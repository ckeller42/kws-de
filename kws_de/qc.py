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
