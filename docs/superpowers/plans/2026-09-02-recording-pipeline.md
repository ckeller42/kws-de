# Recording Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One command pulls a CoreS3 recording session to the workstation, Whisper-checks and segments it, feeds the approved audio into the v3 dataset, and reports held-out **and** user-customised accuracy.

**Architecture:** Bash drives the machines (`ingest.sh`: device→bar→workstation over the serial `mode` command + USB drive; `data-loop.sh`: the whole loop). Python owns the judgement: `kws_de/qc.py` is a pure core (audio gate, prompt normalisation, content matching, word segmentation, approved-tree writer) with an injected transcriber so every rule is unit-tested without a model; `kws-qc` wires in `mlx-whisper`. Existing modules are extended, not replaced: `data._fetch_and_cache` reads `approved/words`, `dataset/train/export` take a `--prefix`, `eval.main` gains `--recordings`.

**Tech Stack:** Python 3.11, numpy, soundfile, librosa (existing), `mlx-whisper` (new `qc` extra, Apple Silicon), bash + rsync + ssh, pytest, sphinx-needs.

**Spec:** `docs/superpowers/specs/2026-09-02-recording-pipeline-design.md`

## Global Constraints

- Public repo: **no speaker names** (numeric `spkNN` only), no machine names/paths (`bar`, `wuerfel`, `/Volumes/...`) anywhere in committed code or docs — hosts and roots come from `KWS_DATA_ROOT` / script flags / env.
- Recordings root is `config.DATA_DIR / "recordings"` (with `KWS_DATA_ROOT` set that is `<root>/data/recordings`). Layout exactly as spec §2, except `incoming/<stamp>/` holds the pull as-is (`spkNN/...` dirs + the pull script's `sessions.csv`).
- `sessions.csv` columns (from `scripts/pull-recordings.sh`): `speaker,pulled,prompt,file,ms,peak_dbfs,set,seed,ts`; `set` ∈ `words|sentences|negatives`; `file` is relative to the pull root, e.g. `spk02/_phrase_/licht-kueche-an_001.wav`.
- Audio gate thresholds: 16 kHz mono 16-bit; duration ≥ 300 ms, ≤ 4000 ms (words) / ≤ 6000 ms (sentences, negatives); RMS ≥ −45 dBFS; peak < −0.5 dBFS.
- Content rule: words → keyword present (edit distance ≤ 1 for words ≥ 5 letters); sentences → every command keyword of the prompt present **in order**; negatives → **no** command keyword present. Normalisation: lower-case, punctuation stripped, `ß`→`ss` on both sides, `prozent` dropped.
- Whisper: `mlx-community/whisper-large-v3-mlx`, language `de`, `word_timestamps=True`, `temperature=0`; model id recorded in `report.md`.
- Two evals, labels verbatim: `held-out` and `user-customised, in-training`.
- Python gates: `uv run --no-sync ruff check . && uv run --no-sync ruff format --check .`, `uv run --no-sync pytest -q`; shell: `shellcheck`; docs: markdownlint config `.markdownlint.json`; `sphinx-build` must succeed. Never `--no-verify`, never `git add -A`. Always `uv sync --extra dev --extra tts --extra docs --extra qc` (all extras) if you sync.
- Paper gate: the branch must touch `docs/paper-notes.md` (Task 7 does).

## File map

| Path | Task | Responsibility |
|---|---|---|
| `kws_de/qc.py` | 1, 2, 3 | pure QC core (gates, matching, segmentation, approved writer) + CLI |
| `tests/test_qc.py` | 1, 2, 3 | rules with a stubbed transcriber; slow Whisper test skipped without `mlx_whisper` |
| `scripts/ingest.sh`, `tests/test_ingest.py` | 4 | device → workstation pull |
| `kws_de/data.py:333`, `kws_de/train.py`, `kws_de/export.py`, `tests/test_data_v3_provenance.py` | 5 | approved dir hook, `_unknown_` from negatives, provenance manifest, `--prefix` |
| `kws_de/eval.py`, `tests/test_eval_recordings.py` | 6 | `--recordings`: user-customised figures |
| `scripts/data-loop.sh`, `docs/sphinx/pipeline.rst`, `docs/sphinx/{requirements,tests,index}.rst`, `DATASHEET.md`, `README.md`, `docs/paper-notes.md` | 7 | driver + docs + traceability |

Execution order 1 → 2 → 3 → 4 → 5 → 6 → 7.

---

### Task 1: QC core — normalisation, audio gate, content gate, `qc.csv`

**Files:**
- Create: `kws_de/qc.py`
- Test: `tests/test_qc.py`

**Interfaces:**
- Consumes: `config.DEVICES/ZONES/ACTIONS/LIGHT_LEVELS`, `soundfile`, `numpy`.
- Produces:

```python
Transcript = dict            # {"text": str, "words": [{"word": str, "start": float, "end": float}]}
Transcriber = Callable[[Path], Transcript]

@dataclass
class Take:
    file: Path; set: str; prompt: str; speaker: str   # set in {"words","sentences","negatives"}

@dataclass
class QcRow:
    file: str; set: str; prompt: str; speaker: str; verdict: str; reason: str
    transcript: str; match_score: float; rms_dbfs: float; peak_dbfs: float; dur_ms: int

def normalise(text: str) -> list[str]
def vocab() -> set[str]                                   # normalised command keywords
def required_tokens(prompt: str, set_name: str) -> list[str]
def audio_gate(path: Path, set_name: str) -> tuple[dict, str | None]   # (measures, reason)
def content_gate(set_name: str, prompt: str, transcript_text: str) -> tuple[float, str | None]
def judge(take: Take, transcriber: Transcriber) -> tuple[QcRow, Transcript]
def read_sessions(incoming: Path) -> list[Take]           # parses sessions.csv
def write_qc_csv(rows: list[QcRow], path: Path) -> None
```

- [ ] **Step 1: Write the failing tests** — `tests/test_qc.py`

```python
import csv
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from kws_de import config, qc


def _wav(path: Path, sig: np.ndarray, sr: int = 16000):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, sig.astype(np.float32), sr, subtype="PCM_16")
    return path


def _tone(ms=800, amp=0.3, sr=16000):
    t = np.arange(int(sr * ms / 1000)) / sr
    return amp * np.sin(2 * np.pi * 440 * t)


def test_normalise_umlauts_sharp_s_prozent_and_punctuation():
    assert qc.normalise("Licht Küche fünfzig Prozent.") == ["licht", "küche", "fünfzig"]
    assert qc.normalise("Außen") == qc.normalise("Aussen") == ["aussen"]


def test_required_tokens_per_set():
    assert qc.required_tokens("Kühlschrank", "words") == ["kühlschrank"]
    assert qc.required_tokens("Licht Küche fünfzig Prozent", "sentences") == ["licht", "küche", "fünfzig"]
    assert qc.required_tokens("wie spät ist es", "negatives") == []


def test_content_gate_rules():
    assert qc.content_gate("words", "Licht", "licht") == (1.0, None)
    assert qc.content_gate("words", "Kühlschrank", "kühlschrenk")[1] is None          # edit distance 1, >=5 letters
    assert qc.content_gate("words", "Licht", "nicht")[1].startswith("wrong_word")       # short word: exact only
    assert qc.content_gate("sentences", "Licht Küche an", "licht küche an")[0] == 1.0
    score, reason = qc.content_gate("sentences", "Licht Küche an", "küche licht an")
    assert reason == "missing:licht küche an (order)" or reason.startswith("missing")
    assert qc.content_gate("sentences", "Licht Küche an", "licht bitte küche an") == (1.0, None)  # filler ok
    assert qc.content_gate("negatives", "wie spät ist es", "wie spät ist es") == (1.0, None)
    assert qc.content_gate("negatives", "wie spät ist es", "mach das licht an")[1] == "contains_command:licht"


def test_audio_gate_ok_clipped_quiet_short(tmp_path):
    ok = _wav(tmp_path / "ok.wav", _tone())
    m, reason = qc.audio_gate(ok, "words")
    assert reason is None and m["sr"] == 16000 and 700 <= m["dur_ms"] <= 900 and m["peak_dbfs"] < -0.5
    assert qc.audio_gate(_wav(tmp_path / "clip.wav", np.clip(_tone(amp=3.0), -1, 1)), "words")[1] == "clipped"
    assert qc.audio_gate(_wav(tmp_path / "quiet.wav", _tone(amp=0.001)), "words")[1] == "too_quiet"
    assert qc.audio_gate(_wav(tmp_path / "short.wav", _tone(ms=100)), "words")[1] == "too_short"
    assert qc.audio_gate(_wav(tmp_path / "long.wav", _tone(ms=4500)), "words")[1] == "too_long"
    assert qc.audio_gate(_wav(tmp_path / "long6.wav", _tone(ms=4500)), "sentences")[1] is None


def test_judge_and_sessions_roundtrip(tmp_path):
    inc = tmp_path / "incoming" / "2026-09-02-1500"
    _wav(inc / "spk02" / "licht" / "001.wav", _tone())
    _wav(inc / "spk02" / "_neg_" / "wie-spaet-ist-es_001.wav", _tone())
    (inc / "sessions.csv").write_text(
        "speaker,pulled,prompt,file,ms,peak_dbfs,set,seed,ts\n"
        'spk02,2026-09-02T15:00:00,Licht,spk02/licht/001.wav,800,-10.0,words,1,123\n'
        'spk02,2026-09-02T15:00:00,"wie spät ist es",spk02/_neg_/wie-spaet-ist-es_001.wav,800,-10.0,negatives,1,124\n'
    )
    takes = qc.read_sessions(inc)
    assert [(t.set, t.speaker, t.prompt) for t in takes] == [("words", "spk02", "Licht"), ("negatives", "spk02", "wie spät ist es")]
    heard = {"001.wav": "Licht", "wie-spaet-ist-es_001.wav": "mach das licht an"}
    def transcriber(p: Path):
        return {"text": heard[p.name], "words": []}
    rows = [qc.judge(t, transcriber)[0] for t in takes]
    assert rows[0].verdict == "approve" and rows[0].match_score == 1.0
    assert rows[1].verdict == "reject" and rows[1].reason == "contains_command:licht"
    out = tmp_path / "qc.csv"
    qc.write_qc_csv(rows, out)
    got = list(csv.DictReader(out.open()))
    assert [r["verdict"] for r in got] == ["approve", "reject"] and got[0]["file"].endswith("licht/001.wav")
```

- [ ] **Step 2: Run to verify it fails** — `uv run --no-sync pytest tests/test_qc.py -q` → FAIL `ModuleNotFoundError: kws_de.qc`.

- [ ] **Step 3: Implement `kws_de/qc.py`**

```python
"""Quality control for CoreS3 recording sessions.

Pure core: every rule is a small function over strings/arrays with an injected
transcriber, so it is unit-tested without a model; `kws-qc` (Task 3) wires in
Whisper. Layout and rules: docs/superpowers/specs/2026-09-02-recording-pipeline-design.md.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

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


def _edit1(a: str, b: str) -> bool:
    if abs(len(a) - len(b)) > 1:
        return False
    i = j = diff = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1; j += 1; continue
        diff += 1
        if diff > 1:
            return False
        if len(a) > len(b): i += 1
        elif len(b) > len(a): j += 1
        else: i += 1; j += 1
    return diff + (len(a) - i) + (len(b) - j) <= 1


def _matches(want: str, heard: str) -> bool:
    return want == heard or (len(want) >= 5 and _edit1(want, heard))


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
            found += 1; pos += 1
    score = found / len(need) if need else 1.0
    if found == len(need):
        return 1.0, None
    return score, f"missing:{' '.join(need)} (order)"


def audio_gate(path: Path, set_name: str) -> tuple[dict, str | None]:
    sig, sr = sf.read(path, dtype="float32", always_2d=True)
    info = sf.info(path)
    ch = sig.shape[1]
    mono = sig[:, 0]
    dur_ms = int(1000 * len(mono) / sr)
    peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
    rms = float(np.sqrt(np.mean(mono**2))) if len(mono) else 0.0
    m = {
        "sr": sr, "channels": ch, "subtype": info.subtype, "dur_ms": dur_ms,
        "peak_dbfs": 20 * np.log10(max(peak, 1e-9)), "rms_dbfs": 20 * np.log10(max(rms, 1e-9)),
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
        file=str(take.file), set=take.set, prompt=take.prompt, speaker=take.speaker,
        verdict="approve" if reason is None else "reject", reason=reason or "",
        transcript=tr.get("text", ""), match_score=round(score, 3),
        rms_dbfs=round(m["rms_dbfs"], 1), peak_dbfs=round(m["peak_dbfs"], 1), dur_ms=m["dur_ms"],
    )
    return row, tr


def read_sessions(incoming: Path) -> list[Take]:
    incoming = Path(incoming)
    takes = []
    with (incoming / "sessions.csv").open(newline="") as fh:
        for r in csv.DictReader(fh):
            takes.append(Take(file=incoming / r["file"], set=r["set"], prompt=r["prompt"], speaker=r["speaker"]))
    return takes


def write_qc_csv(rows: list[QcRow], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(QcRow.__dataclass_fields__))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
```

- [ ] **Step 4: Run tests** — `uv run --no-sync pytest tests/test_qc.py -q` → 5 passed. If `test_content_gate_rules`'s order assertion is brittle, keep the reason format `missing:<need joined> (order)` exactly as implemented and adjust the test to that string.

- [ ] **Step 5: Lint + commit**

```bash
uv run --no-sync ruff check kws_de/qc.py tests/test_qc.py && uv run --no-sync ruff format kws_de/qc.py tests/test_qc.py
git add kws_de/qc.py tests/test_qc.py
git commit -m "feat(qc): recording QC core — audio gate, prompt matching, qc.csv"
```

---

### Task 2: Segmentation + approved tree + report (idempotent)

**Files:**
- Modify: `kws_de/qc.py`
- Test: `tests/test_qc.py`

**Interfaces:**
- Consumes: Task 1 (`Take`, `QcRow`, `judge`, `read_sessions`, `write_qc_csv`), `kws_de.recordings.centre`.
- Produces:

```python
def segment_word(sig: np.ndarray, sr: int, start_s: float, end_s: float) -> np.ndarray   # 1 s centred on the span
def label_for_token(tok: str) -> str                    # normalised token -> config label ("küche" -> "Küche")
def run_qc(incoming: Path, qc_dir: Path, approved: Path, transcriber: Transcriber) -> dict
   # writes qc_dir/{qc.csv,words.csv,report.md}, approved/{words,phrases,negatives}; returns counts
```

Approved layout: `approved/words/<Label>/<spk>_<NNN>.wav` (NNN = the take's number from its filename), `approved/phrases/<spk>/<slug>_<NNN>.wav` + `approved/phrases/index.csv` (`file,prompt,speaker`), `approved/negatives/<spk>/<slug>_<NNN>.wav` + `approved/negatives/index.csv`. `words.csv`: `src,word,speaker,start_ms,end_ms,out_file`.

- [ ] **Step 1: Failing tests** — append to `tests/test_qc.py`

```python
def test_segment_word_centres_and_pads():
    sr = 16000
    sig = np.zeros(sr * 2, dtype=np.float32); sig[sr : sr + 1600] = 0.5      # 100 ms burst at 1.0 s
    seg = qc.segment_word(sig, sr, 1.0, 1.1)
    assert seg.shape == (config.CLIP_SAMPLES,)
    assert np.argmax(np.abs(seg)) in range(config.CLIP_SAMPLES // 2 - 1000, config.CLIP_SAMPLES // 2 + 1000)
    edge = qc.segment_word(sig, sr, 0.0, 0.1)                                # window would start before 0
    assert edge.shape == (config.CLIP_SAMPLES,) and np.all(edge[:4000] == 0)


def test_label_for_token():
    assert qc.label_for_token("küche") == "Küche" and qc.label_for_token("licht") == "Licht"
    assert qc.label_for_token("fünfzig") == "fünfzig"


def test_run_qc_writes_approved_tree_and_is_idempotent(tmp_path):
    inc = tmp_path / "incoming" / "s1"
    _wav(inc / "spk02" / "licht" / "001.wav", _tone())
    _wav(inc / "spk02" / "_phrase_" / "licht-kueche-an_001.wav", _tone(ms=1500))
    _wav(inc / "spk02" / "_neg_" / "hallo-welt_001.wav", _tone())
    _wav(inc / "spk02" / "_neg_" / "hallo-welt_002.wav", np.clip(_tone(amp=3.0), -1, 1))   # clipped -> reject
    (inc / "sessions.csv").write_text(
        "speaker,pulled,prompt,file,ms,peak_dbfs,set,seed,ts\n"
        "spk02,t,Licht,spk02/licht/001.wav,800,-10,words,1,1\n"
        "spk02,t,Licht Küche an,spk02/_phrase_/licht-kueche-an_001.wav,1500,-10,sentences,1,2\n"
        "spk02,t,hallo welt,spk02/_neg_/hallo-welt_001.wav,800,-10,negatives,1,3\n"
        "spk02,t,hallo welt,spk02/_neg_/hallo-welt_002.wav,800,-1,negatives,1,4\n"
    )
    def transcriber(p: Path):
        if "_phrase_" in str(p):
            return {"text": "Licht Küche an", "words": [
                {"word": "Licht", "start": 0.2, "end": 0.5}, {"word": "Küche", "start": 0.6, "end": 0.9},
                {"word": "an", "start": 1.0, "end": 1.2}]}
        return {"text": "Licht" if "licht" in str(p) else "hallo welt", "words": []}
    qcd, appr = tmp_path / "qc" / "s1", tmp_path / "approved"
    counts = qc.run_qc(inc, qcd, appr, transcriber)
    assert counts == {"takes": 4, "approved": 3, "rejected": 1, "words_written": 4}
    assert (appr / "words" / "Licht" / "spk02_001.wav").exists()                 # bare word take
    assert (appr / "words" / "Küche" / "spk02_001.wav").exists()                 # segmented from the phrase
    assert (appr / "words" / "an" / "spk02_001.wav").exists()
    assert (appr / "phrases" / "spk02" / "licht-kueche-an_001.wav").exists()
    assert (appr / "negatives" / "spk02" / "hallo-welt_001.wav").exists()
    assert not (appr / "negatives" / "spk02" / "hallo-welt_002.wav").exists()
    idx = list(csv.DictReader((appr / "phrases" / "index.csv").open()))
    assert idx[0]["prompt"] == "Licht Küche an" and idx[0]["speaker"] == "spk02"
    assert (qcd / "report.md").read_text().count("reject") >= 1
    words = list(csv.DictReader((qcd / "words.csv").open()))
    assert {w["word"] for w in words} == {"Licht", "Küche", "an"}
    # idempotent: a second run produces the same tree and counts
    assert qc.run_qc(inc, qcd, appr, transcriber) == counts
```

- [ ] **Step 2: Run** — FAIL `AttributeError: segment_word`.

- [ ] **Step 3: Implement** — append to `kws_de/qc.py`

```python
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


def label_for_token(tok: str) -> str:
    for lab in config.DEVICES + config.ZONES + config.ACTIONS:
        if normalise(lab) == [tok]:
            return lab
    raise KeyError(tok)


def _take_no(path: Path) -> str:
    m = re.search(r"(\d{3})\.wav$", path.name)
    return m.group(1) if m else "001"


def _slug_of(path: Path) -> str:
    return re.sub(r"_\d{3}\.wav$", "", path.name)


def _clear_session(approved: Path, speaker: str, suffix_files: set[str]) -> None:
    """Remove this session's previous outputs (by speaker) so re-runs are idempotent."""
    for sub in ("phrases", "negatives"):
        d = approved / sub / speaker
        if d.is_dir():
            for f in d.glob("*.wav"):
                f.unlink()
    for f in (approved / "words").glob(f"*/{speaker}_*.wav"):
        f.unlink()


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
    takes = read_sessions(incoming)
    for spk in {t.speaker for t in takes}:
        _clear_session(approved, spk, set())
    for sub in ("phrases", "negatives"):
        idx = approved / sub / "index.csv"
        if idx.exists():                       # drop this session's rows, keep other sessions'
            keep = [r for r in csv.DictReader(idx.open()) if r["speaker"] not in {t.speaker for t in takes}]
            idx.unlink()
            for r in keep:
                _append_index(idx, r)
    rows, words_rows, n_words = [], [], 0
    for t in takes:
        row, tr = judge(t, transcriber)
        rows.append(row)
        if row.verdict != "approve":
            continue
        no = _take_no(t.file)
        if t.set == "words":
            lab = label_for_token(required_tokens(t.prompt, "words")[0])
            dst = approved / "words" / lab / f"{t.speaker}_{no}.wav"
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(t.file.read_bytes()); n_words += 1
        elif t.set == "sentences":
            sig, sr = sf.read(t.file, dtype="float32", always_2d=True); sig = sig[:, 0]
            dst = approved / "phrases" / t.speaker / f"{_slug_of(t.file)}_{no}.wav"
            dst.parent.mkdir(parents=True, exist_ok=True); dst.write_bytes(t.file.read_bytes())
            _append_index(approved / "phrases" / "index.csv", {"file": str(dst.relative_to(approved)), "prompt": t.prompt, "speaker": t.speaker})
            need = required_tokens(t.prompt, "sentences")
            spans = [(normalise(w["word"]), w["start"], w["end"]) for w in tr.get("words", [])]
            pos = 0
            for tok in need:
                while pos < len(spans) and not (spans[pos][0] and _matches(tok, spans[pos][0][0])):
                    pos += 1
                if pos >= len(spans):
                    break
                _, s, e = spans[pos]; pos += 1
                lab = label_for_token(tok)
                out = approved / "words" / lab / f"{t.speaker}_{no}.wav"
                out.parent.mkdir(parents=True, exist_ok=True)
                sf.write(out, segment_word(sig, sr, s, e), sr, subtype="PCM_16")
                words_rows.append({"src": str(t.file), "word": lab, "speaker": t.speaker, "start_ms": int(s * 1000), "end_ms": int(e * 1000), "out_file": str(out)})
                n_words += 1
        else:
            dst = approved / "negatives" / t.speaker / f"{_slug_of(t.file)}_{no}.wav"
            dst.parent.mkdir(parents=True, exist_ok=True); dst.write_bytes(t.file.read_bytes())
            _append_index(approved / "negatives" / "index.csv", {"file": str(dst.relative_to(approved)), "prompt": t.prompt, "speaker": t.speaker})
    qc_dir.mkdir(parents=True, exist_ok=True)
    write_qc_csv(rows, qc_dir / "qc.csv")
    with (qc_dir / "words.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["src", "word", "speaker", "start_ms", "end_ms", "out_file"])
        w.writeheader(); w.writerows(words_rows)
    approved_n = sum(r.verdict == "approve" for r in rows)
    rejects = [r for r in rows if r.verdict != "approve"]
    (qc_dir / "report.md").write_text(
        f"# QC {incoming.name}\n\n{len(rows)} takes, {approved_n} approved, {len(rejects)} rejected, "
        f"{n_words} word clips written.\n\n## Rejects\n\n"
        + "".join(f"- `{Path(r.file).relative_to(incoming)}` — reject: {r.reason} (heard: \"{r.transcript}\")\n" for r in rejects)
        + "\n"
    )
    return {"takes": len(rows), "approved": approved_n, "rejected": len(rejects), "words_written": n_words}
```

- [ ] **Step 4: Run** — `uv run --no-sync pytest tests/test_qc.py -q` → 8 passed. (`_clear_session`'s unused `suffix_files` argument: remove it if ruff flags it — call it `_clear_session(approved, spk)`.)

- [ ] **Step 5: Lint + commit** — `git add kws_de/qc.py tests/test_qc.py && git commit -m "feat(qc): word segmentation, approved tree, report; idempotent re-runs"`

---

### Task 3: `kws-qc` CLI with Whisper (mlx) transcriber

**Files:**
- Modify: `kws_de/qc.py`, `pyproject.toml` (script + `qc` extra), `uv.lock`
- Test: `tests/test_qc.py`

**Interfaces:**
- Produces: `whisper_transcriber(model_id: str) -> Transcriber`; CLI `kws-qc INCOMING [--model ID] [--out QC_DIR] [--approved DIR] [--dry-run]`.
- Defaults: `--out` = `config.DATA_DIR/"recordings"/"qc"/<incoming.name>`, `--approved` = `config.DATA_DIR/"recordings"/"approved"`, `--model mlx-community/whisper-large-v3-mlx`.

- [ ] **Step 1: Failing tests** — append

```python
def test_cli_dry_run_lists_takes_without_model(tmp_path, capsys, monkeypatch):
    inc = tmp_path / "incoming" / "s2"
    _wav(inc / "spk03" / "licht" / "001.wav", _tone())
    (inc / "sessions.csv").write_text("speaker,pulled,prompt,file,ms,peak_dbfs,set,seed,ts\nspk03,t,Licht,spk03/licht/001.wav,800,-10,words,1,1\n")
    monkeypatch.setattr("sys.argv", ["kws-qc", str(inc), "--dry-run"])
    qc.main()
    out = capsys.readouterr().out
    assert "1 takes" in out and "licht/001.wav" in out


@pytest.mark.skipif(pytest.importorskip("mlx_whisper", reason="mlx-whisper not installed") is None, reason="no mlx")
def test_whisper_transcriber_smoke(tmp_path):
    tr = qc.whisper_transcriber("mlx-community/whisper-tiny-mlx")          # tiny: quick smoke only
    out = tr(_wav(tmp_path / "t.wav", _tone(ms=1200)))
    assert set(out) >= {"text", "words"}
```

- [ ] **Step 2: Run** — `test_cli_dry_run...` FAILS (`AttributeError: main`); the smoke test skips without mlx.

- [ ] **Step 3: Implement** — append to `kws_de/qc.py`

```python
def whisper_transcriber(model_id: str = "mlx-community/whisper-large-v3-mlx") -> Transcriber:  # pragma: no cover - model
    import mlx_whisper

    def transcribe(path: Path) -> Transcript:
        r = mlx_whisper.transcribe(str(path), path_or_hf_repo=model_id, language="de", word_timestamps=True, temperature=0.0)
        words = [{"word": w["word"].strip(), "start": float(w["start"]), "end": float(w["end"])}
                 for seg in r.get("segments", []) for w in seg.get("words", [])]
        return {"text": r.get("text", ""), "words": words}

    return transcribe


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(prog="kws-qc", description="quality-control a pulled recording session")
    ap.add_argument("incoming")
    ap.add_argument("--model", default="mlx-community/whisper-large-v3-mlx")
    ap.add_argument("--out", default=None, help="qc dir (default data/recordings/qc/<incoming name>)")
    ap.add_argument("--approved", default=None, help="approved tree (default data/recordings/approved)")
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
```

Then `pyproject.toml`: add `kws-qc = "kws_de.qc:main"` under `[project.scripts]` and `qc = ["mlx-whisper>=0.4"]` under optional-dependencies; `uv lock && uv sync --extra dev --extra tts --extra docs --extra qc`.

- [ ] **Step 4: Run** — `uv run --no-sync pytest tests/test_qc.py -q` (dry-run passes; smoke runs only if mlx present — if it runs it downloads whisper-tiny, ~75 MB).

- [ ] **Step 5: Commit** — `git add kws_de/qc.py tests/test_qc.py pyproject.toml uv.lock && git commit -m "feat(qc): kws-qc CLI with mlx-whisper transcriber (qc extra)"`

---

### Task 4: `scripts/ingest.sh` — device → workstation

**Files:**
- Create: `scripts/ingest.sh`
- Test: `tests/test_ingest.py`

**Interfaces:**
- Consumes: the device serial protocol (`mode usb`, `mode menu`), `scripts/pull-recordings.sh <dest>` (on the device host), `KWS_DATA_ROOT`.
- Produces: `$KWS_DATA_ROOT/data/recordings/incoming/<YYYY-MM-DD-HHMM>/` containing the pull (`spkNN/…`, `sessions.csv`, `logs/`). Flags: `-H host` (default `$KWSREC_HOST`, required), `-p port` (default auto `/dev/cu.usbmodem*` on the host), `-d dest root` (default `${KWS_DATA_ROOT:?}/data/recordings`), `-n` dry-run (print commands).

- [ ] **Step 1: Failing test** — `tests/test_ingest.py`

```python
import os
import stat
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ingest.sh"


def _fake_bin(d: Path, name: str, body: str) -> None:
    p = d / name
    p.write_text("#!/usr/bin/env bash\n" + body)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)


def test_ingest_orders_commands_and_pulls(tmp_path):
    bin_ = tmp_path / "bin"; bin_.mkdir()
    log = tmp_path / "calls.log"
    # fake ssh: record the remote command; simulate the pull dir on "host" via a local dir
    remote = tmp_path / "remote"; (remote / "spk02" / "licht").mkdir(parents=True)
    (remote / "spk02" / "licht" / "001.wav").write_bytes(b"RIFF")
    (remote / "sessions.csv").write_text("speaker,pulled,prompt,file,ms,peak_dbfs,set,seed,ts\n")
    _fake_bin(bin_, "ssh", f'echo "ssh $*" >> "{log}"\ncase "$*" in *"ls /dev/cu.usbmodem"*) echo /dev/cu.usbmodem101;; *"ls /Volumes/KWSREC"*) echo spk02;; esac\nexit 0\n')
    _fake_bin(bin_, "rsync", f'echo "rsync $*" >> "{log}"\nsrc="${{@: -2:1}}"; dst="${{@: -1}}"; mkdir -p "$dst"; cp -R "{remote}/." "$dst/"\n')
    env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "KWS_DATA_ROOT": str(tmp_path / "root")}
    r = subprocess.run(["bash", str(SCRIPT), "-H", "devhost"], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    calls = log.read_text()
    assert calls.index("mode usb") < calls.index("pull-recordings.sh") < calls.index("rsync") < calls.index("mode menu")
    incoming = list((tmp_path / "root" / "data" / "recordings" / "incoming").iterdir())
    assert len(incoming) == 1 and (incoming[0] / "spk02" / "licht" / "001.wav").exists()
    assert "--delete" not in calls


def test_ingest_fails_clearly_without_port(tmp_path):
    bin_ = tmp_path / "bin"; bin_.mkdir()
    _fake_bin(bin_, "ssh", 'exit 0\n')          # never prints a port
    env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "KWS_DATA_ROOT": str(tmp_path / "root")}
    r = subprocess.run(["bash", str(SCRIPT), "-H", "devhost"], env=env, capture_output=True, text=True)
    assert r.returncode == 1 and "usbmodem" in r.stderr
```

- [ ] **Step 2: Run** — FAIL (script missing).

- [ ] **Step 3: Implement** — `scripts/ingest.sh`

```bash
#!/usr/bin/env bash
# Pull a CoreS3 recording session from the device host into the workstation's
# data root: put the device in USB-drive mode over its serial console, run the
# host-side pull script, rsync the result here, return the device to the menu.
# Usage: ingest.sh -H host [-p /dev/cu.usbmodemNNN] [-d recordings_root] [-n]
#   host: ssh name of the machine the CoreS3 is plugged into (never hard-coded here)
set -euo pipefail

host=${KWSREC_HOST:-}; port=""; root=${KWS_DATA_ROOT:+$KWS_DATA_ROOT/data/recordings}; dry=0
while getopts "H:p:d:n" o; do case $o in H) host=$OPTARG;; p) port=$OPTARG;; d) root=$OPTARG;; n) dry=1;; *) exit 2;; esac; done
[[ -n $host ]] || { echo "usage: $0 -H host [-p port] [-d root] [-n]  (or set KWSREC_HOST)" >&2; exit 2; }
[[ -n $root ]] || { echo "set KWS_DATA_ROOT or pass -d" >&2; exit 2; }
run() { if (( dry )); then echo "+ $*"; else "$@"; fi; }

if [[ -z $port ]]; then
  port=$(ssh "$host" 'ls /dev/cu.usbmodem* 2>/dev/null | head -1' || true)
  [[ -n $port ]] || { echo "no /dev/cu.usbmodem* on $host — device unplugged or already in USB-drive mode" >&2; exit 1; }
fi
# 1. device -> USB drive mode (serial link disappears while the drive is exported)
run ssh "$host" "printf 'mode usb\n' > '$port'"
for _ in $(seq 1 20); do
  if [[ -n $(ssh "$host" 'ls /Volumes/KWSREC 2>/dev/null' || true) ]]; then break; fi
  sleep 1
done
[[ -n $(ssh "$host" 'ls /Volumes/KWSREC 2>/dev/null' || true) ]] || { echo "KWSREC did not mount on $host within 20 s" >&2; exit 3; }
# 2. host-side pull (rsync spk*/, aggregate sessions.csv, move recognise.log, eject)
run ssh "$host" 'rm -rf ~/kwsrec-pull && bash scripts/pull-recordings.sh ~/kwsrec-pull'
# 3. bring it here, never deleting anything on either side
stamp=$(date +%Y-%m-%d-%H%M); dest="$root/incoming/$stamp"
mkdir -p "$dest"
run rsync -a --ignore-existing "$host:~/kwsrec-pull/" "$dest/"
# 4. device back to the selection screen (port is back once the drive is ejected)
for _ in $(seq 1 20); do
  p=$(ssh "$host" 'ls /dev/cu.usbmodem* 2>/dev/null | head -1' || true); [[ -n $p ]] && { port=$p; break; }; sleep 1
done
run ssh "$host" "printf 'mode menu\n' > '$port'" || echo "warning: could not send 'mode menu' — tap Menu on the device" >&2
n=$(find "$dest" -name '*.wav' | wc -l | tr -d ' ')
(( n > 0 )) || { echo "nothing pulled into $dest" >&2; exit 1; }
echo "ingested $n takes -> $dest"
```

`chmod +x scripts/ingest.sh`; `shellcheck scripts/ingest.sh` clean (add `# shellcheck disable=SC2029` on the two `ssh … "$port"` lines if it flags client-side expansion — it is intended). Note: the fake `ssh` in the test must answer the KWSREC probe — the test's `case` does; the real host runs `pull-recordings.sh` from the kws-de checkout on that host (`scripts/` relative to its home — document that the host needs the repo, or copy the script over with `scp` first: add `run scp -q scripts/pull-recordings.sh "$host:~/pull-recordings.sh"` before step 2 and call `bash ~/pull-recordings.sh`; update the test's expected order accordingly).

- [ ] **Step 4: Run** — `uv run --no-sync pytest tests/test_ingest.py -q` → 2 passed; `shellcheck scripts/ingest.sh` clean.

- [ ] **Step 5: Commit** — `git add scripts/ingest.sh tests/test_ingest.py && git commit -m "feat(scripts): ingest.sh — pull a CoreS3 session into the data root over the serial mode command"`

---

### Task 5: Data prep — approved dir hook, negatives → `_unknown_`, provenance manifest, `--prefix`

**Files:**
- Modify: `kws_de/data.py:333` (`_fetch_and_cache`), `kws_de/dataset.py` (`build_manifest` provenance), `kws_de/train.py:55-66`, `kws_de/export.py:141-160`
- Test: `tests/test_data_v3_provenance.py`

**Interfaces:**
- Consumes: `approved/words/<Label>/<spk>_<NNN>.wav` (Task 2), `approved/negatives/<spk>/*.wav`.
- Produces: `data.recordings_root() -> Path` (approved/words if it exists, else legacy `data/recordings`); `data.negative_windows(root: Path, rng) -> list[tuple[np.ndarray, str]]` (1 s windows at 1 s hops from every approved negative, speaker `rec:<spk>`); manifest entries gain `"sources": {"mswc": n, "tts": n, "recording": n}` and `"speakers": [...]` per split; `kws-train --prefix features_v3 --out command_v3.keras`, `kws-export --prefix features_v3 --model command_v3.keras --firmware`.

- [ ] **Step 1: Failing tests** — `tests/test_data_v3_provenance.py`

```python
from pathlib import Path

import numpy as np
import soundfile as sf

from kws_de import config, data


def test_recordings_root_prefers_approved(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    assert data.recordings_root() == tmp_path / "recordings"
    (tmp_path / "recordings" / "approved" / "words").mkdir(parents=True)
    assert data.recordings_root() == tmp_path / "recordings" / "approved" / "words"


def test_negative_windows_hop_one_second(tmp_path):
    neg = tmp_path / "negatives" / "spk02"; neg.mkdir(parents=True)
    sf.write(neg / "hallo_001.wav", np.random.default_rng(0).standard_normal(16000 * 3).astype(np.float32) * 0.1, 16000, subtype="PCM_16")
    wins = data.negative_windows(tmp_path / "negatives", np.random.default_rng(0))
    assert len(wins) == 3 and all(w.shape == (config.CLIP_SAMPLES,) for w, _ in wins) and wins[0][1] == "rec:spk02"


def test_manifest_records_sources_and_speakers():
    from kws_de.dataset import build_manifest
    X = np.zeros((4, config.N_FRAMES, config.N_MFCC), np.float32); y = np.array([0, 1, 0, 1])
    is_tts = np.array([True, False, False, False])
    speakers = ["tts:a", "rec:spk02", "mswc:x", "rec:spk03"]
    m = build_manifest({"train": (X, y, is_tts)}, seed=0, labels=config.COMMAND_LABELS, speakers={"train": speakers})
    assert m["splits"]["train"]["sources"] == {"tts": 1, "recording": 2, "mswc": 1}
    assert m["splits"]["train"]["speakers"] == ["spk02", "spk03"]      # numeric ids only, sorted, no tts/mswc
```

- [ ] **Step 2: Run** — FAIL (`recordings_root` missing).

- [ ] **Step 3: Implement**

`kws_de/data.py` — add near the top (after imports) and use at line 333:

```python
def recordings_root() -> Path:
    """QC-approved word clips if the pipeline has run, else the legacy hand-dropped layout."""
    approved = config.DATA_DIR / "recordings" / "approved" / "words"
    return approved if approved.is_dir() else config.DATA_DIR / "recordings"


def negative_windows(root: Path, rng) -> list[tuple[np.ndarray, str]]:
    """1 s windows at 1 s hops from every approved negative phrase -> `_unknown_` material,
    speaker id `rec:<spk>` so speaker-disjoint splitting treats them like other real clips."""
    import soundfile as sf

    out = []
    for f in sorted(Path(root).glob("*/*.wav")):
        sig, sr = sf.read(f, dtype="float32", always_2d=True)
        sig = sig[:, 0]
        if sr != config.SAMPLE_RATE:
            continue
        n = config.CLIP_SAMPLES
        for start in range(0, max(len(sig) - n + 1, 1), n):
            win = sig[start : start + n]
            if len(win) < n:
                win = np.pad(win, (0, n - len(win)))
            out.append((win.astype(np.float32), f"rec:{f.parent.name}"))
    return out
```

At line 333 replace `load_recordings(config.DATA_DIR / "recordings", words)` with `load_recordings(recordings_root(), words)`, and right after the merge loop add:

```python
            neg_root = config.DATA_DIR / "recordings" / "approved" / "negatives"
            if neg_root.is_dir():
                clips.setdefault("_unknown_", []).extend(negative_windows(neg_root, np.random.default_rng(0)))
```

`kws_de/dataset.py` — extend `build_manifest(splits, seed, labels, speakers=None)`: for each split, if `speakers` given, add `"sources": {"tts": n, "recording": n, "mswc": n}` (count prefixes `tts:`, `rec:`, otherwise `mswc`) and `"speakers": sorted({s[4:] for s in spk if s.startswith("rec:")})`. In `build()`, pass `speakers={name: [s for _, s in ws] ...}` — `assemble` receives `clips_ws` as `{label: [(clip, speaker)]}`; collect the speaker list per split from `ws` before calling `assemble`.

`kws_de/train.py` — add `--prefix` (default: `features_v2` if `--v2` else `features`) and `--out` (default `command.keras`/`kws.keras`); `kws_de/export.py` — add `--prefix` and `--model` with the same defaults; the `--firmware` calibration/health-gate uses `{prefix}_train`/`{prefix}_test`.

- [ ] **Step 4: Run** — `uv run --no-sync pytest tests/test_data_v3_provenance.py tests/test_dataset.py tests/test_export_firmware.py -q` → all pass; full `uv run --no-sync pytest -q` green.

- [ ] **Step 5: Commit** — `git add kws_de/data.py kws_de/dataset.py kws_de/train.py kws_de/export.py tests/test_data_v3_provenance.py && git commit -m "feat(data): approved recordings into the v3 build, negatives as _unknown_, provenance manifest, --prefix"`

---

### Task 6: `kws-eval --recordings` — user-customised figures

**Files:**
- Modify: `kws_de/eval.py` (new functions + `main`)
- Test: `tests/test_eval_recordings.py`

**Interfaces:**
- Consumes: `approved/words/<Label>/<spk>_<NNN>.wav`, `approved/phrases/index.csv`, `approved/negatives/index.csv`, `kws_de.features.mfcc`, `_stream_events(predict_fn, audio, labels, step_samples)`, `grammar.parse`, `qc.required_tokens`, `qc.label_for_token`.
- Produces:

```python
def prompt_intent(prompt: str)                      # -> Intent via grammar.parse of the prompt's keywords
def eval_recordings(approved: Path, predict_fn, *, step_ms=100) -> dict
   # {"isolated": {spk: {"n": int, "acc": float, "per_word": {label: acc}}},
   #  "e2e": {spk: {"n": int, "acc": float}}, "false_accepts": {spk: {"n": int, "rate": float}},
   #  "label": "user-customised, in-training"}
def render_recordings_section(res: dict) -> str     # markdown with the exact labels
```

- [ ] **Step 1: Failing test** — `tests/test_eval_recordings.py`

```python
import numpy as np
import soundfile as sf

from kws_de import config, eval as ev
from kws_de.grammar import Intent


def test_prompt_intent():
    assert ev.prompt_intent("Licht Küche fünfzig Prozent") == Intent("Licht", "Küche", "fünfzig")


def test_eval_recordings_with_stub_model(tmp_path):
    root = tmp_path / "approved"
    licht = config.COMMAND_LABELS.index("Licht"); sil = config.COMMAND_LABELS.index("_silence_")
    for spk in ("spk02", "spk03"):
        d = root / "words" / "Licht"; d.mkdir(parents=True, exist_ok=True)
        sf.write(d / f"{spk}_001.wav", np.zeros(16000, np.float32), 16000, subtype="PCM_16")
    (root / "phrases" / "spk02").mkdir(parents=True)
    sf.write(root / "phrases" / "spk02" / "licht-an_001.wav", np.zeros(32000, np.float32), 16000, subtype="PCM_16")
    (root / "phrases" / "index.csv").write_text("file,prompt,speaker\nspk02/licht-an_001.wav,Licht an,spk02\n")
    (root / "negatives" / "spk02").mkdir(parents=True)
    sf.write(root / "negatives" / "spk02" / "hallo_001.wav", np.zeros(16000, np.float32), 16000, subtype="PCM_16")
    (root / "negatives" / "index.csv").write_text("file,prompt,speaker\nspk02/hallo_001.wav,hallo,spk02\n")

    def predict_fn(window):          # always "Licht" -> isolated 100%, e2e Rejection(missing action), negatives fire
        p = np.zeros(len(config.COMMAND_LABELS), np.float32); p[licht] = 0.9; p[sil] = 0.1
        return p

    res = ev.eval_recordings(root, predict_fn)
    assert res["label"] == "user-customised, in-training"
    assert res["isolated"]["spk02"]["acc"] == 1.0 and res["isolated"]["spk03"]["n"] == 1
    assert res["e2e"]["spk02"]["n"] == 1 and res["e2e"]["spk02"]["acc"] == 0.0
    assert res["false_accepts"]["spk02"]["n"] == 1
    md = ev.render_recordings_section(res)
    assert "user-customised, in-training" in md and "held-out" not in md.split("user-customised")[0]
```

- [ ] **Step 2: Run** — FAIL (`prompt_intent` missing).

- [ ] **Step 3: Implement** — add to `kws_de/eval.py`

```python
def prompt_intent(prompt: str):
    from kws_de import qc
    from kws_de.grammar import parse

    return parse([qc.label_for_token(t) for t in qc.required_tokens(prompt, "sentences")])


def eval_recordings(approved, predict_fn, *, step_ms: int = 100) -> dict:
    """User-customised figures on a speaker's own QC-approved recordings. These clips may
    be in the training set — that is the point of the personalisation step — so the
    result is labelled and must never be reported as held-out accuracy."""
    import csv
    from collections import defaultdict
    from pathlib import Path

    import soundfile as sf

    from kws_de.features import mfcc
    from kws_de.grammar import Intent

    approved = Path(approved)
    labels = config.COMMAND_LABELS
    step = config.SAMPLE_RATE * step_ms // 1000
    iso = defaultdict(lambda: {"n": 0, "ok": 0, "per_word": defaultdict(lambda: [0, 0])})
    for f in sorted((approved / "words").glob("*/*.wav")):
        lab, spk = f.parent.name, f.stem.split("_")[0]
        sig, _ = sf.read(f, dtype="float32", always_2d=True)
        pred = labels[int(np.argmax(predict_fn(sig[:, 0])))]
        r = iso[spk]; r["n"] += 1; r["ok"] += pred == lab
        r["per_word"][lab][0] += pred == lab; r["per_word"][lab][1] += 1
    isolated = {s: {"n": r["n"], "acc": r["ok"] / r["n"], "per_word": {w: a / n for w, (a, n) in r["per_word"].items()}} for s, r in iso.items()}

    def _events(path):
        sig, _ = sf.read(path, dtype="float32", always_2d=True)
        return _stream_events(predict_fn, sig[:, 0], labels, step)

    e2e = defaultdict(lambda: {"n": 0, "ok": 0})
    idx = approved / "phrases" / "index.csv"
    if idx.exists():
        for r in csv.DictReader(idx.open()):
            from kws_de.grammar import parse
            got = parse(_events(approved / "phrases" / r["file"]))
            e = e2e[r["speaker"]]; e["n"] += 1; e["ok"] += isinstance(got, Intent) and got == prompt_intent(r["prompt"])
    fa = defaultdict(lambda: {"n": 0, "fired": 0})
    nidx = approved / "negatives" / "index.csv"
    if nidx.exists():
        for r in csv.DictReader(nidx.open()):
            from kws_de.grammar import parse
            got = parse(_events(approved / "negatives" / r["file"]))
            n = fa[r["speaker"]]; n["n"] += 1; n["fired"] += isinstance(got, Intent)
    return {
        "label": "user-customised, in-training",
        "isolated": isolated,
        "e2e": {s: {"n": v["n"], "acc": v["ok"] / v["n"]} for s, v in e2e.items()},
        "false_accepts": {s: {"n": v["n"], "rate": v["fired"] / v["n"]} for s, v in fa.items()},
    }


def render_recordings_section(res: dict) -> str:
    out = [f"## {res['label']}\n", "These clips may be in the training set; this is the personalised-device figure, not the held-out one.\n"]
    out.append("| speaker | isolated words n | acc | e2e phrases n | intent acc | negatives n | false-accept rate |\n|---|---|---|---|---|---|---|")
    for spk in sorted(set(res["isolated"]) | set(res["e2e"]) | set(res["false_accepts"])):
        i, e, f = res["isolated"].get(spk, {}), res["e2e"].get(spk, {}), res["false_accepts"].get(spk, {})
        out.append(f"| {spk} | {i.get('n', 0)} | {i.get('acc', float('nan')):.3f} | {e.get('n', 0)} | {e.get('acc', float('nan')):.3f} | {f.get('n', 0)} | {f.get('rate', float('nan')):.3f} |")
    return "\n".join(out) + "\n"
```

In `main()`: add `ap.add_argument("--recordings", default=None, help="approved recordings dir -> user-customised section")` and, when set, build `predict_fn = make_command_predict_fn((config.MODELS_DIR / "command.tflite").read_bytes())`, run `eval_recordings`, and append `render_recordings_section(res)` to the report file (`--out`, default `docs/eval-report-v2.md` for this mode) plus write `<out>.recordings.json`.

Note: the stub `predict_fn` in the test makes `_stream_events` fire "Licht" repeatedly; the grammar then rejects `Licht an`'s missing action → `acc == 0.0` as asserted. If `_stream_events` needs `stream_kwargs` defaults, pass none.

- [ ] **Step 4: Run** — `uv run --no-sync pytest tests/test_eval_recordings.py -q` → 2 passed; full suite green.

- [ ] **Step 5: Commit** — `git add kws_de/eval.py tests/test_eval_recordings.py && git commit -m "feat(eval): --recordings — user-customised isolated/e2e/false-accept figures per speaker"`

---

### Task 7: `data-loop.sh`, docs, traceability, paper-notes

**Files:**
- Create: `scripts/data-loop.sh`, `docs/sphinx/pipeline.rst`
- Modify: `docs/sphinx/index.rst` (toctree), `docs/sphinx/requirements.rst`, `docs/sphinx/tests.rst`, `README.md` (pipeline section), `DATASHEET.md` (recordings provenance), `docs/paper-notes.md`

- [ ] **Step 1: `scripts/data-loop.sh`**

```bash
#!/usr/bin/env bash
# The whole data loop: ingest -> QC -> v3 dataset -> train -> export (health gate) -> evals.
# Usage: data-loop.sh -H host [--skip-ingest] [--skip-train] [--incoming DIR]
set -euo pipefail
host=${KWSREC_HOST:-}; skip_ingest=0; skip_train=0; incoming=""
while [[ $# -gt 0 ]]; do case $1 in -H) host=$2; shift 2;; --skip-ingest) skip_ingest=1; shift;; --skip-train) skip_train=1; shift;; --incoming) incoming=$2; shift 2;; *) echo "usage: $0 -H host [--skip-ingest] [--skip-train] [--incoming DIR]" >&2; exit 2;; esac; done
: "${KWS_DATA_ROOT:?set KWS_DATA_ROOT}"
rec="$KWS_DATA_ROOT/data/recordings"
if (( ! skip_ingest )); then
  [[ -n $host ]] || { echo "-H host required for ingest" >&2; exit 2; }
  incoming=$(scripts/ingest.sh -H "$host" | tail -1 | awk '{print $NF}')
fi
[[ -n $incoming ]] || incoming=$(ls -d "$rec"/incoming/* | tail -1)
echo "== QC $incoming"; uv run --no-sync kws-qc "$incoming"
echo "== dataset v3"; uv run --no-sync kws-dataset build --cache raw_clips_v3.pkl --prefix features_v3
if (( ! skip_train )); then
  echo "== train"; uv run --no-sync kws-train --prefix features_v3 --out command_v3.keras --epochs 40
  echo "== export (health gate)"; uv run --no-sync kws-export --prefix features_v3 --model command_v3.keras --firmware
fi
echo "== evals"; uv run --no-sync kws-eval --recordings "$rec/approved" --out docs/eval-report-v3.md
echo "done: held-out figures in docs/eval-report-v3.md; user-customised section appended. Flash with the flashing skill."
```

`chmod +x`, `shellcheck` clean. (The `kws-dataset build` cache: if `raw_clips_v3.pkl` doesn't exist, `kws-data --fetch --mswc-root …` must have run once — say so in the docs; the loop does not fetch MSWC.)

- [ ] **Step 2: `docs/sphinx/pipeline.rst`** — sections: Overview (the seven stages as a numbered list mirroring spec §3), Data layout (the tree from spec §2 in a `code-block`), Quality control rules (the exact thresholds and matching rules from Global Constraints), The two evaluation figures (a table: held-out vs user-customised, in-training — and *why the in-training one is legitimate for a personalised device and must never be mixed with the held-out figure*), Running it (`data-loop.sh` and each step by hand), Requirements (the `.. req::` directives below, or reference them from requirements.rst). Add to `index.rst` toctree after `traceability`.

- [ ] **Step 3: sphinx-needs** — in `requirements.rst` add `REQ_PIPE_INGEST` (ingest never deletes; commands ordered usb→pull→rsync→menu), `REQ_PIPE_QC_AUDIO` (thresholds), `REQ_PIPE_QC_CONTENT` (word/sentence/negative rules + normalisation), `REQ_PIPE_SEGMENT` (1 s window centred on Whisper's span, zero-padded), `REQ_PIPE_APPROVED_LAYOUT` (the tree; regenerable; numeric speaker ids), `REQ_PIPE_EVAL_LABELS` (the two labels, never mixed). In `tests.rst` add `TEST_QC_RULES` → `tests/test_qc.py` (links AUDIO, CONTENT, SEGMENT, APPROVED_LAYOUT), `TEST_INGEST` → `tests/test_ingest.py` (INGEST), `TEST_EVAL_RECORDINGS` → `tests/test_eval_recordings.py` (EVAL_LABELS), `TEST_V3_PROVENANCE` → `tests/test_data_v3_provenance.py` (APPROVED_LAYOUT). `uv run --no-sync sphinx-build -W --keep-going -b html docs/sphinx docs/sphinx/_build/html` must pass.

- [ ] **Step 4: README + DATASHEET + paper-notes** — README: a "Recording data loop" section (one paragraph + the `data-loop.sh` one-liner, the `KWSREC_HOST` env, the `qc` extra). DATASHEET: under provenance, "Self-recorded speech: numeric speaker ids only; each take passed an audio gate and a Whisper large-v3 content check (model id in the QC report); sentence takes are segmented into word clips by Whisper word timestamps". paper-notes: a method paragraph "Recording loop" (ingest → QC → segment → v3 → two evals) with the first real counts once a session has been run through it (until then: the counts from the QC report of the existing session — run `kws-qc` on it).

- [ ] **Step 5: Gates + commit** — `shellcheck scripts/*.sh`; markdownlint on README/DATASHEET/paper-notes; sphinx-build; full pytest; `git add scripts/data-loop.sh docs/sphinx/pipeline.rst docs/sphinx/index.rst docs/sphinx/requirements.rst docs/sphinx/tests.rst README.md DATASHEET.md docs/paper-notes.md && git commit -m "docs: recording data loop — driver script, pipeline page, requirements traced to tests"`

---

## Self-review notes (controller)

- Spec coverage: §3.1 remote control → firmware branch (done there, not here); §3.2 → Task 4; §3.3 (gates, verdicts, segmentation, determinism) → Tasks 1–3; §3.4 → Task 5; §3.5 → Task 5 (`--prefix`) + existing train/export; §3.6 → Task 6; §3.7 → Task 7; §6 tests → each task; §7 docs → Task 7. Spec §2's `incoming/<stamp>-spkNN/` is realised as `incoming/<stamp>/spkNN/` (the pull is multi-speaker) — Global Constraints say so.
- Type consistency: `Take`/`QcRow`/`Transcriber` (T1) used by T2/T3; `required_tokens`/`label_for_token` (T1/T2) reused by T6's `prompt_intent`; `_stream_events(predict_fn, audio, labels, step_samples)` matches `kws_de/eval.py:156`; `build_manifest(splits, seed, labels, speakers=None)` extended compatibly (existing calls omit `speakers`).
- Placeholders: none; every code step is complete. The two spots where an exact string may need aligning to the implementation are called out in-step (T1 order-reason text, T4 scp order).
