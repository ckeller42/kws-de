"""Quality control for CoreS3 recording sessions.

Pure core: every rule is a small function over strings/arrays with an injected
transcriber, so it is unit-tested without a model; `kws-qc` (Task 3) wires in
Whisper. Layout and rules: docs/superpowers/specs/2026-09-02-recording-pipeline-design.md.
"""

from __future__ import annotations

import csv
import logging
import re
import unicodedata
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from kws_de import config
from kws_de.grammar import Intent, Rejection, parse

log = logging.getLogger(__name__)

Transcript = dict
Transcriber = Callable[[Path], Transcript]

CAP_MS = {
    "words": 4000,
    "sentences": 6000,
    "negatives": 6000,
    "wake": 4000,
    # FIELD_MAX_TAKE_SAMPLES: a window extends on every fire inside it, so a take
    # is not fixed at 3500 ms — the firmware's ring is the only ceiling.
    "field": 9800,
}
MIN_MS = 300
MIN_RMS_DBFS = -45.0
CLIP_DBFS = -0.5
CLIP_FRACTION = 0.0005  # share of samples at/above CLIP_DBFS that counts as clipping (8 of 16,000)
FILLER = {"prozent"}

# Whisper large-v3 writes the light-level number words as numerals ("50" not "fünfzig") -
# map the digits it hears back onto the config vocabulary's spoken forms before matching.
NUM_WORDS = {"25": "fünfundzwanzig", "50": "fünfzig", "75": "fünfundsiebzig", "100": "hundert"}

# "Hey Bus" heard as e.g. "Hej Boss" or "He Bos" - loose enough for common ASR variants,
# tight enough that ordinary German sentences don't accidentally match.
_WAKE_RE = re.compile(r"(hey|hej|he|hei)(bus|buss|bos|boss)", re.IGNORECASE)

# A wake phrase ending later than this into a field take is not that take's wake
# phrase. The take starts FIELD_PREROLL_MS before the fire and the model fires
# 0.23-1.20 s past the end of the phrase (measured, E17), so the phrase ends
# between 1.3 and 2.27 s in; this is the late end plus slack, and it has to move
# with FIELD_PREROLL_MS. Only the take's first one or two words are ever tested
# against it, so a loose bound cannot swallow a phrase said later in the take.
WAKE_MAX_S = 2.5
# Kept after the end of "bus", so the wake clip is not cut mid-plosive.
WAKE_TAIL_S = 0.15
# Kept after the last word of a field take's command, so the phrase clip is not
# cut mid-plosive either.
PHRASE_TAIL_S = 0.3
# firmware/main/field.h's FIELD_PREROLL_MS: audio the device keeps in front of the
# wake fire. `ms < FIELD_PREROLL_MS + window_ms` is how a ring-truncated take is
# read off the two columns the pull carries (REQ_FW_FIELD_CAPTURE).
FIELD_PREROLL_MS = 2500
TRUNCATED_SLACK_MS = 50  # tick/sample rounding between window_ms and the WAV length
MIN_PREROLL_MS = 500  # no build ever kept less in front of the fire; smaller heads are cuts


@dataclass
class Take:
    file: Path
    set: str
    prompt: str
    speaker: str
    device_intent: str = ""  # what the device itself recognised (field takes only)
    device_words: str = ""  # "<word>:<conf>" entries joined by '|'
    window_ms: int = 0  # how long the assist window was really open (field takes only)
    preroll_ms: int = FIELD_PREROLL_MS  # the recording build's pre-roll, inferred per session
    # seconds to cut for the phrase clip; None means the whole file
    span: tuple[float, float] | None = None


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
    device_intent: str = ""  # verbatim from the device; NEVER used as a label
    agrees: str = ""  # "1"/"0" device vs Whisper, "" when there is nothing to compare
    truncated: str = ""  # "1"/"0" the ring cut this field take short, "" for a guided take


def normalise(text: str) -> list[str]:
    t = unicodedata.normalize("NFC", text).lower().replace("ß", "ss")
    t = re.sub(r"[^\w\s]", " ", t)
    return [NUM_WORDS.get(w, w) for w in t.split() if w not in FILLER]


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
    if set_name == "wake":
        return ["hey", "bus"]
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


def field_wake_split(tr: Transcript) -> tuple[float | None, list[str]]:
    """Split a field take's transcript into (seconds at which the wake clip
    ends, the normalised command tokens after it). Returns `(None, all tokens)`
    when the take does not open with the wake phrase inside the first
    WAKE_MAX_S seconds — the take is then all command (or all junk), and
    nothing is cut off as a wake clip. Whisper emits the phrase as either one
    span ("HeyBus") or two ("Hey", "Bus"), so both spellings are tried."""
    words = tr.get("words", [])
    for n in (1, 2):
        if len(words) < n:
            break
        glued = "".join(t for w in words[:n] for t in normalise(w["word"]))
        if _WAKE_RE.fullmatch(glued) and float(words[n - 1]["end"]) <= WAKE_MAX_S:
            rest = [t for w in words[n:] for t in normalise(w["word"])]
            return float(words[n - 1]["end"]) + WAKE_TAIL_S, rest
    return None, normalise(tr.get("text", ""))


def _split_glued(tok: str, v: set[str]) -> list[str]:
    """Whisper welds German compounds ("Lichtküche" for "Licht Küche"). Peel known
    vocabulary words off the front, longest first; only a token that decomposes
    COMPLETELY into vocabulary words is split, so ordinary speech is left alone
    ("anzug" keeps its "an", "ankommen" its "an")."""
    out, i = [], 0
    while i < len(tok):
        w = next((tok[i:j] for j in range(len(tok), i, -1) if tok[i:j] in v), None)
        if w is None:
            return [tok]
        out.append(w)
        i += len(w)
    return out


def field_intent(tokens: list[str]) -> Intent | Rejection:
    """The Whisper-derived label for a field take: the command tokens mapped
    back onto config labels and run through the SAME grammar the device uses
    (`kws_de.grammar.parse`), with non-vocabulary words dropped — the identical
    filter `required_tokens(..., "sentences")` applies to a guided prompt. An
    `Intent` is a phrase label; a `Rejection` means the take is kept as
    negative / `_unknown_` material, never dropped."""
    v = vocab()
    toks = [s for t in tokens for s in ([t] if t in v else _split_glued(t, v))]
    return parse([label_for_token(t) for t in toks if t in v])


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


def _token_covers(h: str, need: list[str], j: int) -> int | None:
    """Does whitespace-delimited heard word `h` account for one or more of the
    required tokens `need`, starting at `need[j]`? Two ways:
    - `h` matches `need[j]` alone (`_matches`: exact, or edit-distance<=1 for a
      token of MORE than 5 letters) -> covers just need[j];
    - `h` is the exact concatenation of need[j], need[j+1], ... (ASR glued two+
      required keywords into one word with no space, e.g. "Lichtdach" for
      "Licht Dach", or "Lichtan" for "Licht an") -> covers need[j:k].
    Boundary-safe: a short keyword can never match merely because it occurs as
    a substring inside an unrelated longer word ("an" in "dank") — only a
    whole heard word, or an exact run of required tokens, counts.
    Returns the new pointer k (need[j:k] consumed), or None."""
    if _matches(need[j], h):
        return j + 1
    acc = need[j]
    k = j + 1
    while k < len(need) and len(acc) < len(h):
        acc += need[k]
        if acc == h:
            return k + 1
        k += 1
    return None


def content_gate(set_name: str, prompt: str, transcript_text: str) -> tuple[float, str | None]:
    heard = normalise(transcript_text)
    if set_name == "negatives":
        v = vocab()
        counts: dict[str, int] = {}
        for h in heard:
            if h in v:
                counts[h] = counts.get(h, 0) + 1
        # a keyword must appear as a whole token (heard is already whitespace-split, so
        # this holds by construction); a 2-letter keyword ("an", "zu") alone is too easy
        # to hallucinate to reject on — it must appear at least twice, or be >=3 letters.
        for h in heard:
            if h in counts and (len(h) >= 3 or counts[h] >= 2):
                return 0.0, f"contains_command:{h}"
        return 1.0, None
    if set_name == "field":
        # Everything is kept: a field take is real usage, and speech the grammar
        # cannot parse is exactly the negative/`_unknown_` material the model
        # needs. Only silence (or a transcriber that returned nothing) rejects.
        return (1.0, None) if heard else (0.0, "empty_transcript")
    if set_name == "wake":
        glued = "".join(heard)
        return (1.0, None) if _WAKE_RE.fullmatch(glued) else (0.0, f"wrong_word:{glued or '-'}")
    need = required_tokens(prompt, set_name)
    if set_name == "words":
        ok = bool(need) and any(_token_covers(h, need, 0) is not None for h in heard)
        return (1.0, None) if ok else (0.0, f"wrong_word:{' '.join(heard) or '-'}")
    found = 0
    for h in heard:
        if found >= len(need):
            break
        k = _token_covers(h, need, found)
        if k is not None:
            found = k
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
    # Clipping parks many samples at the rail; a 1-2 sample click (pop, seam
    # glitch) is not a reason to throw away a whole take.
    rail = 10 ** (CLIP_DBFS / 20)
    if len(mono) and float(np.mean(np.abs(mono) >= rail)) >= CLIP_FRACTION:
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
    # A field take the device cut to fit its audio ring holds less than the
    # pre-roll plus the window it recorded. Marked here so the truncated ones are
    # distinguishable downstream from takes the recogniser simply never answered.
    # window_ms is tick-based and dur_ms floors the sample count, so a whole
    # take can read 1-30 ms "short"; only a real cut is more than that.
    truncated = ""
    if take.set == "field" and take.window_ms:
        short_by = take.preroll_ms + take.window_ms - m.get("dur_ms", 0)
        truncated = "1" if short_by > TRUNCATED_SLACK_MS else "0"
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
        device_intent=take.device_intent,
        truncated=truncated,
    )
    return row, tr


def read_sessions(incoming: Path) -> list[Take]:
    incoming = Path(incoming)
    takes = []
    heads = []  # ms - window_ms of each field take = that build's pre-roll, unless cut
    with (incoming / "sessions.csv").open(newline="") as fh:
        for r in csv.DictReader(fh):
            takes.append(
                Take(
                    file=incoming / r["file"],
                    set=r["set"],
                    prompt=r["prompt"],
                    speaker=r["speaker"],
                    # a nine-column guided row never reaches these: DictReader
                    # yields None for a column the row does not have
                    device_intent=r.get("device_intent") or "",
                    device_words=r.get("device_words") or "",
                    window_ms=int(r.get("window_ms") or 0),
                )
            )
            if r["set"] == "field" and r.get("window_ms"):
                heads.append(int(r["ms"]) - int(r["window_ms"]))
    # Sessions recorded before the pre-roll grew carry a shorter head; a cut take
    # has less than the pre-roll in front, so the session's longest head is it.
    plausible = [h for h in heads if MIN_PREROLL_MS <= h <= FIELD_PREROLL_MS]
    if plausible:
        pre = max(plausible)
        for t in takes:
            if t.set == "field":
                t.preroll_ms = pre
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
    # "hallo-welt_001.wav" -> "hallo-welt"; a field take's "123456.wav" -> "123456"
    return re.sub(r"_\d{3}\.wav$", "", path.name).removesuffix(".wav")


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
    for sub in ("phrases", "negatives", "wake"):
        idx = approved / sub / "index.csv"
        if idx.exists():
            with idx.open() as fh:
                keep = [r for r in csv.DictReader(fh) if r["file"] not in prev]
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
    from kws_de.eval import intent_text  # local: kws_de.eval imports qc back

    incoming, qc_dir, approved = Path(incoming), Path(qc_dir), Path(approved)
    _clear_stamp(approved, qc_dir)

    takes = read_sessions(incoming)
    rows, words_rows, written, gap_files = [], [], [], []
    n_words = n_skipped = n_wake = 0
    # "field takes" is EVERY field row in the session, approved or not; approved
    # is reported next to it, never instead of it. kws_de.eval.field_figures
    # counts the same way off qc.csv, so the two reports agree on one session.
    n_field = n_field_approved = n_field_truncated = 0
    n_field_wake = n_field_parsable = 0
    n_field_compared = n_field_agree = n_field_unfiled = 0
    for t in takes:
        try:
            row, tr = judge(t, transcriber)
        except Exception as e:  # noqa: BLE001 - one bad take must not abort the batch
            log.error("judge failed for %s: %s", t.file, e)
            row = QcRow(
                file=str(t.file),
                set=t.set,
                prompt=t.prompt,
                speaker=t.speaker,
                verdict="reject",
                reason=f"error: {type(e).__name__}",
                transcript="",
                match_score=0.0,
                rms_dbfs=0.0,
                peak_dbfs=0.0,
                dur_ms=0,
            )
            tr = {"text": "", "words": []}
        rows.append(row)
        if t.set == "field":
            n_field += 1
            n_field_truncated += row.truncated == "1"
        if row.verdict != "approve":
            continue
        if t.set == "field":
            n_field_approved += 1
            cut_s, tokens = field_wake_split(tr)
            if cut_s is not None:
                # the wake phrase is a real "Hey Bus" positive: file it exactly
                # where the guided wake set goes, so it trains the wake model too
                sig, sr = sf.read(t.file, dtype="float32", always_2d=True)
                d = approved / "wake" / t.speaker
                dst = d / f"{t.speaker}_{_next_no(d, t.speaker)}.wav"
                dst.parent.mkdir(parents=True, exist_ok=True)
                sf.write(dst, sig[: int(cut_s * sr), 0], sr, subtype="PCM_16")
                written.append(str(dst.relative_to(approved)))
                _append_index(
                    approved / "wake" / "index.csv",
                    {
                        "file": str(dst.relative_to(approved)),
                        "prompt": config.WAKE_WORD,
                        "speaker": t.speaker,
                    },
                )
                n_wake += 1
                n_field_wake += 1
            got = field_intent(tokens)
            if isinstance(got, Intent):
                n_field_parsable += 1
                # The derived label is the row's prompt too, so `qc.csv` says on
                # its own face whether a field take parsed — `agrees` keeps its
                # single meaning (the device comparison) instead of two.
                row.prompt = intent_text(got)
                # Provenance only: the device's own words go through the SAME
                # grammar, and the two Intents are compared. The device never
                # supplies the label — `got` does. A truncated take carries no
                # device_intent at all: nothing to compare, so `agrees` stays "".
                if t.device_intent.strip():
                    row.agrees = "1" if field_intent(normalise(t.device_intent)) == got else "0"
                    n_field_compared += 1
                    n_field_agree += int(row.agrees == "1")
                # From here the take IS an approved sentence take with a derived
                # prompt: same phrase copy, same index row, same word segmentation.
                # The clip is cut to the COMMAND — from the end of the wake phrase
                # to the last word Whisper heard — because eval streams every
                # approved/phrases/ row through the command model end to end: the
                # pre-roll and up to several seconds of trailing silence would be
                # streamed too, and a spurious event there would score a take the
                # model got right as an e2e miss. A guided sentence contains only
                # the sentence; so does this. Word clips still come off the FULL
                # take, whose Whisper timestamps they are indexed by.
                ends = [float(w["end"]) for w in tr.get("words", [])]
                t = Take(
                    file=t.file,
                    set="sentences",
                    prompt=intent_text(got),
                    speaker=t.speaker,
                    # no word timestamps -> keep the whole take rather than a 0.3 s stub
                    span=(cut_s or 0.0, max(ends) + PHRASE_TAIL_S) if ends else None,
                )
            elif content_gate("negatives", "", row.transcript)[1] is None:
                # Kept, not dropped: unparsable field speech is `_unknown_`
                # material, with the transcript itself as its prompt.
                t = Take(file=t.file, set="negatives", prompt=row.transcript, speaker=t.speaker)
            else:
                # The grammar rejected it, but it still contains command words
                # ("an Licht Küche"). Filing that under negatives/ would teach the
                # model a real command is _unknown_ (data.py) and score a correct
                # recognition of it as a false accept (eval.py). Leave it unfiled.
                n_field_unfiled += 1
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
            if t.span is None:
                dst.write_bytes(t.file.read_bytes())
            else:  # a field take: only the command part of it is the phrase
                lo, hi = (int(s * sr) for s in t.span)
                sf.write(dst, sig[max(lo, 0) : hi], sr, subtype="PCM_16")
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
        elif t.set == "wake":
            d = approved / "wake" / t.speaker
            dst = d / f"{t.speaker}_{_next_no(d, t.speaker)}.wav"
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(t.file.read_bytes())
            written.append(str(dst.relative_to(approved)))
            _append_index(
                approved / "wake" / "index.csv",
                {
                    "file": str(dst.relative_to(approved)),
                    "prompt": t.prompt,
                    "speaker": t.speaker,
                },
            )
            n_wake += 1
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
    if n_field:
        # denominator = takes the device actually answered, not every parsable
        # one: a truncated take has no device intent to agree or disagree with.
        agree_rate = f"{n_field_agree / n_field_compared:.3f}" if n_field_compared else "n/a"
        field_section = (
            f"\n## Field\n\n{n_field} field takes, {n_field_approved} approved, "
            f"{n_field_parsable} parsable, {n_field_wake} wake clips, "
            f"{n_field_unfiled} unparsed (vocab present), "
            f"{n_field_truncated} ring-truncated, "
            f"device-Whisper agreement {agree_rate} over {n_field_compared} compared.\n"
        )
    else:
        field_section = ""
    (qc_dir / "report.md").write_text(
        f"# QC {incoming.name}\n\n{len(rows)} takes, {approved_n} approved, "
        f"{len(rejects)} rejected, {n_words} word clips written, "
        f"{n_skipped} word clips skipped, {n_wake} wake clips written "
        "(word and wake counts mix guided takes with field-derived clips; "
        "the Field section below separates them).\n\n## Rejects\n\n"
        + "".join(
            f"- `{Path(r.file).relative_to(incoming)}` — reject: {r.reason} "
            f'(heard: "{r.transcript}")\n'
            for r in rejects
        )
        + "\n## Segmentation gaps\n\n"
        + ("".join(f"- `{f}`\n" for f in gap_files) or "(none)\n")
        + field_section
    )
    return {
        "takes": len(rows),
        "approved": approved_n,
        "rejected": len(rejects),
        "words_written": n_words,
        "words_skipped": n_skipped,
        "wake_written": n_wake,
        "field_takes": n_field,  # every field row, approved or not
        "field_approved": n_field_approved,
        "field_truncated": n_field_truncated,
        "field_parsable": n_field_parsable,
        "field_agree": n_field_agree,
    }


# Only the words Whisper actually mangles (the light-level numerals + the wake word) -
# the full command vocabulary caused prompt-echo hallucination on weak/ambiguous audio
# (Whisper regurgitating chunks of the prompt as the "transcript"), including false
# rejects on genuinely clean negatives.
_QC_PROMPT = ", ".join([*config.LIGHT_LEVELS, "Prozent", config.WAKE_WORD]) + "."
_PAD_SAMPLES = config.SAMPLE_RATE // 2  # 500 ms of silence on each side


def whisper_transcriber(
    model_id: str = "mlx-community/whisper-large-v3-mlx",
) -> Transcriber:  # pragma: no cover - model
    import mlx_whisper

    def transcribe(path: Path) -> Transcript:
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        if sr != config.SAMPLE_RATE:
            raise ValueError(
                f"{path}: sample rate {sr} != {config.SAMPLE_RATE} (mono 16 kHz PCM expected)"
            )
        pad = np.zeros(_PAD_SAMPLES, dtype=np.float32)
        padded = np.concatenate([pad, audio[:, 0], pad])
        r = mlx_whisper.transcribe(
            padded,
            path_or_hf_repo=model_id,
            language="de",
            word_timestamps=True,
            temperature=0.0,
            initial_prompt=_QC_PROMPT,
        )
        offset = _PAD_SAMPLES / config.SAMPLE_RATE
        words = [
            {
                "word": w["word"].strip(),
                "start": max(0.0, float(w["start"]) - offset),
                "end": max(0.0, float(w["end"]) - offset),
            }
            for seg in r.get("segments", [])
            for w in seg.get("words", [])
        ]
        return {"text": r.get("text", ""), "words": words}

    return transcribe


def main() -> None:  # pragma: no cover - I/O wrapper
    import argparse
    import sys

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
        print(f"{inc}: no sessions.csv (exit 2)", file=sys.stderr)
        raise SystemExit(2)
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
        print(f"could not load {a.model}: {e} (exit 4)", file=sys.stderr)
        raise SystemExit(4) from e
    counts = run_qc(inc, qc_dir, approved, tr)
    with (qc_dir / "report.md").open("a") as fh:
        fh.write(f"\nModel: `{a.model}`\n")
    print(f"qc: {counts} -> {qc_dir}")
