"""Re-gate every TTS clip in a raw-clip cache with the per-clip synthetic-clip gate
(`kws_de.qc.tts_gate`: German, transcript says the word) and drop the failures. E66: the
voice gate passes a voice on one sentence, but Piper `mls-medium#N` turned single words
into babble in 202 of 202 sampled clips. Real clips (MSWC, `rec:`) are never touched.

Backs the cache up as `<cache>.pre-regate.pkl` and writes `<cache>.regate.csv`
(word, speaker, ok, reason, transcript).

Usage:
  uv run --no-sync python scripts/regate-tts-cache.py <cache.pkl> [--dry-run]
"""

import argparse
import csv
import pickle
import shutil
import tempfile
from pathlib import Path

import soundfile as sf

from kws_de import config
from kws_de.data import tts_gate_transcriber, tts_text_for
from kws_de.qc import tts_gate


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cache", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="report only, keep the cache as is")
    a = ap.parse_args()
    transcriber = tts_gate_transcriber()
    if transcriber is None:
        raise SystemExit("no Whisper backend (or KWS_TTS_GATE=0): nothing to gate with")
    # pickle: this project's own local cache, never untrusted input (see kws_de/data.py).
    cached = pickle.load(open(a.cache, "rb"))  # noqa: S301
    clips = cached["clips"]
    tmp = Path(tempfile.mkdtemp())
    rows, counts = [], {}
    for word, items in clips.items():
        text = tts_text_for(word).lower()
        kept = []
        for clip, speaker in items:
            if not speaker.startswith("tts:"):
                kept.append((clip, speaker))
                continue
            wav = tmp / "clip.wav"
            sf.write(wav, clip, config.SAMPLE_RATE, subtype="PCM_16")
            ok, reason = tts_gate(wav, text, transcriber)
            rows.append({"word": word, "speaker": speaker, "ok": int(ok), "reason": reason or ""})
            if ok:
                kept.append((clip, speaker))
        counts[word] = (len(items), len(kept))
        if not a.dry_run:
            clips[word] = kept
    shutil.rmtree(tmp, ignore_errors=True)
    with open(a.cache.with_suffix(".regate.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["word", "speaker", "ok", "reason"])
        w.writeheader()
        w.writerows(rows)
    for word, (before, after) in counts.items():
        if before != after:
            print(f"{word:16s} {before:4d} -> {after:4d}")
    n_ok = sum(r["ok"] for r in rows)
    print(f"tts clips: {n_ok} ok / {len(rows) - n_ok} dropped")
    if not a.dry_run:
        shutil.copy2(a.cache, a.cache.with_suffix(".pre-regate.pkl"))
        with open(a.cache, "wb") as fh:
            pickle.dump(cached, fh)
        print(f"wrote {a.cache} (backup {a.cache.with_suffix('.pre-regate.pkl')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
