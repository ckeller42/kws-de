#!/usr/bin/env python3
"""Voice-cloned TTS of the vocabulary in a device speaker's voice (XTTS-v2, zero-shot).
Spec: docs/superpowers/specs/2026-09-29-voice-clone-tts-design.md. Spike code: kept
only if the clone arm wins on the real scoreboard.

Runs in its OWN venv (torch/CUDA 13 must not share the TF venv):
    uv venv .xtts && uv pip install --python .xtts/bin/python \\
        "coqui-tts[codec]" "transformers<5" torch torchaudio soundfile
    (transformers 5 removed `isin_mps_friendly`; torch>=2.9 wants torchcodec for audio IO,
    which needs system ffmpeg libs -- `synth` loads references with soundfile instead.)

    xtts_clone.py synth OUT [--all|--words a,b] [--speakers spk01,spk02] [--takes 4] [--trim]
    xtts_clone.py score OUT            # strict transcript==text on OUT/tts_check.csv
    xtts_clone.py keep  OUT DEST       # passing COMMAND-word clips -> DEST/<word>/spkNN_NNN.wav

Between synth and score run the language+transcript gate from the kws venv:
    kws-tts-check OUT      (writes OUT/tts_check.csv; needs LD_LIBRARY_PATH=<venv nvidia libs>
                            for CUDA, else it falls back to CPU)
Data root: $KWS_DATA_ROOT, else ~/kws-data (the Linux default of kws_de.config).
"""

import argparse
import collections
import csv
import glob
import os
import re
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("COQUI_TOS_AGREED", "1")  # CPML, non-commercial: private use only (spec)

DATA_ROOT = Path(os.environ.get("KWS_DATA_ROOT", "~/kws-data")).expanduser()
WORDS_ROOT = DATA_ROOT / "data/recordings/approved/words"
COMMAND = [
    "Licht",
    "Kühlschrank",
    "Heizung",
    "Aufstelldach",
    "Küche",
    "Dach",
    "Außen",
    "Lesen",
    "an",
    "aus",
    "auf",
    "zu",
    "heller",
    "dunkler",
    "wärmer",
    "kälter",
    "leise",
    "fünfundzwanzig",
    "fünfzig",
    "fünfundsiebzig",
    "hundert",
]
COMPOUNDS = [
    "Küchenlicht an",
    "Küchenlicht aus",
    "Außenlicht an",
    "Außenlicht aus",
    "Leselicht an",
    "Leselicht aus",
]
SCENES = ["Gute Nacht", "Guten Morgen", "Leseratte", "Nachtlicht"]
SMOKE = [
    "Licht",
    "Kühlschrank",
    "an",
    "aus",
    "fünfundzwanzig",
    "Aufstelldach",
    "Küchenlicht an",
    "Leselicht aus",
    "Gute Nacht",
    "Leseratte",
]
MANIFEST = ["file", "text", "voice", "engine"]


def norm(text: str) -> str:
    """Whisper writes "Lese Licht aus." for "Leselicht aus": compare letters only."""
    return re.sub(r"[^a-zäöüß]", "", text.lower())


def first_utterance(y, sr=16000, gap_ms=300, floor_db=-35.0, pad_ms=120):
    """Cut at the first silence >= gap_ms after speech onset. XTTS keeps talking past a
    one-word text (it recites the reference vocabulary), so everything after the first
    pause is babble, not the word. Leaves the clip alone when there is no pause."""
    import numpy as np

    hop = sr // 100  # 10 ms frames
    n = len(y) // hop
    rms = np.array([np.sqrt(np.mean(y[i * hop : (i + 1) * hop] ** 2)) + 1e-9 for i in range(n)])
    db = 20 * np.log10(rms / (rms.max() + 1e-9))
    voiced = db > floor_db
    on = int(np.argmax(voiced)) if voiced.any() else 0
    need, run, end = gap_ms // 10, 0, n
    for i in range(on, n):
        run = run + 1 if not voiced[i] else 0
        if run >= need:
            end = i - run + 1
            break
    pad = pad_ms * sr // 1000
    return y[max(0, on * hop - pad) : min(len(y), end * hop + pad)]


def synth(a) -> None:
    import soundfile as sf
    import torch
    import torchaudio

    def _load(path, *_a, **_k):  # torchaudio.load -> torchcodec -> system ffmpeg; avoid it
        y, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return torch.from_numpy(y.T.copy()), sr

    torchaudio.load = _load
    from TTS.api import TTS

    words = a.words.split(",") if a.words else (COMMAND + COMPOUNDS + SCENES if a.all else SMOKE)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(dev)
    print(f"[xtts] device={dev}, model ready in {time.time() - t0:.0f}s", flush=True)
    man = out / "manifest.csv"
    new = not man.exists()
    with man.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST)
        if new:
            w.writeheader()
        for spk in a.speakers.split(","):
            refs = sorted(glob.glob(str(Path(a.ref_root) / a.ref_glob.format(spk=spk))))
            if not refs:
                print(f"[xtts] {spk}: no reference clips under {a.ref_root}", file=sys.stderr)
                continue
            t0 = time.time()
            for text in words:
                for k in range(a.takes):
                    fn = f"{text.replace(' ', '_')}__clone_{spk}_{k:02d}.wav"
                    wav = tts.tts(text=text, speaker_wav=refs, language="de", split_sentences=False)
                    y = torchaudio.functional.resample(
                        torch.tensor(wav, dtype=torch.float32),
                        tts.synthesizer.output_sample_rate,
                        16000,
                    ).numpy()
                    if a.trim:
                        y = first_utterance(y)
                    sf.write(out / fn, y, 16000, subtype="PCM_16")
                    w.writerow(
                        {"file": fn, "text": text, "voice": f"clone:{spk}", "engine": "xtts_v2"}
                    )
                    fh.flush()
            print(
                f"[xtts] {spk}: {len(refs)} refs -> {len(words) * a.takes} clips "
                f"in {time.time() - t0:.0f}s",
                flush=True,
            )


def _rows(out: Path):
    man = {r["file"]: r["text"] for r in csv.DictReader(open(out / "manifest.csv"))}
    for r in csv.DictReader(open(out / "tts_check.csv")):
        yield r, man[r["file"]], norm(r["transcript"]) == norm(man[r["file"]])


def score(a) -> None:
    rows = list(_rows(Path(a.out)))
    per = collections.Counter()
    for r, text, ok in rows:
        per[(text, r["voice"])] += ok
    print(
        f"strict {sum(ok for *_, ok in rows)}/{len(rows)}  "
        f"lenient {sum(r['ok'] == '1' for r, *_ in rows)}/{len(rows)}  "
        f"word×speaker with >=1 pass {sum(v > 0 for v in per.values())}/{len(per)}"
    )
    for r, _text, ok in rows:
        if not ok:
            print(f"  x {r['file']:<40} -> {r['transcript'][:60]!r}")


def keep(a) -> None:
    dest = Path(a.dest)
    n = collections.Counter()
    for r, text, ok in _rows(Path(a.out)):
        if not ok or text not in COMMAND:
            continue
        spk = r["voice"].split(":")[1]
        n[(text, spk)] += 1
        (dest / text).mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(a.out) / r["file"], dest / text / f"{spk}_{n[(text, spk)]:03d}.wav")
    print(f"kept {sum(n.values())} clips -> {dest}")
    for w in COMMAND:
        print(f"  {w:<16}", {s: n[(w, s)] for s in sorted({s for _, s in n})})


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synth")
    s.add_argument("out")
    s.add_argument("--speakers", default="spk01,spk02,spk22")
    s.add_argument("--takes", type=int, default=1)
    s.add_argument("--words", default="")
    s.add_argument("--all", action="store_true")
    s.add_argument("--trim", action="store_true")
    s.add_argument("--device", default="auto")
    s.add_argument("--ref-root", default=str(WORDS_ROOT))
    s.add_argument("--ref-glob", default="*/{spk}_*.wav")
    s.set_defaults(fn=synth)
    c = sub.add_parser("score")
    c.add_argument("out")
    c.set_defaults(fn=score)
    k = sub.add_parser("keep")
    k.add_argument("out")
    k.add_argument("dest")
    k.set_defaults(fn=keep)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
