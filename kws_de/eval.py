# NOTE: pickle here only reads this repo's own local, gitignored data/ cache
# (raw_clips.pkl / noise.pkl, written by kws_de.data) — never untrusted input.
import argparse
import os
import pickle
import re

import numpy as np

from kws_de import config


def metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = config.NUM_CLASSES
    cm = np.zeros((n, n), int)
    for t, p in zip(y_true, y_pred, strict=True):
        cm[t, p] += 1
    acc = float((y_true == y_pred).mean())
    per_class = {
        config.LABELS[i]: (float(cm[i, i] / cm[i].sum()) if cm[i].sum() else 0.0) for i in range(n)
    }
    unk = config.label_index("_unknown_")
    cmd_idx = {config.label_index(c) for c in config.COMMANDS}
    unk_mask = y_true == unk
    fa = float(np.mean([p in cmd_idx for p in y_pred[unk_mask]])) if unk_mask.any() else 0.0
    return {
        "accuracy": acc,
        "per_class": per_class,
        "unknown_false_accept": fa,
        "confusion": cm,
    }


def snr_sweep(eval_fn, snrs) -> dict:
    return {float(s): float(eval_fn(s)) for s in snrs}


def intent_accuracy(true_intents, pred_intents) -> float:
    """Fraction of predictions where device+zone+action all match the true intent
    (an `Intent`/`Rejection` mismatch, e.g. a wrong slot or a rejected parse,
    counts as wrong)."""
    pairs = list(zip(true_intents, pred_intents, strict=True))
    if not pairs:
        return 0.0
    return sum(t == p for t, p in pairs) / len(pairs)


def evaluate_streaming(predict_fn, clips_sequences) -> dict:  # pragma: no cover
    """Run `KeywordStream` over held-out command-clip sequences, `grammar.parse`
    the resulting events, and score against each sequence's true intent. Thin
    I/O glue over `intent_accuracy` — not wired up yet (v2 Task 6 follow-on)."""
    raise NotImplementedError("wire KeywordStream + grammar.parse; see plan Task 6")


# --- v2 Task 8 step 5: the full command catalog, end-to-end -----------------


def build_catalog() -> list:
    """Every VALID intent enumerated from `config.DEVICE_ACTIONS` (+ zones for
    `config.ZONED_DEVICES` only): e.g. Intent("Licht", None, "an"),
    Intent("Licht", "Küche", "an"), Intent("Heizung", None, "waermer"), ...
    Bare `device action` (no zone) is included for every device, plus one
    zoned variant per zone for zoned devices."""
    from kws_de.grammar import Intent

    catalog = []
    for device in config.DEVICES:
        for action in config.DEVICE_ACTIONS[device]:
            catalog.append(Intent(device, None, action))
            if device in config.ZONED_DEVICES:
                for zone in config.ZONES:
                    catalog.append(Intent(device, zone, action))
    return catalog


def intent_text(intent) -> str:
    """The spoken sentence for an intent, as a person would say it: `Licht Küche
    fünfzig Prozent`, `Heizung wärmer`. Light levels get the natural "Prozent";
    it is not a command keyword (the level word alone is unambiguous, so the
    grammar drops it as an unknown token) — it just makes the prompt read like
    real speech. Single source for the recorder's sentence prompts."""
    words = [t for t in (intent.device, intent.zone, intent.action) if t]
    if intent.action in config.LIGHT_LEVELS:
        words.append("Prozent")
    return " ".join(words)


def prompt_intent(prompt: str):
    """Recover the true `Intent` a sentence prompt asks for, via the same
    qc.required_tokens/label_for_token mapping QC uses to segment/label words —
    so this is the same ground truth QC already agreed the recording matches."""
    from kws_de import qc
    from kws_de.grammar import parse

    return parse([qc.label_for_token(t) for t in qc.required_tokens(prompt, "sentences")])


HELD_OUT = "held-out"
IN_TRAINING = "user-customised, in-training"


def _relative_to_data_root(p) -> str:
    """Portable display form of a path under the data root (`data/manifest.json`,
    `models/command_v3.tflite`, `data/recordings/approved`, ...) for reports/sidecars
    that get git-committed — NEVER the absolute resolution, which embeds the local
    `KWS_DATA_ROOT` and username (`config.py`: "Never commit a machine path here").
    Falls back to just the basename for a path outside the data root (e.g. a test's
    tmp_path), which is still safe (no machine-local directory structure leaked)."""
    from pathlib import Path

    p = Path(p)
    for name, base in (("data", config.DATA_DIR), ("models", config.MODELS_DIR)):
        try:
            return str(Path(name) / p.relative_to(base))
        except ValueError:
            continue
    return p.name


def _trained_speakers(manifest_path) -> tuple[set[str], bool, str | None]:
    """Speaker ids (`rec:` stripped) listed in the training manifest's `train`
    split. This is a SPEAKER-LEVEL match, not a per-clip one — `build_manifest`
    records no per-file/take field, only which speakers' device recordings went
    into the split (`kws_de/manifest.py`). So "in `trained_speakers`" means "this
    speaker had SOME word/negative clip in the training build as of
    `built_at`" — it does NOT mean this specific clip was in that build. A new
    take QC-approved for an already-trained speaker after the manifest's
    `built_at` is real, was never seen by the current model, and is still
    reported as `IN_TRAINING` by `eval_recordings` purely because the speaker
    id matches; only a fresh `kws-dataset build` (+ retrain) makes the match
    exact again. Returns `(set(), False, None)` if no path is given or the file
    doesn't exist — callers must never guess "in-training" without the manifest
    saying so."""
    import json
    from pathlib import Path

    if manifest_path is None or not Path(manifest_path).exists():
        return set(), False, None
    manifest = json.loads(Path(manifest_path).read_text())
    speakers = set(manifest.get("splits", {}).get("train", {}).get("speakers", []))
    return speakers, True, manifest.get("built_at")


def eval_recordings(approved, predict_fn, *, step_ms: int = 100, manifest_path=None) -> dict:
    """Recordings-based eval, split into the two honest figures: `IN_TRAINING`
    (`"user-customised, in-training"`) for clips whose speaker's device recordings
    are listed in `manifest_path`'s `train` split, `HELD_OUT` (`"held-out"`)
    for everything else. Phrase clips are always `HELD_OUT` — the training build
    (`kws_de.dataset.build`) never reads `approved/phrases/`, only `approved/words/`
    and `approved/negatives/` (see `kws_de.data.recordings_root`/`negative_windows`),
    so a phrase clip is never actually training material regardless of speaker.
    With no manifest (missing path or file absent), every clip is `HELD_OUT`."""
    import csv
    from collections import defaultdict
    from pathlib import Path

    import soundfile as sf

    from kws_de.grammar import Intent, parse

    approved = Path(approved)
    labels = config.COMMAND_LABELS
    step = config.SAMPLE_RATE * step_ms // 1000
    trained_speakers, manifest_found, built_at = _trained_speakers(manifest_path)

    def figure_for(set_name: str, spk: str) -> str:
        return IN_TRAINING if set_name != "phrases" and spk in trained_speakers else HELD_OUT

    figures = {
        label: {"isolated": {}, "e2e": {}, "false_accepts": {}} for label in (IN_TRAINING, HELD_OUT)
    }

    def _new_word_row():
        return {"n": 0, "ok": 0, "per_word": defaultdict(lambda: [0, 0])}

    iso = defaultdict(lambda: defaultdict(_new_word_row))
    for f in sorted((approved / "words").glob("*/*.wav")):
        lab, spk = f.parent.name, f.stem.split("_")[0]
        sig, _ = sf.read(f, dtype="float32", always_2d=True)
        pred = labels[int(np.argmax(predict_fn(sig[:, 0])))]
        r = iso[figure_for("words", spk)][spk]
        r["n"] += 1
        r["ok"] += pred == lab
        r["per_word"][lab][0] += pred == lab
        r["per_word"][lab][1] += 1
    for label, per_spk in iso.items():
        figures[label]["isolated"] = {
            s: {
                "n": r["n"],
                "acc": r["ok"] / r["n"],
                "per_word": {w: a / n for w, (a, n) in r["per_word"].items()},
            }
            for s, r in per_spk.items()
        }

    def _events(path):
        sig, _ = sf.read(path, dtype="float32", always_2d=True)
        return _stream_events(predict_fn, sig[:, 0], labels, step)

    e2e = defaultdict(lambda: defaultdict(lambda: {"n": 0, "ok": 0}))
    idx = approved / "phrases" / "index.csv"
    if idx.exists():
        # index `file` is already relative to approved/ — qc.run_qc writes
        # `str(dst.relative_to(approved))`, so it carries the "phrases/" segment itself.
        with idx.open() as fh:
            for r in csv.DictReader(fh):
                got = parse(_events(approved / r["file"]))
                e = e2e[figure_for("phrases", r["speaker"])][r["speaker"]]
                e["n"] += 1
                e["ok"] += isinstance(got, Intent) and got == prompt_intent(r["prompt"])
    for label, per_spk in e2e.items():
        figures[label]["e2e"] = {
            s: {"n": v["n"], "acc": v["ok"] / v["n"]} for s, v in per_spk.items()
        }

    fa = defaultdict(lambda: defaultdict(lambda: {"n": 0, "fired": 0}))
    nidx = approved / "negatives" / "index.csv"
    if nidx.exists():
        with nidx.open() as fh:  # `file` relative to approved/, as for phrases
            for r in csv.DictReader(fh):
                got = parse(_events(approved / r["file"]))
                n = fa[figure_for("negatives", r["speaker"])][r["speaker"]]
                n["n"] += 1
                n["fired"] += isinstance(got, Intent)
    for label, per_spk in fa.items():
        figures[label]["false_accepts"] = {
            s: {"n": v["n"], "rate": v["fired"] / v["n"]} for s, v in per_spk.items()
        }

    return {
        "manifest_path": (
            _relative_to_data_root(manifest_path) if manifest_path is not None else None
        ),
        "manifest_found": manifest_found,
        "manifest_built_at": built_at,
        "figures": figures,
    }


def _figure_totals(fig: dict) -> tuple[int, set[str]]:
    n_clips = sum(
        v["n"] for bucket in ("isolated", "e2e", "false_accepts") for v in fig[bucket].values()
    )
    speakers = set(fig["isolated"]) | set(fig["e2e"]) | set(fig["false_accepts"])
    return n_clips, speakers


RECORDINGS_SECTION_START = "<!-- kws-eval:recordings:start -->"
RECORDINGS_SECTION_END = "<!-- kws-eval:recordings:end -->"


def replace_recordings_section(report: str, section: str) -> str:
    """Put `section` into `report` between the marker comments, replacing whatever
    was there — one run of `kws-eval --recordings` leaves exactly ONE recordings
    section, so the markdown can never disagree with the JSON sidecar (which is
    overwritten every run). Appends the marked block if the report has none yet."""
    body = f"{RECORDINGS_SECTION_START}\n\n{section}\n{RECORDINGS_SECTION_END}\n"
    start = report.find(RECORDINGS_SECTION_START)
    end = report.find(RECORDINGS_SECTION_END)
    if start != -1 and end > start:
        return report[:start] + body + report[end + len(RECORDINGS_SECTION_END) + 1 :]
    return (report.rstrip("\n") + "\n\n" if report.strip() else "") + body


def render_recordings_section(res: dict) -> str:
    """Markdown for the recordings-based eval: two figures, labelled verbatim
    `IN_TRAINING` (`"user-customised, in-training"`) and `HELD_OUT` (`"held-out"`),
    each stating its clip count, speaker count, and (once, up top) the
    data-root-relative manifest path checked and the model file actually measured
    (never an absolute, machine-local path -- see `_relative_to_data_root`) -- or
    that no manifest was found, in which case every clip is held-out."""
    if res["manifest_found"]:
        built = f", built {res['manifest_built_at']}" if res.get("manifest_built_at") else ""
        out = [f"Training manifest checked: `{res['manifest_path']}`{built}.\n"]
        out.append(
            "Match is speaker-level, not per-clip: a clip counts `user-customised, "
            "in-training` if its speaker has ANY word/negative clip in this "
            "manifest's train split, including takes recorded/QC-approved after "
            f"{res['manifest_built_at'] or 'the manifest was built'} — those clips "
            "were never actually seen by the current model. Re-run `kws-dataset "
            "build` + retrain to make the match exact again. Phrase clips are "
            "always `held-out` (never used for training).\n"
        )
    elif res["manifest_path"] is not None:
        out = [f"No training manifest found at `{res['manifest_path']}`; all clips held-out.\n"]
    else:
        out = ["No training manifest given; all clips held-out.\n"]

    if res.get("model_path"):
        sha = res.get("model_sha256", "")
        out.insert(
            0,
            f"Model measured: `{res['model_path']}` (sha256 `{sha[:12]}`)"
            f"{', evaluated ' + res['evaluated_at'] if res.get('evaluated_at') else ''}.\n",
        )

    for label in (IN_TRAINING, HELD_OUT):
        fig = res["figures"][label]
        n_clips, speakers = _figure_totals(fig)
        out.append(f"\n## {label}\n")  # leading blank line: markdownlint MD022/MD058
        out.append(f"{n_clips} clips across {len(speakers)} speakers.\n")
        out.append(
            "| speaker | isolated words n | acc | e2e phrases n | intent acc "
            "| negatives n | false-accept rate |\n|---|---|---|---|---|---|---|"
        )
        for spk in sorted(speakers):
            i = fig["isolated"].get(spk, {})
            e = fig["e2e"].get(spk, {})
            f = fig["false_accepts"].get(spk, {})
            out.append(
                f"| {spk} | {i.get('n', 0)} | {i.get('acc', float('nan')):.3f} "
                f"| {e.get('n', 0)} | {e.get('acc', float('nan')):.3f} "
                f"| {f.get('n', 0)} | {f.get('rate', float('nan')):.3f} |"
            )
    # one blank line between blocks, never two (markdownlint MD012/MD022/MD058)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out) + "\n")


def _tts_word_clip(word: str, voice: str, rate: int = 170):  # pragma: no cover - shells out
    """Synthesize one word via macOS `say` -> 16 kHz mono float32 array."""
    import subprocess
    import tempfile

    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".wav") as tf_:
        subprocess.run(
            [
                "say",
                "-v",
                voice,
                "-r",
                str(rate),
                "--data-format=LEI16@16000",
                "-o",
                tf_.name,
                word,
            ],
            check=True,
            capture_output=True,
        )
        y, _sr = sf.read(tf_.name)
    return y.astype(np.float32)


def _intent_audio(intent, voice: str, word_cache: dict, gap_ms=250, pad_ms=400):  # pragma: no cover
    """Concatenate the per-word TTS clips for one catalog intent (device [zone]
    action), separated by silence gaps, padded with silence front/back — the
    continuous audio a streaming detector would see."""
    tokens = [intent.device, *([intent.zone] if intent.zone else []), intent.action]
    gap = np.zeros(int(config.SAMPLE_RATE * gap_ms / 1000), np.float32)
    pad = np.zeros(int(config.SAMPLE_RATE * pad_ms / 1000), np.float32)
    parts = [pad]
    for tok in tokens:
        parts.append(word_cache[(tok, voice)])
        parts.append(gap)
    parts.append(pad)
    return np.concatenate(parts)


def make_command_predict_fn(tflite_bytes: bytes):  # pragma: no cover - needs tflite runtime
    """INT8-tflite command model -> predict_fn(1s window) -> float posterior
    over config.COMMAND_LABELS (dequantized from the int8 output tensor)."""
    import tensorflow as tf

    from kws_de.features import mfcc

    itp = tf.lite.Interpreter(model_content=tflite_bytes, num_threads=os.cpu_count())
    itp.allocate_tensors()
    inp = itp.get_input_details()[0]
    out = itp.get_output_details()[0]
    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = out["quantization"]

    def predict_fn(window: np.ndarray) -> np.ndarray:
        feat = mfcc(window)[None, ..., None].astype(np.float32)
        q = np.round(feat / in_scale + in_zp).astype(np.int8)
        itp.set_tensor(inp["index"], q)
        itp.invoke()
        raw = itp.get_tensor(out["index"])[0].astype(np.float32)
        return (raw - out_zp) * out_scale

    return predict_fn


def _stream_events(predict_fn, audio, labels, step_samples, **stream_kwargs) -> list:
    # pragma: no cover - I/O glue (real model + real audio)
    """Slide a trailing 1s window over `audio` every `step_samples`, running
    `predict_fn` + `KeywordStream` to collect the ordered keyword events."""
    from kws_de.stream import KeywordStream

    ks = KeywordStream(predict_fn, labels, **stream_kwargs)
    events = []
    n = len(audio)
    pos = config.CLIP_SAMPLES
    if n < pos:
        audio = np.pad(audio, (pos - n, 0))
        n = pos
    while pos <= n:
        window = audio[pos - config.CLIP_SAMPLES : pos]
        events += ks.push(predict_fn(window))
        pos += step_samples
    return events


def run_catalog_eval(  # pragma: no cover - orchestration (real TTS + real model)
    predict_fn,
    voices,
    noises=None,
    snr=None,
    seed=0,
    step_ms=100,
    smooth_win=3,
    threshold=0.5,
    min_consecutive=2,
    gap_steps=2,
) -> dict:
    """Run the FULL command catalog end-to-end: synthesize each catalog intent
    (several voices), run audio -> mfcc -> KeywordStream -> grammar.parse, and
    score against the true intent. Returns overall + per-entry + per-slot
    accuracy. `noises`/`snr` optionally mix the synthesized audio for the SNR
    sweep (see docs/eval-report-v2.md)."""
    from kws_de.augment import mix_at_snr
    from kws_de.data import command_words
    from kws_de.grammar import Intent, parse

    rng = np.random.default_rng(seed)
    catalog = build_catalog()
    words = command_words()
    word_cache = {(w, v): _tts_word_clip(w, v) for w in words for v in voices}
    step_samples = int(config.SAMPLE_RATE * step_ms / 1000)

    per_entry = []
    total = correct = 0
    device_correct = action_correct = 0
    zone_correct = zone_total = 0
    for intent in catalog:
        entry_trials = entry_correct = 0
        for voice in voices:
            audio = _intent_audio(intent, voice, word_cache)
            if noises is not None and snr is not None:
                noise_clip = noises[int(rng.integers(0, len(noises)))]
                audio = mix_at_snr(audio, noise_clip, snr, rng)
            events = _stream_events(
                predict_fn,
                audio,
                config.COMMAND_LABELS,
                step_samples,
                smooth_win=smooth_win,
                threshold=threshold,
                min_consecutive=min_consecutive,
                gap_steps=gap_steps,
            )
            pred = parse(events)
            total += 1
            entry_trials += 1
            if pred == intent:
                correct += 1
                entry_correct += 1
            if isinstance(pred, Intent):
                device_correct += int(pred.device == intent.device)
                action_correct += int(pred.action == intent.action)
            if intent.zone is not None:
                zone_total += 1
                if isinstance(pred, Intent) and pred.zone == intent.zone:
                    zone_correct += 1
        per_entry.append(
            {
                "intent": intent,
                "trials": entry_trials,
                "correct": entry_correct,
                "accuracy": entry_correct / entry_trials if entry_trials else 0.0,
            }
        )
    return {
        "overall_accuracy": correct / total if total else 0.0,
        "total_trials": total,
        "per_entry": per_entry,
        "device_accuracy": device_correct / total if total else 0.0,
        "action_accuracy": action_correct / total if total else 0.0,
        "zone_accuracy": (zone_correct / zone_total) if zone_total else None,
        "voices": list(voices),
        "smooth_win": smooth_win,
        "threshold": threshold,
        "min_consecutive": min_consecutive,
        "gap_steps": gap_steps,
        "step_ms": step_ms,
    }


def render_report(results: dict) -> str:
    lines = [
        "## Evaluation summary",
        "",
        f"**Accuracy:** {results.get('accuracy', 0):.3f}",
        "",
        "## SNR sweep",
        "",
        "| SNR (dB) | Accuracy |",
        "|---|---|",
    ]
    for snr, acc in sorted(results.get("snr_sweep", {}).items(), reverse=True):
        lines.append(f"| {snr:.0f} | {acc:.3f} |")
    return "\n".join(lines) + "\n"


def _tflite_predict(tflite_bytes: bytes, X) -> np.ndarray:
    """Run the INT8 tflite model sample-by-sample, returning class indices.

    Argmax over an affine-quantized (single scale/zero-point) output tensor
    matches the float argmax, so no dequantization is needed for accuracy.
    """
    import tensorflow as tf

    itp = tf.lite.Interpreter(model_content=tflite_bytes, num_threads=os.cpu_count())
    itp.allocate_tensors()
    inp = itp.get_input_details()[0]
    out = itp.get_output_details()[0]
    scale, zero_point = inp["quantization"]
    Xc = np.asarray(X, np.float32)[..., None]
    preds = np.empty(Xc.shape[0], dtype=np.int64)
    for i in range(Xc.shape[0]):
        q = np.round(Xc[i : i + 1] / scale + zero_point).astype(np.int8)
        itp.set_tensor(inp["index"], q)
        itp.invoke()
        preds[i] = int(np.argmax(itp.get_tensor(out["index"])[0]))
    return preds


def _keras_predict(model, X) -> np.ndarray:
    Xc = np.asarray(X, np.float32)[..., None]
    return np.argmax(model.predict(Xc, verbose=0), axis=1)


def _command_clips_at_snr(clips: dict, noises, rng, snr) -> tuple:
    """Raw held-out command clips mixed at one SNR (snr=None -> unmixed/clean),
    MFCC-extracted. Used for the SNR sweep (unknown/silence excluded — the
    sweep measures command-recognition degradation under noise)."""
    from kws_de.augment import mix_at_snr
    from kws_de.features import mfcc

    X, y = [], []
    for cmd in config.COMMANDS:
        for clip in clips.get(cmd, []):
            sig = clip
            if snr is not None:
                noise = noises[int(rng.integers(0, len(noises)))]
                sig = mix_at_snr(clip, noise, snr, rng)
            X.append(mfcc(sig))
            y.append(config.label_index(cmd))
    return np.asarray(X, np.float32), np.asarray(y, np.int64)


def _fmt_confusion(cm) -> str:
    header = "| true \\ pred | " + " | ".join(config.LABELS) + " |"
    sep = "|---" * (len(config.LABELS) + 1) + "|"
    rows = [header, sep]
    for i, label in enumerate(config.LABELS):
        rows.append("| " + label + " | " + " | ".join(str(x) for x in cm[i]) + " |")
    return "\n".join(rows)


def _origin_counts(cached_clips: dict) -> dict:
    """Per-command-word real (MSWC) vs TTS-synthesized clip counts, derived from
    speaker id prefix ("tts:...") in the cached raw-clips dict."""
    counts = {}
    for label in [*config.COMMANDS, "_unknown_"]:
        items = cached_clips.get(label, [])
        n_tts = sum(1 for _, spk in items if spk.startswith("tts:"))
        counts[label] = {"real": len(items) - n_tts, "tts": n_tts, "total": len(items)}
    return counts


def _origin_counts_for(cached_clips: dict, words: list) -> dict:
    """Like `_origin_counts` but over an arbitrary word list (v2 vocab)."""
    counts = {}
    for label in [*words, "_unknown_"]:
        items = cached_clips.get(label, [])
        n_tts = sum(1 for _, spk in items if spk.startswith("tts:"))
        counts[label] = {"real": len(items) - n_tts, "tts": n_tts, "total": len(items)}
    return counts


def _write_catalog_report(out: str) -> None:  # pragma: no cover - I/O wrapper (manual/integration)
    """Task 8 step 5: build+run the full command catalog end-to-end (audio ->
    mfcc -> KeywordStream -> grammar.parse -> Intent) against the trained
    INT8 command model, and write docs/eval-report-v2.md."""
    import pickle

    from kws_de import budgets
    from kws_de.data import _TTS_RATES, _TTS_VOICES, command_words

    tflite_bytes = (config.MODELS_DIR / "command.tflite").read_bytes()
    predict_fn = make_command_predict_fn(tflite_bytes)

    with open(config.DATA_DIR / "noise.pkl", "rb") as fh:
        noises = pickle.load(fh)

    eval_voices = _TTS_VOICES[:4]
    sweep_voices = _TTS_VOICES[:2]

    clean = run_catalog_eval(predict_fn, eval_voices, seed=0)
    snr_points = [20.0, 10.0, 0.0]
    sweep = {}
    for snr in snr_points:
        r = run_catalog_eval(predict_fn, sweep_voices, noises=noises, snr=snr, seed=int(snr) + 1)
        sweep[snr] = r["overall_accuracy"]

    # provenance: real (MSWC) vs TTS clip counts for the v2 vocab. Prefer the
    # single merged+TTS-filled cache (v1 real clips reused + v2 real fetch +
    # TTS fill, all in one file) if present so word counts aren't double
    # counted/overwritten across separate v1/v2 cache files.
    words = command_words()
    origin = {}
    merged_path = config.DATA_DIR / "raw_clips_merged.pkl"
    if merged_path.exists():
        with open(merged_path, "rb") as fh:
            cached = pickle.load(fh)["clips"]
        origin = _origin_counts_for(cached, words)
    else:
        for cache_name in ("raw_clips.pkl", "raw_clips_v2.pkl"):
            p = config.DATA_DIR / cache_name
            if p.exists():
                with open(p, "rb") as fh:
                    cached = pickle.load(fh)["clips"]
                origin.update(_origin_counts_for(cached, words))

    model_bytes = len(tflite_bytes)
    is_int8 = budgets.is_full_int8(tflite_bytes)
    ops = sorted(budgets.tflite_op_types(tflite_bytes))

    wake_report = None
    wake_path = config.MODELS_DIR / "hey_bus.tflite"
    if wake_path.exists():
        wake_report = budgets.check_wake_budgets(wake_path.read_bytes())

    report = "# kws-de v2 Evaluation Report — command catalog (Task 8)\n\n"
    report += (
        "**Method:** every entry below is a VALID intent enumerated from "
        "`config.DEVICE_ACTIONS` (+ zones for `Licht` only). For each entry, the "
        "device/zone/action WORDS are TTS-synthesized (macOS `say`, several German "
        f"voices: {', '.join(eval_voices)}) and concatenated with silence gaps into "
        "one continuous utterance — the audio a streaming detector would see. That "
        "audio is run through the FULL pipeline: sliding-window `kws_de.features.mfcc` "
        "-> the trained INT8 command model -> `kws_de.stream.KeywordStream` -> "
        "`kws_de.grammar.parse` -> `Intent`, compared against the true intent. "
        "**All catalog phrases are TTS, not real recorded commands** — this measures "
        "the streaming+grammar composition end-to-end, not raw word-recognition "
        "accuracy on natural speech (see the per-word real/TTS provenance table for "
        "how much of the underlying vocabulary is real MSWC speech vs TTS-filled).\n\n"
    )

    report += f"## Overall full-intent accuracy: {clean['overall_accuracy']:.3f}\n\n"
    report += f"({clean['total_trials']} trials = {len(clean['per_entry'])} catalog entries "
    report += f"x {len(eval_voices)} voices)\n\n"
    report += "## Per-slot accuracy (clean)\n\n"
    report += f"- Device: {clean['device_accuracy']:.3f}\n"
    report += f"- Action: {clean['action_accuracy']:.3f}\n"
    zone_acc = clean["zone_accuracy"]
    report += f"- Zone (Licht only): {zone_acc:.3f}\n" if zone_acc is not None else ""

    report += "\n## Command catalog — per-entry full-intent accuracy\n\n"
    report += "| Device | Zone | Action | Accuracy | Trials |\n|---|---|---|---|---|\n"
    for row in clean["per_entry"]:
        i = row["intent"]
        report += (
            f"| {i.device} | {i.zone or '-'} | {i.action} | "
            f"{row['accuracy']:.3f} | {row['trials']} |\n"
        )

    report += "\n## SNR sweep — overall full-intent accuracy\n\n"
    report += f"(2 voices per entry: {', '.join(sweep_voices)})\n\n"
    report += "| SNR (dB) | Full-intent accuracy |\n|---|---|\n"
    report += f"| clean | {clean['overall_accuracy']:.3f} |\n"
    for snr in snr_points:
        report += f"| {snr:.0f} | {sweep[snr]:.3f} |\n"

    report += "\n## Command model budget (INT8)\n\n"
    report += f"- Model size: {model_bytes} bytes (budget {config.MAX_MODEL_BYTES})\n"
    report += f"- Full INT8: {is_int8}\n"
    report += f"- Ops: {', '.join(ops)}\n"

    report += '\n## Wake model ("Hey Bus") budget\n\n'
    if wake_report is not None:
        report += f"- Model size: {wake_report['model_bytes']} bytes (budget 150 000)\n"
        report += (
            f"- 8-bit quantized: {wake_report['quantized_8bit']} "
            f"(int8 I/O: {wake_report['int8']}; microWakeWord exports uint8)\n"
        )
    else:
        report += (
            "Not trained in this run — microWakeWord's trainer requires Piper "
            "sample generation (a separate ~cloned repo + TTS voice checkpoint, "
            "not present locally) plus several GB of pre-generated negative/ambient "
            "spectrogram feature sets from HuggingFace (`kahrendt/microwakeword`: "
            "dinner_party, dinner_party_eval, no_speech, speech). Neither was fetched "
            "in this run (out of scope for the time budget here). The local 3.10 "
            "venv with `microwakeword` installed and importable is proven "
            "(`train/mww/setup.sh`), and the runtime integration path — "
            "`kws_de.wake.WakeDetector` + `load_wake_tflite` + "
            "`kws_de.budgets.check_wake_budgets` — is unit-tested against a "
            "stand-in INT8 tflite, but no real `hey_bus.tflite` was produced.\n"
        )

    report += "\n## Vocabulary provenance (real MSWC vs TTS-added)\n\n"
    report += "| Word | Real (MSWC) | TTS-added | Total |\n|---|---|---|---|\n"
    for w in words:
        o = origin.get(w, {"real": 0, "tts": 0, "total": 0})
        report += f"| {w} | {o['real']} | {o['tts']} | {o['total']} |\n"
    report += (
        f"\nTTS source for both training-data fill and catalog synthesis: macOS `say`, "
        f"German voices ({', '.join(_TTS_VOICES)}) at rates "
        f"{_TTS_RATES[0]}-{_TTS_RATES[-1]} wpm.\n"
    )

    report += (
        "\n## vs v1 / MultiNet\n\n"
        "v1 measured word-level command accuracy on 5 devices (no grammar, no "
        "streaming composition). v2 measures a strictly harder, end-to-end task: "
        "full intent (device+zone+action) recovered from continuous synthesized "
        "speech through the streaming detector and grammar — a single wrong/missed "
        "word anywhere in the phrase fails the whole entry. The headline number "
        "above is therefore not directly comparable to v1's per-word accuracy or "
        "to MultiNet's isolated-word numbers; it is the number that matters for "
        "actually using the assistant.\n"
    )

    out_path = config.DATA_DIR.parent / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"wrote {out_path}")


def main() -> None:  # pragma: no cover - I/O wrapper (manual/integration)
    import tensorflow as tf

    from kws_de import budgets
    from kws_de.data import _TTS_RATES, _TTS_VOICES, split_by_speaker

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/eval-report.md")
    ap.add_argument(
        "--v2-catalog",
        action="store_true",
        help="run the v2 full command-catalog end-to-end eval instead of the v1 report",
    )
    ap.add_argument(
        "--recordings",
        default=None,
        help="QC-approved recordings dir -> append the held-out / in-training sections",
    )
    ap.add_argument(
        "--prefix",
        default=None,
        help="npz/manifest prefix the model was trained on (default: features -> "
        "manifest.json; e.g. features_v3 -> manifest_v3.json), see kws_de.dataset.build",
    )
    ap.add_argument(
        "--qat",
        action="store_true",
        help="evaluate the QAT export (command<suffix>_qat.tflite) instead of the PTQ one",
    )
    args = ap.parse_args()
    if args.v2_catalog:
        out = args.out if args.out != "docs/eval-report.md" else "docs/eval-report-v2.md"
        _write_catalog_report(out)
        return
    if args.recordings:
        import hashlib
        import json
        from datetime import UTC, datetime
        from pathlib import Path

        # `--prefix P` selects P's manifest, P's model artefact AND P's report:
        # v3 recordings figures must never land in the v2 catalog report.
        suffix = (args.prefix or "features").removeprefix("features")
        if args.out != "docs/eval-report.md":
            out = args.out
        else:
            out = f"docs/eval-report{suffix.replace('_', '-') or '-v2'}.md"
        out_path = config.DATA_DIR.parent / out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        approved_dir = Path(args.recordings)
        if not approved_dir.is_dir():
            note = f"no approved recordings found under {_relative_to_data_root(approved_dir)}\n"
            report = out_path.read_text() if out_path.exists() else ""
            out_path.write_text(replace_recordings_section(report, note))
            print(note.strip())
            return
        if args.qat:
            suffix = f"{suffix}_qat"  # the QAT export of the same prefix (kws-export --qat)
        model_path = config.MODELS_DIR / (f"command{suffix}.tflite" if suffix else "command.tflite")
        try:
            tflite_bytes = model_path.read_bytes()
            predict_fn = make_command_predict_fn(tflite_bytes)
        except Exception as e:  # noqa: BLE001 - re-raised with context, not swallowed
            raise SystemExit(f"could not load command model for --recordings: {e}") from e
        manifest_path = config.DATA_DIR / f"manifest{suffix}.json"
        res = eval_recordings(approved_dir, predict_fn, manifest_path=manifest_path)
        res["model_path"] = _relative_to_data_root(model_path)
        res["model_sha256"] = hashlib.sha256(tflite_bytes).hexdigest()
        res["evaluated_at"] = datetime.now(UTC).isoformat()
        section = render_recordings_section(res)
        report = out_path.read_text() if out_path.exists() else ""
        out_path.write_text(replace_recordings_section(report, section))
        json_path = Path(f"{out_path}.recordings.json")
        json_path.write_text(json.dumps(res, indent=2))
        print(f"wrote recordings section to {out_path}, data to {json_path}")
        return

    model = tf.keras.models.load_model(config.MODELS_DIR / "kws.keras")
    tflite_bytes = (config.MODELS_DIR / "model.tflite").read_bytes()

    test = np.load(config.DATA_DIR / "features_test.npz")
    X_test, y_test, is_tts_test = test["X"], test["y"], test["is_tts"]

    y_pred_float = _keras_predict(model, X_test)
    y_pred_int8 = _tflite_predict(tflite_bytes, X_test)
    m_float = metrics(y_test, y_pred_float)
    m_int8 = metrics(y_test, y_pred_int8)

    # Headline, MultiNet-comparable number: REAL-SPEECH-ONLY accuracy, across ALL
    # classes, filtered row-by-row on is_tts (not a hardcoded per-word whitelist) —
    # so each command contributes exactly the real MSWC evidence it actually has
    # (Licht/Kühlschrank/Wasser/`_unknown_` are ~fully real; Heizung is a real
    # majority-of-training/minority-of-real mix; Camping's real contribution is
    # thin, ~22 clips total before the speaker split). See per-class real-row
    # counts below for exactly how much each class contributes here.
    headline_mask = ~is_tts_test
    n_headline = int(headline_mask.sum())
    headline_acc_float = float((y_pred_float[headline_mask] == y_test[headline_mask]).mean())
    headline_acc_int8 = float((y_pred_int8[headline_mask] == y_test[headline_mask]).mean())
    headline_real_rows_per_class = {
        config.LABELS[i]: int(((y_test == i) & headline_mask).sum())
        for i in range(config.NUM_CLASSES)
    }

    # SNR sweep on the held-out RAW clips (speaker-disjoint from train), re-augmented
    # fresh at each SNR. "clean" is approximated as 40 dB (noise ~1% amplitude).
    with open(config.DATA_DIR / "raw_clips.pkl", "rb") as fh:
        cached_clips = pickle.load(fh)["clips"]
    with open(config.DATA_DIR / "noise.pkl", "rb") as fh:
        noises = pickle.load(fh)
    origin = _origin_counts(cached_clips)
    split_rng = np.random.default_rng(0)  # must match kws_de.data._build_and_split's seed
    _, test_clips_raw = split_by_speaker(cached_clips, split_rng, test_frac=0.2)

    snr_points = [40.0, 20.0, 10.0, 0.0]
    sweep_float, sweep_int8 = {}, {}
    for snr in snr_points:
        rng = np.random.default_rng(int(snr) + 1000)
        Xs, ys = _command_clips_at_snr(test_clips_raw, noises, rng, snr)
        sweep_float[snr] = float((_keras_predict(model, Xs) == ys).mean())
        sweep_int8[snr] = float((_tflite_predict(tflite_bytes, Xs) == ys).mean())

    macs = budgets.estimate_macs(model)
    is_int8 = budgets.is_full_int8(tflite_bytes)
    ops = sorted(budgets.tflite_op_types(tflite_bytes))
    model_bytes = len(tflite_bytes)
    within_budget = model_bytes <= config.MAX_MODEL_BYTES and macs <= config.MAX_MACS and is_int8

    def source_label(word: str) -> str:
        o = origin[word]
        if o["tts"] == 0:
            return f"real ({o['real']})"
        if o["real"] == 0:
            return f"TTS-only ({o['tts']})"
        return f"real+TTS mix ({o['real']} real + {o['tts']} TTS)"

    report = "# kws-de Evaluation Report\n\n"

    report += "## Headline: real-speech accuracy (MultiNet-comparable)\n\n"
    report += (
        "Restricted to test rows built from REAL MSWC speech (TTS-synthesized rows "
        "excluded), filtered per-row rather than by a fixed per-word whitelist — so "
        "each command contributes exactly the real MSWC evidence it actually has. "
        f"n={n_headline} held-out real-speech examples (mixed 20/10/0 dB SNR). Real-row "
        "breakdown by class:\n\n"
    )
    report += "| Label | Real rows in headline set |\n|---|---|\n"
    for label in config.LABELS:
        report += f"| {label} | {headline_real_rows_per_class[label]} |\n"
    report += (
        "\n(Camping contributes very few real rows — only 22 real MSWC clips exist "
        "before the speaker split — so its headline contribution is thin; treat the "
        "headline number as strongest for Licht/Kühlschrank/Wasser/`_unknown_`, which "
        "are ~fully real.)\n\n"
    )
    report += "| Model | Accuracy |\n|---|---|\n"
    report += f"| Float (keras) | {headline_acc_float:.3f} |\n"
    report += f"| **INT8 (shipped)** | **{headline_acc_int8:.3f}** |\n"

    report += "\n## Full-model snapshot (`kws_de.eval.render_report`)\n\n"
    report += render_report({"accuracy": m_int8["accuracy"], "snr_sweep": sweep_int8})
    report += (
        "\n(All 7 classes, INT8, command-only SNR sweep — see the full breakdown below "
        "for why this overall number mixes real and synthetic speech.)\n"
    )

    report += (
        "\n## Full 5-word model — overall + per-command accuracy "
        "(held-out test set, mixed SNRs)\n\n"
        "**Camping and Heizung are real+TTS mixes** — Camping had only 22 real MSWC "
        "clips (278 TTS-added to reach 300), Heizung had 120 real (180 TTS-added), so "
        "their rows below blend real and synthetic-voice performance. **Licht, "
        "Kühlschrank, Wasser, and `_unknown_` are ~fully real MSWC speech** (Wasser "
        "reached 300 real clips on a deeper corpus scan — no TTS was needed for it "
        "after all).\n\n"
    )
    report += f"**Overall accuracy — float:** {m_float['accuracy']:.3f}\n\n"
    report += f"**Overall accuracy — INT8 (shipped):** {m_int8['accuracy']:.3f}\n\n"
    report += (
        "| Label | Float accuracy | INT8 accuracy | Data source (clips) |\n|---|---|---|---|\n"
    )
    for label in config.LABELS:
        if label in origin:
            src = source_label(label)
        elif label == "_silence_":
            src = "synthetic (noise)"
        else:
            src = "n/a"
        report += (
            f"| {label} | {m_float['per_class'][label]:.3f} | "
            f"{m_int8['per_class'][label]:.3f} | {src} |\n"
        )

    report += f"\n**Unknown false-accept (float):** {m_float['unknown_false_accept']:.3f}\n"
    report += f"**Unknown false-accept (INT8):** {m_int8['unknown_false_accept']:.3f}\n"

    report += "\n## Confusion matrix — INT8 model (rows=true, cols=predicted)\n\n"
    report += _fmt_confusion(m_int8["confusion"]) + "\n"

    report += (
        "\n## SNR sweep — command-only accuracy (float vs INT8)\n\n"
        "Built from ALL held-out command clips (real + TTS per the source column above), "
        "re-augmented fresh at each SNR; excludes `_unknown_`/`_silence_`.\n\n"
    )
    report += "| SNR (dB) | Float accuracy | INT8 accuracy |\n|---|---|---|\n"
    for snr in sorted(snr_points, reverse=True):
        label = "clean (~40dB)" if snr == 40.0 else f"{snr:.0f}"
        report += f"| {label} | {sweep_float[snr]:.3f} | {sweep_int8[snr]:.3f} |\n"

    report += "\n## Model budget\n\n"
    report += f"- Model size (tflite): {model_bytes} bytes (budget {config.MAX_MODEL_BYTES})\n"
    report += f"- MACs: {macs} (budget {config.MAX_MACS})\n"
    report += f"- Full INT8: {is_int8}\n"
    report += f"- Ops: {', '.join(ops)}\n"
    report += f"- Within all budgets: {within_budget}\n"

    report += "\n## MSWC real-clip counts obtained (of the 300/word target, 600 for _unknown_)\n\n"
    report += "| Label | Real (MSWC) | TTS-added | Total |\n|---|---|---|---|\n"
    for label, o in origin.items():
        report += f"| {label} | {o['real']} | {o['tts']} | {o['total']} |\n"
    report += (
        "\nNoise source: ESC-50 (2000 environmental-sound clips, resampled to 16 kHz). "
        "TTS source: macOS `say`, German voices "
        f"({', '.join(_TTS_VOICES)}) at rates {_TTS_RATES[0]}-{_TTS_RATES[-1]} wpm, "
        "varied punctuation, used only to top up Camping/Heizung to 300 clips since "
        "MSWC German had far fewer real recordings of those two words (a deeper "
        "corpus scan later found 300 real Wasser clips too, so no TTS was needed "
        "for Wasser in this run).\n"
    )

    report += (
        "\n## Comparison to MultiNet\n\n"
        "MultiNet's English command-recognition accuracy is reported at roughly "
        "85-95% on clean speech. The comparable number here is the **headline "
        f"real-speech INT8 accuracy: {headline_acc_int8:.3f} ({headline_acc_int8 * 100:.1f}%)** "
        "across all 7 classes filtered to real MSWC speech only (Licht/Kühlschrank/"
        "Wasser/`_unknown_` ~fully real, Heizung/Camping partially — see the real-row "
        "breakdown above), mixed 20/10/0 dB SNR (harder than MultiNet's clean-speech "
        f"condition). The full 5-word model's INT8 accuracy is {m_int8['accuracy']:.3f}, "
        "but that number includes TTS-augmented Camping/Heizung rows and should not be "
        "quoted as a pure real-speech comparison to MultiNet — use the headline number "
        "for that.\n"
    )

    out_path = config.DATA_DIR.parent / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"wrote {out_path}")
