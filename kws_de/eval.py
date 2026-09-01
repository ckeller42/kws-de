# NOTE: pickle here only reads this repo's own local, gitignored data/ cache
# (raw_clips.pkl / noise.pkl, written by kws_de.data) — never untrusted input.
import argparse
import pickle

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


def render_report(results: dict) -> str:
    lines = [
        "# kws-de Evaluation Report",
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

    itp = tf.lite.Interpreter(model_content=tflite_bytes)
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


def main() -> None:  # pragma: no cover - I/O wrapper (manual/integration)
    import tensorflow as tf

    from kws_de import budgets
    from kws_de.data import _TTS_RATES, _TTS_VOICES, split_by_speaker

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/eval-report.md")
    args = ap.parse_args()

    model = tf.keras.models.load_model(config.MODELS_DIR / "kws.keras")
    tflite_bytes = (config.MODELS_DIR / "model.tflite").read_bytes()

    test = np.load(config.DATA_DIR / "features_test.npz")
    X_test, y_test, is_tts_test = test["X"], test["y"], test["is_tts"]

    y_pred_float = _keras_predict(model, X_test)
    y_pred_int8 = _tflite_predict(tflite_bytes, X_test)
    m_float = metrics(y_test, y_pred_float)
    m_int8 = metrics(y_test, y_pred_int8)

    # Headline, MultiNet-comparable number: REAL-SPEECH-ONLY accuracy on the
    # MSWC-validated subset {Licht, Kühlschrank, Heizung} + _unknown_ + _silence_.
    # Camping (22 real clips) and Wasser (0 real clips) are excluded here — their
    # command classes exist only via TTS synthesis, see the full-model table below.
    headline_labels = ["Licht", "Kühlschrank", "Heizung", "_unknown_", "_silence_"]
    headline_idx = {config.label_index(label) for label in headline_labels}
    headline_mask = np.isin(y_test, list(headline_idx)) & ~is_tts_test
    n_headline = int(headline_mask.sum())
    headline_acc_float = float((y_pred_float[headline_mask] == y_test[headline_mask]).mean())
    headline_acc_int8 = float((y_pred_int8[headline_mask] == y_test[headline_mask]).mean())

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
        "Evaluated on the MSWC-validated subset only — **Licht, Kühlschrank, Heizung** + "
        "`_unknown_` + `_silence_` — restricted to test rows built from REAL MSWC speech "
        "(TTS-synthesized rows excluded, including Heizung's TTS top-up). "
        "**Camping and Wasser are excluded from this number** — see the full-model table "
        f"below. n={n_headline} held-out real-speech examples (mixed 20/10/0 dB SNR).\n\n"
    )
    report += "| Model | Accuracy |\n|---|---|\n"
    report += f"| Float (keras) | {headline_acc_float:.3f} |\n"
    report += f"| **INT8 (shipped)** | **{headline_acc_int8:.3f}** |\n"

    report += "\n## Full-model snapshot (`kws_de.eval.render_report`)\n\n"
    report += render_report({"accuracy": m_int8["accuracy"], "snr_sweep": sweep_int8})
    report += (
        "(All 7 classes, INT8, command-only SNR sweep — see the full breakdown below "
        "for why this overall number mixes real and synthetic speech.)\n"
    )

    report += (
        "\n## Full 5-word model — overall + per-command accuracy "
        "(held-out test set, mixed SNRs)\n\n"
        "**Camping and Wasser are TTS-augmented (synthetic speech)** — Camping had only "
        "22 real MSWC clips, Wasser had 0, so their rows below reflect synthetic-voice "
        "performance and must NOT be read as real-speech accuracy. Heizung is a "
        "real+TTS mix (120 real + 180 TTS, topped up to 300).\n\n"
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
        "varied punctuation, used only to top up Camping/Heizung/Wasser to 300 clips "
        "since MSWC German had far fewer (or zero) real recordings of those words.\n"
    )

    report += (
        "\n## Comparison to MultiNet\n\n"
        "MultiNet's English command-recognition accuracy is reported at roughly "
        "85-95% on clean speech. The comparable number here is the **headline "
        f"real-speech INT8 accuracy: {headline_acc_int8:.3f} ({headline_acc_int8 * 100:.1f}%)** "
        "on Licht/Kühlschrank/Heizung + _unknown_/_silence_, real MSWC speech only, mixed "
        "20/10/0 dB SNR (harder than MultiNet's clean-speech condition). The full 5-word "
        f"model's INT8 accuracy is {m_int8['accuracy']:.3f}, but that number is inflated/"
        "deflated by two TTS-only or TTS-heavy classes (Camping, Wasser) and should not be "
        "quoted as a real-speech comparison to MultiNet.\n"
    )

    out_path = config.DATA_DIR.parent / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"wrote {out_path}")
