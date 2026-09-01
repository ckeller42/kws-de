"""CTC training + catalog eval (Phase 2, Task 5) -- the numbers.

`ctc_train` fits `kws_de.architectures.ctc_encoder.build_ctc_encoder` on
variable-length phrase batches (`kws_de.phrases.build_phrase_batch`) with
`tf.nn.ctc_loss` (blank index 0). `evaluate_catalog` scores greedy-decoded
full intents against the command catalog (`kws_de.eval.build_catalog`).
`main` is the `kws-transducer` CLI entry point: train on the train-split
phrase data, INT8-export (honest failure if the streaming body can't),
evaluate on held-out test-split audio, and write docs/transducer-report.md.
"""

import numpy as np
import tensorflow as tf

from kws_de.architectures.ctc_encoder import build_ctc_encoder
from kws_de.ctc import CTC_TOKENS, logits_to_intent
from kws_de.grammar import Intent
from kws_de.phrases import make_phrase, phrase_features


def _pad_batch(batches: list[tuple[np.ndarray, list]]) -> tuple:
    """[(feat_seq (T, F), target_ids)] -> zero-padded (X, logit_lengths, Y,
    label_lengths) ready for `tf.nn.ctc_loss`. `logit_lengths`/`label_lengths`
    carry each sample's true (unpadded) length -- this is the length masking:
    `tf.nn.ctc_loss` only scores the first `logit_length[i]` frames / first
    `label_length[i]` labels of each padded row, so the zero padding never
    corrupts the loss."""
    feats = [np.asarray(f, np.float32) for f, _ in batches]
    targets = [np.asarray(t, np.int32) for _, t in batches]
    n = len(batches)
    n_mfcc = feats[0].shape[1]
    t_max = max(f.shape[0] for f in feats)
    l_max = max((len(t) for t in targets), default=0)
    X = np.zeros((n, t_max, n_mfcc, 1), np.float32)
    logit_lengths = np.zeros(n, np.int32)
    Y = np.zeros((n, l_max), np.int32)
    label_lengths = np.zeros(n, np.int32)
    for i, (f, t) in enumerate(zip(feats, targets, strict=True)):
        X[i, : f.shape[0], :, 0] = f
        logit_lengths[i] = f.shape[0]
        Y[i, : len(t)] = t
        label_lengths[i] = len(t)
    return X, logit_lengths, Y, label_lengths


def ctc_loss_for(model: tf.keras.Model, batches: list[tuple[np.ndarray, list]]) -> float:
    """Mean CTC loss of `model` over `batches` -- no gradient step. Shared by
    `ctc_train`'s loop and callers wanting to score a held-out batch."""
    X, logit_lengths, Y, label_lengths = _pad_batch(batches)
    logits = model(tf.constant(X), training=False)
    loss = tf.nn.ctc_loss(
        labels=tf.constant(Y),
        logits=logits,
        label_length=tf.constant(label_lengths),
        logit_length=tf.constant(logit_lengths),
        logits_time_major=False,
        blank_index=0,
    )
    return float(tf.reduce_mean(loss).numpy())


def ctc_train(
    batches: list[tuple[np.ndarray, list]],
    n_tokens: int,
    *,
    encoder: str = "matchboxnet",
    epochs: int = 30,
    seed: int = 0,
    learning_rate: float = 1e-3,
    history: list[float] | None = None,
) -> tf.keras.Model:
    """Build `build_ctc_encoder(n_tokens, encoder=encoder)` and fit it on
    `batches` (variable-length `(feat_seq, target_ids)` pairs, e.g. from
    `kws_de.phrases.build_phrase_batch`) with full-batch gradient descent on
    `tf.nn.ctc_loss` (blank index 0, length-masked -- see `_pad_batch`). If
    `history` is given, each epoch's mean loss is appended to it."""
    tf.keras.utils.set_random_seed(seed)
    model = build_ctc_encoder(n_tokens, encoder=encoder)
    X, logit_lengths, Y, label_lengths = _pad_batch(batches)
    X_t = tf.constant(X)
    Y_t = tf.constant(Y)
    logit_lengths_t = tf.constant(logit_lengths)
    label_lengths_t = tf.constant(label_lengths)
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    for _ in range(epochs):
        with tf.GradientTape() as tape:
            logits = model(X_t, training=True)
            loss = tf.nn.ctc_loss(
                labels=Y_t,
                logits=logits,
                label_length=label_lengths_t,
                logit_length=logit_lengths_t,
                logits_time_major=False,
                blank_index=0,
            )
            mean_loss = tf.reduce_mean(loss)
        grads = tape.gradient(mean_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables, strict=True))
        if history is not None:
            history.append(float(mean_loss.numpy()))
    return model


def make_ctc_predict_fn(model: tf.keras.Model):
    """`model` -> `predict_fn(feat_seq) -> logits (T, n_tokens)` -- the float
    per-frame prediction, one phrase at a time (batch size 1)."""

    def predict_fn(feat_seq: np.ndarray) -> np.ndarray:
        x = np.asarray(feat_seq, np.float32)[None, ..., None]
        return model(tf.constant(x), training=False).numpy()[0]

    return predict_fn


def evaluate_catalog(  # pragma: no cover - orchestration (real word-clip audio)
    predict_fn,
    clips_by_word: dict,
    rng,
    catalog: list | None = None,
    n_trials: int = 3,
) -> dict:
    """For each catalog `Intent` (`kws_de.eval.build_catalog` by default):
    synthesize `n_trials` phrases from `clips_by_word` (Task 3), get
    per-frame logits from `predict_fn`, `logits_to_intent`, score full-intent
    against the true intent. Mirrors `kws_de.eval.run_catalog_eval`'s return
    shape (overall/per-entry/per-slot accuracy) so reports can compare
    directly."""
    from kws_de.eval import build_catalog

    catalog = catalog if catalog is not None else build_catalog()
    per_entry = []
    total = correct = 0
    device_correct = action_correct = 0
    zone_correct = zone_total = 0
    for intent in catalog:
        tokens = [intent.device, *([intent.zone] if intent.zone else []), intent.action]
        if any(not clips_by_word.get(t) for t in tokens):
            continue
        entry_trials = entry_correct = 0
        for _ in range(n_trials):
            waveform = make_phrase(tokens, clips_by_word, rng)
            feat_seq = phrase_features(waveform)
            logits = predict_fn(feat_seq)
            pred = logits_to_intent(logits)
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
    }


def main() -> None:  # pragma: no cover - I/O wrapper (real training + real audio)
    """`kws-transducer`: build phrase batches from the TRAIN-split word clips,
    `ctc_train` the matchboxnet-body CTC encoder, attempt an INT8 export of
    the streaming body (recorded honestly if it can't), `evaluate_catalog` on
    held-out TEST-split audio, and write docs/transducer-report.md."""
    import json
    import pickle

    from kws_de import budgets, config
    from kws_de.data import command_words, split_by_speaker
    from kws_de.eval import build_catalog
    from kws_de.export import to_int8_tflite
    from kws_de.phrases import build_phrase_batch

    # pickle: this repo's own local, gitignored data/ cache (written by
    # kws_de.data), never untrusted input -- same pattern as kws_de.data/
    # kws_de.dataset/kws_de.eval.
    with open(config.DATA_DIR / "raw_clips_merged.pkl", "rb") as fh:
        clips_ws = pickle.load(fh)["clips"]  # noqa: S301

    words = command_words()
    split_rng = np.random.default_rng(0)
    train_clips, test_clips = split_by_speaker(clips_ws, split_rng, test_frac=0.2)
    train_words = {w: train_clips.get(w, []) for w in words}
    test_words = {w: test_clips.get(w, []) for w in words}

    catalog = build_catalog()
    n_tokens = len(CTC_TOKENS)

    # Training data: repeated draws of build_phrase_batch (train-split clips
    # only -- no leakage) give many distinct word-clip combinations per
    # catalog entry instead of just one.
    batch_rng = np.random.default_rng(1)
    n_repeats = 8
    batches = []
    for _ in range(n_repeats):
        batches += build_phrase_batch(catalog, train_words, batch_rng)

    epochs = 60
    history: list[float] = []
    model = ctc_train(
        batches, n_tokens, encoder="matchboxnet", epochs=epochs, seed=0, history=history
    )
    predict_fn = make_ctc_predict_fn(model)

    # Streaming INT8 export: the model trains at variable T, but full-INT8
    # quantisation needs static shapes, so export a fixed-T (+ batch-1) clone --
    # the honest on-device shape (one chunk at a time, fixed chunk + ring
    # buffer). The clone's weights are T/batch-independent, so set_weights
    # transfers the trained weights directly. The 1x1-Conv2D per-frame head
    # keeps the graph inside the TFLM INT8 builtin set (the old TimeDistributed
    # head unrolled to a tf.while loop -> TensorListReserve, which blocked E8).
    int8_report: dict = {"ok": False, "error": None, "bytes": None, "ops": None, "tflm_ok": None}
    # rep samples must share one T for to_int8_tflite's np.asarray stack --
    # phrase feature sequences are variable-length, so trim the first 8 batch
    # entries to their shortest common length; that length is the export window.
    rep_sample = [f for f, _ in batches[:8]]
    t_common = min((f.shape[0] for f in rep_sample), default=0)
    rep = np.stack([f[:t_common] for f in rep_sample], axis=0) if t_common else None
    try:
        if rep is None:
            raise ValueError("no representative phrase batch available for int8 export")
        export_model = build_ctc_encoder(n_tokens, encoder="matchboxnet", t_frames=t_common)
        export_model.set_weights(model.get_weights())
        tflite_bytes = to_int8_tflite(export_model, rep)
        ops = budgets.tflite_op_types(tflite_bytes)
        tflm_ops = {
            "CONV_2D",
            "DEPTHWISE_CONV_2D",
            "FULLY_CONNECTED",
            "MEAN",
            "SOFTMAX",
            "RESHAPE",
            "ADD",
            "DELEGATE",
        }
        int8_report = {
            "ok": True,
            "error": None,
            "bytes": len(tflite_bytes),
            "ops": sorted(ops),
            "tflm_ok": ops <= tflm_ops,
            "int8": budgets.is_full_int8(tflite_bytes),
        }
    except Exception as exc:  # noqa: BLE001 - report the real reason, don't hide it
        int8_report["error"] = f"{type(exc).__name__}: {exc}"

    eval_rng = np.random.default_rng(2)
    results = evaluate_catalog(predict_fn, test_words, eval_rng, catalog=catalog, n_trials=3)

    baseline_full_intent = 0.689
    probe_intent = catalog[0]
    probe_tokens = [
        probe_intent.device,
        *([probe_intent.zone] if probe_intent.zone else []),
        probe_intent.action,
    ]
    probe_waveform = make_phrase(probe_tokens, test_words, np.random.default_rng(3))
    probe_logits = predict_fn(phrase_features(probe_waveform))
    from kws_de.ctc import greedy_decode

    probe_decoded = greedy_decode(probe_logits)

    report_lines = [
        "# kws-de Phase 2 -- Streaming CTC Command Recogniser Report\n",
        "\n## Method\n\n",
        "Encoder: matchboxnet conv body (stride-1 in time, no global pool) + "
        "a 1x1 Conv2D per-frame head emitting `(T, n_tokens)` logits "
        "(`kws_de.architectures.ctc_encoder.build_ctc_encoder`). Trained with "
        "`tf.nn.ctc_loss` (blank index 0, length-masked) on phrase audio "
        "(`device [zone] action`) synthesized from TRAIN-split word clips "
        f"only (no leakage). {len(batches)} training phrases "
        f"({n_repeats}x over the {len(catalog)}-entry catalog), {epochs} epochs.\n",
        "\nGreedy CTC decode -> `kws_de.grammar.parse` -> `Intent`, evaluated on "
        "the SAME command catalog as the frame-classifier baseline, but "
        "synthesized from held-out TEST-split real word-clip audio (not TTS "
        "voices -- the baseline used TTS `say` voices; see vs-baseline note "
        "below for why the two numbers aren't a strict apples-to-apples "
        "comparison of audio source, only of the decode/grammar composition).\n",
        "\n## Training loss\n\n",
        f"- First epoch loss: {history[0]:.4f}\n" if history else "- (no training epochs run)\n",
        f"- Last epoch loss: {history[-1]:.4f}\n" if history else "",
        f"- Loss decreased: {history[-1] < history[0] if len(history) > 1 else 'n/a'}\n",
        "\n## Catalog full-intent accuracy\n\n",
        f"**CTC model: {results['overall_accuracy']:.3f}** "
        f"({results['total_trials']} trials over {len(results['per_entry'])} catalog entries)\n",
        f"\n**Frame-classifier baseline (docs/eval-report-v2.md): {baseline_full_intent:.3f}**\n",
        "\n## Per-slot accuracy (CTC model)\n\n",
        f"- Device: {results['device_accuracy']:.3f}\n",
        f"- Action: {results['action_accuracy']:.3f}\n",
    ]
    if results["zone_accuracy"] is not None:
        report_lines.append(f"- Zone (Licht only): {results['zone_accuracy']:.3f}\n")

    report_lines.append("\n## Streaming INT8 export\n\n")
    if int8_report["ok"]:
        report_lines.append(
            f"- Exported a fixed-T (window = {t_common} frames), batch-1 clone: "
            f"{int8_report['bytes']} bytes, int8 I/O: {int8_report['int8']}\n"
        )
        report_lines.append(f"- Ops: {', '.join(int8_report['ops'])}\n")
        report_lines.append(f"- All ops within the TFLM builtin set: {int8_report['tflm_ok']}\n")
        if int8_report["tflm_ok"]:
            report_lines.append(
                "- Device-runnable: the 1x1-Conv2D per-frame head keeps the graph "
                "inside the TFLM INT8 builtin set. The earlier TimeDistributed(Dense) "
                "head unrolled to a `tf.while` loop -> `TensorListReserve`, which the "
                "INT8-builtins-only converter could not legalize; a fixed-T + batch-1 "
                "export (the honest on-device shape: one chunk at a time, ring buffer) "
                "removes every dynamic-shape op too.\n"
            )
        else:
            report_lines.append(
                "- NOT device-runnable as exported: some ops fall outside the TFLM "
                "set (same situation as `kwt`, see kws_de/architectures/kwt.py) -- "
                "the float number above is the honest result; only the pooled "
                "frame-classifier baseline currently ships on-device.\n"
            )
    else:
        report_lines.append("- **INT8 export failed.**\n")
        report_lines.append(f"- Reason: `{int8_report['error']}`\n")
        report_lines.append(
            "- Reporting the float number honestly instead of forcing a broken "
            "export (same policy as `kwt`, see kws_de/architectures/kwt.py).\n"
        )

    report_lines.append("\n## Probe: one decoded phrase\n\n")
    report_lines.append(f"- Catalog intent: `{probe_intent}`\n")
    report_lines.append(f"- Greedy-decoded token sequence: `{probe_decoded}`\n")
    report_lines.append(f"- `logits_to_intent`: `{logits_to_intent(probe_logits)}`\n")

    report_lines.append("\n## vs frame-classifier baseline (0.689)\n\n")
    delta = results["overall_accuracy"] - baseline_full_intent
    verdict = "better" if delta > 0 else ("worse" if delta < 0 else "tied")
    report_lines.append(
        f"CTC full-intent accuracy is **{verdict}** than the baseline "
        f"({results['overall_accuracy']:.3f} vs {baseline_full_intent:.3f}, "
        f"delta {delta:+.3f}). Audio source differs (real held-out word clips "
        "here vs TTS voices for the baseline), so this is not a fully "
        "controlled comparison -- see docs/paper-notes.md for the baseline's "
        "own caveats (Licht-dominated catalog, brittle to long words/word "
        "boundaries).\n"
    )

    out_path = config.DATA_DIR.parent / "docs" / "transducer-report.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(report_lines))
    print(f"wrote {out_path}")

    bench = {
        "overall_accuracy": results["overall_accuracy"],
        "total_trials": results["total_trials"],
        "device_accuracy": results["device_accuracy"],
        "action_accuracy": results["action_accuracy"],
        "zone_accuracy": results["zone_accuracy"],
        "baseline_full_intent": baseline_full_intent,
        "training_loss_first": history[0] if history else None,
        "training_loss_last": history[-1] if history else None,
        "epochs": epochs,
        "n_training_phrases": len(batches),
        "int8_export": {k: v for k, v in int8_report.items() if k != "ops"},
        "probe_intent": str(probe_intent),
        "probe_decoded": probe_decoded,
    }
    bench_path = config.DATA_DIR.parent / "docs" / "transducer-benchmark.json"
    bench_path.write_text(json.dumps(bench, indent=2, ensure_ascii=False))
    print(f"wrote {bench_path}")
