import argparse

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
        config.LABELS[i]: (float(cm[i, i] / cm[i].sum()) if cm[i].sum() else 0.0)
        for i in range(n)
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


def main() -> None:  # pragma: no cover - I/O wrapper
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/eval-report.md")
    args = ap.parse_args()
    # load model + held-out features, compute metrics + SNR sweep, then:
    # open(args.out, "w").write(render_report(results))
    raise NotImplementedError("wire held-out eval; see spec §9")
