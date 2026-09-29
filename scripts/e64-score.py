"""E64 scoring: the four control/clone runs and the deployed model on the real guided-only
scoreboard (docs/paper-notes.md E64, spec 2026-09-29-voice-clone-tts-design).

Same code path as scripts/compare_command_models.py (`eval_recordings` +
`make_command_predict_fn`), every speaker in `approved/words/` counted (no SPEAKERS
restriction, see E52), and scripts/recipe-grid.py's `passes()` as the deploy rule.

Usage:
  uv run --no-sync python scripts/e64-score.py --runs-dir $KWS_DATA_ROOT/models/e64 --json out.json
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

from kws_de import config
from kws_de.codegen import model_bytes
from kws_de.eval import _tflite_predict, eval_recordings, make_command_predict_fn

REPO_ROOT = Path(__file__).resolve().parent.parent
APPROVED = config.DATA_DIR / "recordings" / "approved"
ARMS = ("control", "clone")
SEEDS = (0, 1)


def _passes():
    spec = importlib.util.spec_from_file_location(
        "recipe_grid", REPO_ROOT / "scripts/recipe-grid.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.passes


def score(blob: bytes, test_npz: Path, manifest: Path) -> dict:
    d = np.load(test_npz)
    int8 = float((_tflite_predict(blob, d["X"]) == d["y"]).mean())
    res = eval_recordings(APPROVED, make_command_predict_fn(blob), manifest_path=manifest)
    spk_rows, per_word = {}, {}
    n = ok = fa_n = fa = ph_n = ph_ok = 0
    for fig in res["figures"].values():
        for spk, row in fig["isolated"].items():
            k = round(row["acc"] * row["n"])
            spk_rows[spk] = {"n": row["n"], "ok": k, "acc": row["acc"]}
            per_word[spk] = row["per_word"]
            n, ok = n + row["n"], ok + k
        for row in fig["false_accepts"].values():
            fa_n += row["n"]
            fa += round(row["rate"] * row["n"])
        for row in fig["e2e"].values():
            ph_n += row["n"]
            ph_ok += round(row["acc"] * row["n"])
    return {
        "sha256": hashlib.sha256(blob).hexdigest()[:8],
        "bytes": len(blob),
        "int8_test_acc": int8,
        "int8_test_n": int(len(d["y"])),
        "speakers": spk_rows,
        "per_word": per_word,
        "aggregate_words": ok / n,
        "aggregate_ok": ok,
        "aggregate_n": n,
        "false_accepts": fa,
        "false_accepts_n": fa_n,
        "phrases_ok": ph_ok,
        "phrases_n": ph_n,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", type=Path, default=config.MODELS_DIR / "e64")
    ap.add_argument(
        "--deployed-header", type=Path, default=REPO_ROOT / "firmware/main/gen/model_data.h"
    )
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    passes = _passes()

    control_test = config.DATA_DIR / "features_v3_control_test.npz"
    control_manifest = config.DATA_DIR / "manifest_v3_control.json"
    out = {"deployed": score(model_bytes(args.deployed_header), control_test, control_manifest)}
    for arm in ARMS:
        for seed in SEEDS:
            name = f"{arm}_s{seed}"
            tflite = args.runs_dir / name / f"command_v3_{arm}_w48_qat.tflite"
            if not tflite.exists():
                print(f"{name}: {tflite} missing, skipped")
                continue
            out[name] = score(
                tflite.read_bytes(),
                config.DATA_DIR / f"features_v3_{arm}_test.npz",
                config.DATA_DIR / f"manifest_v3_{arm}.json",
            )
    ref = args.runs_dir / "ref_e62_default_cache" / "command_v3_w48_qat.tflite"
    if ref.exists():  # E62's model: same recipe, default-cache (`say`-only TTS) build
        out["ref_e62"] = score(
            ref.read_bytes(),
            config.DATA_DIR / "features_v3_test.npz",
            config.DATA_DIR / "manifest_v3.json",
        )
    for name, s in out.items():
        s["passes"] = passes(s)
        spk = " ".join(f"{k}={v['ok']}/{v['n']}" for k, v in sorted(s["speakers"].items()))
        print(
            f"{name:11s} {s['sha256']} int8={s['int8_test_acc']:.4f} {spk} "
            f"aggregate={s['aggregate_ok']}/{s['aggregate_n']}={s['aggregate_words']:.4f} "
            f"FA={s['false_accepts']}/{s['false_accepts_n']} "
            f"phrases={s['phrases_ok']}/{s['phrases_n']} {'PASS' if s['passes'] else 'FAIL'}"
        )
    if args.json:
        args.json.write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
