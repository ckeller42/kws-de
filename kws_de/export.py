import argparse
import json
import pathlib

import numpy as np
import tensorflow as tf

from kws_de import config


def to_int8_tflite(model, rep_samples) -> bytes:
    rep = np.asarray(rep_samples, np.float32)[..., None]

    def rep_gen():
        for i in range(rep.shape[0]):
            yield [rep[i : i + 1]]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep_gen
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def write_c_array(tflite: bytes, path) -> None:
    body = ", ".join(str(b) for b in tflite)
    with open(path, "w") as fh:
        fh.write("// Auto-generated. Do not edit.\n")
        fh.write(f"const unsigned char g_model[] = {{{body}}};\n")
        fh.write(f"const unsigned int g_model_len = {len(tflite)};\n")


def write_metadata(path) -> None:
    meta = {
        "labels": config.LABELS,
        "mfcc": {
            "n_mfcc": config.N_MFCC,
            "n_frames": config.N_FRAMES,
            "win": config.WIN_SAMPLES,
            "hop": config.HOP_SAMPLES,
            "n_mels": config.N_MELS,
            "sample_rate": config.SAMPLE_RATE,
        },
        "budgets": {
            "model_bytes": config.MAX_MODEL_BYTES,
            "arena_bytes": config.MAX_ARENA_BYTES,
            "macs": config.MAX_MACS,
        },
    }
    with open(path, "w") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)


def main() -> None:  # pragma: no cover - I/O wrapper
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(config.MODELS_DIR))
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = tf.keras.models.load_model(config.MODELS_DIR / "kws.keras")
    feats = np.load(config.DATA_DIR / "features_train.npz")["X"][:200]
    blob = to_int8_tflite(model, feats)
    (out / "model.tflite").write_bytes(blob)
    write_c_array(blob, out / "model_data.h")
    write_metadata(out / "metadata.json")
