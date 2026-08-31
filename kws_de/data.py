import argparse

import numpy as np

from kws_de import config
from kws_de.augment import mix_at_snr
from kws_de.features import mfcc


def build_dataset(clips, noises, rng, snrs=(20, 10, 0)):
    X, y = [], []

    def add(sig, label):
        X.append(mfcc(sig))
        y.append(config.label_index(label))

    for cmd in config.COMMANDS:
        for clip in clips.get(cmd, []):
            for snr in snrs:
                noise = noises[int(rng.integers(0, len(noises)))]
                add(mix_at_snr(clip, noise, snr, rng), cmd)
    for clip in clips.get("_unknown_", []):
        add(clip, "_unknown_")
    n_sil = max(1, len(clips.get("_unknown_", [])))
    for _ in range(n_sil):
        noise = noises[int(rng.integers(0, len(noises)))]
        sil = mix_at_snr(np.zeros(config.CLIP_SAMPLES, np.float32), noise, 0.0, rng)
        add(sil, "_silence_")
    return np.asarray(X, np.float32), np.asarray(y, np.int64)


def main() -> None:  # pragma: no cover - thin I/O wrapper (manual/integration)
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    args = ap.parse_args()
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    if args.fetch:
        _fetch_and_cache()


def _fetch_and_cache() -> None:  # pragma: no cover
    # MSWC German per-keyword subset via HF datasets streaming + ESC-50 noise.
    # from datasets import load_dataset
    #   ds = load_dataset("MLCommons/ml_spoken_words", "de_wav", streaming=True, split="train")
    #   keep clips whose "keyword" matches our commands (+ a sample for _unknown_)
    # Cache raw clips and/or extracted features under config.DATA_DIR.
    raise NotImplementedError("wire MSWC + ESC-50 fetch; see spec §4")
