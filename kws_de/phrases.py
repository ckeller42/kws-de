"""Phrase synthesis for CTC training (Phase 2, Task 3): device [zone] action
-> one continuous waveform + a token-id target sequence, built from word
clips the caller supplies (the plan requires TRAIN-split speakers only --
enforced by the caller passing a train-split `clips_by_word`, e.g. from
`kws_de.data.split_by_speaker`, never test)."""

import numpy as np

from kws_de import config
from kws_de.ctc import token_id
from kws_de.features import mfcc_sequence


def make_phrase(tokens: list[str], word_clips: dict, rng, gap_ms: int = 250) -> np.ndarray:
    """Concatenate one random clip per token (drawn from `word_clips[token]`)
    with `gap_ms` of silence between words -> a single waveform."""
    gap = np.zeros(int(config.SAMPLE_RATE * gap_ms / 1000), np.float32)
    parts = []
    for i, tok in enumerate(tokens):
        clips = word_clips[tok]
        clip = clips[int(rng.integers(0, len(clips)))]
        parts.append(np.asarray(clip, np.float32).ravel())
        if i < len(tokens) - 1:
            parts.append(gap)
    return np.concatenate(parts)


def phrase_features(waveform: np.ndarray) -> np.ndarray:
    """Variable-length MFCC frame sequence `(T, N_MFCC)` over the whole phrase."""
    return mfcc_sequence(waveform)


def build_phrase_batch(catalog: list, clips_by_word: dict, rng) -> list[tuple[np.ndarray, list]]:
    """One (feat_seq, target_ids) pair per catalog `Intent` (device [zone]
    action). Entries whose words have no clips in `clips_by_word` are skipped
    (e.g. a word absent from a small train split)."""
    batch = []
    for intent in catalog:
        tokens = [intent.device, *([intent.zone] if intent.zone else []), intent.action]
        if any(not clips_by_word.get(t) for t in tokens):
            continue
        waveform = make_phrase(tokens, clips_by_word, rng)
        feat_seq = phrase_features(waveform)
        target_ids = [token_id(t) for t in tokens]
        batch.append((feat_seq, target_ids))
    return batch
