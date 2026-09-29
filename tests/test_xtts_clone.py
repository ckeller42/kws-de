import importlib.util
import sys
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "xtts_clone", Path(__file__).resolve().parents[1] / "scripts" / "xtts_clone.py"
)
xc = importlib.util.module_from_spec(spec)
sys.modules["xtts_clone"] = xc
spec.loader.exec_module(xc)


def test_norm_ignores_case_punctuation_and_whisper_spacing():
    assert xc.norm("Lese Licht aus.") == xc.norm("Leselicht aus")
    assert xc.norm("Küchenlicht an Heizung.") != xc.norm("Küchenlicht an")


def test_first_utterance_cuts_at_first_pause_and_keeps_pauseless_clip():
    sr = 16000
    tone = 0.3 * np.sin(2 * np.pi * 220 * np.arange(int(0.5 * sr)) / sr).astype(np.float32)
    gap = np.zeros(int(0.5 * sr), np.float32)
    y = np.concatenate([gap[: sr // 10], tone, gap, tone, tone])  # word, pause, babble
    cut = xc.first_utterance(y, sr)
    assert 0.5 * sr < len(cut) < 1.0 * sr  # the word plus padding, not the babble
    assert len(xc.first_utterance(np.concatenate([tone, tone]), sr)) == 2 * len(tone)
