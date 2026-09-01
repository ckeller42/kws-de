import csv
from pathlib import Path

import numpy as np
import soundfile as sf

from kws_de import config
from kws_de.mswc import _folder_index, _pick, mine


def test_folder_index_is_case_insensitive(tmp_path: Path):
    (tmp_path / "clips" / "küche").mkdir(parents=True)
    (tmp_path / "clips" / "Licht").mkdir()
    (tmp_path / "clips" / "not_a_dir.txt").write_text("x")
    idx = _folder_index(tmp_path)
    assert idx["küche"] == tmp_path / "clips" / "küche"
    assert idx["licht"] == tmp_path / "clips" / "Licht"
    assert "not_a_dir.txt" not in idx


def test_pick_is_deterministic_bounded_and_non_mutating():
    items = list(range(10))
    a = _pick(items, 4, np.random.default_rng(3))
    b = _pick(items, 4, np.random.default_rng(3))
    c = _pick(items, 4, np.random.default_rng(4))
    assert a == b and len(a) == 4
    assert a != c
    assert items == list(range(10))
    assert _pick(items, 50, np.random.default_rng(0)) != items  # shuffled, all 10
    assert sorted(_pick(items, 50, np.random.default_rng(0))) == items


def _write_tree(root: Path, words: dict[str, list[tuple[str, str, bool]]]):
    """words: folder -> [(filename, speaker, valid)]. Writes 0.5 s tone WAVs
    (MSWC ships .opus; `_decode` goes through soundfile either way) and the csv."""
    rows = []
    for folder, files in words.items():
        d = root / "clips" / folder
        d.mkdir(parents=True)
        for fname, spk, valid in files:
            t = np.arange(8000) / config.SAMPLE_RATE
            sf.write(
                d / fname,
                (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32),
                16000,
            )
            rows.append(
                (
                    "TRAIN",
                    f"{folder}/{fname}",
                    folder,
                    "TRUE" if valid else "FALSE",
                    spk,
                    "",
                )
            )
    with open(root / "de_splits.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["SET", "LINK", "WORD", "VALID", "SPEAKER", "GENDER"])
        w.writerows(rows)


def test_mine_counts_speakers_validity_and_unknown_cap(tmp_path: Path):
    _write_tree(
        tmp_path,
        {
            "licht": [("a.wav", "s1", True), ("b.wav", "s2", True), ("c.wav", "s3", False)],
            "küche": [("a.wav", "s4", True)],
            "haus": [("a.wav", "s5", True), ("b.wav", "s6", True), ("c.wav", "s7", True)],
            "baum": [("a.wav", "s8", True)],
        },
    )
    clips = mine(
        tmp_path,
        ["Licht", "Küche", "Aufstelldach"],
        n_per_word=5,
        n_unknown=10,
        unknown_per_word_cap=2,
        seed=0,
    )
    assert len(clips["Licht"]) == 2  # invalid row excluded
    assert {spk for _, spk in clips["Licht"]} == {"s1", "s2"}
    assert len(clips["Küche"]) == 1 and clips["Küche"][0][1] == "s4"
    assert clips["Aufstelldach"] == []  # folder absent -> empty, not KeyError
    # _unknown_: haus capped at 2, baum 1; target-word folders never leak in
    assert len(clips["_unknown_"]) == 3
    unk = {spk for _, spk in clips["_unknown_"]}
    assert "s8" in unk and len(unk & {"s5", "s6", "s7"}) == 2
    for clip, _ in clips["Licht"] + clips["_unknown_"]:
        assert clip.dtype == np.float32 and clip.ndim == 1 and clip.shape[0] == 8000


def test_mine_is_deterministic_in_seed(tmp_path: Path):
    _write_tree(tmp_path, {"licht": [(f"{i}.wav", f"s{i}", True) for i in range(6)]})
    a = mine(tmp_path, ["Licht"], n_per_word=3, n_unknown=0, seed=1)
    b = mine(tmp_path, ["Licht"], n_per_word=3, n_unknown=0, seed=1)
    c = mine(tmp_path, ["Licht"], n_per_word=3, n_unknown=0, seed=2)
    assert [s for _, s in a["Licht"]] == [s for _, s in b["Licht"]]
    assert [s for _, s in a["Licht"]] != [s for _, s in c["Licht"]]
