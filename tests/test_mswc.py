from pathlib import Path

import numpy as np

from kws_de.mswc import _folder_index, _pick


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
