import importlib.util
import pathlib

_SPEC = importlib.util.spec_from_file_location(
    "recipe_grid", pathlib.Path(__file__).parent.parent / "scripts" / "recipe-grid.py"
)
grid = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(grid)


def _row(ok, n=51, int8=0.83):
    return {"aggregate_words": ok / n, "aggregate_n": n, "int8_test_acc": int8}


def test_beats_deployed_needs_two_seeds_within_one_clip_and_held_out_parity():
    deployed = _row(51, int8=0.819)
    assert grid.beats_deployed([_row(49), _row(51)], deployed)  # mean 50 >= 51 - 1, int8 up
    assert not grid.beats_deployed([_row(48), _row(49)], deployed)  # mean 48.5 < 50
    assert not grid.beats_deployed([_row(51), _row(51, int8=0.80)], deployed)  # int8 below
    assert not grid.beats_deployed([_row(51)], deployed)  # one run is one draw
