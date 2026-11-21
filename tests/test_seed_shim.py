"""`kws_de.model.set_random_seed` shims tf_keras's seed generator for Python 3.12."""

import random
import sys
import warnings

import pytest

tf_keras = pytest.importorskip("tf_keras")

from kws_de.model import _IntRandom, set_random_seed  # noqa: E402


def test_shim_matches_original_sequence():
    ref = random.Random(7)
    shim = _IntRandom(7)
    assert [shim.randint(1, 1e9) for _ in range(5)] == [ref.randint(1, 10**9) for _ in range(5)]


@pytest.mark.skipif(sys.version_info >= (3, 12), reason="3.12 rejects the float call")
def test_shim_matches_float_call_on_311():
    ref = random.Random(7)
    shim = _IntRandom(7)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        expected = [ref.randint(1, 1e9) for _ in range(5)]
    assert [shim.randint(1, 1e9) for _ in range(5)] == expected


def test_set_random_seed_installs_shim():
    from tf_keras.src import backend

    set_random_seed(7)
    gen = backend._SEED_GENERATOR.generator
    assert isinstance(gen, _IntRandom)
    assert gen.randint(1, 1e9) == random.Random(7).randint(1, 10**9)
    # The call that fails on 3.12 without the shim.
    assert backend.RandomGenerator(seed=None)._create_seed(None) > 0
