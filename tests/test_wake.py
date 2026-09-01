from kws_de.wake import WakeDetector


def test_fires_once_on_sustained_wake():
    d = WakeDetector(cutoff=0.8, refractory=5)
    fired = [d.push(0.95) for _ in range(6)]
    assert fired.count(True) == 1  # one wake despite sustained high prob


def test_no_fire_below_cutoff():
    d = WakeDetector(cutoff=0.8, refractory=5)
    assert not any(d.push(0.5) for _ in range(6))


def test_refractory_then_new_wake():
    d = WakeDetector(cutoff=0.8, refractory=3)
    seq = [0.95, 0.1, 0.1, 0.1, 0.95]  # wake, gap past refractory, wake
    assert [d.push(p) for p in seq].count(True) == 2
