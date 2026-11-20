import numpy as np

from ben import temperature as tp


def test_fit_recovers_known_temperature():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 3, size=(20000, 5))
    y = (rng.random(z.shape) < tp.apply_temperature(z, 2.5)).astype(float)
    t, hit = tp.fit_temperature(z, y)
    assert abs(t - 2.5) < 0.1 and not hit


def test_apply_temperature_keeps_ranking_and_identity():
    z = np.array([-2.0, 0.0, 1.0, 4.0])
    assert np.allclose(tp.apply_temperature(z, 1.0), 1 / (1 + np.exp(-z)))
    assert (np.argsort(tp.apply_temperature(z, 3.0)) == np.argsort(z)).all()


def test_ece_matches_attempt1_definition():
    p = np.array([0.05, 0.15, 0.95, 1.0])
    y = np.array([0, 1, 1, 1])
    # bins 0, 1, 9, 9: |0.05| + |0.15-1| + |1.95-2| = 0.05 + 0.85 + 0.05
    assert np.isclose(tp.ece(p, y), 0.95 / 4)
