"""Deterministic contracts for complex-FRF sensor selection.

All signals below are synthetic; held-out arrays are never passed to selection.
"""
import numpy as np

from sparseldv_qa import select_points


def _training():
    rng = np.random.default_rng(20261123)
    real = rng.normal(size=(3, 7, 9))
    imaginary = rng.normal(size=(3, 7, 9))
    return real + 1j * imaginary


def test_global_complex_phase_and_gain_do_not_change_selection():
    training = _training()
    baseline = select_points(training, count=4)
    transformed = training * (2.75 * np.exp(1j * 0.73))
    np.testing.assert_array_equal(select_points(transformed, count=4), baseline)


def test_held_out_response_changes_cannot_affect_training_only_selection():
    training = _training()
    holdout = np.ones((2, 7, 9), dtype=complex)
    baseline = select_points(training, count=4)
    holdout[:] = 1e9 * (1 + 1j)
    np.testing.assert_array_equal(select_points(training, count=4), baseline)


def test_zero_quality_points_are_never_selected():
    training = _training()
    quality = np.array([1.0, 0.0, 0.8, 0.0, 0.7, 0.6, 0.9])
    chosen = select_points(training, count=4, quality=quality)
    assert len(chosen) == len(set(chosen.tolist())) == 4
    assert set(chosen).isdisjoint({1, 3})
