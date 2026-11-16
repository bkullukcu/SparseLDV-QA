"""Deterministic complex-FRF metric contracts on synthetic arrays.

These are algebraic fixtures, not experimental accuracy benchmarks.
"""
import numpy as np
import pytest

from sparseldv_qa import complex_frac, evaluate_reconstruction


def test_complex_frac_distinguishes_orthogonal_frequency_shapes():
    reference = np.array([1 + 2j, 0j])
    orthogonal = np.array([0j, 3 - 1j])
    assert complex_frac(reference, orthogonal) == pytest.approx(0.0)


def test_complex_frac_is_invariant_to_global_complex_gain():
    reference = np.array([1 + 2j, -3 + 0.5j, 2j])
    transformed = (2 - 5j) * reference
    assert complex_frac(reference, transformed) == pytest.approx(1.0)


def test_complex_nrmse_uses_unmeasured_points_and_complex_phase():
    reference = np.array([
        [2 + 0j, 2 + 0j],       # measured: must not affect error
        [1 + 1j, 0 + 0j],
        [0 + 0j, 0 + 2j],
    ])
    predicted = reference.copy()
    predicted[0] = 1000 + 1000j
    predicted[1, 0] = 1 - 1j  # phase error of magnitude 2
    result = evaluate_reconstruction(reference, predicted, [0])
    assert result["unmeasured_points"] == 2
    assert result["nrmse"] == pytest.approx(2 / np.sqrt(6))
    # FRAC ignores a global phase rotation of a single nonzero bin;
    # complex NRMSE deliberately does not.
    assert result["median_frac"] == pytest.approx(1.0)


def test_complex_frac_rejects_nonfinite_complex_input():
    with pytest.raises(ValueError, match="finite"):
        complex_frac(np.array([1 + np.nan * 1j]), np.array([1 + 0j]))
