import numpy as np
import pytest
from sparseldv_qa.baselines import qr_pod_points,farthest_geometry_points,_matrix

def test_pod_qr_repeatable_and_unseen():
    rng=np.random.default_rng(23)
    h=rng.normal(size=(4,15,20))+1j*rng.normal(size=(4,15,20))
    a=qr_pod_points(h,5)
    assert len(a)==5 and len(set(a))==5
    assert np.array_equal(a,qr_pod_points(h,5))

def test_geometry_unique_and_deterministic():
    pts=np.array([(i,j) for i in range(6) for j in range(3)])
    assert len(set(farthest_geometry_points(pts,6)))==6
    assert np.array_equal(farthest_geometry_points(pts,4),farthest_geometry_points(pts,4))

def test_validation():
    with pytest.raises(ValueError):qr_pod_points(np.ones((3,4)),2)
    with pytest.raises(ValueError):farthest_geometry_points(np.zeros((3,2)),2)
