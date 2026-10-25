import unittest
import numpy as np
from sparseldv_qa import *

class TestNumerics(unittest.TestCase):
 def setUp(self):
  self.rng=np.random.default_rng(4);self.h=self.rng.normal(size=(4,9,20))+1j*self.rng.normal(size=(4,9,20));self.xy=np.array([(i,j) for i in range(3) for j in range(3)])
 def test_frac_phase_scale_invariance(self):np.testing.assert_allclose(complex_frac(self.h,3j*self.h),1,atol=1e-12)
 def test_frac_zero(self):self.assertTrue(np.isnan(complex_frac(np.zeros(4),np.ones(4))))
 def test_quality_exclusion(self):
  q=np.ones(9);q[0]=0;s=select_points(self.h,8,q);self.assertNotIn(0,s);self.assertEqual(len(set(s)),8)
 def test_selection_deterministic(self):np.testing.assert_array_equal(select_points(self.h,4),select_points(self.h,4))
 def test_constant_field(self):
  l=graph_laplacian(self.xy,3);truth=np.ones((9,10))*(2+3j);s=np.array([0,4,8]);pred=reconstruct(truth[s],s,l)
  np.testing.assert_allclose(pred,truth,atol=1e-7)
 def test_groups_disjoint(self):
  g=np.repeat(np.arange(4),7)
  for tr,te in group_splits(g):self.assertFalse(set(g[tr])&set(g[te]))
 def test_evaluation_excludes_observations(self):
  a=np.ones((9,2),complex);b=a.copy();b[0]=100
  self.assertEqual(evaluate_reconstruction(a,b,np.array([0]))['nrmse'],0)
 def test_bad_quality(self):
  with self.assertRaises(ValueError):quality_weights([1,1.2])
 def test_duplicate_coordinates(self):
  with self.assertRaises(ValueError):graph_laplacian(np.ones((3,2)),2)
 def test_bad_observation_indices(self):
  with self.assertRaises(ValueError):reconstruct(np.ones((2,3)),np.array([0,0]),graph_laplacian(self.xy,3))
 def test_full_observation_score_rejected(self):
  with self.assertRaises(ValueError):evaluate_reconstruction(self.h[0],self.h[0],np.arange(9))
if __name__=='__main__':unittest.main()
