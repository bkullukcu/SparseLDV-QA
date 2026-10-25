import unittest
import numpy as np
from sparseldv_qa.alternative_reconstruction import idw_operator,rbf_operator,graph_operator,pod_operator,reconstruction_operators
from sparseldv_qa.core import reconstruct,graph_laplacian
class TestAlternatives(unittest.TestCase):
 def setUp(self):
  self.xy=np.stack(np.meshgrid(np.linspace(0,1,4),np.linspace(0,1,4)),axis=-1).reshape(-1,2)
  self.s=np.array([0,3,5,9,12,15])
  f=np.linspace(0,1,24)
  self.h=np.array([np.exp(1j*(f[None,:]*10+i*.1))*np.cos(self.xy[:,0,None]*5+j*.05) for i,j in [(0,0),(1,2),(2,3)]])
 def test_shapes_and_finite(self):
  for key,op in reconstruction_operators(self.xy,self.s,self.h).items():
   self.assertEqual(op.shape,(16,6),key)
   self.assertTrue(np.isfinite(op).all(),key)
   self.assertTrue(np.isfinite(op@self.h[0,self.s]).all(),key)
 def test_exact_IDW_on_observed(self):
  W=idw_operator(self.xy,self.s)
  np.testing.assert_allclose(W[self.s],np.eye(len(self.s)))
 def test_graph_matches_original(self):
  W=graph_operator(self.xy,self.s)
  original=reconstruct(self.h[0,self.s],self.s,graph_laplacian(self.xy))
  np.testing.assert_allclose(W@self.h[0,self.s],original,atol=1e-9,rtol=1e-9)
 def test_invalid(self):
  with self.assertRaises(ValueError):idw_operator(self.xy,[0,0])
  with self.assertRaises(ValueError):rbf_operator(self.xy,np.array([1,2]),regularization=0)
if __name__=='__main__':unittest.main()
