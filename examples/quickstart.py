"""Standalone known-truth complex FRF reconstruction sanity check."""
import numpy as np
from sparseldv_qa import graph_laplacian, reconstruct, evaluate_reconstruction

points=np.array([[0,0],[1,0],[0,1],[1,1],[.5,.5]],dtype=float)
truth=np.tile(np.array([1.+2.j, 1.5+1.2j, 2.+1.j]),(len(points),1))
selected=np.array([0,2,4])
L=graph_laplacian(points, 3)
pred=reconstruct(truth[selected],selected,L,alpha=.1)
metrics=evaluate_reconstruction(truth,pred,selected)
assert np.isfinite(metrics["nrmse"])
print(f"SparseLDV-QA synthetic reconstruction OK: NRMSE={metrics['nrmse']:.6f}")
