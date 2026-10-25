import numpy as np
import pandas as pd
import pytest
from sparseldv_qa.budget import budget_attainment
from sparseldv_qa.alternative_reconstruction import rbf_operator

def test_rbf_uses_nearest_not_second_neighbor():
    x=np.array([[0.,0],[1,0],[4,0]])
    s=np.array([0,2]);ell=2.
    d=abs(x[:,0,None]-x[:,0]);k=np.exp(-.5*(d/ell)**2)
    expected=np.linalg.solve(k[np.ix_(s,s)]+1e-4*np.eye(2),k[:,s].T).T
    np.testing.assert_allclose(rbf_operator(x,s),expected)

def test_unattained_and_nonmonotone_budgets():
    d=pd.DataFrame({'group':['a']*3+['b']*3,'budget':[2,4,6]*2,'selector':['g']*6,'reconstruction':['p']*6,'nrmse':[.8,.3,.6,.8,.9,.7]})
    out=budget_attainment(d,.5)
    assert out.iloc[0].minimum_tested_budget==4
    assert not out.iloc[1].attained and np.isnan(out.iloc[1].minimum_tested_budget)
    with pytest.raises(ValueError):budget_attainment(pd.concat([d,d]),.5)
