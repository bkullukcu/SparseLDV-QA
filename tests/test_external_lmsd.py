import numpy as np
import pytest
from sparseldv_qa.external_lmsd import load_lmsd_baseline,inspect_lmsd_damage

def test_lmsd_valid_schema(tmp_path):
    t=np.linspace(0,100,20);x=np.array([(i,j) for i in range(12) for j in range(12)],float)*50
    h=(np.ones((7,144,20))+1j*np.ones((7,144,20)))*np.linspace(1,2,20)
    p=tmp_path/'baseline.npz';np.savez(p,frf=h,frequency=t,node_position=x,probing_nodes=np.arange(7))
    data=load_lmsd_baseline(p,fmin=10,fmax=95,max_bins=8)
    assert data['frf'].shape==(7,144,8)
    assert np.isclose(data['coordinates_m'][13,0],.05)
    assert data['reference_split_unit']=='excitation_reference'

def test_lmsd_rejects_magnitude_only(tmp_path):
    p=tmp_path/'baseline.npz';np.savez(p,frf=np.ones((7,144,10)),frequency=np.arange(10),node_position=np.ones((144,2)),probing_nodes=np.arange(7))
    with pytest.raises(ValueError,match='Complex'):load_lmsd_baseline(p,fmin=0,fmax=9)

def test_lmsd_damage_not_ground_truth(tmp_path):
    p=tmp_path/'damage.npz';np.savez(p,frf=np.ones((7,7,20)),mass_nodes=np.array([4,12]))
    assert inspect_lmsd_damage(p)['eligible_fullfield_benchmark'] is False
