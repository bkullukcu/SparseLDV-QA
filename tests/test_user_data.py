import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sparseldv_qa.user_data import convert, InputContractError

FIX = Path(__file__).resolve().parents[1] / 'examples' / 'step3_user_data'


def test_csv_contract_and_original_cli_compatibility(tmp_path):
    out = tmp_path / 'converted.npz'
    rep = convert(FIX/'frf_long.csv', FIX/'schema.json', out)
    assert rep['n_training_scans'] == 2 and rep['n_points'] == 5
    assert rep['group_holdout_checked'] and rep['quality_provided']
    assert out.with_suffix('.npz.manifest.json').exists()
    with np.load(out, allow_pickle=False) as z:
        assert z['training'].shape == (2,5,4)
        assert z['truth'].shape == (5,4)
        assert np.isclose(z['coordinates'][-1,0], 0.1)
        assert np.allclose(z['quality'], [.8,.82,.84,.86,.88])
        assert z['truth_sample_id'].item() == 'holdout_C'


def test_npz_contract(tmp_path):
    a = np.ones((2,4,3),complex) + 1j * np.ones((2,4,3))
    source=tmp_path/'original.npz'
    np.savez(source, training=a,truth=a[0],coordinates=np.array([[0,0],[10,0],[20,0],[30,0]]),
             frequency_hz=[40,80,150],point_ids=['a','b','c','d'],training_sample_ids=['A','B'],truth_sample_id='C')
    cfg={'coordinate_unit':'mm','frequency_unit':'Hz','response_unit':'m/s/N'}
    schema=tmp_path/'schema.json';schema.write_text(json.dumps(cfg))
    rep=convert(source,schema,tmp_path/'valid.npz')
    assert rep['n_points']==4
    with np.load(tmp_path/'valid.npz') as z:
        assert np.isclose(z['coordinates'][3,0],.03)


def test_missing_frequency_is_fatal(tmp_path):
    d = pd.read_csv(FIX/'frf_long.csv').iloc[1:]
    bad=tmp_path/'bad.csv'; d.to_csv(bad,index=False)
    with pytest.raises(InputContractError,match='missing or extra'):
        convert(bad,FIX/'schema.json',tmp_path/'out.npz')


def test_duplicate_point_frequency_is_fatal(tmp_path):
    d=pd.read_csv(FIX/'frf_long.csv');d=pd.concat([d,d.iloc[[0]]],ignore_index=True)
    bad=tmp_path/'bad.csv';d.to_csv(bad,index=False)
    with pytest.raises(InputContractError,match='duplicate'):
        convert(bad,FIX/'schema.json',tmp_path/'out.npz')


def test_train_holdout_leakage_is_fatal(tmp_path):
    cfg=json.loads((FIX/'schema.json').read_text());cfg['sample_groups']['holdout_C']='session_A'
    mapping=tmp_path/'config.json';mapping.write_text(json.dumps(cfg))
    with pytest.raises(InputContractError,match='overlap'):
        convert(FIX/'frf_long.csv',mapping,tmp_path/'out.npz')


def test_contradictory_coordinates_are_fatal(tmp_path):
    d=pd.read_csv(FIX/'frf_long.csv');d.loc[0,'x_mm']+=3
    bad=tmp_path/'bad.csv';d.to_csv(bad,index=False)
    with pytest.raises(InputContractError,match='contradictory'):
        convert(bad,FIX/'schema.json',tmp_path/'out.npz')


def test_coherence_requires_declared_meaning(tmp_path):
    cfg=json.loads((FIX/'schema.json').read_text());cfg.pop('quality_kind')
    mapping=tmp_path/'config.json';mapping.write_text(json.dumps(cfg))
    with pytest.raises(InputContractError,match='quality_kind'):
        convert(FIX/'frf_long.csv',mapping,tmp_path/'out.npz')


def test_polar_csv_requires_declared_phase_and_preserves_complex(tmp_path):
    d=pd.read_csv(FIX/'frf_long.csv')
    orig=d.real_part.to_numpy()+1j*d.imag_part.to_numpy()
    d['amp_linear']=np.abs(orig);d['phase_deg']=np.rad2deg(np.angle(orig))
    d=d.drop(columns=['real_part','imag_part']);raw=tmp_path/'polar.csv';d.to_csv(raw,index=False)
    cfg=json.loads((FIX/'schema.json').read_text())
    cfg['complex_format']='magnitude_phase';cfg['phase_unit']='degree';cfg['phase_sign']=1
    cfg['phase_reference']='response/force complex transfer convention'
    del cfg['columns']['real'];del cfg['columns']['imag']
    cfg['columns']['magnitude']='amp_linear';cfg['columns']['phase']='phase_deg'
    schema=tmp_path/'polar.json';schema.write_text(json.dumps(cfg))
    r=convert(raw,schema,tmp_path/'converted.npz')
    assert r['complex_response']=='magnitude_phase_to_rectangular'
    with np.load(tmp_path/'converted.npz') as z:
        assert np.any(abs(z['training'].imag)>0)
    cfg.pop('phase_reference');schema.write_text(json.dumps(cfg))
    with pytest.raises(InputContractError,match='phase_reference'):
        convert(raw,schema,tmp_path/'bad.npz')


def test_npz_real_only_must_be_opted_in(tmp_path):
    source=tmp_path/'real_only.npz';np.savez(source,training=np.ones((2,4,3)),truth=np.ones((4,3)),
        coordinates=[[0,0],[1,0],[2,0],[3,0]],frequency_hz=[1,2,3],
        point_ids=['A','B','C','D'],training_sample_ids=['train1','train2'],truth_sample_id='test')
    cfg={'coordinate_unit':'m','frequency_unit':'Hz','response_unit':'m/N'}
    sch=tmp_path/'sch.json';sch.write_text(json.dumps(cfg))
    with pytest.raises(InputContractError,match='complex arrays'):
        convert(source,sch,tmp_path/'out.npz')
    cfg['allow_real_only']=True;sch.write_text(json.dumps(cfg))
    assert convert(source,sch,tmp_path/'out.npz')['n_training_scans']==2
