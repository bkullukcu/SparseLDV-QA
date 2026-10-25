"""End-to-end integration tests with mocked Polytec ZIP (NOT experimental data)."""
from pathlib import Path
from zipfile import ZipFile
import csv
import numpy as np
import pytest
from sparseldv_qa.zenodo_import import read_polytec, scan_archive, convert_archive, load_coordinates


def polytec_txt(values, *, phase=False):
    header = ['Polytec PSV', 'Weg / Kraft' if not phase else 'Phase', 'Unit m/N',
              'Frequency in Hz', 'Export', 'Sample', '---']
    body = [f'{j+1}\t{v:.12g}' for j,v in enumerate(values)]
    return ('\n'.join(header+body)+'\n').encode('cp1252')


def fixture(tmp_path, *, states=4,points=8,bins=150):
    ar=tmp_path/'00_raw_exports.zip'
    rng=np.random.default_rng(11)
    # Consistent physically meaningful low-rank, complex amplitudes (mock only).
    xyz=np.array([(i%4,i//4) for i in range(points)],float)
    csvfile=tmp_path/'scan_points.csv'
    with csvfile.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['Point Number','X (mm)','Y (mm)'])
        for i,xy in enumerate(xyz,1):w.writerow([i,*xy])
    with ZipFile(ar,'w') as z:
        for state in range(states):
            for p in range(points):
                freqs=np.arange(1,bins+1,dtype=float)
                h=(0.25+0.1*p+0.005*state)*np.exp(1j*(0.017*freqs+0.03*p+0.1*state))
                z.writestr(f'Amplitude/State_{state:02d}/Point_{p+1:02d}.txt',polytec_txt(abs(h)))
                z.writestr(f'Phase/State_{state:02d}/Point_{p+1:02d}.txt',polytec_txt(np.rad2deg(np.angle(h)),phase=True))
        # add extra 9th point only in final state; should not change intersection
        z.writestr(f'Amplitude/State_{states-1:02d}/Point_09.txt',polytec_txt(np.ones(bins)))
        z.writestr(f'Phase/State_{states-1:02d}/Point_09.txt',polytec_txt(np.zeros(bins),phase=True))
    return ar,csvfile


def test_parse_and_convert(mocked_plate):
    ar,coord,tmp_path=mocked_plate
    manifest=tmp_path/'manifest.csv'
    rows=scan_archive(ar,manifest)
    assert len(rows)==4*8+1
    out=tmp_path/'complex.npz'
    details=convert_archive(ar,manifest,coord,out,freq_min=1,freq_max=150,stride=5,expected_states=4,expected_points=8)
    assert details['states']==4 and details['points']==8 and details['frequency_bins']==30
    with np.load(out,allow_pickle=False) as x:
        h=x['frf']
        assert h.shape==(4,8,30)
        assert np.iscomplexobj(h)
        assert np.allclose(x['frequency_hz'],1+5*np.arange(30))
        assert np.allclose(h[0,0,:],0.25*np.exp(1j*0.017*x['frequency_hz']),atol=1e-9)


@pytest.fixture
def mocked_plate(tmp_path):return *fixture(tmp_path),tmp_path


def test_missing_coordinates_rejected(mocked_plate):
    ar,coords,tmp_path=mocked_plate
    m=tmp_path/'manifest.csv';scan_archive(ar,m)
    with coords.open('a') as f:f.write('1,9,9\n')
    with pytest.raises(ValueError,match='Duplicate'):convert_archive(ar,m,coords,tmp_path/'bad.npz',expected_states=4,expected_points=8)


def test_incomplete_pair_rejected(tmp_path):
    zpath=tmp_path/'bad.zip'
    with ZipFile(zpath,'w') as z:z.writestr('Amplitude/State_1/Point_1.txt',polytec_txt(np.ones(10)))
    with pytest.raises(ValueError,match='Incomplete'):scan_archive(zpath,tmp_path/'manifest.csv')


def test_freq_validation():
    raw=polytec_txt(np.ones(40))
    freq,val=read_polytec(raw,freq_min=1,freq_max=40,stride=4)
    assert len(freq)==10 and np.allclose(freq,1+np.arange(10)*4)
    with pytest.raises(ValueError):read_polytec(raw,stride=0)


def test_coordinates_mm_headers(mocked_plate):
    _,path,_=mocked_plate
    c=load_coordinates(path)
    assert len(c)==8 and np.allclose(c[1],[0,0])


def test_end_to_end_benchmark_uses_compared_groups(mocked_plate, tmp_path):
    import importlib.util
    ar,coord,_=mocked_plate
    manifest=tmp_path/'manifest.csv';scan_archive(ar,manifest)
    out=tmp_path/'complex.npz'
    convert_archive(ar,manifest,coord,out,freq_min=1,freq_max=150,stride=5,
                    expected_states=4,expected_points=8)
    spec=importlib.util.spec_from_file_location('realbenchmark', Path(__file__).resolve().parents[1]/'examples'/'benchmark_real_frf.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    df=mod.run(out,tmp_path/'results', counts=(3,),seed=42)
    assert set(df['method'])=={'logdet','qr_pod_inhouse','geometry','random'}
    assert len(df)==4*4
    assert set(df['group'])=={f'State_{i:02d}' for i in range(4)}
    assert (df['nrmse']>=0).all() and df['nrmse'].notna().all()
    assert (tmp_path/'results'/'actual_group_holdout.csv').exists()
