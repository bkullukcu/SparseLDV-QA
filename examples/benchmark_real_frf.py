"""Real-data benchmark entry point; REQUIRES user's original complex FRF NPZ.

Expected keys: frf (sample,point,freq) complex; coordinates (point,2/3);
group_ids (sample,) independent test configuration labels.
No real measurements are bundled with this software.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from sparseldv_qa import select_points, graph_laplacian, reconstruct, evaluate_reconstruction
from sparseldv_qa.baselines import qr_pod_points, farthest_geometry_points, optional_pysensors_points


def run(input_npz, out, *, stride=1, counts=(6,12,20),seed=2026,include_pysensors=False):
    inp=Path(input_npz)
    if not inp.is_file():raise FileNotFoundError('Original complex FRF NPZ not found')
    with np.load(inp,allow_pickle=False) as v:
        if not {'frf','coordinates','group_ids'} <= set(v.files):
            raise ValueError('NPZ must contain frf, coordinates, group_ids')
        h=np.asarray(v['frf']);xyz=np.asarray(v['coordinates']);groups=np.asarray(v['group_ids'])
    if (h.ndim!=3 or xyz.ndim!=2 or groups.shape!=(len(h),) or
        h.shape[1]!=len(xyz) or not np.isfinite(h).all() or not np.isfinite(xyz).all() or
        len(np.unique(groups))<3 or any(count>=len(xyz) or count<1 for count in counts) or stride<1):
        raise ValueError('invalid scan arrays, grouping, counts or frequency stride')
    if not np.iscomplexobj(h):
        raise ValueError('FRF array must retain amplitude AND phase as complex samples')
    h=h[:,:,::stride]
    if h.shape[-1]<5:raise ValueError('at least 5 frequency bins are required after stride')
    lap=graph_laplacian(xyz)
    rng=np.random.default_rng(seed)
    rows=[]
    for g in np.unique(groups):
        tr=h[groups!=g];te=h[groups==g]
        for k in counts:
            selects={'logdet':select_points(tr,k),
                     'qr_pod_inhouse':qr_pod_points(tr,k),
                     'geometry':farthest_geometry_points(xyz,k),
                     'random':rng.choice(len(xyz),size=k,replace=False)}
            if include_pysensors:
                selects['pysensors_sspor']=optional_pysensors_points(tr,k)
            for method,ids in selects.items():
                for local_i,test in enumerate(te):
                    pred=reconstruct(test[ids],ids,lap,alpha=.1)
                    scores=evaluate_reconstruction(test,pred,ids)
                    rows.append(dict(group=str(g),heldout_scan=local_i,budget=k,method=method,
                                     nrmse=scores['nrmse'],median_frac=scores['median_frac']))
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'actual_group_holdout.csv',index=False)
    digest=hashlib.sha256()
    with inp.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): digest.update(block)
    (out/'provenance.json').write_text(json.dumps({
       'data_type':'experimental complex FRF NPZ (source provenance in .provenance.json)',
       'sha256':digest.hexdigest(),
       'n_groups':int(len(np.unique(groups))),'n_scans':int(len(h)),
       'n_points':int(h.shape[1]),'frequency_bins_used':int(h.shape[-1]),
       'stride':int(stride),'seed':seed,
       'split':'held-out independent group; train-only sensor selection',
       'model_alpha':'fixed 0.1; not optimized on test groups',
       'pysensors_executed':bool(include_pysensors),
       'method_labels':['logdet','qr_pod_inhouse','geometry','random']+(['pysensors_sspor'] if include_pysensors else []),
       'counts':list(counts),
       'limitations':'Not independent specimens unless group IDs encode specimens. No prospective sparse scanning.'
    },indent=2),encoding='utf8')
    return pd.DataFrame(rows)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True)
    ap.add_argument('--out',default='benchmarks/actual_plate')
    ap.add_argument('--stride',type=int,default=1,
                    help='Additional stride AFTER NPZ conversion; normally use 1')
    ap.add_argument('--counts',type=int,nargs='+',default=[6,12,20],
                    help='Numbers of retained sensors')
    ap.add_argument('--include-pysensors',action='store_true',
                    help='Run actual PySensors SSPOR if installed')
    x=ap.parse_args();df=run(x.input,x.out,stride=x.stride,
                              counts=tuple(x.counts),include_pysensors=x.include_pysensors)
    print(df.groupby(['method','budget'])['nrmse'].median().to_string())
