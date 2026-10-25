"""KU Leuven LMSD 2021 (doi:10.48804/GDE9TW) adapter.

Data CC BY 4.0, helper scripts MIT. No raw data distributed here.
Healthy baseline has 7 excitation/response reference channels, 144 spatial nodes.
Only baseline is eligible for 144-point withheld-node testing: damaged
responses have seven spatial nodes; NEVER impute the missing 137 as ground truth.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from .baselines import qr_pod_points, farthest_geometry_points
from .core import select_points, evaluate_reconstruction
from .alternative_reconstruction import reconstruction_operators


def load_lmsd_baseline(path, *, fmin=40., fmax=1400., max_bins=96):
    with np.load(Path(path), allow_pickle=False) as z:
        required={'frf','frequency','node_position','probing_nodes'}
        if not required.issubset(z.files):
            raise ValueError(f'Incorrect LMSD baseline keys; missing {required-set(z.files)}')
        response=np.asarray(z['frf'])
        freqs=np.asarray(z['frequency']).reshape(-1).astype(float)
        xyz=np.asarray(z['node_position'],float)
        probe=np.asarray(z['probing_nodes'],int).reshape(-1)
    if response.ndim!=3 or response.shape[0]!=len(probe) or response.shape[1]!=len(xyz) or response.shape[2]!=len(freqs):
        raise ValueError('Expected (reference,point,freq) FRF shape; do not transpose implicitly')
    if len(xyz)!=144 or len(probe)!=7 or xyz.shape[1]!=2:
        raise ValueError('LMSD 2021 full-field baseline must have 144 nodes and 7 references')
    if not np.iscomplexobj(response) or not np.isfinite(response).all():
        raise ValueError('Complex finite FRFs required; real magnitude is insufficient for FRAC')
    if not np.isfinite(xyz).all() or not np.all(np.diff(freqs)>0):
        raise ValueError('Invalid coordinates/frequency')
    if len(set(probe))!=len(probe) or np.any((probe<0)|(probe>=144)):
        raise ValueError('NPZ probing nodes must use zero-based indexing')
    keep=np.flatnonzero((freqs>=fmin)&(freqs<=fmax))
    if len(keep)<3: raise ValueError('Insufficient frequencies in selected range')
    if max_bins is not None and len(keep)>max_bins:
        keep=keep[np.unique(np.linspace(0,len(keep)-1,max_bins).astype(int))]
    # Metadata provided in mm; canonical internal unit is metres
    return {'frf':response[:,:,keep], 'frequency_hz':freqs[keep],
            'coordinates_m':xyz*1e-3,'reference_node_indices':probe,
            'source':'doi:10.48804/GDE9TW', 'reference_split_unit':'excitation_reference'}


def inspect_lmsd_damage(path):
    """Damage files are NOT 144-node full fields; report honest coverage."""
    with np.load(Path(path),allow_pickle=False) as z:
        h=np.asarray(z['frf'])
        if h.ndim!=3 or h.shape[0]!=7 or h.shape[1]!=7:
            raise ValueError('Unexpected damaged multistatic FRF dimensions')
        mass=z['mass_nodes'].tolist() if 'mass_nodes' in z.files else []
    return {'shape':list(h.shape),'mass_nodes':mass,'eligible_fullfield_benchmark':False}


def benchmark_independent_lmsd(baseline, *, counts=(6,12,20), random_repeats=2, seed=2026):
    """Leave-one-excitation-reference-out; training never contains test FRFs.
    Returns observations, no unqualified claim of damage-state generalization.
    """
    h=baseline['frf'];xyz=baseline['coordinates_m'];rng=np.random.default_rng(seed)
    if h.shape[0]<3:raise ValueError('>=3 independent excitation references required')
    rows=[]
    for held in range(len(h)):
        train=np.delete(h,held,axis=0); truth=h[held]
        for k in counts:
            if not 1<=k<len(xyz):raise ValueError('invalid measured node budget')
            selection={'QR_POD':[qr_pod_points(train,k)],
                       'logdet':[select_points(train,k)],
                       'geometry':[farthest_geometry_points(xyz,k)],
                       'random':[rng.choice(len(xyz),k,replace=False) for _ in range(random_repeats)]}
            for selector, variants in selection.items():
                for repeat,s in enumerate(variants):
                    for recon,op in reconstruction_operators(xyz,s,train).items():
                        pred=op@truth[s]
                        quality=evaluate_reconstruction(truth,pred,s)
                        rows.append({'test_reference':held,'budget':k,'selector':selector,
                                     'reconstruction':recon,'random_repeat':repeat,
                                     'nrmse':quality['nrmse'],'median_FRAC':quality['median_frac']})
    df=pd.DataFrame(rows)
    groups=df.groupby(['test_reference','budget','selector','reconstruction'],as_index=False)[['nrmse','median_FRAC']].median()
    summ=groups.groupby(['budget','selector','reconstruction'],as_index=False).agg(median_NRMSE=('nrmse','median'),median_FRAC=('median_FRAC','median'))
    return df,summ
