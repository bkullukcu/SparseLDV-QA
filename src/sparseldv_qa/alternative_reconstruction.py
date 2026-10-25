"""Frozen, auditable alternative spatial/learned complex-FRF reconstruction rules.

Operators map selected complex FRFs of shape (k, frequency) to (point, frequency).
No held-out response enters any operator construction; the POD spatial basis is
estimated from outer-fold training scans only.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist
from .core import graph_laplacian


def _validate_coordinates_indices(coordinates, selected):
    xyz = np.asarray(coordinates, float)
    s = np.asarray(selected)
    if xyz.ndim != 2 or xyz.shape[0] < 3 or not np.isfinite(xyz).all() or len(np.unique(xyz,axis=0)) != len(xyz):
        raise ValueError('coordinates must be finite, distinct (n_points,dimensions)')
    if s.ndim != 1 or s.dtype.kind not in 'iu' or len(s) == 0 or len(np.unique(s))!=len(s) or s.min()<0 or s.max()>=len(xyz):
        raise ValueError('selected must be nonempty unique valid integer point indices')
    if len(s) >= len(xyz):
        raise ValueError('unobserved points required for evaluation')
    return xyz, s


def idw_operator(coordinates, selected, neighbors=4, power=2):
    """Fixed local inverse distance interpolation (physical coordinates; no FRF fitting)."""
    x,s=_validate_coordinates_indices(coordinates,selected)
    if neighbors<1 or power<=0: raise ValueError('positive neighbors, power required')
    d=cdist(x,x[s]); W=np.zeros((len(x),len(s)))
    for i in range(len(x)):
        if i in s:
            W[i,np.flatnonzero(s==i)[0]]=1.
        else:
            near=np.argsort(d[i])[:min(neighbors,len(s))]
            weights=np.power(d[i,near],-power)
            W[i,near]=weights/weights.sum()
    return W


def rbf_operator(coordinates,selected,length_scale_factor=2.0,regularization=1e-4):
    """Gaussian RBF kernel ridge interpolation with geometry-fixed bandwidth."""
    x,s=_validate_coordinates_indices(coordinates,selected)
    if length_scale_factor<=0 or regularization<=0:raise ValueError('positive bandwidth and ridge required')
    ds=cdist(x,x)
    nearest=np.min(np.where(np.eye(len(x),dtype=bool),np.inf,ds),axis=1)
    # physical dimension is inferred from coordinate units; ell = 2 * median nearest spacing
    ell=float(np.median(nearest)*length_scale_factor)
    K=np.exp(-0.5*(ds/ell)**2)
    return np.linalg.solve(K[np.ix_(s,s)]+regularization*np.eye(len(s)), K[:,s].T).T


def graph_operator(coordinates, selected, alpha=0.1, neighbors=6):
    """Tikhonov smoothing operator equivalent to original graph reconstruction."""
    x,s=_validate_coordinates_indices(coordinates,selected)
    if alpha<=0:raise ValueError('alpha > 0 required')
    L=graph_laplacian(x,neighbors=neighbors)
    A=alpha*L+1e-10*np.eye(len(x))
    A[s,s]+=1
    B=np.zeros((len(x),len(s)))
    B[s,np.arange(len(s))]=1
    return np.linalg.solve(A,B)


def pod_basis(training, n_modes=8):
    """Real spatial POD, computed from all training configurations and complex spectra."""
    h=np.asarray(training)
    if h.ndim!=3 or not np.iscomplexobj(h) or not np.isfinite(h).all():
        raise ValueError('training must be finite complex (sample, point, frequency)')
    n=h.shape[1]
    if not 1<=n_modes<=n:raise ValueError('1<=n_modes<=n_points')
    rms=np.sqrt(np.mean(abs(h)**2,axis=(1,2),keepdims=True))
    if np.any(rms==0):raise ValueError('training sample must be nonzero')
    A=(h/rms).transpose(1,0,2).reshape(n,-1)
    R=np.concatenate([A.real,A.imag],axis=1)
    gram=R@R.T
    values, vectors=np.linalg.eigh(gram)
    return vectors[:,np.argsort(values)[-n_modes:][::-1]]


def pod_operator(coordinates,selected,training,n_modes=8,regularization=0.025):
    """Fit complex coefficients to a train-only real spatial POD basis."""
    x,s=_validate_coordinates_indices(coordinates,selected)
    h=np.asarray(training)
    if h.shape[1]!=len(x):raise ValueError('training point dimension mismatch')
    if regularization<=0:raise ValueError('ridge > 0 required')
    U=pod_basis(h,min(n_modes,len(s)))
    Us=U[s]
    return U@np.linalg.solve(Us.T@Us+regularization*np.eye(U.shape[1]),Us.T)


def reconstruction_operators(coordinates,selected,training):
    return {
        'graph_Tikhonov':graph_operator(coordinates,selected),
        'IDW4':idw_operator(coordinates,selected),
        'Gaussian_RBF':rbf_operator(coordinates,selected),
        'train_POD':pod_operator(coordinates,selected,training),
    }
