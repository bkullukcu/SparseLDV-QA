"""Quality-aware spatial selection and complex graph reconstruction.

Arrays use (sample, point, frequency). Fit selection on training scans only.
A coherence value is a reliability descriptor, not an independent replicate.
"""
import numpy as np
from scipy.spatial.distance import cdist

__all__ = ['complex_frac', 'quality_weights', 'graph_laplacian', 'select_points',
           'reconstruct', 'group_splits', 'evaluate_reconstruction']


def complex_frac(a, b, axis=-1):
    """Squared complex correlation; zero-energy comparisons return NaN."""
    a, b = np.broadcast_arrays(np.asarray(a, complex), np.asarray(b, complex))
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError('FRFs must be finite')
    numerator = np.abs(np.sum(a.conj()*b, axis=axis))**2
    denominator = np.sum(abs(a)**2, axis=axis)*np.sum(abs(b)**2, axis=axis)
    return np.clip(np.divide(numerator, denominator, out=np.full_like(numerator, np.nan, dtype=float), where=denominator>0), 0, 1)


def quality_weights(coherence, floor=0.0):
    """Median coherence across all axes except point (axis 0); reject NaNs."""
    c = np.asarray(coherence, float)
    if c.ndim < 1 or not np.isfinite(c).all() or np.any((c<0)|(c>1)):
        raise ValueError('coherence must be finite in [0, 1], with points on axis 0')
    if not 0 <= floor < 1:
        raise ValueError('floor must be in [0, 1)')
    q = np.median(c.reshape(c.shape[0], -1), axis=1)
    return np.where(q >= floor, q, 0.)


def graph_laplacian(coordinates, neighbors=6):
    """Symmetric Gaussian k-nearest-neighbor graph; coordinate units cancel."""
    x = np.asarray(coordinates, float)
    if x.ndim != 2 or len(x)<3 or not np.isfinite(x).all():
        raise ValueError('finite (point, coordinate) array with >=3 points required')
    if len(np.unique(x,axis=0)) != len(x):
        raise ValueError('duplicate coordinates')
    if not isinstance(neighbors, (int,np.integer)) or not 1<=neighbors<len(x):
        raise ValueError('neighbors must be in [1, n_points-1]')
    d = cdist(x,x); order=np.argsort(d,axis=1)[:,1:neighbors+1]
    sigma=np.median(np.take_along_axis(d,order,axis=1))
    w=np.zeros_like(d); rows=np.arange(len(x))[:,None]
    w[rows,order]=np.exp(-0.5*(d[rows,order]/sigma)**2)
    w=np.maximum(w,w.T)
    return np.diag(w.sum(axis=1))-w


def select_points(training, count, quality=None, ridge=0.05):
    """Greedy regularized log-determinant selection on complex response features.

    Each training sample is energy-normalized; features are point rows and
    real/imaginary sample-frequency columns. Quality scales rows by sqrt(q).
    Selection uses no held-out scan. Returned indices are ordered by selection.
    """
    h=np.asarray(training,complex)
    if h.ndim!=3 or not np.isfinite(h).all():
        raise ValueError('training must be finite (sample, point, frequency)')
    n=h.shape[1]
    if not isinstance(count,(int,np.integer)) or not 1<=count<=n or ridge<=0:
        raise ValueError('invalid count or ridge')
    q=np.ones(n) if quality is None else np.asarray(quality,float)
    if q.shape!=(n,) or not np.isfinite(q).all() or np.any((q<0)|(q>1)) or np.sum(q>0)<count:
        raise ValueError('quality must provide enough positive weights in [0,1]')
    norms=np.sqrt(np.mean(abs(h)**2,axis=(1,2),keepdims=True))
    if np.any(norms==0):raise ValueError('zero-energy training sample')
    a=(h/norms).transpose(1,0,2).reshape(n,-1)
    f=np.concatenate([a.real,a.imag],axis=1)*np.sqrt(q[:,None])
    k=f@f.T/f.shape[1]
    residual=k.copy(); chosen=[]
    # Schur complement updates equal greedy logdet gains for K_SS + ridge I.
    for _ in range(count):
        gain=np.diag(residual).copy();gain[q==0]=-np.inf;gain[chosen]=-np.inf
        j=int(np.argmax(gain));chosen.append(j)
        col=residual[:,j].copy();residual-=np.outer(col,col)/(ridge+residual[j,j])
    return np.asarray(chosen,int)


def reconstruct(observed, indices, laplacian, alpha=0.1, quality=None):
    """Solve (D + alpha L + eps I) H = D H_obs for all frequency bins.

    observed is (selected_point, frequency). Regularization alpha is relative to
    the unnormalised graph. Set it before outer validation, or tune inside it.
    """
    y=np.asarray(observed,complex);s=np.asarray(indices)
    l=np.asarray(laplacian,float)
    if l.ndim!=2 or l.shape[0]!=l.shape[1] or not np.isfinite(l).all() or not np.allclose(l,l.T):
        raise ValueError('symmetric finite Laplacian required')
    if not np.allclose(l.sum(axis=1),0,atol=1e-8) or np.linalg.eigvalsh(l).min() < -1e-8:
        raise ValueError('matrix is not a positive semidefinite Laplacian')
    n=len(l)
    if s.ndim!=1 or s.dtype.kind not in 'iu' or len(np.unique(s))!=len(s) or np.any((s<0)|(s>=n)):
        raise ValueError('unique integer point indices required')
    if y.ndim!=2 or y.shape[0]!=len(s) or len(s)==0 or not np.isfinite(y).all() or alpha<=0:
        raise ValueError('invalid observations or alpha')
    q=np.ones(len(s)) if quality is None else np.asarray(quality,float)
    if q.shape!=(len(s),) or not np.isfinite(q).all() or np.any((q<=0)|(q>1)):
        raise ValueError('observation quality must be in (0,1]')
    d=np.zeros(n);d[s]=q;rhs=np.zeros((n,y.shape[1]),complex);rhs[s]=q[:,None]*y
    return np.linalg.solve(np.diag(d)+alpha*l+1e-10*np.eye(n),rhs)


def group_splits(groups):
    """Leave-one-group-out indices: a scan's resonance windows stay together."""
    g=np.asarray(groups)
    if g.ndim!=1 or len(np.unique(g))<2:raise ValueError('at least two groups required')
    for key in np.unique(g):yield np.flatnonzero(g!=key),np.flatnonzero(g==key)


def evaluate_reconstruction(truth,prediction,observed_indices):
    """Score only unmeasured points; NRMSE uses total complex energy."""
    a,b=np.asarray(truth,complex),np.asarray(prediction,complex)
    if a.shape!=b.shape or a.ndim!=2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('matching finite (point,frequency) arrays required')
    s=np.asarray(observed_indices)
    if s.dtype.kind not in 'iu' or np.any((s<0)|(s>=len(a))):raise ValueError('invalid indices')
    mask=np.ones(len(a),bool);mask[s]=False
    if not mask.any():raise ValueError('no unmeasured points remain')
    den=np.linalg.norm(a[mask])
    if den==0:raise ValueError('zero-energy truth')
    return {'nrmse':float(np.linalg.norm(a[mask]-b[mask])/den),
            'median_frac':float(np.nanmedian(complex_frac(a[mask],b[mask]))),
            'unmeasured_points':int(mask.sum())}
