"""Regenerate the two README figures from deterministic, synthetic complex FRFs.

From the repository root:
    python examples/make_readme_figures.py

Use --latex to typeset labels with a local TeX installation. The default uses
Matplotlib's MathText and works without TeX (including on CI runners).
No experimental dataset is read and no benchmark claim is made.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sparseldv_qa import graph_laplacian, select_points, reconstruct, evaluate_reconstruction

BLUE = '#245778'
TEAL = '#0d8278'
ORANGE = '#ce7950'
DARK = '#23313f'
MUTED = '#5f6d78'


def configure(latex: bool):
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
        'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10,
        'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#66727b', 'axes.labelcolor': DARK,
        'text.color': DARK, 'xtick.color': MUTED, 'ytick.color': MUTED,
        'svg.fonttype': 'path', 'text.usetex': latex,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
    })


def synthetic_response(points: np.ndarray, f: np.ndarray, excitation: float) -> np.ndarray:
    """Two damped resonances with smooth spatially varying complex amplitudes."""
    x, y = points.T
    phi1 = 1.0 + .18*np.cos(np.pi*x)*np.cos(np.pi*y) + .09*x
    phi2 = .30 + .24*np.cos(np.pi*x)*np.sin(np.pi*y) + .07*y
    f1, f2 = 68.0, 131.0
    t1 = f1**2 / (f1**2 - f**2 + 2j*.035*f1*f)
    t2 = f2**2 / (f2**2 - f**2 + 2j*.030*f2*f)
    return (1+.07*excitation)*phi1[:,None]*t1[None,:] + \
           (.45+.045*excitation)*phi2[:,None]*t2[None,:] + .05


def compute_demo():
    u = np.linspace(0, 1, 6)
    xx, yy = np.meshgrid(u, u)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    frequencies = np.linspace(28, 174, 350)
    training = np.stack([synthetic_response(points, frequencies, e) for e in [-1.5, -.5, .5, 1.5]])
    held_out = synthetic_response(points, frequencies, 2.4)
    selected = select_points(training, count=10)
    lap = graph_laplacian(points, neighbors=5)
    predicted = reconstruct(held_out[selected], selected, lap, alpha=.018)
    metrics = evaluate_reconstruction(held_out, predicted, selected)
    return points, frequencies, selected, held_out, predicted, metrics


def render(out: Path):
    points, f, selected, truth, pred, metrics = compute_demo()

    fig, ax = plt.subplots(figsize=(7.4, 4.2), constrained_layout=True)
    ax.scatter(points[:,0], points[:,1], s=36, facecolors='#dde6e9',
               edgecolors='#9baab3', linewidth=.85, label='Available grid positions', zorder=2)
    ax.scatter(points[selected,0], points[selected,1], s=105, facecolors=ORANGE,
               edgecolors='white', linewidth=1.2, marker='o', label=f'Selected ({len(selected)} of {len(points)})', zorder=3)
    for i, (x,y) in enumerate(points[selected], start=1):
        ax.text(x, y, str(i), ha='center', va='center', fontsize=7.2, color='white', zorder=4, weight='bold')
    ax.set(xlim=(-.07,1.07), ylim=(-.07,1.07), xlabel='Normalized specimen coordinate $x$',
           ylabel='Normalized specimen coordinate $y$', title='Synthetic selection of spatial FRF measurements')
    ax.set_aspect('equal')
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(.5, -.17), ncol=2)
    fig.savefig(out/'sensor_selection.svg', format='svg', metadata={'Date':None,'Creator':'SparseLDV-QA deterministic example'})
    plt.close(fig)

    mask=np.ones(len(points), dtype=bool); mask[selected]=False
    unseen=np.flatnonzero(mask)
    per_point_err=np.linalg.norm(truth[unseen]-pred[unseen],axis=1)/(np.linalg.norm(truth[unseen],axis=1)+1e-12)
    point=int(unseen[np.argsort(per_point_err)[len(unseen)//2]])
    mag=lambda a: 20*np.log10(np.maximum(np.abs(a),1e-9))
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(8.2,5.2),sharex=True,gridspec_kw={'height_ratios':[1.6,1]})
    ax1.plot(f,mag(truth[point]),color=BLUE,lw=2,label='Reference')
    ax1.plot(f,mag(pred[point]),color=ORANGE,lw=1.75,ls='--',label='Reconstructed')
    ax1.set(ylabel='FRF magnitude (dB, relative)',title=f'Complex FRF reconstruction at an unmeasured grid position')
    ax1.legend(frameon=False,loc='upper right')
    ax1.grid(color='#dfe5e9',linewidth=.65,alpha=.7)
    ax2.plot(f,np.unwrap(np.angle(truth[point])),color=BLUE,lw=1.9)
    ax2.plot(f,np.unwrap(np.angle(pred[point])),color=ORANGE,lw=1.6,ls='--')
    ax2.set(xlabel='Frequency (Hz)',ylabel='Unwrapped phase (rad)')
    ax2.grid(color='#dfe5e9',linewidth=.65,alpha=.7)
    fig.subplots_adjust(hspace=.14,top=.79,bottom=.14,left=.13,right=.96)
    fig.text(.13,.94,f'Synthetic held-out scan  |  unmeasured-point NRMSE = {metrics["nrmse"]:.3f}', fontsize=9,color=MUTED)
    fig.text(.96,.03,'Illustration, not an experimental performance estimate',ha='right',color=MUTED,fontsize=8)
    fig.savefig(out/'frf_reconstruction.svg',format='svg',metadata={'Date':None,'Creator':'SparseLDV-QA deterministic example'})
    plt.close(fig)
    print(f'Wrote two SVG figures; held-out synthetic NRMSE={metrics["nrmse"]:.4f}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path,default=Path(__file__).resolve().parents[1]/'docs'/'figures')
    p.add_argument('--latex', action='store_true',help='Use system LaTeX for text rendering')
    args=p.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    configure(args.latex);render(args.output_dir)


if __name__ == '__main__':main()
