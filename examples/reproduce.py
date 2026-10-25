"""Self-contained synthetic QA example: python examples/reproduce.py --output ..."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sparseldv_qa import *

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',default='examples/output');a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 rng=np.random.default_rng(2026);xy=np.array([(i,j) for i in np.linspace(0,1,7) for j in np.linspace(0,1,7)]);f=np.linspace(30,150,121)
 shapes=np.stack([np.sin(np.pi*xy[:,0])*np.sin(np.pi*xy[:,1]),np.cos(2*np.pi*xy[:,0])*np.sin(np.pi*xy[:,1]),np.sin(np.pi*xy[:,0])*np.cos(2*np.pi*xy[:,1])],axis=1)
 training=np.stack([shapes@((1+.05*rng.normal(size=3))[:,None]/((np.array([55,95,130])[:,None])**2-f**2+1j*2*.02*np.array([55,95,130])[:,None]*f)) for _ in range(8)])
 quality=np.ones(49);quality[[10,17,24,31,38]]=0
 s=select_points(training,12,quality);l=graph_laplacian(xy);pred=reconstruct(training[0,s],s,l);metrics=evaluate_reconstruction(training[0],pred,s)
 fig,ax=plt.subplots(1,2,figsize=(7.2,3.1),layout='constrained');ax[0].scatter(*xy.T,c=quality,cmap='viridis',s=35,vmin=0,vmax=1);ax[0].scatter(*xy[s].T,facecolors='none',edgecolors='#D55E00',s=100,label='Selected');ax[0].set(xlabel='x / a',ylabel='y / b',title='A  Quality-constrained selection',aspect='equal');ax[0].legend(fontsize=8)
 k=26;ax[1].plot(f,abs(training[0,k]),label='Truth');ax[1].plot(f,abs(pred[k]),'--',label='Reconstruction');ax[1].set(xlabel='Frequency (Hz)',ylabel='Response (arbitrary units)',title='B  Synthetic held-point spectrum');ax[1].legend(fontsize=8);fig.suptitle('SYNTHETIC VERIFICATION - no experimental coherence',fontsize=10)
 for ext in ['pdf','svg','png']:fig.savefig(out/('synthetic_qa.'+ext),dpi=400,bbox_inches='tight')
 np.savez_compressed(out/'demo_input.npz',training=training[1:],truth=training[0],coordinates=xy,quality=quality)
 (out/'synthetic_results.json').write_text(json.dumps(metrics,indent=2));plt.close(fig)
if __name__=='__main__':main()
