import argparse,json,sys
import numpy as np
from .core import graph_laplacian,select_points,reconstruct,evaluate_reconstruction

from . import __version__


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'convert':
        from .user_data import convert
        q = argparse.ArgumentParser(description='Convert external FRF CSV/NPZ using a declared schema')
        q.add_argument('command', choices=['convert'])
        q.add_argument('input')
        q.add_argument('--schema', required=True)
        q.add_argument('--output', default='converted_frfs.npz')
        a = q.parse_args()
        print(json.dumps(convert(a.input, a.schema, a.output), indent=2))
        return
    p=argparse.ArgumentParser(description='Reconstruct held-out FRF from independently supplied training scans',
        epilog='Import your own data first with: sparseldv-qa convert INPUT --schema SCHEMA.json --output OUT.npz (see docs/USER_DATA_IMPORT.md).')
    p.add_argument('--version', action='version', version='%(prog)s '+__version__)
    p.add_argument('input',help='NPZ: training (sample,point,freq), truth (point,freq), coordinates (point,dim), optional quality (point,)')
    p.add_argument('--count',type=int,default=8);p.add_argument('--alpha',type=float,default=.1);p.add_argument('--output',default='reconstruction.npz');a=p.parse_args()
    with np.load(a.input,allow_pickle=False) as z:
        q=z['quality'] if 'quality' in z else None;s=select_points(z['training'],a.count,q)
        pred=reconstruct(z['truth'][s],s,graph_laplacian(z['coordinates'],min(6,len(z['coordinates'])-1)),a.alpha,None if q is None else q[s])
        metrics=evaluate_reconstruction(z['truth'],pred,s)
    np.savez_compressed(a.output,prediction=pred,selected=s);print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
