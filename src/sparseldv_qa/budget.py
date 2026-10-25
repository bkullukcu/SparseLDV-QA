"""Observed budget attainment; no interpolation or prospective guarantee."""
import numpy as np
import pandas as pd

def budget_attainment(results, threshold, group='group'):
    """First tested budget meeting an externally specified NRMSE threshold.

    Missing attainment remains NaN. Replicates must be collapsed first. Results
    describe supplied held-out errors; do not select/tune threshold on test data.
    """
    keys=[group,'selector','reconstruction'];need=keys+['budget','nrmse']
    if not set(need)<=set(results):raise ValueError('missing result columns')
    if not np.isfinite(threshold) or threshold<0:raise ValueError('nonnegative finite threshold required')
    if results.empty or results[need].isna().any().any() or results.duplicated(keys+['budget']).any():
        raise ValueError('nonempty unique collapsed rows required')
    a=results[['budget','nrmse']].to_numpy(float)
    if not np.isfinite(a).all() or (a[:,0]<1).any() or (a[:,0]%1!=0).any() or (a[:,1]<0).any():
        raise ValueError('positive integer budgets and finite nonnegative errors required')
    rows=[]
    for key,df in results.groupby(keys):
        ok=df.loc[df.nrmse<=threshold,'budget']
        rows.append(dict(zip(keys,key),threshold=float(threshold),minimum_tested_budget=float(ok.min()) if len(ok) else np.nan,
                         attained=bool(len(ok)),maximum_tested_budget=int(df.budget.max())))
    return pd.DataFrame(rows)
