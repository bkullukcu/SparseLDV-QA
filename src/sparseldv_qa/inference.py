"""Configuration-blocked *sensitivity* of paired benchmark scores.

Multiple configurations on the same specimen are not independent specimens;
bootstrap intervals are exploratory conditional on this specimen.
"""
import numpy as np
import pandas as pd


def paired_configuration_sensitivity(table, method_a="logdet", method_b="geometry",
                                     count=24, metric="nrmse", n_boot=4000, seed=2026):
    """Block-resample paired configuration means, never individual resonance windows.

    Report A - B: positive NRMSE favors B. CI quantifies grouping sensitivity,
    NOT confidence intervals for new specimens or independent acquisitions.
    """
    required={"method", "count", "condition", metric}
    if not required.issubset(table.columns):
        raise ValueError("Missing benchmark columns")
    if method_a == method_b or int(count) != count or count < 1 or n_boot < 2:
        raise ValueError("Distinct methods, positive count and at least 2 resamples required")
    a=table.loc[(table.method==method_a)&(table["count"]==count), ["condition",metric]]
    b=table.loc[(table.method==method_b)&(table["count"]==count), ["condition",metric]]
    if a.condition.duplicated().any() or b.condition.duplicated().any():
        raise ValueError("One row per configuration/method/count expected")
    paired=a.merge(b,how="inner",on="condition",suffixes=("_a","_b"),validate="one_to_one")
    if len(paired)<3 or len(paired)!=len(a) or len(paired)!=len(b):
        raise ValueError("Need >=3 exactly matched configurations")
    d=(paired[f"{metric}_a"]-paired[f"{metric}_b"]).to_numpy(dtype=float)
    if not np.isfinite(d).all():raise ValueError("Nonfinite scores")
    rng=np.random.default_rng(seed)
    bidx=rng.integers(0,len(d),size=(int(n_boot),len(d)))
    draws=np.median(d[bidx],axis=1)
    return {"count":int(count),"method_a":method_a,"method_b":method_b,
            "metric":metric,"n_configuration":len(d),"median_paired_difference":float(np.median(d)),
            "resampling_interval_95":np.quantile(draws,[.025,.975]).tolist(),
            "positive_fraction":float(np.mean(d>0)),"negative_fraction":float(np.mean(d<0)),
            "interpretation":"within-specimen blocked resampling sensitivity, not a population confidence interval"}
