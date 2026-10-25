# Research notes

The supplied CSV/NPZ files are small synthetic measurement examples. They check import rules, point selection and reconstruction; they cannot establish accuracy on an independent specimen.

The software was developed alongside studies of complex FRF measurements, including a KU Leuven LMSD dataset and measurements of a bolted structure. None of the raw records, derived experimental response arrays or benchmark tables are distributed here. They remain subject to their original reuse terms. Scientific reanalysis requires obtaining the source measurements and documenting the exact frequency selections and train/test groups.

Several excitation references on one specimen are not independent specimens. Sensor budgets must be evaluated under the acquisition constraints of the actual experiment. Coherence weighting should use measured coherence rather than synthetic proxies.

A self-contained numerical example can be reproduced using:

```bash
python examples/reproduce.py --output example_output
```

For a separately supplied complex FRF NPZ containing `frf`, `coordinates` and `group_ids`, the grouped reconstruction comparison is available at:

```bash
python examples/benchmark_real_frf.py --input /path/to/experimental_frfs.npz --out local_benchmark
```

Neither command is a substitute for independent experimental validation.
