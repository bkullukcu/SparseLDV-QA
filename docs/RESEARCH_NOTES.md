# Research notes

The CSV/NPZ fixtures use generated two-resonance complex FRFs to demonstrate import, point selection and reconstruction. Experimental performance can be assessed by providing independently measured scans.

The development also involved complex FRF studies from a KU Leuven LMSD dataset and a bolted structure. Those measurements are available through their respective data custodians, subject to applicable access terms. For reanalysis, document the source, frequency grid, geometry and training/test groups.

When grouping experimental observations, distinguish excitation references from physically independent specimens. Evaluate sensor budgets under the acquisition constraints of the measurement system; use measured coherence to derive coherence-based weights.

A self-contained numerical example can be reproduced using:

```bash
python examples/reproduce.py --output example_output
```

For a separately supplied complex FRF NPZ containing `frf`, `coordinates` and `group_ids`, the grouped reconstruction comparison is available at:

```bash
python examples/benchmark_real_frf.py --input /path/to/experimental_frfs.npz --out local_benchmark
```

For experimental validation, apply the same protocol to independent measurement groups and report the acquisition conditions.
