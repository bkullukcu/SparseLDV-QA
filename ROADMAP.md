# SparseLDV-QA — development roadmap

Priority order is a planning aid, not evidence of completed experimental validation.

## P0: Data contracts and numerical verification
- Enforce citation/version/README DOI consistency and Python syntax across supported interpreters.
- Add failure-case tests for inconsistent CSV/NPZ axes, duplicate points, non-finite FRFs and missing unit declarations.
- Test selection, coherence weighting and hyperparameter tuning for train/holdout leakage.
- Establish known-truth complex NRMSE and FRAC reference fixtures.

## P1: Scientific comparisons
- Compare graph-regularized reconstruction with geometric and QR/POD baselines at equal measurement budgets.
- Audit geometric sensitivity, spatial sampling and regularization with group-blocked validation.
- Track numerical conditioning and uncertainty without treating synthetic scores as experimental evidence.

## P2: Usability and reproducibility
- Export machine-readable provenance: input hashes, units, sensor budgets, selected points and held-out groups.
- Document input preparation for independently acquired LDV data without bundling restricted recordings.
- Maintain stable CLI/public APIs and regenerate example figures from versioned scripts.

## Acceptance criteria
Each numerical change requires a deterministic regression test and an explanation of the statistical or physical assumption; experimental benchmarks require source, permissions and full protocol. CI must pass before merging.
