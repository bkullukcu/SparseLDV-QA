# SparseLDV-QA

**Sensor selection and complex FRF reconstruction for scanning laser vibrometry.**

SparseLDV-QA works with frequency-response measurements sampled at spatial locations. Given several complex FRF scans and the measurement coordinates, it selects a subset of points and reconstructs the response elsewhere. It includes geometry-based and response-based selection methods, reconstruction baselines, and metrics for held-out measurements.

This is a research-stage package (`v0.8.2`). The API may change before version 1.0.

## Install

Python 3.10 or newer is required. From the repository directory:

```bash
python -m pip install -e .
python examples/quickstart.py
```

The quickstart uses a small synthetic FRF. It checks that the reconstruction API runs; it is not an experimental accuracy benchmark.

For development and tests:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

## Use your own measurements

The input converter accepts CSV and NumPy archives. A JSON schema maps the original columns, units and sample identifiers. Complex responses can be supplied as real/imaginary pairs, or as linear magnitude and phase with an explicitly declared phase convention.

```bash
sparseldv-qa convert examples/step3_user_data/frf_long.csv \
  --schema examples/step3_user_data/schema.json \
  --output example_frfs.npz

sparseldv-qa example_frfs.npz --count 2 --output example_reconstruction.npz
```

The first command creates a validated NPZ file and a provenance manifest. The second selects measurement points and reconstructs the held-out response in that file. Replace the sample CSV and schema with your own files once their column definitions have been checked.

See [the input guide](docs/USER_DATA_IMPORT.md) for array dimensions, required metadata and NumPy input. The included fixtures are synthetic.

## What the results mean

The reconstruction error is meaningful only when the reference response was withheld from sensor selection and model fitting. Measurements from several excitation points on one specimen are not independent specimen-level validation. Coherence-based weights require measured coherence; they cannot be inferred from FRF amplitude.

No experimental recordings or derived benchmark tables are shipped. The numerical tests use synthetic inputs. For the original-study context and validation limits, see [research notes](docs/RESEARCH_NOTES.md).

## Repository contents

- `src/sparseldv_qa/` — numerical routines, input conversion and command-line interface
- `examples/` — synthetic run, sample CSV/NPZ inputs and a grouped FRF evaluation script
- `tests/` — unit and regression tests
- `docs/` — input format and methodological notes

## Citation and license

Citation metadata is in [CITATION.cff](CITATION.cff). Cite the original datasets separately when using their measurements. The source is supplied under the MIT license; confirm institutional authorization before making a public release. The software has not yet undergone independent multi-laboratory validation.
