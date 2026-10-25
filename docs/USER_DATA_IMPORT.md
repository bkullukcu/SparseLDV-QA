# Importing complex FRF measurements

SparseLDV-QA converts instrument exports to the complex arrays used for sensor selection and spatial reconstruction. Each CSV row identifies a scan, physical measurement point and frequency bin. Complex FRFs can be provided as real/imaginary parts or linear magnitude/phase.

## Quick start

```bash
python -m pip install -e .
sparseldv-qa convert examples/step3_user_data/frf_long.csv \
  --schema examples/step3_user_data/schema.json \
  --output converted_frfs.npz
sparseldv-qa converted_frfs.npz --count 2 --output reconstruction.npz
```

The converter writes `converted_frfs.npz` and `converted_frfs.npz.manifest.json`. The analysis command reads `training` (scan, point, frequency), `truth` (point, frequency), `coordinates` (point, 2/3) and optional `quality` (point). The example uses a declared holdout scan for reconstruction and evaluation.

## CSV schema

Start from `examples/step3_user_data/schema.json`. Map the fields `sample`, `point`, `frequency`, `real`, `imag`, `x`, `y`; optional fields are `z` and `coherence`. For magnitude/phase exports, configure `complex_format` and the corresponding columns instead of real/imaginary parts.

Specify `coordinate_unit` (`m` or `mm`), `frequency_unit` (`Hz` or `kHz`), `response_unit` (for example, mobility `m/s/N`) and `truth_sample`. Optionally specify `training_samples`, `sample_groups` and `enforce_group_holdout`. The holdout scan should be selected according to the experiment design before fitting. With group identifiers, the converter checks separation between training and evaluation groups.

For measured coherence, set `quality_kind` to `magnitude_squared_coherence`; values are summarized as a point-wise median across scans and frequencies. Repeated sample–point–frequency observations should be aggregated upstream using a documented procedure.

## NumPy input

An NPZ file can store `training` (S, P, F), `truth` (P, F), `coordinates` (P, 2/3), `frequency_hz` (F) and `point_ids` (P). It can also contain `training_sample_ids` (S), scalar `truth_sample_id`, `quality` (P) and scan/group identifiers. Include the JSON schema to declare units and the physical transfer quantity. For frequencies expressed in hertz, use `frequency_unit: Hz`.

The importer checks array shape, identifiers, frequency ordering, finite complex values and quality definitions. Record the instrument's excitation/reference convention in the experiment metadata.

```python
from sparseldv_qa.user_data import convert
report = convert('my_scans.csv', 'my_schema.json', 'converted_frfs.npz')
```

A complete generated NPZ example is supplied as `examples/step3_user_data/frf_numpy.npz`:

```bash
sparseldv-qa convert examples/step3_user_data/frf_numpy.npz \
  --schema examples/step3_user_data/schema_numpy.json \
  --output mine.npz
```

## Magnitude and phase

For linear magnitude/phase exports, set `"complex_format": "magnitude_phase"` and declare `phase_unit` (`degree`/`radian`), `phase_sign` (`1`/`-1`) and `phase_reference`. Conversion uses \(H=|H|e^{is\phi}\). Provide nonnegative linear magnitudes and a recorded phase reference. NumPy arrays are complex-valued by default; use `allow_real_only: true` when the measured response is intentionally real-valued.

## Experimental evaluation

Use a withheld full-field scan to calculate NRMSE and FRAC at unmeasured positions. Supply measured coherence when available. For transfer studies, hold out entire specimens, sessions or mounting configurations as appropriate, and document the acquisition conditions.
