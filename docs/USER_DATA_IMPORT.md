# Importing complex FRF measurements

The converter maps instrument data to the arrays used by SparseLDV-QA. It validates the declared format but does not perform sensor selection or fit a reconstruction model. Input rows must represent a rectangular complex FRF (`real` and `imag`), one **physical point** and one frequency bin per scan. Magnitudes without phase **cannot** be converted to complex FRFs.

## Quick start

```bash
python -m pip install -e .
sparseldv-qa convert examples/step3_user_data/frf_long.csv \
  --schema examples/step3_user_data/schema.json \
  --output converted_frfs.npz
sparseldv-qa converted_frfs.npz --count 2 --output reconstruction.npz
```

Conversion generates `converted_frfs.npz` and `converted_frfs.npz.manifest.json`. The existing analysis command can then load `training` (scan,point,frequency), `truth` (point,frequency), `coordinates` (point,2/3) and optional `quality` (point). Additional arrays include point IDs, frequencies in Hz, and scan/group IDs. The second command is only a synthetic demonstration of reconstruction against a separately declared holdout scan.

## Declared schema

Copy `examples/step3_user_data/schema.json` and map your original CSV column headings in `columns`. Required conceptual fields: `sample`, `point`, `frequency`, `real`, `imag`, `x`, `y`; optional `z` and `coherence`. Declare `coordinate_unit` (`m`/`mm`), `frequency_unit` (`Hz`/`kHz`), `response_unit` (physical transfer function, e.g. mobility `m/s/N`), `truth_sample` (holdout scan ID) and optionally `training_samples`, `sample_groups` and `enforce_group_holdout`. If `coherence` is supplied, `quality_kind` **must** be `magnitude_squared_coherence`; it is aggregated as a point-wise median across scans and frequencies, not mislabelled as a calibrated probability.

`truth_sample` must be scientifically held out by the experimenter, not just a random choice of a convenient row. If `sample_groups` are supplied, a group intersection between train and holdout fails by default. If there are repeated measurements at one sample-point-frequency combination, average them **upstream with documented methodology** instead of dropping rows during conversion.

## NumPy input

An NPZ may instead store `training` (S,P,F), `truth` (P,F), `coordinates` (P,2/3), `frequency_hz` (F) and `point_ids` (P), plus `training_sample_ids` (S) and scalar `truth_sample_id`. The JSON still declares `coordinate_unit`, `frequency_unit` and `response_unit`. Despite the key `frequency_hz`, the declared unit is applied to the values; consistently supply `frequency_unit: Hz` for arrays already measured in Hz. Optional: `quality` (P), and scan/group identifiers. The adapter checks shapes, distinct points, positive strictly ascending frequency bins, finite complex data, and requested quality meaning. The caller must verify the original experimental reference/response conventions.

```python
from sparseldv_qa.user_data import convert
report = convert('my_scans.csv', 'my_schema.json', 'converted_frfs.npz')
```

## Measurement limitations

- Reconstructed experimental metrics require a genuine withheld `truth`; with only observed sparse points, do **not** claim unmeasured-point NRMSE.
- Coherence is missing in many export formats. Never invent values; omit it from both source and schema.
- Point ID and coordinate consistency is an internal check; it is not instrument calibration.
- Use held-out specimens/sessions where appropriate. Separate scans from one specimen do not establish inter-specimen generalisation.

## Magnitude/phase exports

If your scanning-LDV instrument exports **linear** magnitude and phase, use `"complex_format": "magnitude_phase"`, map `magnitude` and `phase` columns, and explicitly declare `"phase_unit": "degree"` or `"radian"`, `"phase_sign": 1` or `-1`, and a nonempty `"phase_reference"` describing the instrument reference convention. Conversion is \(H=|H|e^{i s\phi}\) with the declared sign. Negative magnitudes, dB magnitudes, and undeclared phase units are rejected. For NumPy, complex arrays are required unless `allow_real_only: true` explicitly records intentional zero imaginary parts.

A fully synthetic NumPy fixture `examples/step3_user_data/frf_numpy.npz` and its `schema_numpy.json` can be converted as follows:

```bash
sparseldv-qa convert examples/step3_user_data/frf_numpy.npz --schema examples/step3_user_data/schema_numpy.json --output mine.npz
```
