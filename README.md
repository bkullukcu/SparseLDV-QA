# SparseLDV-QA

**Sensor placement and complex frequency-response reconstruction for sparse vibrometry.**

SparseLDV-QA selects spatial measurement locations from training FRFs and reconstructs the complex response at unmeasured locations. The package exposes response-based selection, geometric and QR/POD baselines, graph-regularized reconstruction, and held-out error measures. It is intended for researchers working with scanning laser Doppler vibrometry and other spatially sampled FRF data.

**Research software preview · v0.8.2 · Python 3.10–3.13 · MIT**

## Example: select points, reconstruct responses

![Synthetic sensor placement on a measurement grid](docs/figures/sensor_selection.svg)

*Ten points selected from a synthetic 36-point grid using training scans only. Numbers show the selection order, not measurement priority or experimental quality.*

![Complex FRF reconstruction at an unmeasured position](docs/figures/frf_reconstruction.svg)

*Held-out synthetic FRF. The reference and reconstruction are compared at a representative unmeasured position; global NRMSE is evaluated only at unmeasured points. This is a numerical demonstration, not an experimental benchmark.*

Both figures are generated using the actual package API. Reproduce them with `python examples/make_readme_figures.py`. The generator uses fixed synthetic inputs and reads no laboratory data.

## Install and run

Clone this repository and run the following commands from its root directory:

```bash
python -m pip install -e .
python examples/quickstart.py
```

The quickstart runs a known-truth synthetic reconstruction and prints the NRMSE. To run the test suite:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

## Import your FRFs

Supply either a CSV measurement table or an NPZ archive, with a JSON schema describing coordinate columns, frequencies, units and complex representation.

```bash
sparseldv-qa convert examples/step3_user_data/frf_long.csv \
  --schema examples/step3_user_data/schema.json \
  --output example_frfs.npz

sparseldv-qa example_frfs.npz --count 2 \
  --output example_reconstruction.npz
```

The converter checks the frequency grid and spatial sample identifiers before writing a normalized NPZ and a provenance manifest. Complex data may be stored as real/imaginary parts or as magnitude/phase with an explicit phase convention. See [Data import](docs/USER_DATA_IMPORT.md) for the input contract and the NumPy example.

## Method in brief

For observed FRFs $\mathbf{H}_{\mathrm{obs}}$ at selected spatial positions, reconstruction solves

$$
\left(\mathbf{D}+\alpha\mathbf{L}+\varepsilon\mathbf{I}\right)\widehat{\mathbf{H}}=\mathbf{B},
$$

where $\mathbf{L}$ is a graph Laplacian derived from spatial coordinates, $\mathbf{D}$ contains observation weights at the measured positions, and $\mathbf{B}$ contains the weighted measured complex FRFs. The regularization parameter $\alpha$ controls spatial smoothing. Sensor selection and parameter tuning must use training data only.

The evaluation routine reports the complex-response NRMSE and median FRAC on **unmeasured** positions. These metrics describe agreement with a held-out reference; they are not a guarantee that sparse measurements will be sufficient for another specimen.

## Scope

- Input: complex FRFs, coordinates, and (optionally) measured coherence.
- Output: selected point indices, reconstructed complex FRFs, and comparison metrics.
- Limit: results depend on the training structures, excitation conditions, measurement quality and chosen reconstruction parameters. The public repository does not contain experimental recordings or source-derived benchmark tables.

See [Research notes](docs/RESEARCH_NOTES.md) for validation boundaries and [FAQ](docs/FAQ.md) for common import questions. For a more detailed training-group split with your own measurements, see [`examples/benchmark_real_frf.py`](examples/benchmark_real_frf.py).

## Reproduce the figures

```bash
python examples/make_readme_figures.py
# Optional: use an installed LaTeX distribution to typeset figure labels
python examples/make_readme_figures.py --latex
```

The two vector SVGs are kept under `docs/figures/`; the plotting code is in `examples/`. No additional plotting data are needed.

## Citation and license

See [`CITATION.cff`](CITATION.cff) for the software citation and [`LICENSE`](LICENSE) for the code license. Cite any external measurement dataset separately. This preview has not been independently validated across laboratories, and its API may change before 1.0.
