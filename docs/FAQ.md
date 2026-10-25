# Frequently asked questions

**How much training data is needed for sensor selection?**

Use complex FRFs from earlier scans with known spatial coordinates. Reconstruction then needs measurements at the selected locations. To calculate NRMSE at withheld positions, include a separate reference scan covering those positions.

**Which FRF formats are supported?**

Supply real and imaginary components, or linear magnitude and phase with a declared phase unit and sign convention. The complex reconstruction uses both amplitude and phase.

**How should I split a dataset?**

Fit selection and reconstruction parameters on training scans. Set aside the evaluation scan before fitting. When the experiment involves multiple specimens or mountings, use specimen- or configuration-level groups.

**Where do the README figures come from?**

`python examples/make_readme_figures.py` generates two-resonance complex FRFs and passes them through the package's selection and reconstruction functions. The plots show the numerical workflow on known synthetic responses.

**How do I reproduce an experimental comparison?**

Collect the source measurements, spatial coordinates, complex FRF processing settings, selection budget and train/test assignments. The full procedure is outlined in [Research notes](RESEARCH_NOTES.md).
