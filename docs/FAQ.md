# FAQ

**Do I need a complete spatial FRF scan to use this package?**

Point selection is trained on complex FRFs from earlier scans. For reconstruction, only the selected locations are required. Evaluation of reconstruction accuracy needs reference FRFs at the unmeasured locations; without a reference there is no measured NRMSE.

**Which FRF representation can I import?**

Use real/imaginary components or linear magnitude and phase. Declare the phase unit and convention explicitly in the JSON schema. A magnitude-only spectrum is insufficient for complex reconstruction.

**Does sensor selection use the test scan?**

It must not. Fit the selector with training scans, then reconstruct the held-out scan using its selected locations. If measurements are grouped by specimen or mounting configuration, keep complete groups out of training.

**Are the README plots experimental results?**

No. They come from generated two-resonance complex FRFs. Regenerate them with `python examples/make_readme_figures.py`. They illustrate the workflow but are not evidence of measurement-time savings or accuracy on a new specimen.

**What is required to reproduce a published experimental comparison?**

The source dataset, geometry, complex FRF processing settings, training/test split, selection parameters, and permission to reuse those measurements. These experimental datasets are not distributed here.
