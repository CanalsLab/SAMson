# samson-mri 1.0.0: package, app and documentation

The software of the SAMson article: the Python package `samson-mri` (command `samson`, Python API and a Streamlit
application for brain extraction of rodent MRI with SAMson).

| Path | Content |
|---|---|
| `guides/01_installation.md` | installation on Linux, macOS and Windows (also without Python), GPU or CPU, the CUDA / driver constraint, the SAM checkpoint |
| `guides/02_quickstart_cli_api.md` | command-line and Python quick start |
| `guides/03_app_guide.md` | the five steps of the app; choice of the anteroposterior range with subject and template views; orientation page |
| `guides/04_input_output_conventions.md` | orientation, slice order, ×10 headers, specification JSON, coverage rules, outputs |
| `guides/05_validation_and_reproducibility.md` | identity with the article's masks, environments, test suite |
| `SOFTWARE_DOCUMENTATION.md` | description of the software (purpose, architecture, CLI/API, app, conventions, tests, limitations, availability) |
| `app_tutorial/` | `SAMson_app_tutorial.mp4` (video tutorial of the app, 8 min 12 s) with captions (`.srt`) and the example image used in it (`example_image/`) |
| `dist/` | `samson_mri-1.0.0-py3-none-any.whl` and its `SHA256SUMS` |
| `source/` | the released source tree (the content of the 1.0.0 source distribution): `src/samson_mri`, `tests/`, `docs/` (user guide for non-programmers `docs/user_guide.md`, CLI reference, I/O conventions, images), `examples/`, `validation/`, `pyproject.toml`, `LICENSE` (MIT), `CHANGELOG.md`, `CITATION.cff`, `requirements-lock.txt`, `environment.yml` |

## Install and first run

```bash
pip install "dist/samson_mri-1.0.0-py3-none-any.whl[app]"   # or, from PyPI once published: pip install "samson-mri[app]"
samson download-model                                      # SAM ViT-H checkpoint (2.4 GB) to ~/.cache/samson_mri
samson example                                             # segments the bundled example (D1 subject A110123)
samson app                                                 # the graphical application (tutorial: app_tutorial/)
```

Pinned environment: `conda env create -f source/environment.yml` (Python 3.10, PyTorch 2.5.1), then
`pip install --no-deps dist/samson_mri-1.0.0-py3-none-any.whl`; or `pip install -r source/requirements-lock.txt`.
Details, GPU/CPU notes and troubleshooting: `guides/01_installation.md`.

Checksums (`sha256sum -c dist/SHA256SUMS`):

```
2622b4faf745eec62dbe6ff87dee595adbb67ad52d22c63a9697c14c615eaae0  samson_mri-1.0.0-py3-none-any.whl
```

## Verified for this release

* The checksums of `dist/` match; the wheel installs in a clean virtual environment (Python 3.10.22, PyTorch 2.5.1,
  Streamlit 1.65.0) and its test suite, run from `source/tests` against the installed wheel, gives 133 passed (every test,
  including the `model` test with the SAM checkpoint on the GPU); `samson example` gives the published masks.
* `source/` is the content of the 1.0.0 source distribution (`pip install ./source` installs the same package as the wheel).
* The identity of the package's masks with the published SAMson masks of D1–D3 (150/150 in each of two environments, on
  the GPU) is recorded in `source/validation/` and `guides/05_validation_and_reproducibility.md`; the validation script
  was run for this release on one subject of each of D1, D2 and D3 of the data deposit (6/6 identical).
