# Validation

samson-mri must give the masks of the article's tool. Its algorithm (`src/samson_mri/_algorithm.py`) is the code of
the SAMson notebook, and two kinds of checks confirm that the package reproduces it.

## 1. Tests against a frozen copy of the notebook (every test run)

`tests/reference_notebook.py` is the notebook's code, unchanged. `tests/test_algorithm.py` and `tests/test_pipeline.py`
check that the package gives **identical** results to it: template box shapes, box prompts (several anteroposterior
ranges, both median_otsu settings, parts of the volume), post-processing, slice-wise prediction (with a deterministic fake
SAM predictor), saved NIfTI masks, the notebook's `reorient=True`, Semi-Auto selection by a reference mask, and metrics.
Other tests cover what the package adds: re-orientation (all 48 axis orders and flips, round trips, scrambled copies of
an image give the same mask on their own grid), slice ranges, 4D inputs, the JSON specification, output files,
reloading and re-selection, the command line, and the app (Streamlit's headless app tester runs its five steps).

## 2. Real SAM ViT-H model against the published masks

`validation/validate_package.py` runs the package with SAM's ViT-H checkpoint and compares its masks voxel by voxel with
the published SAMson masks of the data deposit (DIGITAL.CSIC, doi:10.20350/digitalCSIC/17000): all 75 subjects of D1
(53), D2 (10) and D3 (12), with the article's settings (presets `article-d1`, `article-d2`, `article-d3`). Full-Auto
masks are compared with `outputs/masks/samson_full_auto`, and Semi-Auto masks (best of the 3 candidates of each slice
by the reference mask) with `outputs/masks/samson_semi_auto`.

### Results (GPU, NVIDIA RTX A6000)

| Environment | Time | Result |
|---|---|---|
| Environment of the article's analyses: Python 3.11.7, PyTorch 2.4.1 + CUDA 12.1, numpy 1.26.3, dipy 1.9.0, scikit-image 0.22.0, OpenCV 4.10.0 | 15 min | **150/150 masks identical** |
| Fresh environment with the tested versions of `requirements-lock.txt`: Python 3.10.22, PyTorch 2.5.1 + CUDA 12.4, numpy 2.2.6, dipy 1.10.0, scikit-image 0.25.0, SciPy 1.14.1, OpenCV 4.10.0 | 15 min | **150/150 masks identical** |

No voxel differs (Full-Auto and Semi-Auto, every subject of D1, D2 and D3). Per-subject results:
`validation/results_*.csv`.

### Other environments and hardware

* **Newest versions** of the dependencies at the time of the release (what `pip install samson-mri` installs without the
  lock file: PyTorch 2.14.1, dipy 1.11.0, scikit-image 0.25.2, SciPy 1.15.3, OpenCV 5.0.0, nibabel 5.4.2), run on the
  CPU (that PyTorch build needs a newer GPU driver than the test machine's): all 130 tests pass; the example subject's
  Full-Auto mask is identical to the published one; its Semi-Auto mask differs by 1 voxel (mean Jaccard index unchanged,
  0.9529).
* **CPU vs GPU**: the same 1-voxel difference appears on the CPU with the article's environment, so it comes from the
  floating-point arithmetic of the CPU, not from the library versions. On the GPU, both environments reproduce the
  published masks exactly.

To repeat the validation (about 15 min on one GPU; `<deposit folder>` contains the `datasets/` and `outputs/` folders
of the data deposit):

```bash
python validation/validate_package.py --data <deposit folder> --checkpoint sam_vit_h_4b8939.pth \
    --out validation/results_<environment>.csv
python validation/summarise.py validation/results_<environment>.csv
```

The quick version is `samson example` (or `pytest -m model`): the bundled example subject must give the published
Full-Auto and Semi-Auto masks.
