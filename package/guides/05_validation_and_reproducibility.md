# 5. Validation and reproducibility

Full report: `../source/docs/validation.md`; per-subject results: `../source/validation/results_*.csv`; script:
`../source/validation/validate_package.py` (+ `summarise.py`). Status of each item below is stated explicitly.

## 5.1 Identity with the article's masks (recorded by the package authors, GPU, RTX A6000)

The package's algorithm module (`_algorithm.py`) is the code of the SAMson notebook. The validation script ran the package
with SAM ViT-H and compared masks voxel by voxel with the published masks.

| Environment | Content | Result |
|---|---|---|
| Article environment (Python 3.11.7, PyTorch 2.4.1+cu121, numpy 1.26.3, dipy 1.9.0, scikit-image 0.22.0, OpenCV 4.10.0), 15 min | all 75 subjects of D1 (53), D2 (10) and D3 (12), Full-Auto and Semi-Auto (150 masks) | **150 / 150 identical** (0 differing voxels) |
| Fresh environment with `requirements-lock.txt` (Python 3.10.22, PyTorch 2.5.1+cu124, numpy 2.2.6, dipy 1.10.0, scikit-image 0.25.0), 15 min | the same 150 masks | **150 / 150 identical** |

Verified for this release: the two result CSV files contain 150 rows each, all with `identical = True`; the script was
run on one subject of each of D1, D2 and D3 of the data deposit (6/6 masks identical). The re-run of all tools on the
harmonized datasets is documented separately (`../../notebooks/`, `../../results/`).

Caveats recorded by the authors: (i) with the newest dependency versions at release time on the **CPU**, the example's
Full-Auto mask is identical but its Semi-Auto mask differs by 1 voxel (mean Jaccard unchanged, 0.9529); the same 1-voxel
difference appears on the CPU with the article's environment, i.e. it is CPU floating-point arithmetic, not the library
versions. (ii) Only the Linux x86-64 platform was validated.

## 5.2 Test suite (verified for this publication)

| Setting | Result |
|---|---|
| Wheel installed in a clean virtual environment (Python 3.10.22, torch 2.5.1, Streamlit 1.65.0), `pytest`, GPU | **133 passed** (every test, including the `model` test with the real ViT-H checkpoint and the command-line tests with the installed `samson` script), 110 s |
| The same, `pytest -m "not model"`, CPU, `samson` script not on the `PATH` | 131 passed, 1 skipped (`console script not installed`), 1 deselected, 85 s |
| `samson` environment (Python 3.11.7, torch 2.4.1; **no Streamlit**), `pytest -m "not model"`, CPU | 127 passed, 3 skipped (tests needing Streamlit or an installed `samson` script), 1 deselected |

The `model` test checks that the bundled example gives the published Full-Auto and Semi-Auto masks.
Content: 67 test functions in 8 files: algorithm (identity with the frozen notebook code, `tests/reference_notebook.py`),
pipeline, I/O and orientation (all 48 axis/flip combinations), spec, anteroposterior helpers, CLI, app (headless
Streamlit tester, 5 steps) and model. A fake deterministic SAM predictor is used, so no GPU or checkpoint is needed.

## 5.3 Commands checked against `--help` (verified)

All 11 help texts (`samson`, and the 10 sub-commands) were regenerated (CPU) and compared with
`../source/docs/cli_reference.md`: 10 are identical; the `samson app --address` description in the reference is outdated
(the program's help adds "...i.e. only this computer; the app can read and write files on this computer, so do not
open it to a network you do not trust"). `samson --version` prints `samson-mri 1.0.0`. `samson spec`, `samson spec --show`,
`samson preview` (bundled example, range 4-33) and `samson evaluate` (published Full-Auto mask against the manual mask:
Jaccard 0.9120, Dice 0.9508, accuracy 91.01 %, 3D Jaccard 0.9244; 16 slices) were executed successfully on CPU.

## 5.4 How to reproduce

```bash
pip install -r source/requirements-lock.txt && pip install --no-deps dist/*.whl
cd source && pytest -m "not model"                # tests, minutes, CPU
samson download-model && samson example           # real model, bundled example: both masks identical to the published ones
python validation/validate_package.py --data <deposit folder> --checkpoint sam_vit_h_4b8939.pth \
       --out validation/results_mine.csv          # identity on the 75 D1-D3 subjects (about 15 min on one GPU); then validation/summarise.py
```

Determinism: the same checkpoint (`sam_vit_h_4b8939.pth`, SHA-256 verified by `samson download-model`), settings and
software versions give the same mask; each `_samson.json` records versions, device, spec and boxes.
