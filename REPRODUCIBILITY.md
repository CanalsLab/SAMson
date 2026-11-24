# SAMson: data, masks, code and software of the article

Release of the study *SAMson: An automated brain extraction tool for rodents using SAM* (Computer Methods and Programs in
Biomedicine). It contains what is needed to

1. reproduce the published evaluation (Tables 1–3, Figs. 2, 4 and 5 and every number of the text) from the masks;
2. use the curated data: the harmonised images of the seven datasets (D1–D7; 10 dataset-contrasts, 409 scans) with their
   reference masks, and the published masks of SAMson and of the nine other methods;
3. install and use SAMson (`samson-mri` 1.0.0: command line, Python API and application, with user guides and a video
   tutorial).

Every path is relative to this folder. The code (GitHub) and the data (DIGITAL.CSIC) are published separately: to
reproduce the results, download the DIGITAL.CSIC deposit and place its `datasets/` and `outputs/` folders in this folder,
next to `notebooks/`. Nothing else outside this folder is read to reproduce the results.

Licence: code MIT (`LICENSE`); data: terms of the DIGITAL.CSIC record. Citation: `CITATION.cff`.

## Folder map

| Folder | Size | Content |
|---|---|---|
| `datasets/` | 518 MB | D1–D7: images and reference masks (`<Dk>/<anat\|dwi\|func>/[<age>_month/]` and `manual_masks/`), the SAMson template, `participants.tsv` (one row per scan), `datasets.tsv` (one row per dataset-contrast), `dataset_settings.json` (orientation, SAMson anteroposterior range, median_otsu, evaluated slices), `acquisition.tsv` (acquisition parameters of Table 1). See `datasets/README.md`. |
| `outputs/` | 35 MB | `masks/<method>/`: the 409 masks of each of the 12 methods (SAMson Full-Auto, Semi-Auto and Oracle, SAM2 and MedSAM with SAMson's boxes, BET, BET4Animal, deepbet, BEN, RBM, ANTs, SHERM), on the grid and with the header of the dataset image; `boxes/`: SAMson's box prompts; `semi_auto_selection.csv`; `sherm/`: logs, environments and run times of the two SHERM runs. See `outputs/README.md`. |
| `results/` | 23 MB | metrics of every mask: `subjects.csv`, `slices.csv`, `boxes.csv` (written by `run_all.py --evaluate`), `exp_samson_*.csv` (`SAMson_experiments.ipynb`), `exp_tools_*.csv` and `sherm_runtimes.csv` (`SAMson_benchmark.ipynb`); run times (`runtimes.csv`) and computing environment (`environment.json`) of the full run |
| `notebooks/` | 38 MB | `SAMson.ipynb` (SAMson user notebook), `SAMson_experiments.ipynb`, `SAMson_benchmark.ipynb` and `SAMson_figures_and_statistics.ipynb` (results), `run_all.py`, `lib/` and `runners/` (mask generation), `figures/` and `tables/` (outputs of `SAMson_figures_and_statistics.ipynb`). See `notebooks/README.md` and `notebooks/REPRODUCIBILITY.md`. |
| `docs/` | <0.1 MB | `tools_and_conventions.md` (input conventions, tool versions, settings, environments and run protocol), `tool_parameters.json` (tool parameters, read by the code), `tools.bib` (references of the tools), `environments/` (conda environment files) |
| `package/` | 82 MB | `samson-mri` 1.0.0: `source/`, `dist/` (wheel, `SHA256SUMS`), `guides/`, `SOFTWARE_DOCUMENTATION.md`, `app_tutorial/` (video tutorial, captions, example image). See `package/README.md`. |

`MANIFEST.tsv`: relative path, size (bytes) and SHA-256 of every file of the release.

## Where each part is published

| Destination | Content |
|---|---|
| GitHub (code repository, https://github.com/CanalsLab/SAMson) | `package/source/` (the package repository), `package/dist/` and `package/app_tutorial/` (release assets; the 74 MB video is best attached to the release or stored with Git LFS), `package/guides/`, `package/SOFTWARE_DOCUMENTATION.md`; `notebooks/` (code, notebooks, `figures/`, `tables/`); `results/` (CSV and JSON); `docs/`; `README.md`, this `REPRODUCIBILITY.md`, `LICENSE`, `CITATION.cff`, `.gitignore` |
| DIGITAL.CSIC (data repository of the article, doi:10.20350/digitalCSIC/17000) | `datasets/` (images, reference masks, template, tables, acquisition parameters, README); `outputs/` (masks of every method, SAMson boxes, Semi-Auto selection, SHERM records) |

Data availability and terms of each dataset: `datasets/README.md` (section "Licence").

## Environments

* Results (notebooks `SAMson_experiments`, `SAMson_benchmark`, `SAMson_figures_and_statistics`) and the user notebook
  `SAMson.ipynb`: conda environment `samson` (`conda env create -f docs/environments/samson.yml`), Python 3.11.7 with numpy 1.26.3, scipy 1.12.0, pandas 2.2.2,
  statsmodels 0.14.4, matplotlib, seaborn, nibabel 5.2.1, dipy 1.9.0, scikit-image 0.22.0, opencv-python 4.10, jupyter
  (and torch 2.4.1 with segment-anything 1.0 to re-run SAMson). Exact versions: `results/environment.json`,
  `docs/tools_and_conventions.md`.
* Masks of the other methods (only to re-compute them): one environment per tool, `docs/environments/`
  (`samson_sam2.yml`, `deepbet_1_env.yml`, `ben.yml`, `fsl_6.0.7.4.yml`, `fsl_6.0.7.17.yml`), plus ANTs 2.5.0 and SHERM
  (MATLAB). Model weights and third-party code to download, and how to point the code to them (environment variables):
  `notebooks/REPRODUCIBILITY.md`, section "External software".
* The SAMson package: `package/guides/01_installation.md` (`pip install` of the wheel, or `source/environment.yml`).

## Reproduction order

1. `datasets/` (from DIGITAL.CSIC) is the input; `MANIFEST.tsv` gives the size and SHA-256 of every file.
2. Optional (hours; GPU and the tools' environments): `notebooks/run_all.py` re-computes the masks of `outputs/`
   (SHERM: `notebooks/runners/sherm/run_sherm_samson.m`). Masks that exist are skipped; use `--root <empty folder>` to
   recompute them elsewhere and compare. A re-run reproduces the published masks voxel for voxel (checked on SAMson,
   BET and BET4Animal for D1–D3 scans, including the five pairs with dataset-specific settings).
3. `notebooks/SAMson_experiments.ipynb` (about 2 min): SAMson metrics → `results/exp_samson_*.csv`.
4. `notebooks/SAMson_benchmark.ipynb` (about 5 min): metrics of the nine other methods → `results/exp_tools_*.csv`,
   SHERM run times → `results/sherm_runtimes.csv`.
5. `notebooks/SAMson_figures_and_statistics.ipynb` (about 2 min): figures, tables and statistics.

```bash
cd notebooks
for nb in SAMson_experiments SAMson_benchmark SAMson_figures_and_statistics; do
    jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 $nb.ipynb
done
```

## Manuscript items

All of them are produced by `notebooks/SAMson_figures_and_statistics.ipynb` (section in brackets); tables are written to
`notebooks/tables/` and figures to `notebooks/figures/`.

| Manuscript | Section | Output |
|---|---|---|
| Table 1 (datasets) | Data, Table 1 | `datasets.csv` (from `datasets/datasets.tsv`, `datasets/acquisition.tsv`) |
| Table 2 (SAMson in every dataset-contrast) | 2.7 | `samson_summary.csv`; every metric of SAMson Full-Auto, Semi-Auto, Oracle and boxes with SD and t-based 95% CI: `samson_all_metrics.csv` |
| Section 3.3 (benchmark: 90 paired comparisons of the Jaccard index, D1 at 3 m.o.a. and D2–D7; every metric of every method; tests of 3D Dice and HD95) | 2.6 | `benchmark_jaccard_tests.csv`, `benchmark_all_metrics.csv` (the full benchmark table, mean and t-based 95% CI), `benchmark_dice3d_hd95_tests.csv`, `slice_position.csv` |
| Fig. 2 | 1. Figures | `fig2.pdf`, `.png` |
| Fig. 4 | 1. Figures | `fig4.pdf`, `.png` |
| Fig. 5 | 1. Figures | `fig5.pdf`, `.png` |
| Figs. 1 and 3 | – | not produced by code (diagram; example slices); the Jaccard values of the Fig. 3 examples are printed (section 3) |
| Semi-Auto vs Full-Auto; object detector (D2 vs D3, box–mask slopes); age effect; Oracle comparisons | 2.1–2.5 | `semi_vs_full.csv`, `d2_vs_d3.csv`, `box_mask_slopes.csv`, `slope_contrasts.csv`, `age_*.csv`, `oracle_comparisons.csv` |
| Contrasts (D4, D6) and resolution (D1 3 m.o.a., D6 T2w, D7) | 2.2, 2.7 | `contrasts_d4_pairs.csv`, `contrasts_d6.csv`, `resolution.csv`, `resolution_pairs.csv` |
| Run times, hardware and software (Methods) | 2.8 | `runtime.csv` and the printed environment |
| Counts of evaluated slices, caption counts of Figs. 2 and 4 | 3 | printed |

## Notes

* **Voxel sizes ×10.** Images and masks store voxel sizes multiplied by 10 in the header (real size = header / 10), the
  input convention of every tool; metrics in mm use the real voxel size.
* **D2 input of the tools.** The methods run by `run_all.py` received the D2 images with a header voxel size of
  0.1302 × 0.1302 × 0.8 mm (×10); `notebooks/lib/config.py` (`TOOL_INPUT_AFFINE_X10`) keeps this header so that a re-run
  reproduces the published D2 masks. Only BET, BET4Animal, RBM and ANTs use the voxel size. SHERM and every metric use
  the voxel size of the dataset files (0.078125 × 0.078125 × 0.7 mm).
* **Dataset-specific settings.** Five (method, dataset) pairs, for which the common settings gave less accurate masks,
  use dataset-specific settings: BET on D1 and D2 and BET4Animal on D2 and D3 (start centre and thresholds; BET4Animal
  with its mouse scaling on the real voxel size), RBM on D3 (axis 1 not reversed); `notebooks/lib/config.py`,
  `DATASET_SPECIFIC_SETTINGS`. Every other mask uses the common settings of `docs/tool_parameters.json`.
* **Evaluated slices.** Every slice with brain in the reference mask, in every dataset and for every method. The animal is
  the statistical unit (scans of one animal averaged first).
* **SHERM** ran on two other computers (run 1: D1, D3–D6; run 2: D2 and D7); its records are in `outputs/sherm/`.
* **Records of the run:** `results/runtimes.csv` (wall time of every scan, dataset and tool) and
  `results/environment.json` (hardware, software and parameters of the run).
