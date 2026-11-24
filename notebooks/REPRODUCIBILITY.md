# Notebooks and code

| File | Content |
|---|---|
| `SAMson.ipynb` | user notebook of SAMson on the 12 scans of D3, showing the scan with the median Full-Auto Jaccard index of D3, with an orientation check against the template; its last cell checks its masks against `outputs/masks` (12/12 Full-Auto and 12/12 Semi-Auto voxel-identical). Produces no number of the article. Writes `notebooks/SAMson_outputs/` (not distributed). |
| `SAMson_experiments.ipynb` | SAMson Full-Auto, Semi-Auto, box prompts and Oracle on all 409 scans; for each dataset-contrast it shows the slice grids of the Full-Auto and Semi-Auto masks of the scan with the median Full-Auto Jaccard index; writes `results/exp_samson_{subjects,slices}.csv`. `REGENERATE = False` (default) loads the masks and boxes of `outputs/`; `True` re-runs SAM (about 70 min on the GPU) and checks that masks and boxes are identical to `outputs/`. Ends with a check against `results/subjects.csv` and `results/boxes.csv`. |
| `SAMson_benchmark.ipynb` | the nine other methods: versions, environments, commands; mask generation through `lib/` (`REGENERATE = False` by default: only missing masks are computed; `True` re-runs a tool into `work/regenerated/`); evaluation; for each tool, a 2 × 5 grid (dataset-contrasts arranged as in Fig. 5) with the central evaluated slice of the tool's median scan; writes `results/exp_tools_{subjects,slices}.csv` and `results/sherm_runtimes.csv` (SHERM run times of the 409 masks, from `outputs/sherm/`). Ends with a check against `results/subjects.csv`. |
| `SAMson_figures_and_statistics.ipynb` | every figure, table and statistic of the article's evaluation; reads the metrics written by the two notebooks above (`results/exp_samson_*.csv`, `results/exp_tools_*.csv`, `results/sherm_runtimes.csv`), the run times and environment of the full run (`results/runtimes.csv`, `results/environment.json`) and `datasets/`; writes Figs. 2, 4 and 5 (`figures/fig2`, `fig4`, `fig5`, `.pdf` and `.png`) and the tables in `tables/` (among them `samson_all_metrics.csv`, every metric of SAMson in every dataset-contrast, and `benchmark_all_metrics.csv`, every metric of every method). |
| `run_all.py`, `lib/` | command-line driver and library of the full run that produced `outputs/masks` (all methods except SHERM), `outputs/boxes`, `outputs/semi_auto_selection.csv` and `results/{subjects,slices,boxes,runtimes}.csv`, `results/environment.json` |
| `runners/` | scripts executed in the tools' own environments (deepbet, SAM2, RBM); `runners/sherm/`: SHERM (MATLAB; instructions in its README) |
| `figures/`, `tables/` | outputs of `SAMson_figures_and_statistics.ipynb` |

All paths are relative to the release folder (the notebooks use `os.path.abspath("..")`, the library the location of `lib/config.py`).

## Reproducing the results

1. Environment `samson` (Python 3.11.7; `docs/tools_and_conventions.md`): numpy 1.26.3, scipy 1.12.0, pandas 2.2.2, statsmodels 0.14.4, matplotlib, seaborn, nibabel 5.2.1, dipy 1.9.0, scikit-image 0.22.0, opencv-python, torch 2.4.1 and segment-anything 1.0 (the last two only to re-run SAMson), jupyter.
2. Optional, hours: `run_all.py` re-computes the masks of the 11 methods run on this computer (below), SHERM is run with `runners/sherm/run_sherm_samson.m`. Masks that exist are skipped, so on the released `outputs/` nothing is recomputed; run it with `--root <empty folder>` to recompute everything.
3. `SAMson_experiments.ipynb` (about 2 min) and `SAMson_benchmark.ipynb` (about 5 min): metrics of every mask.
4. `SAMson_figures_and_statistics.ipynb` (about 2 min): figures, tables and statistics.

```bash
cd notebooks
for nb in SAMson_experiments SAMson_benchmark SAMson_figures_and_statistics; do
    jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 $nb.ipynb
done
```

## External software

Only needed to re-compute masks (`run_all.py`, `REGENERATE = True` in `SAMson_experiments` and `SAMson_benchmark`), except the SAM ViT-H checkpoint, which `SAMson.ipynb` also needs. `lib/config.py` locates the software with environment variables; the defaults are the installation of the publication run. Conda environments: `docs/environments/*.yml` (`conda env create -f <file>`).

Downloads (SHA-256 of every file in `docs/tool_parameters.json`):

| What | Source | Place |
|---|---|---|
| SAM ViT-H `sam_vit_h_4b8939.pth` | `samson download-model` (package), or https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth | `SAMSON_CHECKPOINTS` folder |
| MedSAM `medsam_vit_b.pth` | model checkpoint of github.com/bowang-lab/MedSAM (link in its README) | `SAMSON_CHECKPOINTS` folder |
| SAM 2.1 `sam2.1_hiera_large.pt` | https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt | `SAMSON_CHECKPOINTS` folder |
| BEN code | `git clone https://github.com/yu02019/BEN`, `git checkout f23ed39` | `<SAMSON_THIRD_PARTY>/BEN` |
| BEN weight Mouse-T2WI-7T | pretrained weights of BEN (Google Drive link in `dataset_release/README.md` of the BEN repository); not in its git repository | `<SAMSON_THIRD_PARTY>/BEN/weight/Mouse_T2WI_7T_2022_07191047/.hdf5` |
| RBM code and model | `git clone https://github.com/CAMRIatUNC/RodentMRISkullStripping`, `git checkout dfa1a12` (model `rbm/scripts/rat_brain-2d_unet.hdf5` included) | `<SAMSON_THIRD_PARTY>/RodentMRISkullStripping` |
| ANTs 2.5.0 | github.com/ANTsX/ANTs (release v2.5.0) | `SAMSON_ANTS_BIN` |
| SHERM | `runners/sherm/README.md` | MATLAB, another computer |

| Variable | Default (publication run) | Content |
|---|---|---|
| `SAMSON_CHECKPOINTS` | `~/.cache/samson_mri` | `sam_vit_h_4b8939.pth` (SAM ViT-H; `samson download-model`), `medsam_vit_b.pth` (MedSAM), `sam2.1_hiera_large.pt` (SAM 2.1) |
| `SAMSON_CONDA_ENVS` | the `envs` folder of the conda installation that runs the code | folder of the conda environments below |
| `SAMSON_PY_SAMSON`, `SAMSON_PY_SAM2`, `SAMSON_PY_BEN`, `SAMSON_PY_DEEPBET` | `<envs>/samson/bin/python`, `<envs>/samson_sam2/bin/python`, `<envs>/ben/bin/python`, `<envs>/deepbet_1_env/bin/python` | Python of SAMson and MedSAM (and this library), SAM2, BEN and RBM, deepbet |
| `SAMSON_FSL_6074`, `SAMSON_FSL_60717` | `<envs>/fsl_6.0.7.4`, `<envs>/fsl_6.0.7.17` | FSL environments of BET and BET4Animal |
| `SAMSON_ANTS_BIN` | `$ANTSPATH`, else the folder of `antsBrainExtraction.sh` on the `PATH` | ANTs 2.5.0 |
| `SAMSON_THIRD_PARTY` | `<release>/third_party` (not distributed) | contains `BEN/` (github.com/yu02019/BEN, commit f23ed39) and `RodentMRISkullStripping/` (github.com/CAMRIatUNC/RodentMRISkullStripping, commit dfa1a12); or set `SAMSON_BEN_REPO` and `SAMSON_RBM_MODEL` (`rbm/scripts/rat_brain-2d_unet.hdf5`) |

Versions, commits, checksums and settings of every tool: `docs/tools_and_conventions.md`; parameters read by the code: `docs/tool_parameters.json`.

## The full run (`run_all.py`)

```bash
cd notebooks
touch ../.FULL_RUN_ACTIVE      # the run owns hardware threads 16-95 and GPU 0 (other jobs must not use them)
python run_all.py --cores 16-95 --gpu 0 --full-run --environment --evaluate --work ../work/full_run [--pool all]
rm ../.FULL_RUN_ACTIVE
```

* `run_all.py` runs the tools one after another (`--tools` to select; SAMson Full-Auto and Semi-Auto in one embedding pass, as in the article, with the same time recorded for both) and, for each tool, the dataset-contrasts one after another (`--dcs`), so that each (tool, dataset-contrast) batch runs alone on the allocation and its wall time is that of the tool on that dataset. ANTs (about 2–3 h per scan on one thread) processes the scans of all dataset-contrasts in one pool of 80 workers, longest first (`--pool ants`, default); its dataset wall time is then the span of that dataset's scans within the pool. The wall time of every tool's whole phase is recorded too (`level = tool`), and load snapshots of the machine at the start and end of every batch (`results/load_snapshots.jsonl`), (`docs/tools_and_conventions.md`, section 4).
* Masks that exist are never recomputed: an interrupted run is resumed by running the same command again. Every scan's and every batch's wall time is appended to `results/runtimes.csv` (readers keep the last row of a scan).
* `--work` holds the staged (re-oriented) inputs, the tools' own output files and one log per scan and tool (`<work>/logs/<tool>/<Dk>/<folder>/<stem>.log`); it is not needed afterwards.
* Test runs: `--root <test folder>` writes `outputs/` and `results/` there (`--limit N` scans per dataset-contrast, `--stems`); they stop as soon as `.FULL_RUN_ACTIVE` appears. Example: `python run_all.py --root ../work/test --cores 56-79 --limit 1 --tools samson_full_auto,samson_semi_auto,bet --evaluate`.
* Settings of the published masks: `docs/tool_parameters.json`, except five (tool, dataset-contrast) pairs with dataset-specific settings (BET on D1 and D2, BET4Animal on D2 and D3, RBM on D3; `lib/config.py`, `DATASET_SPECIFIC_SETTINGS`), and D2, whose tools receive a header voxel size of 0.1302 × 0.1302 × 0.8 mm (×10; `TOOL_INPUT_AFFINE_X10`), so that a re-run reproduces the published masks (only BET, BET4Animal, RBM and ANTs use the voxel size; every metric uses the voxel size of the dataset files). In `results/runtimes.csv`, the five pairs have the times of their run with the common settings (the masks with the dataset-specific settings were not timed separately).

## Pipeline per scan (all tools)

1. Load the dataset image (`datasets/<Dk>/<folder>/<stem>.nii.gz`, `<folder>` = `anat`, `func` or `dwi`, D1: `anat/<age>`; stored orientation, voxel sizes ×10 in the header; real voxel size = header / 10).
2. Re-orient it losslessly to the canonical orientation declared in `datasets/dataset_settings.json` (`perm`, `flips`; affine re-oriented with `inv_ornt_aff`) and write it once as float32 (`<work>/inputs/x10/...`). This file, with its ×10 header, is the input of every tool (SAMson and MedSAM read the same array in memory).
3. Run the tool (parameters: `docs/tool_parameters.json`; ANTs: D4's 6-slice images zero-padded to 8 slices and the mask cropped back; BEN and RBM, which read the slices along axis 2 without rotation, receive the canonical image with axis 1 reversed, i.e. dorsal-up slices, and their masks are reversed back; deepbet receives anatomical header labels (the direction cosines of axes 1 and 2 are exchanged for D1, D2 and D5, which are labelled in the template's convention; voxel data unchanged), all other tools the headers as stored; the labels of every scan are checked first, `lib/datasets.py: check_header_labels`).
4. Map the tool's mask back with the inverse transform and save it on the grid of the dataset image, with its header: `outputs/masks/<tool>/<Dk>/<folder>/<stem>_mask.nii.gz` (uint8 {0, 1}; the folder structure of `datasets/` is mirrored). SAMson Full-Auto also saves its box prompts (`outputs/boxes/...`), Semi-Auto its candidate selection (`outputs/semi_auto_selection.csv`).

Resources: CPU tools run as single-thread workers, one process per scan pinned to one hardware thread of the allocation (`taskset`; thread variables = 1); GPU tools run one process per batch on the GPU, pinned to the whole allocation. Per-scan time: the scan's process (CPU tools, including model loading) or the scan's processing inside the tool's process (GPU tools; model load counted in the batch time only). Re-orientation and output conversion are not timed.

SHERM (MATLAB) runs outside `run_all.py`: `runners/sherm/run_sherm_samson.m` reads the `datasets` folder and writes masks in the layout above, one JSON log per scan, `sherm_runtimes.csv` and `sherm_environment.json`. The published SHERM masks come from two runs on two other computers; their records are combined in `outputs/sherm/` (`lib/sherm.py`).

## Library (`lib/`)

| Module | Content |
|---|---|
| `config.py` | paths, external software, tool registry, tool parameters (`docs/tool_parameters.json`, parsed explicitly), `DATASET_SPECIFIC_SETTINGS`, `TOOL_INPUT_AFFINE_X10` |
| `datasets.py` | `Scan` description, scan list of the harmonized datasets, re-orientation and its inverse, staging of the tool inputs, saving masks, evaluated slices |
| `samson_core.py` | SAMson's functions copied verbatim from `SAMson.ipynb` (do not edit) |
| `tools.py` | tool runners and the batch driver (skip-if-exists, timing, per-scan logs) |
| `resources.py` | allocation, pinned workers, `runtimes.csv`, `environment.json`, load snapshots |
| `metrics.py`, `evaluation.py` | metrics of the article (slice accuracy/Jaccard/Dice, failed slices, 3D Jaccard/Dice, HD95, volume bias), box analysis, `results/{subjects,slices,boxes}.csv` |
| `sherm.py` | SHERM records: scan-level run times from the logs of `outputs/sherm/` |

`runners/rbm_unet_run_padded.py` is the padded RBM runner; `runners/sam2_run.py` and `runners/deepbet_run.py` run SAM2 and deepbet in their environments and record the per-scan wall time.

Metric conventions (identical for every method): per-slice Jaccard and Dice are rounded to 4 decimals before averaging over the slices of a scan (effect ≤ 5 × 10⁻⁵); box rectangles have end-exclusive edges.
