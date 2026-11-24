# SAMson (samson-mri)

**SAMson** (*SAM for Segmentation Of Neuroimages*) extracts the brain from rodent MRI with the Segment Anything Model
(SAM). This package turns the tool of the article into an installable Python package with:

* an **interactive app** (in the web browser) to set up a dataset: orientation check, **choice of the anteroposterior
  range on side views of a subject and of the template**, segmentation, and review of the masks (including the
  slice-by-slice choice of the Semi-Auto mode);
* a **command line tool**, `samson`, for batches of images;
* a **Python API** (`from samson_mri import SAMson`);
* explicit **input/output conventions**, a bundled **example** subject, and **tests**.

The algorithm is the one of the article, unchanged: on the article's 75 subjects the package
gives the published masks voxel for voxel (see [docs/validation.md](docs/validation.md)).

> Panadero Soler D, Kotb Selim M, Martínez-Tazo P, Muñoz-Moreno E, Ramos-Cabrer P, López-Larrubia P, De Santis S,
> Canals S, Pertusa A. *SAMson: an automated brain extraction tool for rodents using SAM.*
> [doi:10.1101/2024.03.07.583982](https://doi.org/10.1101/2024.03.07.583982). Data: [doi:10.20350/digitalCSIC/17000](https://doi.org/10.20350/digitalCSIC/17000).

New to Python? Read the step-by-step **[user guide](docs/user_guide.md)**. Other documents:
[command line reference](docs/cli_reference.md), [input/output conventions](docs/io_conventions.md), [choosing the anteroposterior range](docs/anteroposterior_range.md),
[validation](docs/validation.md), [changelog](CHANGELOG.md).

---

## Installation

You need Python 3.9–3.12 and, for reasonable speed, an NVIDIA GPU. SAMson also runs on the CPU, with the same masks but
slower: about 9 s per slice with 16 CPU threads (2.5 min for the 16-slice example), against a fraction of a second on a GPU.

```bash
conda create -n samson python=3.10 -c conda-forge      # a clean environment (or python -m venv samson)
conda activate samson
pip install "samson-mri[app]"                          # from PyPI, once published; or, from this folder:
pip install ".[app]"                                   # (or the release file: pip install "dist/samson_mri-1.0.0-py3-none-any.whl[app]")
samson download-model                                  # SAM's ViT-H checkpoint, 2.4 GB, once
samson example                                         # check the installation (about 1 minute on a GPU)
```

**GPU note.** `pip` installs the newest PyTorch, which may need a newer NVIDIA driver than yours (then SAMson silently runs on
the CPU; `samson info` tells you). In that case install first a PyTorch build for your driver, e.g.
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` (see https://pytorch.org/get-started/locally),
or the tested versions below.

To reproduce the article's environment exactly, install the tested versions first:
`pip install -r requirements-lock.txt` (or `conda env create -f environment.yml`).

## Quick start

### 1. The app

```bash
samson app --folder /path/to/my/images
```

and open the address it prints (http://localhost:8501) in a web browser. The app guides you through five steps:

| Step | What you do |
|---|---|
| 1 · Images | choose the folder of a dataset and a representative image |
| 2 · Orientation | check that SAMson sees coronal slices, dorsal side up, caudal end first (fix it with a few clicks if not) |
| 3 · Anteroposterior range | slide the range until each subject slice and its matched template slice show the same anatomy (side views + coronal pairs + box prompts) |
| 4 · Segment | Full-Auto or Semi-Auto, one image or the whole folder |
| 5 · Review | check the masks; in Semi-Auto, pick the best of SAM's candidates for any slice; save |

The settings are saved as `samson_spec.json` with the results, and the app shows the equivalent command line.

### 2. The command line

```bash
# Check the anteroposterior range on a representative image (writes a figure; no GPU needed)
samson preview sub-01_T2w.nii.gz --template 4 33

# Segment a folder (Full-Auto)
samson segment /data/T2w/ -o /data/T2w_masks --template 4 33

# ...or with the settings saved by the app, in Semi-Auto mode
samson segment /data/T2w/ -o /data/T2w_masks --spec samson_spec.json --mode semi-auto
samson select /data/T2w_masks/sub-01_T2w_samson.json --choices F,F,2,1,F,...   # F = Full-Auto mask; or use the app's Review step

# Compare with manual masks
samson evaluate --masks /data/T2w_masks/sub-01_T2w_mask.nii.gz --references manual/sub-01_mask.nii.gz
```

`samson --help` lists all commands; `samson <command> --help` their options.

### 3. Python

```python
from samson_mri import SAMson, DatasetSpec

spec = DatasetSpec(template=(4, 33))                 # + orientation, slices, otsu, volume (see docs/io_conventions.md)
samson = SAMson()                                     # SAM ViT-H, on the GPU if there is one
result = samson.segment("sub-01_T2w.nii.gz", spec)   # Full-Auto; mode="semi-auto" for the candidates
result.save("masks/")                                 # masks/sub-01_T2w_mask.nii.gz, _samson.json, _qc.png
mask = result.mask                                    # numpy array on the grid of the input image
```

## What SAMson needs to know about a dataset

All of it fits in a small JSON file (`samson_spec.json`), written by the app or by `samson spec`:

| Setting | Meaning | Default |
|---|---|---|
| `template` | template slices (0–39, caudal to rostral) that match the first and last processed slice | **required** |
| `slices` | first and last slice to process (0-based, SAMson orientation; -1 = last) | all slices |
| `orientation` | how the input axes map to SAMson's (left-right, ventral-dorsal, caudal-rostral) | as stored |
| `otsu` | median_otsu parameters | `auto` (rule of the article) |
| `volume` | volume of 4D images to use | 0 (first) |

The anteroposterior range matters most: SAMson places its box prompts using the template slices that match each image
slice. See [docs/anteroposterior_range.md](docs/anteroposterior_range.md) and the app's step 3.

## Outputs

For each input `<name>.nii[.gz]`, in the output folder:

| File | Content |
|---|---|
| `<name>_mask.nii.gz` | brain mask, uint8 0/1, on the voxel grid and with the header of the input |
| `<name>_candidates.nii.gz` | Semi-Auto only: 4D, volume 0 = Full-Auto mask, volumes 1–3 = SAM's three candidates |
| `<name>_samson.json` | settings, box prompts, Semi-Auto selection, versions, timings (everything needed to reproduce the mask) |
| `<name>_qc.png` | quality-control figure: every processed slice with its mask and box |
| `samson_summary.csv`, `samson_spec.json` | batch summary and the specification used |

Details: [docs/io_conventions.md](docs/io_conventions.md).

## Tests

```bash
pip install ".[app,test]"
pytest                    # 131 tests, about 1.5 minutes; no GPU or checkpoint needed (a fake SAM predictor is used)
pytest -m model           # with the real ViT-H checkpoint: the example must give the published masks exactly
```

## Licence and citation

Code: MIT licence ([LICENSE](LICENSE)). Bundled data (template and example subject): SAMson's public data,
doi:10.20350/digitalCSIC/17000. SAM: Apache 2.0 (Meta AI). Please cite the article ([CITATION.cff](CITATION.cff)).
