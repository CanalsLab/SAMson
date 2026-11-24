# 1. Installation

Package: `samson-mri` 1.0.0 (command `samson`; Python import name `samson_mri`). Python 3.9 to 3.12. The release files
are in `../dist/` (wheel and source archive, checksums in `../dist/SHA256SUMS`).

SAMson needs SAM's ViT-H checkpoint (2.4 GB, downloaded once, not shipped with the package) and PyTorch. An NVIDIA GPU
is recommended; the CPU gives the same masks but is slower (see section 5).

## 1. A machine without Python (all platforms)

Install **Miniforge** (a minimal conda distribution, free), from https://github.com/conda-forge/miniforge .

* **Linux / macOS**: download the installer for your system and run `bash Miniforge3-*.sh`; open a new terminal.
* **Windows**: run the `.exe` installer and open the *Miniforge Prompt* (use it for every command below).

Check: `conda --version` prints a version. (Anaconda or Miniconda work equally; if Python 3.9-3.12 and `venv` are
already available, `python -m venv samson` followed by activating it is an alternative to conda.)

## 2. Install SAMson

```bash
conda create -n samson python=3.10 -c conda-forge     # a clean environment
conda activate samson                                 # repeat in every new terminal
# either from the release file (offline copy of this folder):
pip install "<path to>/package/dist/samson_mri-1.0.0-py3-none-any.whl[app]"
# or from the source tree:   pip install "<path to>/package/source[app]"
# or from PyPI, once published:  pip install "samson-mri[app]"
samson download-model                                 # SAM ViT-H checkpoint, once (about 2.4 GB)
samson example                                        # self-test on the bundled mouse (see below)
```

`[app]` adds Streamlit (the app). Without it only the command line and the Python API are installed. On Windows, quote
the argument as shown; on macOS with zsh the quotes are also needed.

`samson download-model` writes `sam_vit_h_4b8939.pth` to `~/.cache/samson_mri/` (checksum verified). An existing
checkpoint can be used with `--checkpoint FILE` or the environment variable `SAMSON_CHECKPOINT`. The machine needs
network access only for this download (and for `pip`).

`samson example` segments the bundled example (mouse A110123 of dataset D1) in Full-Auto and Semi-Auto mode and must
report that both masks are identical to the published ones (GPU; see 05 for the CPU caveat of 1 voxel in Semi-Auto).

## 3. Exactly the tested versions (reproducibility)

```bash
conda env create -f <path to>/package/source/environment.yml      # Python 3.10, torch 2.5.1 (CUDA 12.4 build) and pins
conda activate samson && pip install --no-deps <path to>/package/dist/samson_mri-1.0.0-py3-none-any.whl
# alternatively
pip install -r <path to>/package/source/requirements-lock.txt && pip install --no-deps <wheel>
```


## 4. GPU and CPU notes; the CUDA / driver constraint

* `pip install samson-mri` installs the **newest** PyTorch wheel. A recent PyTorch is built against a recent CUDA
  toolkit and needs an NVIDIA driver at least as new; with an older driver PyTorch silently falls back to the CPU.
  Typical symptom: `samson info` prints "Device: cpu. An NVIDIA GPU seems to be present, but this PyTorch build ... cannot
  use it". The reference machine (driver 535, CUDA 12.2) is an example: PyTorch wheels built for CUDA 12.1 or 12.4 (the
  `cu121`, `cu124` wheels) run on it; wheels for a newer CUDA do not.
* Remedy: install a PyTorch build matching your driver **before** installing SAMson, for instance
  `pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121` (choose from
  https://pytorch.org/get-started/locally), then install SAMson. Alternatively update the driver.
  The same message also appears when no GPU is visible to the process (e.g. `CUDA_VISIBLE_DEVICES` empty).
* Force the device with `--device cpu | cuda | cuda:1 | mps` (CLI) or `SAMson(device=...)` (API). Apple-silicon
  (`mps`) is accepted by the code but was **not tested** for this release.
* Memory: ViT-H needs a few GB of GPU memory; `CUDA out of memory` means another program uses the GPU.
* CPU-only machines: install the CPU wheel first (`pip install torch torchvision --index-url
  https://download.pytorch.org/whl/cpu`); the masks are the same as on the GPU, with the exception noted in guide 05.

## 5. Platforms

Developed and validated on Linux x86-64 (Ubuntu). The package is pure Python (`py3-none-any`), and its test suite
(CPU, a fake SAM predictor) passes on Linux (guide 05). Windows and macOS: installation instructions above are
standard, but the package has **not been tested** on them for this release.

## 6. Run time and hardware (as documented)

* GPU (NVIDIA RTX A6000): median 8.6 s per article subject (mean 12 s) in the validation runs, including loading
  overheads (`../source/validation/results_*.csv`, column `seconds`); the bundled example takes about 1 minute in total
  including model loading.
* CPU: about 9 s per slice with 16 threads (README of the package), i.e. about 2.5 minutes for the 16-slice example.
* Full-Auto and Semi-Auto cost the same (SAM predicts one image embedding per slice; Semi-Auto keeps the 3 candidates).
* Full benchmark timings under identical resources are reported separately in `../../results/runtimes.csv`.

## 7. Uninstall / troubleshooting

`pip uninstall samson-mri`; `conda env remove -n samson`; delete `~/.cache/samson_mri` to remove the checkpoint.
Common problems and fixes: `source/docs/user_guide.md`, section 6.
