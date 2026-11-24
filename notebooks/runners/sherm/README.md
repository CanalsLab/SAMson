# SHERM on the SAMson datasets (MATLAB)

`run_sherm_samson.m` runs SHERM (Liu et al., 2020) on every scan of the `datasets` folder and writes the masks in the
layout used for all other tools of the benchmark:

```
<OUTPUTS>/masks/sherm/<Dk>/<anat|dwi|func>[/<age>]/<stem>_mask.nii.gz
```

Each mask is a 0/1 image on the grid of the dataset image and with its header.

**Published SHERM records.** The masks of the publication (`outputs/masks/sherm/`) come from two runs on two computers: run 1 (D1, D3–D6; MATLAB R2021a) and run 2 (D2 and D7; MATLAB R2023b). `outputs/sherm/` holds the JSON log of each of the 409 masks (`logs/<Dk>_<stem>.json`, as written by this script), the environment of each run (`environment_run1.json`, `environment_run2.json`) and the scan-level run times (`sherm_runtimes.csv`, with the run in column `sherm_run`).

## What you need

* MATLAB R2017a or later, with the Image Processing Toolbox and the Statistics and Machine Learning Toolbox. The Parallel Computing Toolbox is optional; with it, several scans run at the same time.
* SHERM: `git clone https://github.com/liu-yikang/SHERM-rodentSkullStrip`, then `git checkout c44afe8` (commit of 17 February 2021).
* VLFeat 0.9.21: the binary package `vlfeat-0.9.21-bin.tar.gz` from vlfeat.org. If `vl_version` fails in MATLAB (for example on Apple-silicon Macs), compile it with `vl_compile`.
* NIfTI_tools: MATLAB File Exchange no. 8797, "Tools for NIfTI and ANALYZE image" (`load_untouch_nii`, `save_untouch_nii`).
* The `datasets` folder exactly as received. Do not change, convert or re-save any image.

## How to run

1. Open `run_sherm_samson.m` and set the paths of SHERM, VLFeat and NIfTI_tools in its SETTINGS block (`DATASETS` and `OUTPUTS` default to `datasets/` and `work/sherm/` of this release). Set `N_WORKERS` to the number of physical cores you want to use, and `MEMORY_GB` to the RAM the run may use.
2. **Pilot.** With `MODE = 'pilot'`, type `run_sherm_samson` in the MATLAB command window. This runs 10 scans: one per dataset and contrast. Its files go to `OUTPUTS/pilot/`. Check the orientation and the masks there (`OUTPUTS/pilot/sherm_inputs/` holds the exact images given to SHERM).
3. **Full run.** Set `MODE = 'all'` and run it again: 409 scans. The pilot scans are run again here, so that every dataset is timed in one go.
   * Keep `N_WORKERS` and `MEMORY_GB` unchanged for the whole full run.
   * If possible, run nothing else heavy on the computer meanwhile: the times are reported in the article.
   * The run can be stopped and restarted at any time. Scans whose mask already exists are skipped, and scans that failed are tried again.
4. The `OUTPUTS` folder then holds:
   * `masks/sherm/`: the masks;
   * `sherm_runtimes.csv`: times and resources per scan, per dataset and per run (next section);
   * `sherm_environment.json`: the computer and software (next section);
   * `logs/`: one log per scan, and one environment file per execution.

Do not change `ANIMAL` or `ISOTROPIC`, the SHERM code, or the order of the steps. These settings are fixed so that SHERM's results can be compared with those of the other tools.

## Running only some groups

`ONLY` restricts the run to some groups, in the order given, e.g. `ONLY = {'D7_T2w'}`. Masks that exist are skipped as
usual. With `ONLY = {}` (default) every group runs, D7 last.

D7 needs much more time and memory than the other datasets: SHERM resamples the slices to the in-plane voxel size
(0.052 mm: 16 slices become about 164) and filters 70 copies of this volume, about 35 GB per scan, and an estimated 10 h per
scan on one thread (D4 T2w: about 11 min). Run D7 only on a computer with at least 48 GB of free RAM per scan running at
the same time (`MEMORY_GB` at least 35), or not at all; with less memory the computer swaps and the run may not finish.

## What is recorded

The script records the following by itself. Nothing has to be noted by hand.

**`sherm_environment.json`** (one copy per execution in `logs/sherm/`):

* the computer: operating system, CPU model, physical cores and logical CPUs, RAM, and the GPUs present;
* the software: MATLAB version, toolboxes, VLFeat version and SHERM commit;
* the resources used: workers requested and started, one thread per worker, memory limit, and the time to start the workers;
* the settings, and the start, end and elapsed time of the execution.

SHERM has no GPU code: it runs on the CPU only. The GPUs are listed only to describe the computer.

**`sherm_runtimes.csv`**, with three kinds of row (`level`):

| `level` | One row per | `seconds` |
|---|---|---|
| `scan` | scan | wall-clock time of the scan |
| `dataset` | dataset-contrast | wall-clock time while at least one of its scans was running |
| `run` | run | the same, over all its scans |

In `dataset` and `run` rows, scans that run at the same time are counted once, and pauses between a stop and a restart are not counted. The other columns are:

* `cpu_seconds`: CPU time used (summed over scans);
* `peak_memory_gb`: peak memory of the MATLAB process during the scan, MATLAB itself included (Linux only; the largest scan for `dataset` and `run` rows);
* `workers`: scans that ran at the same time; `threads_per_worker`: 1; `gpu`: none;
* `load_1min`, `available_memory_gb`: load of the whole computer at the end of each scan (load on Linux and macOS, free memory on Linux and Windows), which shows whether other jobs were running;
* `status`, `run_id` (which execution), `matlab`, and the SHERM settings.

## What the script does with each scan

1. It reads the image as stored. The voxel sizes in the stored header are 10 × the real ones.
2. It re-orients the image without interpolation to the orientation used for every tool: the 3rd axis holds the coronal slices, from caudal to rostral. The 2nd axis is reversed so that the dorsal side is on the first row, as in SHERM's own example images.
3. It writes a temporary copy with the real voxel size (header ÷ 10). SHERM's parameters are fixed in millimetres:
   * kernel radii of 0.2–0.7 mm;
   * an expected brain volume of 300–550 mm³ for `'mouse'` and 1500–1900 mm³ for `'rat'`.
4. It calls `sherm(input, output, descriptor, animal, 0)`. All datasets have slices thicker than the in-plane voxel size, so SHERM runs in its anisotropic mode.
5. It maps the mask back to the stored orientation and saves it with the header of the dataset image.

Status values in `sherm_runtimes.csv`:

| Status | Meaning |
|---|---|
| `ok` | A mask was saved. |
| `no_candidate` | No region met SHERM's volume and shape criteria. An empty mask is saved; this is a result, not an error. |
| `out_of_memory` or `error` | No mask was saved. The message column says why, and the scan is tried again on the next run. |

The run stops with a message if a toolbox, function or licence is missing.

## Time and memory

Each scan runs on one thread. The memory needed per scan is estimated before each dataset is processed, and only as many scans run at the same time as `MEMORY_GB` allows:

* about 35 GB per D7 scan (384 × 384 matrix, 16 slices);
* about 10 GB per D2 scan;
* 4 GB or less for the other datasets.

With `MEMORY_GB = 64`, D7 therefore runs one scan at a time.

## Data

Data availability and terms of use: `datasets/README.md`.
