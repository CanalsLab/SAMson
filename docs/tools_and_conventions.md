# Tools, settings and conventions of the evaluation

How the masks of `outputs/masks/` were produced: the input given to every method, the version and settings of every
tool, the computing environment and the run protocol. The parameters read by the code are in `tool_parameters.json`;
references of the tools are in `tools.bib`.

## 1. Image conventions

* **×10 header.** Every image, mask and the template store voxel sizes multiplied by 10 in the header (pixdim and
  affine, sform = qform); real voxel size = header / 10. Metrics in mm use the real voxel size.
* **Canonical orientation.** SAMson's array convention, shared by the template: axis 0 left–right, axis 1 ventral →
  dorsal, axis 2 coronal slices from caudal (slice 0) to rostral, so that `numpy.rot90` of a slice shows the dorsal side
  up. Each dataset declares a lossless re-orientation `(perm, flips)` in `datasets/dataset_settings.json`: none for D1,
  D2, D3 and D5; axis 1 reversed for D4; axes 1 and 2 reversed for D6 and D7.
* **Header orientation labels.** The template, D1, D2 and D5 label the coronal-slice axis as S and the dorsoventral axis
  as A (scanner convention); D3, D4, D6 and D7 are labelled anatomically. The declared `(perm, flips)`, checked visually
  against the template, defines the orientation.
* **SAMson's anteroposterior specification** (image slices → template slices, 0-based; `dataset_settings.json`):

  | Dataset | Slices × spacing | Image → template | Length image / template | Ratio |
  |---|---|---|---|---|
  | D1 T2w, D5 DWI | 16 × 0.8 mm | 0–15 → 4–33 | 12.0 / 11.6 mm | 1.03 |
  | D2 T2w | 22 × 0.7 mm | 0–21 → 2–39 | 14.7 / 14.8 mm | 0.99 |
  | D3 T2w | 48 × 0.3 mm | 0–47 → 3–35 | 14.1 / 12.8 mm | 1.10 |
  | D4 T2w, T2\*w, DWI | 6 × 1.0 mm | 0–5 → 12–22 | 5.0 / 4.0 mm | 1.25 |
  | D6 T2w, BOLD | 22 × 0.5 mm | 0–21 → 5–34 | 10.5 / 11.6 mm | 0.91 |
  | D7 T2w | 16 × 0.5 mm | 0–15 → 12–31 | 7.5 / 7.6 mm | 0.99 |

  D4 and D7 cover only part of the brain; their template ranges were chosen from the anatomy and the reference masks.
  The specification and SAMson's conventions are described for users in `package/guides/04_input_output_conventions.md`
  and `package/source/docs/anteroposterior_range.md`.

## 2. Input of every method

1. The dataset image is re-oriented losslessly to the canonical orientation and written once as float32 with its ×10
   header. This file is the input of every method.
2. **BEN and RBM** read the slices along axis 2 without rotating them; they receive the canonical image with axis 1
   reversed (dorsal-up coronal slices, BEN's documented rodent orientation), and their mask is reversed back.
3. **deepbet** reorients by the header labels; it receives anatomical labels (direction-cosine columns of axes 1 and 2
   exchanged for D1, D2 and D5; voxel data unchanged). All other methods receive the headers as stored.
4. **ANTs:** the 6-slice D4 images are zero-padded to 8 slices (one empty slice at each end) and the mask is cropped
   back.
5. **SHERM** receives the canonical image with axis 1 reversed (the orientation of its example images) and the real
   voxel size (header / 10), because its parameters are fixed in millimetres.
6. **D2:** the methods run by `run_all.py` receive the D2 images with a header voxel size of 0.1302 × 0.1302 × 0.8 mm
   (×10; `notebooks/lib/config.py`, `TOOL_INPUT_AFFINE_X10`), with which the published D2 masks were computed; only
   BET, BET4Animal, RBM and ANTs use the voxel size. SHERM and every metric use the voxel size of the dataset files.
7. The mask of every method is binarised, mapped back with the inverse transform and saved on the grid and with the
   header of the dataset image.

## 3. Tools and settings

One setting per tool for every dataset, except the five dataset-specific settings listed below the table. SHA-256 of
models and scripts: first 12 hexadecimal digits (full values in `tool_parameters.json`).

| Method | Version / source | Model or template | Call and parameters | Device | Reference |
|---|---|---|---|---|---|
| SAMson Full-Auto | functions of `notebooks/SAMson.ipynb` (identical to `samson-mri` 1.0.0) | SAM ViT-H `sam_vit_h_4b8939.pth` (`a7bf3b02f3eb`) | AP specification and median_otsu rule of `dataset_settings.json`; one box prompt per slice; SAMson post-processing | GPU | `kirillov`, `garyfallidis`, `otsu` |
| SAMson Semi-Auto | as Full-Auto (same embedding pass) | as Full-Auto | 3 candidates per slice; the candidate with the highest Jaccard index against the reference mask is kept (oracle selection) | GPU | `kirillov` |
| SAMson Oracle | as Full-Auto | as Full-Auto | box prompt = bounding box of the reference mask in each slice | GPU | `kirillov` |
| MedSAM | `segment_anything` with MedSAM weights | `medsam_vit_b.pth` (`34b34b78c1d1`) | SAMson Full-Auto boxes, single output, SAMson post-processing | GPU | `ma` |
| SAM2 | SAM 2 commit `2b90b9f5`; `notebooks/runners/sam2_run.py` | `sam2.1_hiera_large.pt` (`2647878d5dfa`) | SAMson Full-Auto boxes, SAMson post-processing | GPU | `ravi` |
| BET | FSL 6.0.7.4 packages (fsl-bet2 2111.4) | – | `bet <in> <out> -m -r 40 -f 0.2 -g 0.07` | CPU, 1 thread | `smith`, `jenkinson` |
| BET4Animal | FSL 6.0.7.17 packages (fsl-bet2 2111.8), `bet4animal` 1.1 | – | `bet4animal <in> <out>.nii.gz "-m -x 1.6,1.2,2 -r 40 -f 0.4 -g 0"` (on the ×10 header, the mouse scaling `-z 6 -r 4` of the real voxel size) | CPU, 1 thread | `fsl_bet4animal`, `smith` |
| deepbet | deepbet 1.0.2 (PyPI) | bundled models (`de8ee7c69654`, `39a7b2b5f799`) | `run_bet(..., threshold=0.95, n_dilate=0)` | GPU | `deepbet` |
| BEN | github.com/yu02019/BEN commit `f23ed394` | Mouse-T2WI-7T weight (`61c8780d2e73`) | `BEN_infer.py -weight weight/Mouse_T2WI_7T_2022_07191047/`, threshold 0.5 | CPU, 1 thread | `ben` |
| RBM | github.com/CAMRIatUNC/RodentMRISkullStripping commit `dfa1a123`; `notebooks/runners/rbm_unet_run_padded.py` | `rat_brain-2d_unet.hdf5` (`e60ef6e3accb`) | in-plane resampling to `voxsize` 1.0 header units (0.1 mm), 128 × 128 patches, stride 32, threshold 0.5 | CPU, 1 thread | `rbm` |
| ANTs | ANTs 2.5.0, `antsBrainExtraction.sh` (`237e4174aac2`) | SAMson template (×10) and its mask (template > 0.5) | `antsBrainExtraction.sh -d 3 -a <in> -e <template> -m <mask> -u 0 -o <prefix>`, fixed seed | CPU, 1 thread | `avants`, `tustison` |
| SHERM | github.com/liu-yikang/SHERM-rodentSkullStrip commit `c44afe8`, VLFeat 0.9.21; MATLAB; `notebooks/runners/sherm/run_sherm_samson.m` | built-in mouse prior | `'mouse'`, anisotropic mode (`isotropic = 0`) | CPU, 1 thread per scan | `liu1` |

**Dataset-specific settings** (`notebooks/lib/config.py`, `DATASET_SPECIFIC_SETTINGS`), used where the common setting
gave less accurate masks:

| Method | Dataset | Setting |
|---|---|---|
| BET | D1 | `-m -r 40 -c 55 42 7 -f 0.44 -g 0.027` |
| BET | D2 | `-m -r 40 -c 126 173 10 -f 0.2 -g 0.07` |
| BET4Animal | D2 | `-m -z 6 -r 6 -c 126 173 10 -f 0.44 -g 0.07`, real voxel size |
| BET4Animal | D3 | `-m -z 6 -r 4 -c 87 87 23 -f 0.4 -g 0`, real voxel size |
| RBM | D3 | canonical image without the reversal of axis 1 |

## 4. Computing environment and run protocol

**Workstation (all methods except SHERM).** AMD EPYC 9474F (48 cores, 96 hardware threads), 755 GiB RAM, NVIDIA RTX
A6000 (48 GB; driver 535.309.01), Ubuntu 24.04.3 LTS. Exact values of the run: `results/environment.json`.

| Environment | Main packages | Methods |
|---|---|---|
| `samson` (Python 3.11.7) | torch 2.4.1+cu121, segment-anything 1.0, dipy 1.9.0, numpy 1.26.3, scipy 1.12.0, scikit-image 0.22.0, nibabel 5.2.1, opencv-python 4.10.0.84 | SAMson, MedSAM; this library |
| `samson_sam2` (Python 3.11.7) | torch 2.5.1+cu121, SAM 2 1.0 | SAM2 |
| `deepbet_1_env` (Python 3.12.7) | deepbet 1.0.2, torch 2.5.0 | deepbet |
| `ben` (Python 3.7.16) | TensorFlow 1.15.4 (CPU), Keras 2.2.4, numpy 1.16.0, SimpleITK 2.0.0 | BEN, RBM |
| `fsl_6.0.7.4`, `fsl_6.0.7.17` | FSL conda packages of releases 6.0.7.4 and 6.0.7.17 | BET, BET4Animal |
| ANTs | ANTs 2.5.0 | ANTs |

Environment files of the conda environments: `docs/environments/` (`conda env create -f <file>`).

**Run protocol** (`notebooks/run_all.py`).
* Every method ran alone on the allocation (hardware threads 16–95 and the GPU), one method after another and, within a
  method, one dataset-contrast after another; ANTs ran all scans in one pool of 80 single-thread workers, longest first.
* CPU methods: one process per scan pinned to one hardware thread (thread variables = 1), up to 80 scans at a time.
  ANTs with one ITK thread and `-u 0` is deterministic; BEN and RBM gave identical masks with 1 and 8 threads.
* GPU methods: one process for all scans of a method, one scan at a time, with 80 CPU threads. SAMson Full-Auto and
  Semi-Auto share one embedding pass (same time recorded for both). deepbet requires the GPU (its CPU masks differ by a
  few voxels).
* Wall time of every scan, dataset-contrast and method: `results/runtimes.csv`. Re-orientation and output conversion
  are not timed.
* SHERM ran on two other computers (run 1: D1, D3–D6; run 2: D2 and D7); records in `outputs/sherm/`.
