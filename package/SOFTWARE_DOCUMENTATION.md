# SAMson (`samson-mri` 1.0.0): software documentation

Formal description of the software released with the manuscript *SAMson: an automated brain extraction tool for rodents
using SAM* (Computer Methods and Programs in Biomedicine). Statements marked "verified" were checked on the
files of this publication; others are quoted from the package documentation of the authors.

## 1. Purpose and scope

SAMson extracts the brain (skull stripping) from three-dimensional rodent MRI by prompting the Segment Anything Model
(SAM, ViT-H) with one bounding box per coronal slice. The boxes are derived without a trained brain detector, from a
brain template and from a single Otsu-based measurement on one slice of the subject. The article's tool, also available as
the notebook `SAMson.ipynb`, is released as an installable, versioned package with a stable interface: a command-line program, a Python API, an interactive browser app, explicit input/output
conventions, a bundled example, automated tests and documented run time. Scope: coronal-slice mouse (and rat-like)
magnitude images in NIfTI format; the only setting that is dataset specific is the anteroposterior (AP) range (plus the
orientation of the stored axes). Out of scope: training, non-NIfTI inputs (DICOM/Bruker conversion is external), and
automatic detection of slices without brain.

## 2. Architecture and modules

Python package `samson_mri` (src layout), pure Python, one command `samson`.

| Module | Role |
|---|---|
| `_algorithm.py` | SAMson itself, the functions of the SAMson notebook unchanged: template box shapes, median_otsu anchor box, linear interpolation along the AP range, SAM prediction, post-processing (largest component, lobe/isolated-pixel removal, closing) |
| `template.py`, `data/` | bundled template `anat_template_FP.nii.gz` (40 coronal slices, 0.4 mm, x10 header) and example subject |
| `spec.py` | `DatasetSpec` (orientation, slice range, template range, median_otsu, volume), JSON read/write, presets |
| `io.py` | NIfTI reading, in-memory re-orientation (permutation and flips, no interpolation), writing on the input grid |
| `model.py` | checkpoint discovery/download with checksum, SAM loading on CPU/GPU |
| `pipeline.py` | `SAMson.segment`, `segment_files`, `SegmentationResult` (Semi-Auto selection, saving), `load_result`, `apply_selection` |
| `ap.py`, `viz.py` | AP-range helpers (checks, length ratio, range suggestion from masks) and figures (side views, coronal pairs, QC) |
| `metrics.py` | slice-wise Jaccard, Dice, accuracy, 3D Jaccard (as in the article) |
| `cli.py`, `app/streamlit_app.py`, `example.py` | command line, web app, self-test |

Size: about 2,800 lines of Python in `src/` (verified, `wc -l`). The release file is `dist/samson_mri-1.0.0-py3-none-any.whl`
(source tree: `source/`).

## 3. Algorithm entry points

1. The input volume is re-oriented in memory (guide 04). 2. `median_otsu` (dipy; radius 2 and 5 passes if fewer than 140
rows, else 1 and 3, or user values) is applied to the processed slices and post-processed on the **anchor slice**, the
processed slice matched closest to the template centre; its box is the central box. 3. The template's per-slice brain
boxes, expressed relative to its central box, are linearly interpolated at the template positions matching each
processed slice (`template_positions`, `interpolated_coordinates`) and rescaled on the central box
(`boxes_from_central_box`). 4. Each slice is embedded by SAM once and prompted with its box (`segment_volume`); in
Full-Auto mode the single mask is used, in Semi-Auto mode SAM's three multimask candidates are also kept. 5. Masks are
post-processed, rotated back, written in the input orientation and grid.
The article's **Semi-Auto** result selects, per slice, the candidate with the highest Jaccard index against the
reference mask (oracle selection; a disclosed methodological choice, `SegmentationResult.select_by_reference`); in
practice the user selects candidates visually in the app.

## 4. Command-line interface

`samson` with the sub-commands `segment`, `preview`, `app`, `spec`, `suggest-range`, `select`, `evaluate`,
`download-model`, `example`, `info` (complete options: `source/docs/cli_reference.md`; verified against `--help`,
guide 05). Typical use: `samson segment <folder> -o <out> --spec samson_spec.json [--mode semi-auto]`. See
`guides/02_quickstart_cli_api.md`.

## 5. Python API

`from samson_mri import SAMson, DatasetSpec, Orientation, load_result, apply_selection, ...`;
`SAMson(model_type, checkpoint, device).segment(image, spec, mode, reference, choices)` returns a `SegmentationResult` with
`mask` (input grid), `candidates`, `select`, `evaluate`, `compare`, `save`. Interface and file formats follow semantic
versioning (`CHANGELOG.md`).

## 6. Streamlit app and how it facilitates use

`samson app` guides the user through five steps (`guides/03_app_guide.md`, screenshots in `source/docs/images/`, narrated
tutorial in `app_tutorial/`):

1. **Images**: choose a folder and a representative image.
2. **Orientation**: shows the image as SAMson sees it beside the template; coronal view dorsal side up, side view with
   the cerebellum on the left; axes set by the user or guessed from the header.
3. **AP range**: the subject's and the template's sagittal views with the processed slices and their matched template
   positions, and coronal subject/template pairs with the box prompts. The user slides the range until each pair shows
   the same anatomy; warnings flag implausible ranges; the range can be suggested from a few manual masks. This replaces
   trial and error with a direct visual check of the single dataset-specific parameter.
4. **Segment**: Full-Auto or Semi-Auto, one image or the whole folder; the equivalent command line is displayed.
5. **Review/export**: slice-by-slice view of masks and boxes; in Semi-Auto the user picks the best of SAM's candidates per
   slice; optional comparison with a reference mask; the masks, candidates, sidecar, QC figure and specification are saved.

The app drives the same code as the CLI (the specification file is the shared interface). Its five steps are exercised by
the test suite (headless Streamlit tester).

## 7. Input and output conventions

Summarised in `guides/04_input_output_conventions.md` (full text: `source/docs/io_conventions.md`): array axes
(left-right, ventral-to-dorsal, caudal-to-rostral with slice 0 caudal), re-orientation by axes/flips or direction letters,
output masks uint8 {0,1} on the grid and with the header of the input, template of 40 slices (0.4 mm), voxel-size x10
convention (masks do not depend on voxel sizes), behaviour for subject coverage larger or smaller than the template
range, and sidecar JSON content.

## 8. Dependencies and supported platforms

Python >= 3.9; numpy, scipy, pandas, nibabel, dipy, scikit-image, opencv-python-headless, matplotlib, torch >= 2.0,
torchvision, segment-anything 1.0; optional Streamlit (app) and pytest. SAM ViT-H checkpoint (Apache 2.0, Meta AI,
2.4 GB) is downloaded on demand and not redistributed. Exact tested versions: `source/requirements-lock.txt` and
`environment.yml` (Python 3.10, PyTorch 2.5.1/CUDA 12.4). Validated on Linux x86-64 (NVIDIA RTX A6000, AMD EPYC 9474F);
CPU-only operation supported; Windows and macOS (including `mps`) are not tested. Driver constraint: recent PyTorch
wheels need a recent NVIDIA driver; otherwise the CPU is used silently (guide 01 section 4).

## 9. Testing and validation

* Test suite (verified for this publication): the installed wheel gives **133 passed** (every test, including the real
  checkpoint on the GPU; Python 3.10, Streamlit installed); without Streamlit, `pytest -m "not model"` gives 127 passed,
  3 skipped, 1 deselected. Tests compare the package with a frozen copy of the notebook code, and cover orientation (all
  48 combinations), specification, outputs, CLI and app.
* Identity with the article (recorded by the authors, GPU): 150/150 masks identical (75 D1–D3 subjects x
  Full-Auto/Semi-Auto) in the article's environment and 150/150 in a fresh environment with the pinned versions
  (guide 05). One documented CPU-only exception: a 1-voxel difference in
  the example's Semi-Auto mask, caused by CPU floating-point arithmetic.
* The re-run of all tools on the harmonized datasets of this publication is documented in `../notebooks/`.

## 10. Performance

As documented (not re-measured here): on an NVIDIA RTX A6000, validation runs took a median of 8.6 s (mean 12 s) per
article subject (16 to 48 slices), including loading overheads; on CPU about 9 s per slice with 16 threads (2.5 min for
the 16-slice example); Semi-Auto costs the same as Full-Auto. Timings of the benchmark under identical resources:
`../results/runtimes.csv`.

## 11. Licence and availability

Code: MIT (`source/LICENSE`); bundled template and example: SAMson public data (doi:10.20350/digitalCSIC/17000); SAM:
Apache 2.0. Repository: https://github.com/CanalsLab/SAMson (tag v1.0.0, to be created). Zenodo DOI: *to be assigned*
(doi:10.5281/zenodo.XXXXXXX). PyPI: `samson-mri` (to be published).
Citation: `source/CITATION.cff`. Release checksums: `dist/SHA256SUMS`.

## 12. Limitations

* One dataset-specific parameter (AP range) must be chosen by the user; a wrong range yields wrongly sized boxes.
* No detection of slices without brain: all processed slices are segmented; exclude empty end slices with the slice range.
* The template has 40 slices (0.4 mm); ranges cannot extend outside it. Images covering a partial brain whose range does
  not include the template centre can have biased box sizes (the app warns).
* Coronal acquisitions of adequate slice coverage are assumed; the box prompt shape comes from one mouse template.
* GPU use depends on a compatible driver; CPU is slower. Only Linux x86-64 validated.
* Semi-Auto accuracy reported in the article is an oracle upper bound; visual selection by users was not evaluated.
* Documentation note: `source/docs/validation.md` reports 130 tests; the suite of this release gives 131 passed, 1 skipped (with Streamlit, `-m "not model"`).

## 13. Documents

Guides: `guides/01_installation.md`, `02_quickstart_cli_api.md`, `03_app_guide.md`, `04_input_output_conventions.md`,
`05_validation_and_reproducibility.md`. User guide for non-programmers:
`source/docs/user_guide.md`. Also `source/docs/{cli_reference,io_conventions,anteroposterior_range,validation}.md`.
