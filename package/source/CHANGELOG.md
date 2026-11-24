# Changelog

All notable changes to samson-mri are listed here. Versions follow [semantic versioning](https://semver.org):
MAJOR.MINOR.PATCH; a change of the masks produced for the same input and settings always needs a new MAJOR version.

## [1.0.0] - 2026-11-09

First packaged release of SAMson (the tool of the article, also available as the notebook `SAMson.ipynb`).

### Added
- Installable Python package `samson-mri` (`pip install`), with the template and an example subject bundled.
- Command line tool `samson` with the commands `segment`, `preview`, `app`, `spec`, `suggest-range`, `select`,
  `evaluate`, `download-model`, `example` and `info`.
- Python API: `SAMson`, `DatasetSpec`, `Orientation`, `SegmentationResult`, `load_result`, `apply_selection`.
- Interactive app (Streamlit, `samson app`): orientation check, choice of the anteroposterior range on side views of a
  subject and of the template (with coronal pairs and the box prompts), segmentation, review of the masks (with a zoom
  on the brain) and slice-by-slice selection of the Semi-Auto candidates.
- Dataset specification files (`samson_spec.json`) with the orientation, slice range, template range, median_otsu
  parameters and volume of 4D images; presets of the article's datasets.
- Re-orientation of the inputs (axis permutations and flips) with masks saved on the grid of the input images; slice
  range (slices outside it are left empty); 4D inputs (one volume).
- Outputs with explicit conventions: mask, Semi-Auto candidates, JSON sidecar (settings, boxes, selection, versions),
  quality-control figure, batch summary; resumable batches.
- Suggestion of the template range from a few manual masks (`samson suggest-range`).
- Checkpoint download with checksum (`samson download-model`).
- Tests (pytest; a fake SAM predictor for the fast tests, the real model for the regression test) and a validation
  against the published masks (docs/validation.md).

### Unchanged
- The algorithm: box generation from the template and median_otsu, SAM ViT-H prediction and post-processing are those
  of the SAMson notebook; the masks of the article's 75 subjects are reproduced voxel for voxel.
