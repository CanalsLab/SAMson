# Input and output conventions

## Inputs

* **Images**: NIfTI files (`.nii` or `.nii.gz`), one subject per file, magnitude images (non-negative intensities).
  3D images, or 4D images of which one volume is used (`volume`, default 0: e.g. the first echo of a multi-echo series
  or the b = 0 image of a diffusion series). NaN or infinite values are set to 0 (with a warning).
* **A dataset** is a set of images with the same acquisition, orientation and anteroposterior coverage. One dataset
  specification (below) is used for all of them.
* **Reference masks** (optional; Semi-Auto selection as in the article, evaluation): NIfTI files on the voxel grid of
  their image (same shape); voxels > 0.5 are brain.
* **Folders**: `samson segment <folder>` takes every `.nii`/`.nii.gz` file in the folder (not in sub-folders), except
  SAMson's own outputs (`*_mask`, `*_candidates`, `*_brain`); `--pattern '*T2w*'` selects files by name.

## SAMson's orientation

SAMson works on coronal slices. Internally, an image is an array `(x, y, z)` where

| axis | anatomy | direction of increasing index |
|---|---|---|
| 0 | left-right | either |
| 1 | ventral-dorsal | towards **dorsal** (so that `np.rot90` of a slice shows the dorsal side up) |
| 2 | caudal-rostral (anteroposterior) | towards **rostral**: slice 0 is the caudal end (cerebellum side) |

This is the convention of the article's notebook. Images stored differently are re-oriented in memory, without
interpolation, by the `orientation` of the specification:

* `axes`: for each SAMson axis (left-right, ventral-dorsal, caudal-rostral), the input axis that becomes it;
* `flip`: for each SAMson axis, whether it is reversed afterwards.

Examples: images already in SAMson's orientation: `{"axes": [0, 1, 2], "flip": [false, false, false]}` (default);
the notebook's `reorient=True` (slices from rostral to caudal): `"flip": [false, false, true]`; slices stacked along the
second axis: `"axes": [0, 2, 1]`.

The orientation can also be written as three letters giving the anatomical direction of each input axis (the direction
of increasing index): R/L (right/left), A/P (rostral/caudal), S or D/I or V (dorsal/ventral), e.g.
`"orientation": "RIA"` or `samson segment --orientation RIA`. The app (step 2) sets it with a few clicks and shows the
result; `Guess from the image header` reads the letters from the NIfTI header, which is often wrong for rodents, so
always check the views.

**Masks are always saved on the grid of the input image** (in its orientation, with its header), whatever the
orientation used internally. Slice indices in the specification and in the JSON outputs (`slices`, `processed_slices`,
`anchor_slice`, `boxes`) refer to SAMson's orientation.

## Dataset specification (`samson_spec.json`)

```json
{
  "samson_spec_version": 1,
  "name": "my T2w dataset",
  "orientation": {"axes": [0, 1, 2], "flip": [false, false, false]},
  "slices": [0, -1],
  "template": [4, 33],
  "otsu": "auto",
  "volume": 0
}
```

| key | meaning |
|---|---|
| `template` | **required**: template slices (0-39; 0 = caudal end, 39 = rostral end) matching the first and the last processed slice. The slices in between are matched linearly. See [anteroposterior_range.md](anteroposterior_range.md). |
| `slices` | first and last processed slice, 0-based and inclusive, in SAMson's orientation; negative values count from the end (`-1` = last slice). Slices outside the range are empty in the mask. |
| `orientation` | see above |
| `otsu` | `"auto"` (the article's rule: median radius 2 and 5 passes if the processed image has fewer than 140 rows, axis 1 in SAMson's orientation; otherwise 1 and 3) or `[median_radius, numpass]` |
| `volume` | volume of 4D images (0 = first) |
| `name` | free text |

Command line options (`--template`, `--slices`, `--otsu`, `--volume`) override the values of a `--spec` file or a
`--preset` (`article-d1`, `article-d2`, `article-d3`). For the orientation: `--orientation LETTERS` replaces it;
`--axes` replaces the axes and keeps the flips of the file; `--flip-ap` and `--flip-dv` set (not toggle) the flip of the
anteroposterior or dorsoventral axis. `--orientation` and `--axes` cannot be combined.

## Outputs

For an input `<name>.nii` or `<name>.nii.gz`, in the output folder:

| file | content |
|---|---|
| `<name>_mask.nii.gz` | the brain mask: uint8, 0/1, same shape (3D) and header (affine, voxel sizes) as the input, voxel-to-voxel aligned with it. In Semi-Auto mode: the selected mask of every slice (the Full-Auto mask until a selection is made). |
| `<name>_candidates.nii.gz` | Semi-Auto mode only: uint8 4D, volume 0 = Full-Auto mask, volumes 1, 2, 3 = SAM's candidates 0, 1, 2 (the notebook's indices). Used to change the selection later (`samson select`, app step 5). |
| `<name>_samson.json` | sidecar: package version, input path and shape, specification, mode, processed slices, median_otsu parameters, anchor slice and its Otsu box, box prompts, selection (one value per processed slice: -1 = Full-Auto mask, 0-2 = candidate; `F` on the command line) and how it was made, reference mask (if any), model and device, number of mask voxels, timings and software versions. |
| `<name>_qc.png` | quality-control figure: every processed slice with the mask (blue) and the box prompt (green). |
| `<name>_brain.nii.gz` | with `--save-brain`: the image multiplied by the mask (float32); rewritten when the Semi-Auto selection changes. |
| `samson_spec.json` | the specification used (written by `samson segment` and by the app). |
| `samson_summary.csv` | one row per image: status (ok, skipped because the mask exists, or the error), time, processed slices, mean Jaccard index if a reference was given. |

Re-running `samson segment` on the same folder skips images whose mask exists (`--overwrite` redoes them), so an
interrupted batch can be resumed.

**Box coordinates** (`boxes` in the JSON file) are `[x_left, y_up, x_right, y_down]` in pixels of the rotated slices
(`np.rot90(volume[:, :, k])` in SAMson's orientation), as in the notebook.

## Python objects

`SAMson.segment()` returns a `SegmentationResult` with: `mask` (input grid), `mask_samson` (SAMson orientation),
`full_auto` and `candidates` (masks of the rotated slices, shape `(n_slices, n, H, W)`), `prompts` (box prompts,
anchor slice, median_otsu parameters, template position of every slice), `selection`, `metrics` (per-slice metrics
against a reference), `select()`, `select_by_reference()`, `compare()` and `save()`. `load_result(<name>_samson.json)`
reloads a saved result.
