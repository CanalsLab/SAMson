# 4. Input and output conventions

Authoritative source: `../source/docs/io_conventions.md` (reproduced and complemented here; statements below were checked
against the code of version 1.0.0).

## Inputs

* NIfTI (`.nii`, `.nii.gz`), one subject per file, magnitude images. 3D, or 4D with one volume used (`volume`,
  default 0, e.g. first echo or b = 0). NaN/Inf are set to 0 with a warning. One dataset = one acquisition protocol,
  orientation and coverage, described by one specification.
* Folders: every `.nii/.nii.gz` in the folder (not sub-folders), except SAMson's own outputs (`*_mask`, `*_candidates`,
  `*_brain`); `--pattern` filters by name.
* Reference masks (optional): same grid as the image; voxels > 0.5 are brain.

## Orientation, slice order, coordinates

SAMson works on coronal slices. Internally an image is an array `(x, y, z)` with

| axis | anatomy | increasing index |
|---|---|---|
| 0 | left-right | either |
| 1 | ventral-dorsal | towards dorsal (`np.rot90(volume[:, :, k])` shows the dorsal side up) |
| 2 | caudal-rostral | towards rostral: **slice 0 = caudal end (cerebellum side)** |

Images stored otherwise are re-oriented in memory, without interpolation, by the specification's `orientation`:
`axes` (input axis that becomes each SAMson axis) and `flip` (reversal afterwards), or three direction letters of the
input axes (`R/L`, `S/D/I/V`, `A/P`; e.g. `RIA`: axis 0 towards right, axis 1 towards inferior = ventral so flipped,
axis 2 towards anterior = rostral). All 48 permutation-and-flip combinations are covered by the tests (round trips,
and masks on the grid of scrambled copies of an image are identical). **The output mask is always saved on the grid of
the input image** (shape, orientation, affine), whatever the internal orientation. Slice indices in the specification
and in the JSON sidecar (`slices`, `processed_slices`, `anchor_slice`, `boxes`) refer to SAMson's orientation.
Box coordinates are `[x_left, y_up, x_right, y_down]` in pixels of the rotated slices.

## Template and anteroposterior coordinate

The bundled template (`anat_template_FP`) has **40 coronal slices, 0.4 mm apart, slice 0 caudal**. The spec's
`template: [t_first, t_last]` gives the template slices matching the first and last **processed** slice (0 <= first <= last
<= 39 is enforced, otherwise `SpecError`); intermediate slices are matched by linear interpolation of the template's box
shapes. The **anchor slice** is the processed slice whose matched position is closest to the template centre; its brain
box is found by median_otsu and scales all boxes.

## Voxel sizes and the x10 header convention

Rodent data are often stored with voxel sizes x10 (pixdim and affine) so that human-oriented software handles them. SAMson
**does not use voxel sizes to compute masks**: the masks are identical whether the header is x10 or true size. Voxel
sizes only serve to display lengths in the app/preview (`--header-scale 0.1`, app *Header scale* = 0.1 for x10 headers),
to compute the subject/template length ratio, and as the geometry copied to the output header. The mask keeps the input
header unchanged (so an x10 input gives an x10 mask; real size = header value / 10). The article's template and D1-D3 data
use the x10 convention; the publication datasets (`../../datasets/`) are all x10 with real size = header / 10.

## Specification (`samson_spec.json`)

```json
{"samson_spec_version": 1, "name": "my T2w dataset",
 "orientation": {"axes": [0, 1, 2], "flip": [false, false, false]},
 "slices": [0, -1], "template": [4, 33], "otsu": "auto", "volume": 0}
```

`slices`: first/last processed slice, 0-based inclusive, negative counts from the end; slices outside stay empty.
`otsu`: `"auto"` (median radius 2 and 5 passes if the processed image has fewer than 140 rows on axis 1, else 1 and 3) or
`[radius, passes]`. CLI options override a `--spec`/`--preset`.

## Coverage of the subject versus the template

* The subject and template ranges are matched linearly (first processed slice to `t_first`, last to `t_last`); SAMson never
  extrapolates beyond the template: the template range must lie inside 0-39.
* **Subject extends beyond the template range** (extra slices at either end, e.g. olfactory bulb beyond slice 39 or
  tissue caudal to the cerebellum): exclude them with `slices`; they are left empty in the mask. Without that, they are
  forced into the template range and the length-ratio warning appears.
* **Subject covers only part of the brain**: choose the matching part of the template (e.g. 12-22 for six slices). If the
  range does not contain the template centre, the anchor slice is an end slice and box sizes may be biased (a warning is
  shown); the article's D4 datasets (six 1 mm slices) use this configuration.
* **Slices without brain**: SAMson has no brain/no-brain detector. Every processed slice receives a box and SAM returns a
  mask for it, which may be spurious if the slice contains no brain; therefore leave such slices out with `slices`.
  This is a documented limitation (SOFTWARE_DOCUMENTATION, section 12).

## Outputs (per input `<name>.nii[.gz]`, in the output folder)

| file | content |
|---|---|
| `<name>_mask.nii.gz` | uint8 {0,1}, same shape (3D) and header as the input, voxel-aligned; Semi-Auto: the selected mask per slice (Full-Auto until selection) |
| `<name>_candidates.nii.gz` | Semi-Auto only: uint8 4D; volume 0 = Full-Auto, volumes 1-3 = SAM candidates 0-2 |
| `<name>_samson.json` | sidecar: version, input, spec, mode, processed slices, otsu parameters, anchor, box prompts, selection (-1 = Full-Auto, 0-2 candidate), reference, model/device, voxel count, timings, software versions |
| `<name>_qc.png` | every processed slice with mask (blue) and box (green) |
| `<name>_brain.nii.gz` | with `--save-brain`: image x mask |
| `samson_spec.json`, `samson_summary.csv` | specification used; one row per image (status, seconds, slices, Jaccard if reference) |

Reproducibility rule: for the same input, specification, checkpoint and software versions the same mask is obtained
(GPU; see guide 05 for the 1-voxel CPU caveat).
