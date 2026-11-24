# 2. Quick start: command line and Python API

All commands below were checked against `samson <command> --help` of version 1.0.0 (the complete help of every command
is in `../source/docs/cli_reference.md`). Commands marked (SAM) need the checkpoint; the others run without it and
without GPU.

## 1. Specify the dataset

SAMson needs to know, per dataset (not per image): the orientation of the input axes, the first and last processed
slice, and the **template slices (0-39, caudal to rostral)** that match them (`--template FIRST LAST`, required).
Everything is stored in a small JSON file (guide 04).

```bash
samson spec --template 4 33 --name "my T2w dataset" -o samson_spec.json      # write
samson spec --show samson_spec.json                                         # display
samson spec --template 12 22 --orientation RIA -o spec_partial.json         # axes given as anatomical letters
```

Presets of the article's datasets: `--preset article-d1 | article-d2 | article-d3` (files in
`../source/examples/specs/`, together with `partial_coverage_6_slices.json`).

## 2. Check the anteroposterior range (no SAM, no GPU)

```bash
samson preview sub-01_T2w.nii.gz --template 4 33 --header-scale 0.1 -o preview.png
```

Writes the figure of the app's step 3 (subject and template side views, coronal pairs with the box prompts). Use
`--header-scale 0.1` if the header stores voxel sizes x10 (guide 04). Options `--x`, `--no-boxes`, `--slices`,
`--orientation`, ... are listed by `samson preview --help`. If the manual masks of a few animals exist,
`samson suggest-range masks/*.nii.gz --spec samson_spec.json` ranks candidate template ranges.

## 3. Segment (SAM)

```bash
# Full-Auto, a folder (every .nii/.nii.gz in it) or a list of files / glob patterns
samson segment /data/T2w/ -o /data/T2w_masks --spec samson_spec.json
samson segment /data/T2w/ -o out --template 4 33 --pattern '*T2w*'        # options override the spec

# Semi-Auto: also stores SAM's 3 candidates per slice
samson segment sub-01_T2w.nii.gz -o out --spec samson_spec.json --mode semi-auto
samson select out/sub-01_T2w_samson.json --choices F,F,2,1,F,...            # one entry per processed slice
samson select out/sub-01_T2w_samson.json --reference sub-01_manual.nii.gz   # best candidate by a reference mask
```

`F` = the Full-Auto mask, `0`, `1`, `2` = SAM's candidates. `--reference` (Jaccard-optimal candidate per slice) is the
**oracle selection** used in the article's Semi-Auto results; for real use, choose candidates visually in the app.
Re-running a folder skips images whose mask exists (`--overwrite` to redo). Other options: `--no-qc`, `--save-brain`,
`--device`, `--checkpoint`, `--model-type`, `--volume`, `--otsu`, `--slices`, `--flip-ap`, `--flip-dv`, `--axes`.

## 4. Evaluate against manual masks

```bash
samson evaluate --masks out/sub-01_T2w_mask.nii.gz --references manual/sub-01_mask.nii.gz -o evaluation.csv
```

Prints and saves the slice-wise mean Jaccard and Dice, accuracy, and 3D Jaccard.

## 5. Other commands

`samson info` (versions, device, checkpoint), `samson example` (self-test), `samson download-model`,
`samson app` (guide 03), `samson --version`.

## 6. Python API

```python
from samson_mri import SAMson, DatasetSpec, Orientation, load_result, apply_selection

spec = DatasetSpec(template=(4, 33))            # + orientation=Orientation(...), slices=(first, last), otsu=(2, 5), volume=0
samson = SAMson()                                # ViT-H, GPU if available; SAMson(device="cpu", checkpoint="...")
result = samson.segment("sub-01_T2w.nii.gz", spec)                  # Full-Auto  -> SegmentationResult
result = samson.segment("sub-01_T2w.nii.gz", spec, mode="semi-auto", reference="manual_mask.nii.gz")
result.select([-1, -1, 2, 1])                    # -1 = Full-Auto mask, 0-2 = candidates (one per processed slice)
mask = result.mask                               # uint8 array on the grid of the input image
result.save("masks/")                            # _mask.nii.gz, _candidates.nii.gz (semi-auto), _samson.json, _qc.png
samson.segment_files(["a.nii.gz", "b.nii.gz"], spec, "masks/")     # batch, resumable, writes samson_summary.csv
result2 = load_result("masks/sub-01_T2w_samson.json")               # reload a saved result
spec.save("samson_spec.json"); DatasetSpec.load("samson_spec.json")
```

Public names (`samson_mri.__all__`): `SAMson`, `SegmentationResult`, `BoxPrompts`, `DatasetSpec`, `Orientation`,
`SpecError`, `PRESETS`, `Subject`, `load_subject`, `find_images`, `compute_boxes`, `apply_selection`, `load_result`,
`FULL_AUTO`, `Template`, `load_template`, `evaluate`, `compare_volumes`. A complete script is
`../source/examples/run_example.py` and a shell version `../source/examples/run_example.sh`.

Semantic versioning applies to the CLI options, the file formats and these names: a change of the masks produced for the
same input and settings is always a new major version (`../source/CHANGELOG.md`).
