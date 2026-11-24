# Command line reference

Generated from `samson <command> --help` (samson-mri 1.0.0).

```text
usage: samson [-h] [--version] <command> ...

SAMson: brain extraction of rodent MRI with the Segment Anything Model.

positional arguments:
  <command>
    segment       segment images and save the masks
    preview       figure to check the anteroposterior range (no SAM needed)
    app           open the interactive app in the web browser
    spec          write (or --show) a dataset specification file
    suggest-range
                  rank template ranges from reference masks
    select        choose the Semi-Auto mask of every slice after a run
    evaluate      compare masks with reference masks
    download-model
                  download SAM's checkpoint (2.4 GB for vit_h)
    example       segment the example subject bundled with the package
    info          versions, device, checkpoint and template

options:
  -h, --help      show this help message and exit
  --version       show program's version number and exit

Run `samson <command> --help` for the options of a command. Documentation: README.md
```

## samson segment

```text
usage: samson segment [-h] -o OUTPUT [--pattern PATTERN] [--mode {full-auto,semi-auto}] [--reference REFERENCE]
                      [--reference-dir REFERENCE_DIR] [--reference-pattern REFERENCE_PATTERN] [--choices CHOICES]
                      [--no-qc] [--save-brain] [--overwrite] [--spec FILE]
                      [--preset {article-d1,article-d2,article-d3}] [--template FIRST LAST] [--slices FIRST LAST]
                      [--orientation CODES] [--axes LR VD CR] [--flip-ap] [--flip-dv] [--otsu V [V ...]]
                      [--volume VOLUME] [--checkpoint FILE] [--model-type {vit_h,vit_l,vit_b}] [--device DEVICE]
                      inputs [inputs ...]

Segment NIfTI images with SAMson.

positional arguments:
  inputs                NIfTI files, folders or glob patterns

options:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        output folder
  --pattern PATTERN     file name pattern for folders (default: all .nii/.nii.gz), e.g. '*T2w*'
  --mode {full-auto,semi-auto}
                        full-auto (default) or semi-auto (also saves SAM's 3 candidates per slice to choose from)
  --reference REFERENCE
                        reference mask of the (single) input image: Semi-Auto selection and evaluation
  --reference-dir REFERENCE_DIR
                        folder with reference masks of the inputs
  --reference-pattern REFERENCE_PATTERN
                        reference file name in --reference-dir; {stem} = image name without extension (default
                        '{stem}_mask.nii.gz')
  --choices CHOICES     Semi-Auto selection: one value for all slices or a comma-separated list, one per processed
                        slice (F = Full-Auto mask, 0-2 = candidate), e.g. F,F,2,1,F
  --no-qc               do not save the quality-control figures
  --save-brain          also save the brain-extracted images (<image>_brain.nii.gz)
  --overwrite           redo images whose mask exists

dataset specification (a --spec file, a --preset, and/or the options below, which override them):
  --spec FILE           dataset specification file (JSON), e.g. written by the app or `samson spec`
  --preset {article-d1,article-d2,article-d3}
                        specification of one of the article's datasets
  --template FIRST LAST
                        template slices (0-39, caudal to rostral) matching the first and last processed slice
  --slices FIRST LAST   first and last slice to process (0-based, SAMson orientation, -1 = last); default: all
  --orientation CODES   anatomical direction of the input axes, e.g. RIA (axis 0 right, 1 ventral, 2 rostral); see the
                        docs
  --axes LR VD CR       input axis that becomes SAMson's left-right, ventral-dorsal and caudal-rostral axis (keeps the
                        flips of the --spec/--preset; not with --orientation)
  --flip-ap             reverse the anteroposterior axis: the input slices go from rostral to caudal (the notebook's
                        reorient=True). Sets the flip (it does not toggle a flip of the --spec file)
  --flip-dv             reverse the dorsoventral axis: the brain is upside down in the coronal view. Sets the flip
  --otsu V [V ...]      'auto' (default) or median_radius numpass, e.g. --otsu 2 5
  --volume VOLUME       volume of 4D images to use (default 0, the first)

model:
  --checkpoint FILE     SAM checkpoint (default: $SAMSON_CHECKPOINT or the downloaded one)
  --model-type {vit_h,vit_l,vit_b}
                        SAM model (default vit_h, as in the article)
  --device DEVICE       auto (default), cuda, cuda:1, cpu or mps
```

## samson preview

```text
usage: samson preview [-h] [-o OUTPUT] [--x X] [--no-boxes] [--header-scale HEADER_SCALE] [--spec FILE]
                      [--preset {article-d1,article-d2,article-d3}] [--template FIRST LAST] [--slices FIRST LAST]
                      [--orientation CODES] [--axes LR VD CR] [--flip-ap] [--flip-dv] [--otsu V [V ...]]
                      [--volume VOLUME]
                      input

positional arguments:
  input                 a representative NIfTI image of the dataset

options:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        figure file (default: <image>_ap_preview.png)
  --x X                 left-right index of the side view (default: middle)
  --no-boxes            do not compute the box prompts (faster)
  --header-scale HEADER_SCALE
                        factor from header voxel sizes to mm (0.1 for x10 headers)

dataset specification (a --spec file, a --preset, and/or the options below, which override them):
  --spec FILE           dataset specification file (JSON), e.g. written by the app or `samson spec`
  --preset {article-d1,article-d2,article-d3}
                        specification of one of the article's datasets
  --template FIRST LAST
                        template slices (0-39, caudal to rostral) matching the first and last processed slice
  --slices FIRST LAST   first and last slice to process (0-based, SAMson orientation, -1 = last); default: all
  --orientation CODES   anatomical direction of the input axes, e.g. RIA (axis 0 right, 1 ventral, 2 rostral); see the
                        docs
  --axes LR VD CR       input axis that becomes SAMson's left-right, ventral-dorsal and caudal-rostral axis (keeps the
                        flips of the --spec/--preset; not with --orientation)
  --flip-ap             reverse the anteroposterior axis: the input slices go from rostral to caudal (the notebook's
                        reorient=True). Sets the flip (it does not toggle a flip of the --spec file)
  --flip-dv             reverse the dorsoventral axis: the brain is upside down in the coronal view. Sets the flip
  --otsu V [V ...]      'auto' (default) or median_radius numpass, e.g. --otsu 2 5
  --volume VOLUME       volume of 4D images to use (default 0, the first)
```

## samson app

```text
usage: samson app [-h] [--port PORT] [--address ADDRESS] [--folder FOLDER]

options:
  -h, --help         show this help message and exit
  --port PORT
  --address ADDRESS  network address to listen on (default: localhost, i.e.
                     only this computer; the app can read and write files on
                     this computer, so do not open it to a network you do not
                     trust)
  --folder FOLDER    folder with the images to show first
```

## samson spec

```text
usage: samson spec [-h] [-o OUTPUT] [--name NAME] [--show FILE] [--spec FILE]
                   [--preset {article-d1,article-d2,article-d3}] [--template FIRST LAST] [--slices FIRST LAST]
                   [--orientation CODES] [--axes LR VD CR] [--flip-ap] [--flip-dv] [--otsu V [V ...]]
                   [--volume VOLUME]

options:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        file to write (default samson_spec.json)
  --name NAME           a name for the dataset
  --show FILE           show a specification file instead of writing one

dataset specification (a --spec file, a --preset, and/or the options below, which override them):
  --spec FILE           dataset specification file (JSON), e.g. written by the app or `samson spec`
  --preset {article-d1,article-d2,article-d3}
                        specification of one of the article's datasets
  --template FIRST LAST
                        template slices (0-39, caudal to rostral) matching the first and last processed slice
  --slices FIRST LAST   first and last slice to process (0-based, SAMson orientation, -1 = last); default: all
  --orientation CODES   anatomical direction of the input axes, e.g. RIA (axis 0 right, 1 ventral, 2 rostral); see the
                        docs
  --axes LR VD CR       input axis that becomes SAMson's left-right, ventral-dorsal and caudal-rostral axis (keeps the
                        flips of the --spec/--preset; not with --orientation)
  --flip-ap             reverse the anteroposterior axis: the input slices go from rostral to caudal (the notebook's
                        reorient=True). Sets the flip (it does not toggle a flip of the --spec file)
  --flip-dv             reverse the dorsoventral axis: the brain is upside down in the coronal view. Sets the flip
  --otsu V [V ...]      'auto' (default) or median_radius numpass, e.g. --otsu 2 5
  --volume VOLUME       volume of 4D images to use (default 0, the first)
```

## samson suggest-range

```text
usage: samson suggest-range [-h] [--slice-mm SLICE_MM] [--top TOP] [--spec FILE]
                            [--preset {article-d1,article-d2,article-d3}] [--template FIRST LAST]
                            [--slices FIRST LAST] [--orientation CODES] [--axes LR VD CR] [--flip-ap] [--flip-dv]
                            [--otsu V [V ...]] [--volume VOLUME]
                            masks [masks ...]

positional arguments:
  masks                 reference masks of a few subjects of the dataset

options:
  -h, --help            show this help message and exit
  --slice-mm SLICE_MM   true slice spacing in mm (leaves out implausible range lengths)
  --top TOP

dataset specification (a --spec file, a --preset, and/or the options below, which override them):
  --spec FILE           dataset specification file (JSON), e.g. written by the app or `samson spec`
  --preset {article-d1,article-d2,article-d3}
                        specification of one of the article's datasets
  --template FIRST LAST
                        template slices (0-39, caudal to rostral) matching the first and last processed slice
  --slices FIRST LAST   first and last slice to process (0-based, SAMson orientation, -1 = last); default: all
  --orientation CODES   anatomical direction of the input axes, e.g. RIA (axis 0 right, 1 ventral, 2 rostral); see the
                        docs
  --axes LR VD CR       input axis that becomes SAMson's left-right, ventral-dorsal and caudal-rostral axis (keeps the
                        flips of the --spec/--preset; not with --orientation)
  --flip-ap             reverse the anteroposterior axis: the input slices go from rostral to caudal (the notebook's
                        reorient=True). Sets the flip (it does not toggle a flip of the --spec file)
  --flip-dv             reverse the dorsoventral axis: the brain is upside down in the coronal view. Sets the flip
  --otsu V [V ...]      'auto' (default) or median_radius numpass, e.g. --otsu 2 5
  --volume VOLUME       volume of 4D images to use (default 0, the first)
```

## samson select

```text
usage: samson select [-h] [--choices CHOICES] [--reference REFERENCE] [--no-qc] sidecar

positional arguments:
  sidecar               the <image>_samson.json file written by `samson segment --mode semi-auto`

options:
  -h, --help            show this help message and exit
  --choices CHOICES     one value or a comma-separated list, one per processed slice (F = Full-Auto mask, 0-2 =
                        candidate), e.g. F,F,2,1,F
  --reference REFERENCE
                        reference mask: the best candidate of every slice (the article's Semi-Auto)
  --no-qc
```

## samson evaluate

```text
usage: samson evaluate [-h] --masks MASKS [MASKS ...] --references REFERENCES [REFERENCES ...] [--ap-axis AP_AXIS]
                       [--spec SPEC] [-o OUTPUT]

options:
  -h, --help            show this help message and exit
  --masks MASKS [MASKS ...]
  --references REFERENCES [REFERENCES ...]
  --ap-axis AP_AXIS     input axis along which the coronal slices are stacked (default 2)
  --spec SPEC           take the anteroposterior axis from a specification file
  -o OUTPUT, --output OUTPUT
                        CSV file with the results
```

## samson download-model

```text
usage: samson download-model [-h] [--model-type {vit_h,vit_l,vit_b}] [--folder FOLDER]

options:
  -h, --help            show this help message and exit
  --model-type {vit_h,vit_l,vit_b}
  --folder FOLDER       destination folder (default: the SAMson cache folder)
```

## samson example

```text
usage: samson example [-h] [-o OUTPUT] [--device DEVICE] [--checkpoint CHECKPOINT]

options:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        output folder (default ./samson_example)
  --device DEVICE
  --checkpoint CHECKPOINT
```

## samson info

```text
usage: samson info [-h]

options:
  -h, --help  show this help message and exit
```
