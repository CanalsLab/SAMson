#!/usr/bin/env bash
# SAMson from the command line, on the example subject bundled with the package.
set -e
IMAGE=$(python -c "from samson_mri.example import example_path; print(example_path('image'))")
MANUAL=$(python -c "from samson_mri.example import example_path; print(example_path('manual_mask'))")

samson info                                                        # versions, GPU, checkpoint
samson download-model                                              # once (2.4 GB)

# Check the anteroposterior range on the image (figure), then save the specification
samson preview "$IMAGE" --template 4 33 --header-scale 0.1 -o example_output/ap_preview.png
samson spec --template 4 33 --name "example (article D1)" -o example_output/spec.json

# Full-Auto, then Semi-Auto with the article's selection (best candidate according to the manual mask)
samson segment "$IMAGE" -o example_output/full_auto --spec example_output/spec.json
samson segment "$IMAGE" -o example_output/semi_auto --spec example_output/spec.json --mode semi-auto --reference "$MANUAL"

# Change the selection of the Semi-Auto mask by hand (one value per slice: F = Full-Auto mask, 0-2 = candidate)
samson select example_output/semi_auto/d1_A110123_3mo_T2w_samson.json --choices F,F,F,F,F,F,F,1,1,1,1,1,1,1,1,1

# Compare with the manual mask
samson evaluate --masks example_output/full_auto/d1_A110123_3mo_T2w_mask.nii.gz example_output/semi_auto/d1_A110123_3mo_T2w_mask.nii.gz \
                --references "$MANUAL" "$MANUAL" -o example_output/evaluation.csv
