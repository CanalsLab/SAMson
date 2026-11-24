"""
SAMson from Python, on the example subject bundled with the package (a mouse of the article's dataset D1).
Run:  python run_example.py      (needs SAM's checkpoint: samson download-model)
"""
from samson_mri import DatasetSpec, SAMson
from samson_mri.example import example_path
from samson_mri.io import load_subject
from samson_mri.pipeline import compute_boxes
from samson_mri.viz import ap_figure, save_figure

image = example_path("image")              # T2w, 108 x 90 x 16 voxels, coronal slices (already in SAMson's orientation)
manual = example_path("manual_mask")       # its manual brain mask

# 1. The dataset specification: template slices 4-33 match the first and last of the 16 slices (whole brain)
spec = DatasetSpec(template=(4, 33))

# 2. Check the anteroposterior range (no SAM needed): subject and template side views and coronal pairs
subject = load_subject(image, spec)
save_figure(ap_figure(subject.volume, subject.voxel_size, *spec.template, prompts=compute_boxes(subject, spec),
                      header_scale=0.1, title="example"), "example_output/ap_preview.png")

# 3. Full-Auto segmentation, evaluated against the manual mask
samson = SAMson()                          # loads SAM ViT-H once (GPU if available)
result = samson.segment(image, spec, reference=manual)
print(f"Full-Auto: mean Jaccard {result.metrics.jaccard.mean():.4f}")
result.save("example_output/full_auto")

# 4. Semi-Auto: SAM's 3 candidates per slice; choose them by hand...
result = samson.segment(image, spec, mode="semi-auto")
result.select([-1] * 7 + [1] * 9)          # -1 = keep the Full-Auto mask of the slice, 0-2 = candidate
result.save("example_output/semi_auto_manual")
# ...or, as in the article, the best one according to the manual mask
result.select_by_reference(manual)
print(f"Semi-Auto (best candidate by the manual mask): selection {result.selection}")
print("Saved:", result.save("example_output/semi_auto_reference"))
