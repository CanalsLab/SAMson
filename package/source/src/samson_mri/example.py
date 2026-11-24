"""
The example bundled with the package: one mouse of the article's dataset D1 (3 months; T2w, 16 coronal slices of
0.8 mm, 108 x 90 voxels of 0.167 mm), its manual brain mask and the SAMson masks published with the article.

``samson example`` segments it in Full-Auto and Semi-Auto modes, compares the masks with the manual one and checks that
they are identical to the published SAMson masks.
"""
from __future__ import annotations

import os
from importlib import resources

import nibabel as nib
import numpy as np

EXAMPLE_FILES = {
    "image": "d1_A110123_3mo_T2w.nii.gz",
    "manual_mask": "d1_A110123_3mo_manual_mask.nii.gz",
    "published_full_auto": "d1_A110123_3mo_published_full_auto_mask.nii.gz",
    "published_semi_auto": "d1_A110123_3mo_published_semi_auto_mask.nii.gz",
}


def example_path(key: str = "image") -> str:
    """Path of a file of the bundled example ('image', 'manual_mask', 'published_full_auto', 'published_semi_auto')."""
    return str(resources.files("samson_mri").joinpath("data", "example", EXAMPLE_FILES[key]))


def example_spec():
    """The article's D1 specification: all 16 slices, template slices 4-33, input already in SAMson's orientation."""
    from .spec import PRESETS
    return PRESETS["article-d1"]


def run_example(output: str = "samson_example", device: str = "auto", checkpoint: str | None = None, predictor=None) -> int:
    """Segment the example subject (Full-Auto and Semi-Auto), save the outputs and report the checks. Returns 0 if all pass."""
    from .io import load_subject
    from .pipeline import SAMson, compute_boxes
    from .viz import ap_figure, save_figure

    spec = example_spec()
    image, manual = example_path("image"), example_path("manual_mask")
    os.makedirs(output, exist_ok=True)
    print(f"Example: {os.path.basename(image)} (article dataset D1), template range {spec.template[0]}-{spec.template[1]}")

    subject = load_subject(image, spec)
    save_figure(ap_figure(subject.volume, subject.voxel_size, *spec.template, prompts=compute_boxes(subject, spec),
                          header_scale=0.1, title="example (D1)"), os.path.join(output, "example_ap_preview.png"))

    samson = SAMson(checkpoint=checkpoint, device=device, predictor=predictor)
    ok = True
    for mode, published in (("full-auto", "published_full_auto"), ("semi-auto", "published_semi_auto")):
        folder = os.path.join(output, mode)
        result = samson.segment(image, spec, mode=mode, reference=manual)
        paths = result.save(folder)
        expected = np.asanyarray(nib.load(example_path(published)).dataobj) > 0
        identical = np.array_equal(result.mask > 0, expected)
        ok &= identical
        jaccard = result.metrics.jaccard.mean()
        print(f"  {mode:<9}: mean Jaccard index vs the manual mask {jaccard:.4f} over {len(result.metrics)} slices; "
              f"identical to the article's published mask: {'yes' if identical else 'NO'}  ->  {paths['mask']}")
        if not identical:
            differing = int(((result.mask > 0) != expected).sum())
            print(f"    {differing} voxels differ from the published mask (different software versions or hardware can change "
                  "a few voxels; see docs/validation.md)")
    print(f"Outputs in {os.path.abspath(output)} (masks, *_samson.json, *_qc.png, example_ap_preview.png)")
    return 0 if ok else 1
