"""
Validation of samson-mri against the published SAMson masks (run with SAM's real ViT-H model, preferably on a GPU).

All 75 subjects of D1 (53), D2 (10) and D3 (12) of the SAMson data deposit (DIGITAL.CSIC,
https://doi.org/10.20350/digitalCSIC/17000), with the article's specifications (presets article-d1, article-d2,
article-d3). For every subject, the package's Full-Auto mask must equal the published Full-Auto mask
(outputs/masks/samson_full_auto) and its Semi-Auto mask (best of the 3 candidates of each slice by the reference mask)
the published Semi-Auto mask (outputs/masks/samson_semi_auto).

Usage (<deposit folder> contains the datasets/ and outputs/ folders of the deposit):
    python validate_package.py --data <deposit folder> --checkpoint sam_vit_h_4b8939.pth --out results_<environment>.csv
    python summarise.py results_<environment>.csv
"""
import argparse
import os
import platform
import time

import nibabel as nib
import numpy as np
import pandas as pd

import samson_mri
from samson_mri import PRESETS, SAMson

PRESET_OF = {"D1": "article-d1", "D2": "article-d2", "D3": "article-d3"}


def as_bool(path):
    data = np.asanyarray(nib.load(path).dataobj)
    return (data.reshape(data.shape[:3]) if data.ndim > 3 else data) > 0


def compare(a, b):
    """(identical, number of differing voxels, Jaccard index) of two masks."""
    a, b = np.asarray(a) > 0, np.asarray(b) > 0
    if a.shape != b.shape:
        return False, -1, np.nan
    union = np.count_nonzero(a | b)
    return bool(np.array_equal(a, b)), int(np.count_nonzero(a != b)), (np.count_nonzero(a & b) / union if union else 1.0)


def validate(samson, root, rows):
    participants = pd.read_csv(os.path.join(root, "datasets", "participants.tsv"), sep="\t", dtype=str, keep_default_na=False)
    for dataset, preset in PRESET_OF.items():
        for _, p in participants[participants.dataset == dataset].iterrows():
            start = time.time()
            result = samson.segment(os.path.join(root, "datasets", p["image"]), PRESETS[preset], mode="semi-auto",
                                    reference=os.path.join(root, "datasets", p["mask"]))
            full = result.subject.to_input_space(result._to_volume(result.full_auto[:, 0]))
            stem = p["image"][:-len(".nii.gz")]   # e.g. D1/anat/3_month/sub-A110123_T2w
            for mode, mask in (("full-auto", full), ("semi-auto", result.mask)):
                published = os.path.join(root, "outputs", "masks", "samson_" + mode.replace("-", "_"), stem + "_mask.nii.gz")
                identical, differing, jaccard = compare(mask, as_bool(published))
                rows.append(dict(dataset=dataset, subject=p["subject"], group=p["group"], mode=mode, identical=identical,
                                 differing_voxels=differing, jaccard_vs_published=jaccard, seconds=round(time.time() - start, 1)))
            print(f"{dataset} {p['subject']}: Full-Auto identical {rows[-2]['identical']}, Semi-Auto identical {rows[-1]['identical']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="folder of the data deposit (with datasets/ and outputs/)")
    parser.add_argument("--checkpoint", help="SAM ViT-H checkpoint (default: the one of `samson download-model`)")
    parser.add_argument("--out", required=True, help="CSV file of the results")
    args = parser.parse_args()
    samson = SAMson(checkpoint=args.checkpoint)
    rows = []
    start = time.time()
    validate(samson, args.data, rows)
    table = pd.DataFrame(rows)
    table.to_csv(args.out, index=False)
    import torch
    print(f"\nsamson-mri {samson_mri.__version__}; Python {platform.python_version()}, torch {torch.__version__}, numpy {np.__version__}; "
          f"GPU {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}; {(time.time() - start) / 60:.1f} min")
    summary = table.groupby(["dataset", "mode"]).agg(subjects=("identical", "size"), identical=("identical", "sum"),
                                                     max_differing_voxels=("differing_voxels", "max"),
                                                     min_jaccard=("jaccard_vs_published", "min"))
    print(summary.to_string())
    print(f"\nIdentical masks: {int(table.identical.sum())}/{len(table)}")


if __name__ == "__main__":
    main()
