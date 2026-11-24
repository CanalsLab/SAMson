"""
Evaluation metrics, identical for every tool.

Slice metrics: the article's accuracy, Jaccard and Dice of every evaluated slice. Failed slice: slice Jaccard < 0.5.
3D metrics over the evaluated slices (stacked): Jaccard, Dice, HD95 (mm, real voxel sizes) and volume bias.
A missing, empty or wrongly shaped mask counts as an empty mask (Jaccard 0); its status is recorded.
"""
import numpy as np
import pandas as pd
from scipy import ndimage

from . import datasets as ds

SUBJECT_COLUMNS = ["tool", "dataset", "contrast", "subject", "session", "group", "n_slices", "jaccard", "dice", "accuracy",
                   "failed_slices", "failure_rate_pct", "jaccard_3d", "dice_3d", "hd95_mm", "volume_ref_mm3", "volume_tool_mm3",
                   "volume_bias_pct", "status"]
SLICE_COLUMNS = ["tool", "dataset", "contrast", "subject", "session", "group", "slice", "slice_rel", "jaccard", "dice", "accuracy",
                 "ref_area_mm2", "tool_area_mm2"]
BOX_COLUMNS = ["dataset", "contrast", "subject", "session", "group", "slice", "slice_rel", "box_jaccard", "box_dice", "box_accuracy",
               "box_area_pct", "box_x_left", "box_y_up", "box_x_right", "box_y_down", "ref_x_left", "ref_y_up", "ref_x_right",
               "ref_y_down", "full_auto_jaccard", "oracle_jaccard"]


def evaluate(mask, reference):
    """Accuracy (%), Jaccard index and Dice coefficient of a mask slice against a reference slice (as in the article)."""
    sam, ref = mask.astype(int), reference.astype(int)
    total = ref + sam
    if not (np.any(ref) and np.any(total)):
        return 0, 0, 0
    overlap = np.count_nonzero(total == 2)
    accuracy = 100 - round((np.count_nonzero(ref - sam) / np.count_nonzero(ref)) * 100, 2)
    jaccard = round((overlap / np.count_nonzero(total)), 4)
    dice = round(2 * (overlap / (np.count_nonzero(sam) + np.count_nonzero(ref))), 4)
    return accuracy, jaccard, dice


def hd95(a, b, spacing):
    """Symmetric 95th percentile Hausdorff distance (mm) between the surfaces of two binary volumes."""
    if not a.any() or not b.any():
        return np.nan
    surface_a, surface_b = a ^ ndimage.binary_erosion(a), b ^ ndimage.binary_erosion(b)
    a_to_b = ndimage.distance_transform_edt(~surface_b, sampling=spacing)[surface_a]
    b_to_a = ndimage.distance_transform_edt(~surface_a, sampling=spacing)[surface_b]
    return max(np.percentile(a_to_b, 95), np.percentile(b_to_a, 95))


def ids(scan):
    return dict(dataset=scan.dataset, contrast=scan.contrast, subject=scan.subject, session=scan.session, group=scan.group)


def score(tool, scan, mask_path, reference=None, spacing=None):
    """(subject row, slice rows) of a tool's mask for a scan."""
    reference = ds.load_reference(scan) if reference is None else reference
    spacing = ds.real_spacing(scan) if spacing is None else spacing
    status = "ok"
    try:
        mask = ds.load_mask(mask_path, scan)
    except Exception as error:   # unreadable file
        mask, status = None, f"unreadable ({type(error).__name__})"
    if mask is None:
        mask, status = np.zeros_like(reference), ("missing" if status == "ok" else status)
    elif mask.shape != reference.shape:
        mask, status = np.zeros_like(reference), f"wrong shape {mask.shape}"
    elif not mask.any():
        status = "empty"
    slices = ds.evaluated_slices(scan, reference)
    pixel_area = spacing[0] * spacing[1]
    rows = []
    for i, k in enumerate(slices):
        accuracy, jaccard, dice = evaluate(mask[:, :, k], reference[:, :, k])
        rows.append(dict(tool=tool, **ids(scan), slice=k + 1, slice_rel=i / (len(slices) - 1) if len(slices) > 1 else 0.0,
                         jaccard=jaccard, dice=dice, accuracy=accuracy, ref_area_mm2=reference[:, :, k].sum() * pixel_area,
                         tool_area_mm2=mask[:, :, k].sum() * pixel_area))
    table = pd.DataFrame(rows, columns=SLICE_COLUMNS)
    pred, ref = mask[:, :, slices], reference[:, :, slices]
    overlap, union = (pred & ref).sum(), (pred | ref).sum()
    voxel = float(np.prod(spacing))
    failed = int((table.jaccard < 0.5).sum())
    subject = dict(tool=tool, **ids(scan), n_slices=len(table), jaccard=table.jaccard.mean(), dice=table.dice.mean(),
                   accuracy=table.accuracy.mean(), failed_slices=failed, failure_rate_pct=100 * failed / len(table),
                   jaccard_3d=overlap / union if union else 0.0, dice_3d=2 * overlap / (pred.sum() + ref.sum()) if union else 0.0,
                   hd95_mm=hd95(pred, ref, spacing), volume_ref_mm3=ref.sum() * voxel, volume_tool_mm3=pred.sum() * voxel,
                   volume_bias_pct=100 * (pred.sum() - ref.sum()) / ref.sum(), status=status)
    return subject, table


# ------------------------------------------------------------------ box analysis (as benchmark_box_analysis.py)

def box_rectangles(box, reference_slice):
    """Filled box prompt and reference box of a rotated slice (end-exclusive edges, as in the article's box analysis)."""
    box_mask, reference_box = np.zeros(reference_slice.shape, np.uint8), np.zeros(reference_slice.shape, np.uint8)
    xl, yu, xr, yd = (max(0, round(c)) for c in box)
    box_mask[yu:yd, xl:xr] = 1
    idx = np.argwhere(reference_slice)
    if not idx.size:   # no brain in the reference slice (only possible with fixed evaluated-slice ranges)
        return box_mask, reference_box, (np.nan,) * 4
    (ryu, rxl), (ryd, rxr) = idx.min(axis=0), idx.max(axis=0)
    reference_box[ryu:ryd, rxl:rxr] = 1
    return box_mask, reference_box, (rxl, ryu, rxr, ryd)


def box_scores(scan, boxes, reference=None):
    """Per evaluated slice: SAMson Full-Auto box prompt vs reference bounding box (rotated-slice coordinates)."""
    reference = ds.load_reference(scan) if reference is None else reference
    z0, _, _, _ = scan.ap_slices(reference.shape[2])
    slices = ds.evaluated_slices(scan, reference)
    rows = []
    for i, k in enumerate(slices):
        box = boxes[k - z0]
        box_mask, reference_box, ref_box = box_rectangles(box, np.rot90(reference[:, :, k]))
        accuracy, jaccard, dice = evaluate(box_mask, reference_box)
        rows.append(dict(**ids(scan), slice=k + 1, slice_rel=i / (len(slices) - 1) if len(slices) > 1 else 0.0,
                         box_jaccard=jaccard, box_dice=dice, box_accuracy=accuracy,
                         box_area_pct=100 * box_mask.sum() / reference_box.sum() if reference_box.any() else np.nan,
                         box_x_left=box[0], box_y_up=box[1], box_x_right=box[2], box_y_down=box[3],
                         ref_x_left=ref_box[0], ref_y_up=ref_box[1], ref_x_right=ref_box[2], ref_y_down=ref_box[3]))
    return pd.DataFrame(rows)


def mask_agreement(a, b):
    """Voxels that differ and Jaccard between two boolean masks of the same grid."""
    if a is None or b is None or a.shape != b.shape:
        return dict(identical=False, differing_voxels=np.nan, jaccard=np.nan)
    union = (a | b).sum()
    return dict(identical=bool(np.array_equal(a, b)), differing_voxels=int((a != b).sum()),
                jaccard=float((a & b).sum() / union) if union else 1.0)
