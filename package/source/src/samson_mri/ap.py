"""
Help to choose SAMson's anteroposterior specification: which template slices match the first and last processed slice.

SAMson maps the processed slices linearly onto the template range: slice ``first`` -> template slice ``t0``, slice
``last`` -> ``t1``. The template box shapes at those positions, scaled on the brain box that median_otsu finds in the
*anchor* slice (the processed slice that falls closest to the template's centre, 19.5), become the box prompts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._algorithm import anchor_slice, bounding_boxes, boxes_from_central_box, interpolated_coordinates
from .template import Template, load_template

TEMPLATE_SPACING_MM = 0.4          # Distance between template slices
TYPICAL_LENGTH_RATIO = (0.8, 1.5)  # Subject range length / template range length in the article and benchmark data (1.0-1.25)


def template_positions(n_slices: int, t0: int, t1: int) -> np.ndarray:
    """Template slice (fractional) matched by each of n_slices processed slices."""
    return np.linspace(t0, t1, n_slices)


def plausible_voxel_size(voxel_size) -> bool:
    """True if the in-plane voxel size looks like a true mouse/rat MRI voxel size in mm (not a x10 header)."""
    return 0.01 <= float(min(voxel_size[:2])) <= 0.5


def length_ratio(n_processed: int, slice_mm: float, t0: int, t1: int) -> float:
    """Length of the processed range (mm) / length of the template range (mm)."""
    return (n_processed - 1) * slice_mm / max((t1 - t0) * TEMPLATE_SPACING_MM, 1e-9)


def linked_range(t0: int, n_processed: int, slice_mm: float, ratio: float = 1.1, n_template: int = 40) -> tuple:
    """Template range starting at t0 whose length matches the processed range (length ratio subject/template = ratio)."""
    span = (n_processed - 1) * slice_mm / (ratio * TEMPLATE_SPACING_MM)
    t1 = int(round(t0 + span))
    return t0, int(min(max(t1, t0), n_template - 1))


def check_range(n_processed: int, t0: int, t1: int, voxel_size=None, header_scale: float = 1.0, template: Template | None = None) -> list:
    """Possible problems of an anteroposterior specification: a list of dicts with a 'short' and a 'detail' message."""
    template = template or load_template()
    warnings = []
    centre = (template.n_slices - 1) / 2
    if n_processed > 1 and not (t0 <= centre <= t1):
        anchor = anchor_slice(n_processed, t0, t1, template.n_slices)
        position = template_positions(n_processed, t0, t1)[anchor]
        warnings.append(dict(
            short=f"the template range does not include the template centre ({centre:g})",
            detail=f"The template range {t0}-{t1} does not include the template's centre ({centre:g}). SAMson measures the brain box "
                   f"on the processed slice closest to it (here slice {anchor} of the range, matched to template slice {position:.1f}) "
                   "and treats it as the central box, so the box sizes may be biased. This is expected only for images that do not "
                   "cover the middle of the brain."))
    if voxel_size is not None and n_processed > 1 and t1 > t0:
        size = [float(v) * header_scale for v in voxel_size]
        if plausible_voxel_size(size):
            ratio = length_ratio(n_processed, size[2], t0, t1)
            low, high = TYPICAL_LENGTH_RATIO
            if not low <= ratio <= high:
                warnings.append(dict(
                    short=f"length ratio {ratio:.2f} outside the usual {low}-{high}",
                    detail=f"The processed slices span {(n_processed - 1) * size[2]:.1f} mm and the template range "
                           f"{(t1 - t0) * TEMPLATE_SPACING_MM:.1f} mm (ratio {ratio:.2f}; usually {low}-{high}). Check that the "
                           "template slices really match the first and last processed slices."))
    return warnings


def reference_boxes(reference_masks, first: int, last: int) -> list:
    """Boxes of the reference masks (SAMson orientation) in the processed slices."""
    return [bounding_boxes(np.asarray(mask[:, :, first:last + 1], dtype=float)) for mask in reference_masks]


def box_agreement(reference_masks, first: int, last: int, t0: int, t1: int, template: Template | None = None, boxes=None) -> float:
    """
    Mean box IoU between the boxes SAMson would build from the template range (anchored on the *reference* box of the
    anchor slice, to leave the Otsu step out) and the boxes of the reference masks, over the processed slices.
    reference_masks: list of reference masks in SAMson's orientation (or boxes: their reference_boxes).
    """
    template = template or load_template()
    boxes = reference_boxes(reference_masks, first, last) if boxes is None else boxes
    n = last - first + 1
    coordinates = interpolated_coordinates(n, t0, t1, template.relative_coordinates)
    anchor = anchor_slice(n, t0, t1, template.n_slices)
    scores = []
    for subject_boxes in boxes:
        if not np.any(subject_boxes[anchor]):
            continue
        predicted = boxes_from_central_box(subject_boxes[anchor], coordinates)
        scores.append(np.mean([_box_iou(p, r) for p, r, has in zip(predicted, subject_boxes, subject_boxes.any(axis=1)) if has]))
    return float(np.mean(scores)) if scores else np.nan


def suggest_from_masks(reference_masks, first: int, last: int, template: Template | None = None, top: int = 10,
                       slice_mm: float | None = None) -> pd.DataFrame:
    """
    Rank template ranges by their box agreement with reference masks (a few manually segmented subjects are enough).
    If slice_mm (true slice spacing) is given, ranges whose length ratio is outside the typical interval are left out.
    """
    template = template or load_template()
    n = last - first + 1
    boxes = reference_boxes(reference_masks, first, last)
    rows = []
    for t0 in range(template.n_slices - 1):
        for t1 in range(t0 + 1, template.n_slices):
            if slice_mm is not None and not (TYPICAL_LENGTH_RATIO[0] <= length_ratio(n, slice_mm, t0, t1) <= TYPICAL_LENGTH_RATIO[1]):
                continue
            rows.append((t0, t1, box_agreement(None, first, last, t0, t1, template, boxes=boxes)))
    table = pd.DataFrame(rows, columns=["template_first", "template_last", "box_iou"]).dropna()
    return table.sort_values("box_iou", ascending=False).head(top).reset_index(drop=True)


def _box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0
