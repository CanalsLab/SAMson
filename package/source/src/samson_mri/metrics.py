"""
Accuracy metrics of the article (functions of ``SAMson_opt.ipynb``) and helpers to evaluate saved masks.

Per-slice metrics are computed on coronal slices: ``accuracy`` (%), the ``jaccard`` index and the ``dice`` coefficient
of a mask slice against the reference slice. Subject values are means over the evaluated slices.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def evaluate(mask, reference):
    """
    Accuracy (%), Jaccard index and Dice coefficient of a mask against a reference mask (0 if the reference is empty).
    Accuracy is 100 minus the percentage of disagreeing pixels relative to the reference mask size.
    """
    sam, ref = mask.astype(int), reference.astype(int)
    total = ref + sam
    if not (np.any(ref) and np.any(total)):
        return 0, 0, 0
    overlap = np.count_nonzero(total == 2)
    accuracy = 100 - round((np.count_nonzero(ref - sam) / np.count_nonzero(ref)) * 100, 2)
    jaccard = round((overlap / np.count_nonzero(total)), 4)
    dice = round(2 * (overlap / (np.count_nonzero(sam) + np.count_nonzero(ref))), 4)
    return accuracy, jaccard, dice


def evaluate_slices(masks, reference, slices=None):
    """
    Metrics of every candidate mask of every slice (or of the given slice indexes) against the reference volume.
    masks: (n_slices, n_candidates, H, W) masks of rotated slices; reference: volume (x, y, n_slices).
    Returns a table with 1-based slice numbers and 0-based candidate indexes (as in best_mask_indexes).
    """
    slices = range(len(masks)) if slices is None else slices
    rows = [(s + 1, c, *evaluate(masks[s, c], np.rot90(reference[:, :, s]))) for s in slices for c in range(masks.shape[1])]
    return pd.DataFrame(rows, columns=["slice", "candidate", "accuracy", "jaccard", "dice"])


def subject_means(metrics):
    """Subject mean accuracy, Jaccard and Dice over slices, rounded as in the article."""
    return {m: round(sum(metrics[m]) / len(metrics), d) for m, d in (("accuracy", 2), ("jaccard", 4), ("dice", 4))}


def find_best_mask_indexes(metrics):
    """Index (0, 1 or 2) of the candidate with the highest Jaccard index in each slice (first one if tied)."""
    return metrics.pivot(index="slice", columns="candidate", values="jaccard").to_numpy().argmax(axis=1).tolist()


def compare_volumes(mask, reference, ap_axis=2, slices=None):
    """
    Compare two binary volumes on the same grid.

    Per-slice metrics are computed on the slices perpendicular to ``ap_axis`` (the coronal slices) that contain brain in
    the reference (or on ``slices``). Returns (per-slice table, summary dict) with the slice means and the 3D Jaccard
    and Dice over the whole volume.
    """
    mask, reference = np.asarray(mask) > 0, np.asarray(reference) > 0
    if mask.shape != reference.shape:
        raise ValueError(f"The mask {mask.shape} and the reference {reference.shape} are not on the same grid.")
    mask, reference = np.moveaxis(mask, ap_axis, 2), np.moveaxis(reference, ap_axis, 2)
    if slices is None:
        slices = [k for k in range(reference.shape[2]) if reference[:, :, k].any()]
    rows = [(k, *evaluate(mask[:, :, k], reference[:, :, k])) for k in slices]
    table = pd.DataFrame(rows, columns=["slice", "accuracy", "jaccard", "dice"])
    overlap, union = np.count_nonzero(mask & reference), np.count_nonzero(mask | reference)
    total = np.count_nonzero(mask) + np.count_nonzero(reference)
    summary = dict(n_slices=len(table),
                   accuracy=float(table.accuracy.mean()) if len(table) else np.nan,
                   jaccard=float(table.jaccard.mean()) if len(table) else np.nan,
                   dice=float(table.dice.mean()) if len(table) else np.nan,
                   failed_slices=int((table.jaccard < 0.5).sum()),
                   jaccard_3d=overlap / union if union else np.nan,
                   dice_3d=2 * overlap / total if total else np.nan)
    return table, summary
