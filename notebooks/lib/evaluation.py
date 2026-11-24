"""
Evaluation of every tool's masks on every scan -> results/{subjects,slices,boxes}.csv, and the final sorting of the
Semi-Auto candidate selection (outputs/semi_auto_selection.csv).
"""
import os
from os.path import join as pjoin
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from . import config, datasets as ds, metrics as mt


def _evaluate_scan(args):
    scan, tools, root = args
    reference = ds.load_reference(scan)
    spacing = ds.real_spacing(scan)
    subjects, slices = [], []
    for tool in tools:
        row, table = mt.score(tool, scan, ds.output_mask_path(root, tool, scan), reference, spacing)
        subjects.append(row)
        slices.append(table)
    boxes = None
    path = ds.boxes_path(root, scan)
    if os.path.exists(path) and "samson_full_auto" in tools:
        boxes = mt.box_scores(scan, np.load(path), reference)
        for tool, column in (("samson_full_auto", "full_auto_jaccard"), ("samson_oracle", "oracle_jaccard")):
            t = next((t for tt, t in zip(tools, slices) if tt == tool), None)
            boxes[column] = boxes.slice.map(dict(zip(t.slice, t.jaccard))) if t is not None and len(t) else np.nan
    return subjects, pd.concat(slices, ignore_index=True), boxes


def evaluate_all(scans, tools, root, workers=8, write=True, prefix=""):
    """Metrics of every tool on every scan (in parallel); written to <root>/results/{subjects,slices,boxes}.csv if write
    (with a prefix: <prefix>_subjects.csv and <prefix>_slices.csv)."""
    with ProcessPoolExecutor(workers) as pool:
        results = list(pool.map(_evaluate_scan, [(s, tools, root) for s in scans], chunksize=1))
    subjects = pd.DataFrame([r for res in results for r in res[0]], columns=mt.SUBJECT_COLUMNS)
    slices = pd.concat([res[1] for res in results], ignore_index=True)[mt.SLICE_COLUMNS]
    box_tables = [res[2] for res in results if res[2] is not None]
    boxes = pd.concat(box_tables, ignore_index=True).reindex(columns=mt.BOX_COLUMNS) if box_tables else pd.DataFrame(columns=mt.BOX_COLUMNS)
    if write:
        folder = pjoin(root, "results")
        os.makedirs(folder, exist_ok=True)
        name = (prefix + "_") if prefix else ""
        subjects.to_csv(pjoin(folder, f"{name}subjects.csv"), index=False)
        slices.to_csv(pjoin(folder, f"{name}slices.csv"), index=False)
        if not prefix:
            boxes.to_csv(pjoin(folder, "boxes.csv"), index=False)
    return subjects, slices, boxes


def finalize_selection(root):
    """outputs/semi_auto_selection.csv without the rows of interrupted or repeated passes (last row of every slice kept)."""
    path = pjoin(root, "outputs", "semi_auto_selection.csv")
    if not os.path.exists(path):
        return None
    t = pd.read_csv(path, dtype={"subject": str, "session": str}, keep_default_na=False)
    t = t.drop_duplicates(["dataset", "contrast", "subject", "session", "slice"], keep="last")
    t.to_csv(path, index=False)
    return t
