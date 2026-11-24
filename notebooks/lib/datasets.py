"""
Datasets, scans, orientation handling and file conventions.

Every scan is described by a `Scan`. Its image is read in its stored ("raw") orientation with its x10 header, re-oriented
losslessly to the canonical SAMson/template orientation (axis permutation `perm`, then reversal of the re-oriented axes
in `flips`: axis 2 = coronal slices, slice 0 = caudal, dorsal up after `np.rot90`), identically for every tool. A tool's
mask (canonical grid) is mapped back with the inverse transform and saved on the raw grid with the dataset image's
header. Metrics are computed in the canonical orientation (slice = coronal slice).
"""
import os
import csv
import json
from dataclasses import dataclass, field, asdict
from os.path import join as pjoin

import numpy as np
import nibabel as nib
from nibabel.orientations import apply_orientation, inv_ornt_aff

from . import config

MODALITY_DIR = {"T2w": "anat", "T2starw": "anat", "DWI": "dwi", "BOLD": "func"}


@dataclass
class Scan:
    dc: str                     # dataset-contrast key, e.g. "D4_T2starw"
    dataset: str                # "D4"
    contrast: str               # "T2starw"
    subject: str                # subject ID (without "sub-")
    session: str                # session label ("" if none)
    group: str                  # group (D1: age in months; otherwise from participants.tsv or "")
    stem: str                   # image file name without .nii(.gz): output masks are <stem>_mask.nii.gz
    image: str                  # dataset image (raw orientation)
    reference: str              # reference (manual) mask on the image grid
    perm: tuple = (0, 1, 2)     # canonical axis i = raw axis perm[i]
    flips: tuple = (0, 0, 0)    # canonical axes reversed after the permutation
    x10_factor: float = 1.0     # factor that turns the stored header into the x10 header (harmonized datasets: 1)
    ap: tuple = (0, -1, 0, 39)  # SAMson AP specification (first, last processed slice; template slices); -1 = last slice
    otsu: tuple = None          # median_otsu (radius, passes); None = SAMson's rule ((2, 5) if rows < 140 else (1, 3))
    evaluation: dict = field(default_factory=dict)   # {"first": int, "last": int (-1 = last), "require_brain": bool}
    extra: dict = field(default_factory=dict)
    rel_dir: str = ""           # folder of the image relative to <datasets>/<Dk> (e.g. "anat", "anat/3_month"); mirrored by the outputs

    def __post_init__(self):
        if not self.rel_dir:
            self.rel_dir = MODALITY_DIR[self.contrast]

    @property
    def modality(self):
        return self.rel_dir.split("/")[0]

    @property
    def key(self):
        return (self.dc, self.stem)

    def ap_slices(self, n_slices):
        z0, z1, t0, t1 = self.ap
        return z0 % n_slices, z1 % n_slices, t0, t1


# ------------------------------------------------------------------ orientation

def orientation(perm, flips):
    """nibabel orientation array of the re-orientation: raw axis i -> (canonical axis, direction)."""
    ornt = np.zeros((3, 2))
    for new, old in enumerate(perm):
        ornt[old] = [new, -1 if flips[new] else 1]
    return ornt


def to_canonical(array, perm, flips):
    """Raw -> canonical array (transpose, then flips of the canonical axes); lossless."""
    array = np.transpose(array, perm)
    for axis, flip in enumerate(flips):
        if flip:
            array = np.flip(array, axis)
    return array


def from_canonical(array, perm, flips):
    """Canonical -> raw array (inverse of to_canonical)."""
    for axis, flip in enumerate(flips):
        if flip:
            array = np.flip(array, axis)
    return np.transpose(array, np.argsort(perm))


def first_volume(data):
    return data.reshape(data.shape[:3] + (-1,))[..., 0] if data.ndim > 3 else data


def canonical_affine(scan, image, raw_affine=None):
    """Affine (x10 header) of the canonical array of a scan's image (raw_affine: instead of the image's stored affine)."""
    scale = np.diag([scan.x10_factor] * 3 + [1])
    raw = image.affine if raw_affine is None else np.asarray(raw_affine, dtype=float)
    return scale @ raw @ inv_ornt_aff(orientation(scan.perm, scan.flips), image.shape[:3])


def real_spacing(scan, image=None):
    """Real voxel size (mm) along the canonical axes: header voxel size x x10_factor / 10, permuted."""
    image = image or nib.load(scan.image)
    zooms = np.array(image.header.get_zooms()[:3], dtype=float) * scan.x10_factor / 10
    return zooms[list(scan.perm)]


def load_canonical(scan, dtype=None):
    """(canonical image array, canonical x10 affine). dtype None: float64 (get_fdata, as SAMson reads images)."""
    image = nib.load(scan.image)
    data = first_volume(image.get_fdata() if dtype is None else np.asanyarray(image.dataobj).astype(dtype))
    return np.ascontiguousarray(to_canonical(data, scan.perm, scan.flips)), canonical_affine(scan, image)


def load_reference(scan):
    """Canonical boolean reference mask."""
    image = nib.load(scan.reference)
    assert image.shape[:3] == nib.load(scan.image).shape[:3], (scan.reference, image.shape)
    return np.ascontiguousarray(to_canonical(first_volume(np.asanyarray(image.dataobj)) > 0, scan.perm, scan.flips))


def load_mask(path, scan):
    """Canonical boolean tool mask saved on the raw grid (None if missing; raises on a wrong shape)."""
    if not os.path.exists(path):
        return None
    data = np.squeeze(np.asanyarray(nib.load(path).dataobj))
    return np.ascontiguousarray(to_canonical(data > 0.5, scan.perm, scan.flips))


def write_nifti(data, affine, path):
    """NIfTI with qform = sform = affine (code 1) and mm units."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = nib.Nifti1Image(data, affine)
    out.set_qform(affine, 1)
    out.set_sform(affine, 1)
    out.header.set_xyzt_units("mm")
    tmp = path + ".part.nii.gz"
    nib.save(out, tmp)
    os.replace(tmp, path)
    return path


def label_convention(affine):
    """Header label convention of a canonical affine: "anatomical" (axis 1 -> S, axis 2 -> A), "template" (axis 1 -> A,
    axis 2 -> S) or the axis codes themselves if neither."""
    codes = "".join(nib.aff2axcodes(affine))
    return {("S", "A"): "anatomical", ("A", "S"): "template"}.get((codes[1], codes[2]), codes)


def relabel(affine, target):
    """Canonical affine with the header labels of `target` ("anatomical" or "template"): if the stored labels are in the
    other convention, the direction cosines of axes 1 and 2 are exchanged (voxel sizes, origin and axis 0 kept)."""
    current = label_convention(affine)
    if current == target:
        return affine
    if current not in ("anatomical", "template"):
        raise ValueError(f"unexpected header labels {current}: relabelling to {target} is not defined")
    out = np.array(affine, dtype=float)
    zooms = np.sqrt((out[:3, :3] ** 2).sum(axis=0))
    c1, c2 = out[:3, 1] / zooms[1], out[:3, 2] / zooms[2]
    out[:3, 1], out[:3, 2] = c2 * zooms[1], c1 * zooms[2]
    assert label_convention(out) == target, (nib.aff2axcodes(affine), nib.aff2axcodes(out))
    return out


def check_header_labels(scans):
    """Stored labels of every scan's canonical affine against the expected convention of its dataset; raises if any
    scan deviates (relabelling would then be undefined or wrong)."""
    import pandas as pd
    rows = []
    for s in scans:
        image = nib.load(s.image)
        conv = label_convention(canonical_affine(s, image))
        rows.append(dict(dc=s.dc, stem=s.stem, labels="".join(nib.aff2axcodes(canonical_affine(s, image))), convention=conv,
                         expected=config.EXPECTED_LABELS.get(s.dataset), ok=conv == config.EXPECTED_LABELS.get(s.dataset)))
    table = pd.DataFrame(rows)
    bad = table[~table.ok]
    if len(bad):
        raise ValueError("header labels differ from the expected convention:\n" + bad.to_string())
    return table


def stage_input(scan, folder, header="x10"):
    """
    The tool input of a scan: canonical image (float32) with the x10 header (header="x10") or with the real voxel size
    (header="true": BET4Animal on D2 and D3, config.DATASET_SPECIFIC_SETTINGS); D2: the affine the tools received in the benchmark
    (config.TOOL_INPUT_AFFINE_X10); suffix "_anat" / "_tmpl": header labels relabelled to the anatomical /
    template convention (columns 1 and 2 of the direction cosines exchanged where needed; voxel data unchanged);
    suffix "_flip1": axis 1 reversed (option B of the BEN/RBM orientation decision). Written once to <folder>/<header>/<Dk>/<image folder relative to Dk>/<stem>.nii.gz.
    """
    path = pjoin(folder, header, scan.dataset, scan.rel_dir, scan.stem + ".nii.gz")
    if os.path.exists(path):
        return path
    data, affine = load_canonical(scan, dtype=np.float32)
    if scan.dataset in config.TOOL_INPUT_AFFINE_X10:   # header the tools received in the benchmark (config.TOOL_INPUT_AFFINE_X10)
        affine = canonical_affine(scan, nib.load(scan.image), config.TOOL_INPUT_AFFINE_X10[scan.dataset])
    if header.startswith("true"):
        affine = np.diag([0.1, 0.1, 0.1, 1]) @ affine
    if "_anat" in header or "_tmpl" in header:   # header labels of the tool (voxel data unchanged)
        expected = config.EXPECTED_LABELS.get(scan.dataset)
        if expected and label_convention(affine) != expected:
            raise ValueError(f"{scan.stem}: header labels {nib.aff2axcodes(affine)} are not the expected {expected} convention")
        affine = relabel(affine, "anatomical" if "_anat" in header else "template")
    if header.endswith("_flip1"):   # option B for BEN/RBM: canonical axis 1 reversed (dorsal at the top row of the array)
        flip = np.array([[1, 0, 0, 0], [0, -1, 0, data.shape[1] - 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        data, affine = np.ascontiguousarray(data[:, ::-1, :]), affine @ flip
    return write_nifti(data.astype(np.float32), affine, path)


def output_mask_path(root, tool, scan):
    return pjoin(root, "outputs", "masks", tool, scan.dataset, scan.rel_dir, scan.stem + "_mask.nii.gz")


def boxes_path(root, scan, ext=".npy"):
    return pjoin(root, "outputs", "boxes", scan.dataset, scan.rel_dir, scan.stem + "_boxes" + ext)


def save_mask(mask_canonical, scan, path):
    """Save a canonical mask on the raw grid of the dataset image (uint8 {0,1}) with the image's (x10) header."""
    image = nib.load(scan.image)
    raw = from_canonical(np.asarray(mask_canonical) > 0, scan.perm, scan.flips).astype(np.uint8)
    assert raw.shape == image.shape[:3], (raw.shape, image.shape)
    header = image.header.copy()
    header.set_data_dtype(np.uint8)
    if len(header.get_data_shape()) > 3:
        header.set_data_shape(image.shape[:3])
    affine = np.diag([scan.x10_factor] * 3 + [1]) @ image.affine
    out = nib.Nifti1Image(raw, affine, header)
    out.header.set_slope_inter(1, 0)
    qcode, scode = int(image.header["qform_code"]) or 1, int(image.header["sform_code"]) or 1
    out.set_qform(affine, qcode)
    out.set_sform(affine, scode)
    out.header.set_xyzt_units("mm")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part.nii.gz"
    nib.save(out, tmp)
    os.replace(tmp, path)
    return path


# ------------------------------------------------------------------ evaluated slices

def evaluated_slices(scan, reference):
    """Evaluated slices (canonical indices) of a scan: the dataset rule, identical for every tool."""
    n = reference.shape[2]
    ev = scan.evaluation or {}
    if "slices" in ev:
        slices = [k % n for k in ev["slices"]]
    else:
        z0, z1, _, _ = scan.ap_slices(n)
        first = ev.get("first", z0) % n
        last = ev.get("last", z1) % n
        slices = list(range(first, last + 1))
    if ev.get("require_brain", True):
        slices = [k for k in slices if reference[:, :, k].any()]
    return slices


# ------------------------------------------------------------------ scan lists

def list_scans(dcs=None):
    """Scans of the harmonized datasets (all 409, or those of the dataset-contrasts `dcs`), in DC_ORDER and stem order."""
    scans = _new_scans()
    if dcs:
        scans = [s for s in scans if s.dc in dcs]
    order = {k: i for i, k in enumerate(config.DC_ORDER)}
    return sorted(scans, key=lambda s: (order.get(s.dc, 99), s.stem))


# ------------------------------------------------------------------ harmonized datasets (<release>/datasets)

def _read_tsv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _by_slices(entries, n, key):
    for e in entries:
        if int(e["n_slices"]) == n:
            return e
    raise KeyError(f"no {key} entry for {n} slices")


def _new_scans():
    """Scans of the harmonized datasets: dataset_settings.json (re-orientation, SAMson AP specification, median_otsu,
    evaluated slices; keyed by dataset-contrast and by the number of slices) and participants.tsv (one row per scan)."""
    settings = json.load(open(pjoin(config.DATASETS, "dataset_settings.json")))
    out = []
    for row in _read_tsv(pjoin(config.DATASETS, "participants.tsv")):
        dc = f"{row['dataset']}_{row['contrast']}"
        cfg = settings[dc]
        n = int(row["n_slices"])
        canonical = cfg["canonical"]
        ap = _by_slices(cfg["samson_ap"], n, "samson_ap")
        otsu = _by_slices(cfg["median_otsu"], n, "median_otsu")
        ev = cfg["evaluated_slices"]
        if ev["rule"] == "fixed":
            evaluation = dict(slices=[int(k) for k in ev["slices"][str(n)]], require_brain=False)
        elif ev["rule"] == "ap_with_reference":
            evaluation = dict(first=ap["image_slices"][0], last=ap["image_slices"][1], require_brain=True)
        else:
            raise ValueError(f"unknown evaluated-slice rule {ev['rule']} ({dc})")
        image = pjoin(config.DATASETS, row["image"])
        out.append(Scan(dc, row["dataset"], row["contrast"], subject=row["subject"], session=row.get("session", "") or "",
                        group=row.get("group", "") or "", stem=os.path.basename(image)[:-7], image=image,
                        reference=pjoin(config.DATASETS, row["mask"]),
                        perm=tuple(int(x) for x in canonical["perm"]), flips=tuple(int(x) for x in canonical["flips"]),
                        x10_factor=1.0, ap=(int(ap["image_slices"][0]), int(ap["image_slices"][1]),
                                            int(ap["template_slices"][0]), int(ap["template_slices"][1])),
                        otsu=(int(otsu["median_radius"]), int(otsu["numpass"])), evaluation=evaluation,
                        extra=dict(animal=row.get("animal", ""), n_slices=n),
                        rel_dir=os.path.relpath(os.path.dirname(image), pjoin(config.DATASETS, row["dataset"]))))
    return out


def scan_table(scans):
    import pandas as pd
    return pd.DataFrame([dict(dc=s.dc, dataset=s.dataset, contrast=s.contrast, subject=s.subject, session=s.session,
                              group=s.group, stem=s.stem, image=s.image) for s in scans])
