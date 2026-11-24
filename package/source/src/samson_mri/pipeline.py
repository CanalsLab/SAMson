"""
SAMson's workflow: box prompts, SAM masks, selection of the candidates and output files.

Typical use::

    from samson_mri import SAMson, DatasetSpec
    spec = DatasetSpec(template=(4, 33))                       # see samson_mri.spec
    samson = SAMson()                                           # SAM ViT-H on the GPU if available
    result = samson.segment("sub-01_T2w.nii.gz", spec)          # Full-Auto
    result.save("masks/")                                       # masks/sub-01_T2w_mask.nii.gz, ..._samson.json, ..._qc.png
"""
from __future__ import annotations

import csv
import datetime
import json
import os
import platform
import time
from dataclasses import dataclass, field

import nibabel as nib
import numpy as np

from ._algorithm import anchor_slice, boxes_from_central_box, central_box, interpolated_coordinates, segment_volume
from ._version import __version__
from .io import Subject, load_mask, load_subject, save_like, subject_stem
from .metrics import compare_volumes, evaluate_slices, find_best_mask_indexes
from .model import DEFAULT_MODEL, load_predictor
from .spec import DatasetSpec
from .template import load_template

MODES = ("full-auto", "semi-auto")
FULL_AUTO = -1   # Selection value of the Full-Auto mask of a slice (0, 1, 2: SAM's three candidates, as in the notebook)


@dataclass
class BoxPrompts:
    """SAMson's box prompts of a subject, with the intermediate results (shown by the app and the preview)."""
    first: int                  # first and last processed slice (SAMson orientation)
    last: int
    boxes: np.ndarray           # (n_processed, 4) [x_left, y_up, x_right, y_down] in rotated-slice coordinates
    otsu: tuple                 # (median_radius, numpass)
    anchor: int                 # index (within the processed slices) of the slice whose Otsu box anchors the boxes
    central_box: tuple          # Otsu box of the anchor slice
    template_positions: np.ndarray   # template slice (0-39, fractional) matched by every processed slice

    @property
    def processed_slices(self) -> list:
        return list(range(self.first, self.last + 1))


def compute_boxes(subject: Subject, spec: DatasetSpec | None = None, template=None) -> BoxPrompts:
    """Box prompts of a subject (no SAM needed): generate_boxes of the notebook on the processed slices."""
    spec = spec or subject.spec
    template = template or load_template()
    spec.check_template(template.n_slices)
    first, last = spec.slice_range(subject.n_slices)
    part = subject.volume[:, :, first:last + 1]
    otsu = tuple(int(v) for v in spec.otsu_parameters(part))
    t0, t1 = spec.template
    coordinates = interpolated_coordinates(part.shape[2], t0, t1, template.relative_coordinates)
    box = central_box(part, otsu[0], otsu[1], t0, t1, template.n_slices)
    boxes = boxes_from_central_box(box, coordinates)
    return BoxPrompts(first=first, last=last, boxes=boxes, otsu=otsu,
                      anchor=anchor_slice(part.shape[2], t0, t1, template.n_slices),
                      central_box=tuple(int(v) for v in box), template_positions=np.linspace(t0, t1, part.shape[2]))


@dataclass
class SegmentationResult:
    """Masks of one subject. Arrays of masks are (n_processed_slices, n, H, W) masks of rotated slices (as in the notebook)."""
    subject: Subject
    spec: DatasetSpec
    mode: str
    prompts: BoxPrompts
    full_auto: np.ndarray                 # (n, 1, H, W) Full-Auto masks
    candidates: np.ndarray | None = None  # (n, 3, H, W) Semi-Auto candidates (None in Full-Auto mode)
    selection: list | None = None         # per processed slice: -1 (Full-Auto mask) or 0, 1, 2 (candidate)
    selection_source: str = "full-auto"   # "full-auto", "reference", "manual"
    model: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    reference_path: str | None = None
    metrics: object = None                # per-slice metrics against a reference (pandas DataFrame), if one was given

    def __post_init__(self):
        if self.selection is None:
            self.selection = [FULL_AUTO] * len(self.full_auto)

    # ---- masks ------------------------------------------------------------------------------------------------------
    @property
    def n_processed(self) -> int:
        return len(self.full_auto)

    def selected_slices(self) -> np.ndarray:
        """(n, H, W) mask of every processed slice according to the selection."""
        out = []
        for s, choice in enumerate(self.selection):
            out.append(self.full_auto[s, 0] if choice == FULL_AUTO else self.candidates[s, choice])
        return np.array(out)

    def _to_volume(self, slices: np.ndarray) -> np.ndarray:
        """(n, H, W) rotated slices -> volume in SAMson's orientation (unprocessed slices empty), as save_nifti_mask."""
        volume = np.zeros(self.subject.volume.shape, dtype=np.uint8)
        volume[:, :, self.prompts.first:self.prompts.last + 1] = np.stack([np.rot90(m, k=-1) for m in slices], axis=2)
        return volume

    @property
    def mask_samson(self) -> np.ndarray:
        """Final mask in SAMson's orientation (uint8 0/1)."""
        return self._to_volume(self.selected_slices())

    @property
    def mask(self) -> np.ndarray:
        """Final mask on the grid and in the orientation of the input image (uint8 0/1)."""
        return self.subject.to_input_space(self.mask_samson)

    def candidates_volume(self) -> np.ndarray:
        """4D uint8 array on the input grid: volume 0 = Full-Auto mask, volumes 1-3 = candidates 0-2 (Semi-Auto only)."""
        if self.candidates is None:
            raise ValueError("Candidates exist in Semi-Auto mode only.")
        volumes = [self._to_volume(self.full_auto[:, 0])] + [self._to_volume(self.candidates[:, c]) for c in range(self.candidates.shape[1])]
        return self.subject.to_input_space(np.stack(volumes, axis=3))

    # ---- selection --------------------------------------------------------------------------------------------------
    def select(self, choices, source: str = "manual") -> "SegmentationResult":
        """Choose the mask of every processed slice: one value for all slices or a list (-1 = Full-Auto, 0-2 = candidate)."""
        choices = [int(choices)] * self.n_processed if np.isscalar(choices) else [int(c) for c in choices]
        if len(choices) != self.n_processed:
            raise ValueError(f"{len(choices)} choices given for {self.n_processed} processed slices.")
        allowed = {FULL_AUTO} | (set(range(self.candidates.shape[1])) if self.candidates is not None else set())
        if not set(choices) <= allowed:
            raise ValueError(f"Choices must be in {sorted(allowed)} (got {sorted(set(choices) - allowed)}).")
        self.selection, self.selection_source = choices, source
        return self

    def select_by_reference(self, reference) -> "SegmentationResult":
        """The article's Semi-Auto: in every slice, the candidate with the highest Jaccard index against a reference mask."""
        if self.candidates is None:
            raise ValueError("Selection by reference needs the Semi-Auto candidates (mode='semi-auto').")
        part = self._reference_part(reference)
        best = find_best_mask_indexes(evaluate_slices(self.candidates, part))
        return self.select(best, source="reference")

    def _reference_part(self, reference) -> np.ndarray:
        if isinstance(reference, (str, os.PathLike)):
            self.reference_path = os.path.abspath(reference)
            reference = load_mask(str(reference), self.subject)
        reference = self.subject.spec.orientation.apply(np.asarray(reference))
        return reference[:, :, self.prompts.first:self.prompts.last + 1]

    def evaluate(self, reference) -> "SegmentationResult":
        """Per-slice metrics of the final mask against a reference mask (stored in ``metrics``)."""
        part = self._reference_part(reference)
        self.metrics = evaluate_slices(self.selected_slices()[:, None], part)
        return self

    def compare(self, reference) -> dict:
        """Summary metrics of the final mask against a reference on the input grid (slices with brain in the reference)."""
        if isinstance(reference, (str, os.PathLike)):
            reference = load_mask(str(reference), self.subject)
        reference_samson = self.subject.spec.orientation.apply(np.asarray(reference) > 0)
        return compare_volumes(self.mask_samson, reference_samson, ap_axis=2)[1]

    # ---- output -----------------------------------------------------------------------------------------------------
    def output_paths(self, output_dir: str) -> dict:
        stem = self.subject.stem
        return {key: os.path.join(output_dir, stem + suffix) for key, suffix in
                (("mask", "_mask.nii.gz"), ("candidates", "_candidates.nii.gz"), ("sidecar", "_samson.json"),
                 ("qc", "_qc.png"), ("brain", "_brain.nii.gz"))}

    def sidecar(self) -> dict:
        """Everything needed to reproduce and interpret the outputs (written as <stem>_samson.json)."""
        n_masks = int(self.mask_samson.sum())
        voxel = float(np.prod(self.subject.image.header.get_zooms()[:3]))
        return {
            "samson_mri_version": __version__,
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "input": {"path": self.subject.path, "shape": list(self.subject.image.shape), "volume_used": self.spec.volume},
            "spec": self.spec.to_dict(),
            "mode": self.mode,
            "processed_slices": [self.prompts.first, self.prompts.last],
            "slice_order": "SAMson orientation (0 = caudal); see docs/io_conventions.md",
            "otsu": list(self.prompts.otsu),
            "anchor_slice": self.prompts.first + self.prompts.anchor,
            "central_box": [int(v) for v in self.prompts.central_box],
            "boxes": np.round(self.prompts.boxes, 3).tolist(),
            "selection": self.selection,
            "selection_source": self.selection_source,
            "selection_legend": {"-1": "Full-Auto mask", "0": "candidate 0", "1": "candidate 1", "2": "candidate 2"},
            "reference": self.reference_path,
            "model": self.model,
            "mask_voxels": n_masks,
            "mask_volume_header_units": n_masks * voxel,
            "timings_s": {k: round(v, 2) for k, v in self.timings.items()},
            "environment": _environment(),
        }

    def save(self, output_dir: str, qc: bool = True, save_brain: bool | None = None) -> dict:
        """
        Write the mask (and, in Semi-Auto, the candidates), the JSON sidecar and the QC figure; returns their paths.
        save_brain: also write the brain-extracted image; None (default) = only if one exists already (so that it is
        updated when the selection changes).
        """
        paths = self.output_paths(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        if save_brain is None:
            save_brain = os.path.exists(paths["brain"])
        written = {"mask": save_like(self.mask, self.subject.image, paths["mask"])}
        if self.candidates is not None:
            written["candidates"] = save_like(self.candidates_volume(), self.subject.image, paths["candidates"])
        if save_brain:
            brain = self.subject.data * self.mask
            written["brain"] = save_like(brain, self.subject.image, paths["brain"], dtype=np.float32)
        if qc:
            from .viz import qc_figure, save_figure
            written["qc"] = save_figure(qc_figure(self), paths["qc"])
        sidecar = self.sidecar()
        sidecar["outputs"] = {k: os.path.basename(v) for k, v in written.items()}
        with open(paths["sidecar"], "w") as f:
            json.dump(sidecar, f, indent=1)
        written["sidecar"] = paths["sidecar"]
        return written


def _environment() -> dict:
    versions = {"python": platform.python_version()}
    for module in ("numpy", "torch", "nibabel", "dipy", "skimage", "cv2", "scipy"):
        try:
            versions[module] = __import__(module).__version__
        except Exception:   # pragma: no cover - optional information
            pass
    return versions


class SAMson:
    """
    SAMson brain extraction.

    Parameters
    ----------
    model_type : SAM model ("vit_h", the article's model; "vit_l" or "vit_b" are smaller and faster).
    checkpoint : path of the SAM checkpoint (default: see samson_mri.model.find_checkpoint).
    device : "auto" (GPU if available), "cuda", "cpu" or "mps".
    predictor : an already loaded SAM predictor (optional; used instead of loading one).
    template : path of the template (default: the bundled one).
    """

    def __init__(self, model_type: str = DEFAULT_MODEL, checkpoint: str | None = None, device: str = "auto",
                 predictor=None, template: str | None = None):
        self.model_type, self.checkpoint, self.device = model_type, checkpoint, device
        self._predictor = predictor
        self.template = load_template(template)

    @property
    def predictor(self):
        if self._predictor is None:
            self._predictor = load_predictor(self.model_type, self.checkpoint, self.device)
        return self._predictor

    def model_info(self) -> dict:
        predictor = self._predictor
        info = {"type": self.model_type, "checkpoint": os.path.basename(getattr(predictor, "samson_checkpoint", "") or "")}
        try:
            info["device"] = str(predictor.model.device)
        except Exception:
            info["device"] = self.device
        return info

    def boxes(self, image, spec: DatasetSpec) -> BoxPrompts:
        """Box prompts of an image (path or Subject), without running SAM."""
        subject = image if isinstance(image, Subject) else load_subject(str(image), spec)
        return compute_boxes(subject, spec, self.template)

    def segment(self, image, spec: DatasetSpec, mode: str = "full-auto", reference=None, choices=None,
                progress=None) -> SegmentationResult:
        """
        Segment one image.

        image : path of a NIfTI image (or a Subject).
        mode : "full-auto" (one mask per slice) or "semi-auto" (also SAM's three candidates per slice).
        reference : a reference mask (path or array on the image grid). In Semi-Auto mode the candidate with the highest
                    Jaccard index is selected in every slice (the article's Semi-Auto); in both modes the result is
                    evaluated against it (``result.metrics``).
        choices : Semi-Auto selection (one value or one per processed slice; -1 = Full-Auto mask, 0-2 = candidate).
        progress : called as progress(done_slices, total_slices).
        """
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES} (got {mode!r}).")
        if choices is not None and reference is not None:
            raise ValueError("Give either a reference mask or choices, not both.")
        subject = image if isinstance(image, Subject) else load_subject(str(image), spec)
        timings = {}
        start = time.time()
        prompts = compute_boxes(subject, spec, self.template)
        timings["boxes"] = time.time() - start
        part = subject.volume[:, :, prompts.first:prompts.last + 1]
        start = time.time()
        prompt_sets = [(prompts.boxes, False)] + ([(prompts.boxes, True)] if mode == "semi-auto" else [])
        masks = segment_volume(part, prompt_sets, self.predictor, progress=progress)
        timings["sam"] = time.time() - start
        result = SegmentationResult(subject=subject, spec=spec, mode=mode, prompts=prompts, full_auto=masks[0],
                                    candidates=masks[1] if mode == "semi-auto" else None,
                                    model=self.model_info(), timings=timings)
        if reference is not None:
            if mode == "semi-auto":
                result.select_by_reference(reference)
            result.evaluate(reference)
        elif choices is not None:
            if mode != "semi-auto" and not (np.isscalar(choices) and int(choices) == FULL_AUTO):
                raise ValueError("Choices need mode='semi-auto'.")
            result.select(choices)
        return result

    def segment_files(self, images, spec: DatasetSpec, output_dir: str, mode: str = "full-auto", references=None,
                      choices=None, qc: bool = True, save_brain: bool | None = None, overwrite: bool = False,
                      progress=None, log=print) -> list:
        """
        Segment several images and save the outputs in output_dir (see SegmentationResult.save). Images whose mask
        already exists are skipped unless overwrite=True, so an interrupted run can be resumed. references: a dict
        {image path: reference path} or a function image path -> reference path (or None). Writes samson_summary.csv
        and returns its rows.
        """
        os.makedirs(output_dir, exist_ok=True)
        if not overwrite and all(os.path.exists(os.path.join(output_dir, subject_stem(p) + "_mask.nii.gz")) for p in images):
            log("All masks exist already (use --overwrite to redo them).")
        else:
            self.predictor   # Load SAM once before the loop (and fail early if the checkpoint is missing)
        rows = []
        for i, path in enumerate(images):
            stem = subject_stem(path)
            mask_path = os.path.join(output_dir, stem + "_mask.nii.gz")
            row = dict(image=path, mask=mask_path, status="", seconds=np.nan, processed_slices="", jaccard=np.nan)
            if os.path.exists(mask_path) and not overwrite:
                row["status"] = "skipped (mask exists)"
                rows.append(row)
                log(f"[{i + 1}/{len(images)}] {stem}: skipped, its mask exists (use --overwrite to redo it)")
                continue
            start = time.time()
            try:
                reference = references.get(path) if isinstance(references, dict) else references(path) if callable(references) else None
                callback = (lambda done, total, i=i, stem=stem: progress(i, len(images), stem, done, total)) if progress else None
                result = self.segment(path, spec, mode=mode, reference=reference, choices=choices, progress=callback)
                result.save(output_dir, qc=qc, save_brain=save_brain)
                row.update(status="ok", seconds=round(time.time() - start, 1),
                           processed_slices=f"{result.prompts.first}-{result.prompts.last}")
                if result.metrics is not None:
                    row["jaccard"] = round(float(result.metrics.jaccard.mean()), 4)
                log(f"[{i + 1}/{len(images)}] {stem}: done in {row['seconds']} s" +
                    (f", mean Jaccard {row['jaccard']}" if result.metrics is not None else ""))
            except Exception as error:   # One failed image does not stop the batch; the error is reported
                row.update(status=f"error: {type(error).__name__}: {error}")
                log(f"[{i + 1}/{len(images)}] {stem}: ERROR {type(error).__name__}: {error}")
            rows.append(row)
        with open(os.path.join(output_dir, "samson_summary.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["image"])
            writer.writeheader()
            writer.writerows(rows)
        return rows


def load_result(sidecar_path: str) -> SegmentationResult:
    """
    Reload the result of a previous run from its files (<stem>_samson.json and the masks next to it), e.g. to review
    the Semi-Auto candidates later. The input image must still be at the path recorded in the JSON file.
    """
    with open(sidecar_path) as f:
        sidecar = json.load(f)
    folder = os.path.dirname(os.path.abspath(sidecar_path))
    spec = DatasetSpec.from_dict(sidecar["spec"])
    image_path = sidecar["input"]["path"]
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"The input image {image_path} (recorded in {os.path.basename(sidecar_path)}) is not there any more.")
    subject = load_subject(image_path, spec)
    first, last = sidecar["processed_slices"]
    outputs = sidecar.get("outputs", {})

    def rotated_slices(volume_input):   # input-space volume(s) -> (n, k, H, W) masks of rotated slices
        volume = spec.orientation.apply(volume_input)
        if volume.ndim == 3:
            volume = volume[..., None]
        return np.moveaxis(np.array([np.rot90(volume[:, :, k, :], axes=(0, 1)) for k in range(first, last + 1)]), 3, 1)

    if "candidates" in outputs:
        stack = rotated_slices(np.asanyarray(nib.load(os.path.join(folder, outputs["candidates"])).dataobj) > 0)
        full_auto, candidates = stack[:, :1], stack[:, 1:]
    else:
        full_auto, candidates = rotated_slices(np.asanyarray(nib.load(os.path.join(folder, outputs["mask"])).dataobj) > 0), None
    prompts = BoxPrompts(first=first, last=last, boxes=np.array(sidecar["boxes"]), otsu=tuple(sidecar["otsu"]),
                         anchor=sidecar["anchor_slice"] - first, central_box=tuple(sidecar.get("central_box", ())),
                         template_positions=np.linspace(*spec.template, last - first + 1))
    result = SegmentationResult(subject=subject, spec=spec, mode=sidecar["mode"], prompts=prompts, full_auto=full_auto,
                                candidates=candidates, model=sidecar.get("model", {}), timings=sidecar.get("timings_s", {}),
                                reference_path=sidecar.get("reference"))
    if candidates is not None:
        result.selection, result.selection_source = list(sidecar["selection"]), sidecar.get("selection_source", "full-auto")
    return result


def apply_selection(sidecar_path: str, choices=None, reference: str | None = None, qc: bool = True) -> dict:
    """
    Choose the Semi-Auto mask of every slice after a run (from the files written by SAMson): rewrites <stem>_mask.nii.gz
    and updates <stem>_samson.json. choices: one value or one per processed slice (-1 = Full-Auto mask, 0-2 =
    candidate); reference: a reference mask (the candidate with the highest Jaccard index in every slice).
    """
    result = load_result(sidecar_path)
    if result.candidates is None:
        raise ValueError("This subject was segmented in Full-Auto mode: there are no candidates to choose from.")
    if reference is not None:
        result.select_by_reference(reference)
        result.evaluate(reference)
    elif choices is not None:
        result.select(choices)
    else:
        raise ValueError("Give the choices or a reference mask.")
    return result.save(os.path.dirname(os.path.abspath(sidecar_path)), qc=qc)
