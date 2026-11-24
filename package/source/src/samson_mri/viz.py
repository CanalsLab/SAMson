"""
Figures: anteroposterior range preview (subject and template side views), quality-control montages and candidates.

All functions return matplotlib ``Figure`` objects (no pyplot state, so they work in scripts, notebooks and the app).
Views follow SAMson's orientation: coronal slices with the dorsal side up; side (sagittal) views with the dorsal side
up and the caudal end on the left.
"""
from __future__ import annotations

import os

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from ._algorithm import anchor_slice
from .ap import TEMPLATE_SPACING_MM, check_range, length_ratio, plausible_voxel_size, template_positions
from .template import Template, load_template

MASK_COLOUR = (30 / 255, 144 / 255, 1.0)      # SAMson masks (blue, as in the notebook)
REFERENCE_COLOUR = (1.0, 234 / 255, 0.0)      # reference masks (yellow)
BOX_COLOUR = "lime"                           # box prompts
RANGE_COLOUR = "#ff7f0e"                      # processed range / matched template range
ANCHOR_COLOUR = "#d62728"                     # anchor slice


def display_range(image: np.ndarray, percentile: float = 99.5) -> tuple:
    values = image[np.isfinite(image)]
    if values.size == 0 or not np.any(values > 0):
        return 0.0, 1.0
    return 0.0, float(np.percentile(values[values > 0], percentile))


def coronal(volume: np.ndarray, k: int) -> np.ndarray:
    """Coronal slice k of a volume in SAMson's orientation, as SAMson sees it (dorsal side up)."""
    return np.rot90(volume[:, :, int(k)])


def sagittal(volume: np.ndarray, x: int | None = None) -> np.ndarray:
    """Side view (y, z) at left-right index x (default: middle); show with origin='lower' (dorsal up, caudal left)."""
    x = volume.shape[0] // 2 if x is None else int(x)
    return volume[x, :, :]


def template_midline(template: Template) -> int:
    """Left-right index of the template's midline (centre of its brain mask)."""
    return int(round(np.argwhere(template.brain).mean(axis=0)[0]))


def _show_sagittal(ax, view, voxel_size, title, pad=6):
    vmin, vmax = display_range(view)
    ax.imshow(view, cmap="gray", origin="lower", vmin=vmin, vmax=vmax, aspect=voxel_size[1] / voxel_size[2], interpolation="nearest")
    ax.set_title(title, fontsize=10, pad=pad)
    ax.set_yticks([])
    ax.tick_params(labelsize=8)


PAIR_COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22"]


def ap_figure(volume: np.ndarray, voxel_size, t0: int, t1: int, first: int = 0, last: int | None = None,
              x: int | None = None, prompts=None, template: Template | None = None, n_pairs: int = 6,
              header_scale: float = 1.0, title: str = "") -> Figure:
    """
    Anteroposterior range preview.

    Top: side views (dorsal up, caudal end on the left) of the subject, with its processed slices, and of the template,
    with the matched template range. Thin lines mark every processed slice and its template position; the coloured
    lines are the slices shown below, in pairs: subject coronal slice (with its box prompt if ``prompts`` is given) above
    the template slice it is matched to. A good range makes every pair show the same anatomy. The anchor slice (dashed)
    is the slice whose median_otsu box anchors all the boxes.
    """
    template = template or load_template()
    last = volume.shape[2] - 1 if last is None else last
    n = last - first + 1
    positions = template_positions(n, t0, t1)
    anchor = anchor_slice(n, t0, t1, template.n_slices)
    pairs = sorted(set(np.round(np.linspace(0, n - 1, min(n_pairs, n))).astype(int).tolist()))
    colour = {i: PAIR_COLOURS[j % len(PAIR_COLOURS)] for j, i in enumerate(pairs)}

    columns = max(len(pairs), 4)
    fig = Figure(figsize=(max(10.0, 1.9 * columns), 10.2), dpi=100)
    grid = fig.add_gridspec(3, columns, height_ratios=[2.3, 1, 1], hspace=0.28, wspace=0.06,
                            left=0.03, right=0.99, top=0.9 if title else 0.94, bottom=0.02)
    half = columns // 2
    ax_subject = fig.add_subplot(grid[0, :half])
    ax_template = fig.add_subplot(grid[0, half:])

    _show_sagittal(ax_subject, sagittal(volume, x), voxel_size,
                   f"Subject side view (x = {volume.shape[0] // 2 if x is None else x})\nprocessed slices {first}-{last}", pad=14)
    _show_sagittal(ax_template, sagittal(template.volume, template_midline(template)), template.voxel_size_mm,
                   f"Template side view\nmatched template slices {t0}-{t1}", pad=14)
    ax_subject.set_xlabel("subject slice (0 = caudal)", fontsize=8, labelpad=1)
    ax_template.set_xlabel("template slice (0 = caudal end, 39 = rostral end)", fontsize=8, labelpad=1)
    ax_template.set_xticks(range(0, template.n_slices, 5))
    height_s, height_t = volume.shape[1], template.volume.shape[1]
    for i, (k, p) in enumerate(zip(range(first, last + 1), positions)):
        style = dict(color=colour.get(i, "#ffd84d"), lw=1.8 if i in colour else 0.7, alpha=0.95 if i in colour else 0.55,
                     ls="--" if i == anchor else "-")
        ax_subject.axvline(k, **style)
        ax_template.axvline(p, **style)
        if i in colour:
            ax_subject.text(k, height_s - 0.5, str(k), color=colour[i], fontsize=8, ha="center", va="bottom", fontweight="bold")
            label = f"{p:.0f}" if abs(p - round(p)) < 0.05 else f"{p:.1f}"
            ax_template.text(p, height_t - 0.5, label, color=colour[i], fontsize=7.5, ha="center", va="bottom", fontweight="bold")

    for column, i in enumerate(pairs):
        k, p = first + i, positions[i]
        ax = fig.add_subplot(grid[1, column])
        image = coronal(volume, k)
        vmin, vmax = display_range(image)
        ax.imshow(image, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest", aspect=voxel_size[1] / voxel_size[0])
        if prompts is not None:
            box = prompts.boxes[i]
            ax.add_patch(Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], edgecolor=BOX_COLOUR, facecolor="none", lw=1.2))
        ax.set_title(f"subject slice {k}" + (" (anchor)" if i == anchor else ""), fontsize=9, color=colour[i], fontweight="bold")
        ax.axis("off")
        ax = fig.add_subplot(grid[2, column])
        ax.imshow(coronal(template.volume, int(round(p))), cmap="gray", interpolation="nearest",
                  aspect=template.voxel_size_mm[1] / template.voxel_size_mm[0])
        ax.set_title(f"template slice {int(round(p))} ({p:.1f})", fontsize=9, color=colour[i], fontweight="bold")
        ax.axis("off")

    notes = []
    size = [float(v) * header_scale for v in voxel_size]
    if plausible_voxel_size(size) and n > 1 and t1 > t0:
        notes.append(f"range length: subject {(n - 1) * size[2]:.1f} mm, template {(t1 - t0) * TEMPLATE_SPACING_MM:.1f} mm "
                     f"(ratio {length_ratio(n, size[2], t0, t1):.2f})")
    warnings = check_range(n, t0, t1, voxel_size, header_scale, template)
    if title or notes or warnings:
        fig.suptitle(" | ".join([title] + notes if title else notes) + ("\n" + "; ".join("⚠ " + w["short"] for w in warnings) if warnings else ""),
                     fontsize=10, color="darkred" if warnings else "black")
    return fig


def brain_window(result, margin: float = 0.3) -> tuple:
    """
    Region (x0, x1, y0, y1) of the coronal views around the brain: the union of the box prompts and of all the masks of the
    result, enlarged by ``margin`` (fraction of its size) on each side, made square and kept inside the slice.
    """
    height, width = coronal(result.subject.volume, result.prompts.first).shape
    boxes = np.asarray(result.prompts.boxes, dtype=float).reshape(-1, 4)
    boxes = boxes[np.any(boxes != 0, axis=1)]
    x0, y0, x1, y1 = (boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max()) if len(boxes) else (0, 0, width, height)
    masks = [result.full_auto] + ([result.candidates] if result.candidates is not None else [])
    found = np.argwhere(np.concatenate([np.asarray(m).reshape(-1, height, width) for m in masks]).any(axis=0))
    if found.size:
        x0, y0 = min(x0, found[:, 1].min()), min(y0, found[:, 0].min())
        x1, y1 = max(x1, found[:, 1].max() + 1), max(y1, found[:, 0].max() + 1)
    side = max(x1 - x0, y1 - y0) * (1 + 2 * margin)
    side = min(side, max(width, height))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0, y0 = min(max(cx - side / 2, 0), max(width - side, 0)), min(max(cy - side / 2, 0), max(height - side, 0))
    return x0 - 0.5, min(x0 + side, width) - 0.5, y0 - 0.5, min(y0 + side, height) - 0.5


def _zoom(ax, window):
    if window is not None:
        ax.set_xlim(window[0], window[1])
        ax.set_ylim(window[3], window[2])


def qc_figure(result, ncols: int = 6, reference=None, zoom: bool = False) -> Figure:
    """
    Every processed slice with the final mask (blue), the box prompt (green) and, optionally, a reference (yellow).
    zoom=True shows only the region around the brain (``brain_window``).
    """
    volume = result.subject.volume
    first, last = result.prompts.first, result.prompts.last
    masks = result.selected_slices()
    reference_samson = None if reference is None else result.subject.spec.orientation.apply(np.asarray(reference) > 0)
    n = last - first + 1
    ncols = min(ncols, n)
    nrows = -(-n // ncols)
    window = brain_window(result) if zoom else None
    height, width = coronal(volume, first).shape
    if window is not None:
        width, height = window[1] - window[0], window[3] - window[2]
    panel = 2.2
    fig = Figure(figsize=(ncols * panel, nrows * panel * height / width + 0.6), dpi=90)
    fig.suptitle(f"{result.subject.stem}: {result.mode}, slices {first}-{last}", fontsize=10)
    for i in range(n):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        image = coronal(volume, first + i)
        vmin, vmax = display_range(image)
        ax.imshow(image, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        if reference_samson is not None:
            ax.contour(coronal(reference_samson, first + i), [0.5], colors=[REFERENCE_COLOUR], linewidths=0.8)
        overlay = np.zeros(masks[i].shape + (4,))
        overlay[masks[i] > 0] = (*MASK_COLOUR, 0.45)
        ax.imshow(overlay, interpolation="nearest")
        box = result.prompts.boxes[i]
        ax.add_patch(Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], edgecolor=BOX_COLOUR, facecolor="none", lw=1))
        choice = result.selection[i]
        label = "" if result.candidates is None else (" F" if choice == -1 else f" c{choice}")
        ax.set_title(f"slice {first + i}{label}", fontsize=8, pad=3)
        _zoom(ax, window)
        ax.axis("off")
    fig.tight_layout(h_pad=1.0, rect=(0, 0, 1, 0.97))
    return fig


def candidates_figure(result, index: int, zoom: bool = False) -> Figure:
    """
    The Full-Auto mask and the three Semi-Auto candidates of processed slice ``index`` (0-based within the range).
    zoom=True shows only the region around the brain (``brain_window``).
    """
    image = coronal(result.subject.volume, result.prompts.first + index)
    window = brain_window(result) if zoom else None
    options = [("Full-Auto", result.full_auto[index, 0])]
    if result.candidates is not None:
        options += [(f"candidate {c}", result.candidates[index, c]) for c in range(result.candidates.shape[1])]
    fig = Figure(figsize=(2.6 * len(options), 2.8), dpi=90)
    vmin, vmax = display_range(image)
    for j, (name, mask) in enumerate(options):
        ax = fig.add_subplot(1, len(options), j + 1)
        ax.imshow(image, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        overlay = np.zeros(mask.shape + (4,))
        overlay[mask > 0] = (*MASK_COLOUR, 0.45)
        ax.imshow(overlay, interpolation="nearest")
        chosen = result.selection[index] == (-1 if j == 0 else j - 1)
        ax.set_title(name + (" ✓" if chosen else ""), fontsize=9, color="green" if chosen else "black")
        _zoom(ax, window)
        ax.axis("off")
    fig.tight_layout()
    return fig


def orientation_figure(volume: np.ndarray, voxel_size, template: Template | None = None) -> Figure:
    """The image as SAMson will see it (coronal, side and top views) next to the template's coronal and side views."""
    template = template or load_template()
    fig = Figure(figsize=(12, 5.6), dpi=90)
    grid = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.15)
    mid = [s // 2 for s in volume.shape[:3]]
    ax = fig.add_subplot(grid[0, 0])
    image = coronal(volume, mid[2])
    ax.imshow(image, cmap="gray", vmin=0, vmax=display_range(image)[1], aspect=voxel_size[1] / voxel_size[0])
    ax.set_title(f"Your image: coronal slice {mid[2]}\n(should show the dorsal side up)", fontsize=9)
    ax.axis("off")
    ax = fig.add_subplot(grid[0, 1:])
    _show_sagittal(ax, sagittal(volume), voxel_size, "Your image: side view (dorsal up; caudal end on the LEFT)")
    ax.set_xlabel("slice", fontsize=8)
    ax = fig.add_subplot(grid[1, 0])
    tmid = template.n_slices // 2
    ax.imshow(coronal(template.volume, tmid), cmap="gray", aspect=template.voxel_size_mm[1] / template.voxel_size_mm[0])
    ax.set_title(f"Template: coronal slice {tmid}", fontsize=9)
    ax.axis("off")
    ax = fig.add_subplot(grid[1, 1:])
    _show_sagittal(ax, sagittal(template.volume, template_midline(template)), template.voxel_size_mm,
                   "Template: side view (cerebellum on the left, olfactory bulbs on the right)")
    ax.set_xlabel("template slice", fontsize=8)
    return fig


def save_figure(fig: Figure, path: str, dpi: int | None = None) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=dpi or fig.dpi, bbox_inches="tight")
    return path
