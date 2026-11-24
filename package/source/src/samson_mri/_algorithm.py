"""
SAMson's algorithm.

These functions are those of ``SAMson_opt.ipynb`` (https://github.com/CanalsLab/SAMson), the notebook that produced
the published SAMson masks. They are kept unchanged so that the package gives the same masks, voxel for voxel. The only
differences are:

* the template box shapes (``relative_coordinates``) and the SAM predictor are passed as arguments instead of being
  notebook globals;
* ``template_relative_coordinates`` receives the template array instead of its path;
* ``segment_volume`` reports progress through an optional callback instead of printing a progress bar;
* two guards for inputs that made the notebook raise an error or divide by zero (an empty mask in
  ``keep_largest_component``, an empty slice in ``to_rgb``); they do not change any other result.

Conventions (as in the notebook): a volume is a 3D array (x, y, z) whose third axis is the anteroposterior axis with the
caudal end at slice 0; ``np.rot90`` of a slice ``volume[:, :, k]`` shows it with the dorsal side up. Boxes are
``[x_left, y_up, x_right, y_down]`` in the coordinates of these rotated slices.
"""
from __future__ import annotations

import cv2
import numpy as np
from dipy.segment.mask import median_otsu
from scipy import ndimage
from skimage.morphology import binary_closing, binary_dilation, binary_erosion

# ----------------------------------------------------------------------------------------------------------------------
# Image and mask utilities
# ----------------------------------------------------------------------------------------------------------------------

def to_rgb(slice_2d):
    """Rotate a slice and convert it to the 8-bit RGB image SAM expects (normalised by the slice maximum)."""
    image = np.rot90(slice_2d).astype(np.float32)
    if not np.max(image) > 0:  # Guard (empty slice): the notebook divided by zero here
        return np.zeros(image.shape + (3,), dtype=np.uint8)
    return cv2.cvtColor(((image / np.max(image)) * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)


def keep_largest_component(mask):
    """Keep only the largest connected component of a binary mask (returned as 0/1 integers)."""
    labels, _ = ndimage.label(mask)
    sizes = np.bincount(labels.ravel())[1:]
    if sizes.size == 0:  # Guard (empty mask): the notebook raised a ValueError here
        return np.zeros(np.shape(mask), dtype=int)
    return np.where(labels == np.argmax(sizes) + 1, 1, 0)


def remove_lobes_and_separate_pixels(mask, otsu=False):
    """
    Remove lobes (regions that detach from the main body after an erosion) and isolated pixel groups from a binary
    mask, with increasing footprint sizes (larger ones for the Otsu mask), keeping the largest connected component.
    """
    if np.any(mask):
        for size in ([2, 5, 20] if otsu else [1, 2, 3]):
            footprint = np.ones((size, size))
            eroded = binary_erosion(mask, footprint=footprint)
            if np.any(eroded):
                lobes = eroded.copy()
                lobes[keep_largest_component(eroded) == 1] = 0  # Eroded regions other than the main body
                mask = np.where(mask == True, 1, 0) - np.where(binary_dilation(lobes, footprint=footprint) == True, 1, 0)  # noqa: E712
            mask = keep_largest_component(mask)
    return mask


def postprocess(mask):
    """SAMson post-processing of a SAM mask: lobe and isolated pixel removal followed by a 2x2 closing."""
    return binary_closing(remove_lobes_and_separate_pixels(mask), footprint=np.ones((2, 2)))


# ----------------------------------------------------------------------------------------------------------------------
# Input prompt generation
# ----------------------------------------------------------------------------------------------------------------------

def bounding_boxes(volume, threshold=0.5):
    """Box [x_left, y_up, x_right, y_down] around the voxels above threshold in every rotated slice ([0, 0, 0, 0] if none)."""
    boxes = []
    for i in range(volume.shape[2]):
        idx = np.argwhere(np.rot90(volume[:, :, i]) > threshold)
        boxes.append([idx[:, 1].min(), idx[:, 0].min(), idx[:, 1].max(), idx[:, 0].max()] if np.any(idx) else [0, 0, 0, 0])
    return np.array(boxes)


def template_relative_coordinates(template):
    """
    Brain box of every template slice relative to the box of the template central slice: distances [left, up, right,
    down] from the central box centre, as fractions of the central box width and height.
    """
    edges = bounding_boxes(template)
    xl, yu, xr, yd = edges[round(len(edges) / 2)]
    width_x, width_y = xr - xl, yd - yu
    central_x, central_y = xl + width_x / 2, yu + width_y / 2
    return np.column_stack([(central_x - edges[:, 0]) / width_x, (central_y - edges[:, 1]) / width_y,
                            (edges[:, 2] - central_x) / width_x, (edges[:, 3] - central_y) / width_y])


def template_positions(n_slices, initial_slice, final_slice, n_template_slices):
    """Normalised template positions (0 = first template slice, 1 = last) of the slices of a subject, as in generate_boxes."""
    z_template = np.linspace(0, 1, n_template_slices)
    return z_template, np.linspace(z_template[initial_slice], z_template[final_slice], n_slices)


def anchor_slice(n_slices, initial_slice, final_slice, n_template_slices):
    """Index (within the processed slices) of the slice whose median_otsu box anchors all the boxes in generate_boxes."""
    _, z_subject = template_positions(n_slices, initial_slice, final_slice, n_template_slices)
    return int(np.abs(z_subject - 0.5).argmin())


def interpolated_coordinates(n_slices, initial_slice, final_slice, relative_coordinates):
    """Template relative box coordinates interpolated at every subject slice (first part of generate_boxes)."""
    z_template, z_subject = template_positions(n_slices, initial_slice, final_slice, len(relative_coordinates))

    # Interpolate the template relative coordinates at each subject slice (nearest slice if z matches exactly)
    previous = np.searchsorted(z_template, z_subject, side='right') - 1
    following = np.searchsorted(z_template, z_subject, side='left')
    with np.errstate(divide='ignore', invalid='ignore'):
        coordinates = relative_coordinates[previous] + (z_subject - z_template[previous])[:, None] * (
            (relative_coordinates[following] - relative_coordinates[previous]) / (z_template[following] - z_template[previous])[:, None])
    exact = previous == following
    coordinates[exact] = relative_coordinates[np.abs(z_template[None, :] - z_subject[exact, None]).argmin(axis=1)]
    return coordinates


def central_box(volume, median_radius, numpass, initial_slice, final_slice, n_template_slices):
    """Box of the brain found by median_otsu in the anchor slice (second part of generate_boxes): x_left, y_up, x_right, y_down."""
    _, z_subject = template_positions(volume.shape[2], initial_slice, final_slice, n_template_slices)
    _, otsu = median_otsu(volume, median_radius=median_radius, numpass=numpass)
    central_mask = remove_lobes_and_separate_pixels(np.rot90(otsu[:, :, np.abs(z_subject - 0.5).argmin()]), otsu=True)
    idx = np.argwhere(central_mask != 0)
    return (idx[:, 1].min(), idx[:, 0].min(), idx[:, 1].max(), idx[:, 0].max()) if np.any(idx) else (0, 0, 0, 0)


def boxes_from_central_box(box, coordinates):
    """Scale and centre the interpolated template box shapes on a central box (last part of generate_boxes)."""
    xl, yu, xr, yd = box
    width_x, width_y = xr - xl, yd - yu
    central_x, central_y = xl + width_x / 2, yu + width_y / 2
    return np.column_stack([central_x - width_x * coordinates[:, 0], central_y - width_y * coordinates[:, 1],
                            central_x + width_x * coordinates[:, 2], central_y + width_y * coordinates[:, 3]])


def generate_boxes(volume, median_radius, numpass, initial_slice, final_slice, relative_coordinates):
    """
    Box prompts [x_left, y_up, x_right, y_down] for every slice of a subject: the template box shapes, linearly
    interpolated along the anteroposterior range [initial_slice, final_slice] of the template, are scaled and centred
    on the box of the brain found by median_otsu in the subject's central slice.

    (The notebook's generate_boxes, split into the three helper functions above so that the app can show each step.)
    """
    coordinates = interpolated_coordinates(volume.shape[2], initial_slice, final_slice, relative_coordinates)
    box = central_box(volume, median_radius, numpass, initial_slice, final_slice, len(relative_coordinates))
    return boxes_from_central_box(box, coordinates)


# ----------------------------------------------------------------------------------------------------------------------
# Mask prediction
# ----------------------------------------------------------------------------------------------------------------------

def segment_volume(volume, prompt_sets, model, progress=None):
    """
    Predict and post-process SAM masks for every slice of a volume.
    prompt_sets is a list of (boxes, multimask_output) pairs: each slice is embedded once and reused for all of them.
    Returns, for each pair, an array (n_slices, 1 or 3 candidates, H, W) of binary masks.
    progress, if given, is called as progress(done_slices, total_slices) after every slice.
    """
    results = [[] for _ in prompt_sets]
    for s in range(volume.shape[2]):
        model.set_image(to_rgb(volume[:, :, s]))
        for result, (boxes, multimask) in zip(results, prompt_sets):
            masks, _, _ = model.predict(point_coords=None, point_labels=None, box=np.array(boxes[s])[None, :], multimask_output=multimask)
            result.append([postprocess(mask) for mask in masks])
        if progress is not None:
            progress(s + 1, volume.shape[2])
    return [np.array(result) for result in results]


def default_otsu_parameters(volume):
    """median_otsu parameters of SAMson_opt.ipynb: (2, 5) for images with fewer than 140 rows (axis 1), else (1, 3)."""
    return (2, 5) if volume.shape[1] < 140 else (1, 3)
