"""
SAMson core functions, copied verbatim (unchanged) from the SAMson notebook (`notebooks/SAMson.ipynb`; identical to
`samson_mri/_algorithm.py` of the package). Do not edit: SAMson's masks must stay voxel-identical to the published ones.

Module globals set by the caller (as in the notebooks): `predictor` (SamPredictor) and `relative_coordinates`
(`template_relative_coordinates(template_path)`).
"""
import cv2
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage
from skimage.morphology import binary_closing, binary_erosion, binary_dilation
from dipy.segment.mask import median_otsu

predictor = None              # set by lib.tools before use
relative_coordinates = None   # set by lib.tools before use


"""""""""       IMAGE  AND  MASK  UTILITIES       """""""""

def load_volume(path, reorient=False):
    """
    Load a NIfTI image as a float64 3D array (a trailing singleton 4th dimension is dropped).
    reorient=True flips the z-axis, for images whose anteroposterior axis goes from rostral to caudal.
    """
    volume = nib.load(path).get_fdata()
    if volume.ndim > 3: volume = np.squeeze(volume, axis=-1)
    return np.flip(volume, axis=2) if reorient else volume

def to_rgb(slice_2d):
    """Rotate a slice and convert it to the 8-bit RGB image SAM expects (normalised by the slice maximum)."""
    image = np.rot90(slice_2d).astype(np.float32)
    return cv2.cvtColor(((image / np.max(image)) * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

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
    jaccard  = round((overlap / np.count_nonzero(total)), 4)
    dice     = round(2 * (overlap / (np.count_nonzero(sam) + np.count_nonzero(ref))), 4)
    return accuracy, jaccard, dice

def evaluate_slices(masks, reference, slices=None):
    """
    Metrics of every candidate mask of every slice (or of the given slice indexes) against the reference volume.
    Returns a table with 1-based slice numbers (as in the plots) and 0-based candidate indexes (as in best_mask_indexes).
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

def keep_largest_component(mask):
    """Keep only the largest connected component of a binary mask (returned as 0/1 integers)."""
    labels, _ = ndimage.label(mask)
    return np.where(labels == np.argmax(np.bincount(labels.ravel())[1:]) + 1, 1, 0)

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
                mask = np.where(mask == True, 1, 0) - np.where(binary_dilation(lobes, footprint=footprint) == True, 1, 0)
            mask = keep_largest_component(mask)
    return mask

def postprocess(mask):
    """SAMson post-processing of a SAM mask: lobe and isolated pixel removal followed by a 2x2 closing."""
    return binary_closing(remove_lobes_and_separate_pixels(mask), footprint=np.ones((2, 2)))

def print_percent_done(index, total, bar_len=20):
    """Display a progress bar that updates on the same line."""
    percent_done = round((index + 1) / total * 100, 1)
    done = round(percent_done / (100 / bar_len))
    bar = f"[{'◼' * done}{' ' * (bar_len - done)}] {percent_done}%"
    if percent_done < 100:
        print(f'⏳  {bar}', end='\r')
    else:
        print(f'✅  {bar}', end=' ')


"""""""""       INPUT  PROMPT  GENERATION       """""""""

def bounding_boxes(volume, threshold=0.5):
    """Box [x_left, y_up, x_right, y_down] around the voxels above threshold in every rotated slice ([0, 0, 0, 0] if none)."""
    boxes = []
    for i in range(volume.shape[2]):
        idx = np.argwhere(np.rot90(volume[:, :, i]) > threshold)
        boxes.append([idx[:, 1].min(), idx[:, 0].min(), idx[:, 1].max(), idx[:, 0].max()] if np.any(idx) else [0, 0, 0, 0])
    return np.array(boxes)

def template_relative_coordinates(template_path):
    """
    Brain box of every template slice relative to the box of the template central slice: distances [left, up, right,
    down] from the central box centre, as fractions of the central box width and height.
    """
    edges = bounding_boxes(nib.load(template_path).get_fdata())
    xl, yu, xr, yd = edges[round(len(edges) / 2)]
    width_x, width_y = xr - xl, yd - yu
    central_x, central_y = xl + width_x / 2, yu + width_y / 2
    return np.column_stack([(central_x - edges[:, 0]) / width_x, (central_y - edges[:, 1]) / width_y,
                            (edges[:, 2] - central_x) / width_x, (edges[:, 3] - central_y) / width_y])

def generate_boxes(volume, median_radius, numpass, initial_slice, final_slice):
    """
    Box prompts [x_left, y_up, x_right, y_down] for every slice of a subject: the template box shapes, linearly
    interpolated along the anteroposterior range [initial_slice, final_slice] of the template, are scaled and centred
    on the box of the brain found by median_otsu in the subject's central slice.
    """
    z_template = np.linspace(0, 1, len(relative_coordinates))
    z_subject  = np.linspace(z_template[initial_slice], z_template[final_slice], volume.shape[2])

    # Interpolate the template relative coordinates at each subject slice (nearest slice if z matches exactly)
    previous = np.searchsorted(z_template, z_subject, side='right') - 1
    following = np.searchsorted(z_template, z_subject, side='left')
    with np.errstate(divide='ignore', invalid='ignore'):
        coordinates = relative_coordinates[previous] + (z_subject - z_template[previous])[:, None] * (
            (relative_coordinates[following] - relative_coordinates[previous]) / (z_template[following] - z_template[previous])[:, None])
    exact = previous == following
    coordinates[exact] = relative_coordinates[np.abs(z_template[None, :] - z_subject[exact, None]).argmin(axis=1)]

    # Brain box of the subject's central slice from the cleaned median_otsu mask
    _, otsu = median_otsu(volume, median_radius=median_radius, numpass=numpass)
    central_mask = remove_lobes_and_separate_pixels(np.rot90(otsu[:, :, np.abs(z_subject - 0.5).argmin()]), otsu=True)
    idx = np.argwhere(central_mask != 0)
    xl, yu, xr, yd = (idx[:, 1].min(), idx[:, 0].min(), idx[:, 1].max(), idx[:, 0].max()) if np.any(idx) else (0, 0, 0, 0)
    width_x, width_y = xr - xl, yd - yu
    central_x, central_y = xl + width_x / 2, yu + width_y / 2

    return np.column_stack([central_x - width_x * coordinates[:, 0], central_y - width_y * coordinates[:, 1],
                            central_x + width_x * coordinates[:, 2], central_y + width_y * coordinates[:, 3]])


"""""""""       MASK  PREDICTION       """""""""

def segment_volume(volume, prompt_sets, model=None, show_progress=True):
    """
    Predict and post-process SAM masks for every slice of a volume.
    prompt_sets is a list of (boxes, multimask_output) pairs: each slice is embedded once and reused for all of them.
    Returns, for each pair, an array (n_slices, 1 or 3 candidates, H, W) of binary masks.
    """
    model = model or predictor
    results = [[] for _ in prompt_sets]
    for s in range(volume.shape[2]):
        if show_progress: print_percent_done(s, volume.shape[2])
        model.set_image(to_rgb(volume[:, :, s]))
        for result, (boxes, multimask) in zip(results, prompt_sets):
            masks, _, _ = model.predict(point_coords=None, point_labels=None, box=np.array(boxes[s])[None, :], multimask_output=multimask)
            result.append([postprocess(mask) for mask in masks])
    return [np.array(result) for result in results]

def save_nifti_mask(mask_slices, reference_nifti, path, reorient=False):
    """Save one mask per slice (in plotting orientation) as a uint8 NIfTI aligned voxel-to-voxel with the input image."""
    volume = np.stack([np.rot90(mask, k=-1) for mask in mask_slices], axis=2).astype(np.uint8)  # Undo the slice rotation
    if reorient: volume = np.flip(volume, axis=2)  # Back to the anteroposterior order of the input image
    nifti = nib.Nifti1Image(volume, reference_nifti.affine, reference_nifti.header)
    nifti.set_data_dtype(np.uint8)
    nib.save(nifti, path)
