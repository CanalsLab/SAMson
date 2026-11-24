"""
SAMson Full-Auto with SAM2 (sam2.1_hiera_large) instead of SAM: given box prompts (SAMson's Full-Auto boxes), SAMson's
post-processing; masks saved on the grid of the (canonical) input image. The main loop records the wall time of every
job (one JSON line per job). Usage: python sam2_run.py jobs.json
"""
import json, sys, time
from datetime import datetime
import cv2
import numpy as np
import nibabel as nib
import torch
from scipy import ndimage
from skimage.morphology import binary_closing, binary_erosion, binary_dilation
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def to_rgb(slice_2d):
    """Rotate a slice and convert it to the 8-bit RGB image SAM expects (normalised by the slice maximum)."""
    image = np.rot90(slice_2d).astype(np.float32)
    return cv2.cvtColor(((image / np.max(image)) * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)


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


config = json.load(open(sys.argv[1]))
device = "cuda" if torch.cuda.is_available() else "cpu"
t0 = time.perf_counter()
model = SAM2ImagePredictor(build_sam2(config.get("config", "configs/sam2.1/sam2.1_hiera_l.yaml"), config["checkpoint"], device=device))
print(json.dumps(dict(event="model_loaded", seconds=time.perf_counter() - t0, device=device)), flush=True)
for job in config["jobs"]:
    start, t = datetime.now().isoformat(timespec="seconds"), time.perf_counter()
    image = nib.load(job["input"])
    volume = image.get_fdata()
    z0, z1 = job["z0"], job["z1"]
    masks, = segment_volume(volume[:, :, z0:z1 + 1], [(np.load(job["boxes"]), False)], model=model, show_progress=False)
    out = np.zeros(image.shape[:3], dtype=np.uint8)
    out[:, :, z0:z1 + 1] = np.stack([np.rot90(m, k=-1) for m in masks[:, 0]], axis=2)
    nifti = nib.Nifti1Image(out, image.affine, image.header); nifti.set_data_dtype(np.uint8)
    nib.save(nifti, job["output"])
    print(json.dumps(dict(event="job", output=job["output"], start=start, end=datetime.now().isoformat(timespec="seconds"),
                          seconds=time.perf_counter() - t)), flush=True)
