"""
Input and output of images and masks (NIfTI).

Inputs: NIfTI images (``.nii`` or ``.nii.gz``), 3D or 4D (one volume is used, see ``DatasetSpec.volume``), magnitude
images (non-negative intensities). Outputs: masks on the voxel grid of the input image, with its header (affine,
voxel sizes), as uint8 0/1 NIfTI files; see ``docs/io_conventions.md`` for the file names.
"""
from __future__ import annotations

import fnmatch
import glob
import os
import warnings
from dataclasses import dataclass

import nibabel as nib
import numpy as np

from .spec import DatasetSpec, SpecError

NIFTI_SUFFIXES = (".nii.gz", ".nii")
OUTPUT_SUFFIXES = ("_mask", "_candidates", "_brain")   # Files written by SAMson (not taken as inputs)


def subject_stem(path: str) -> str:
    """File name without folder and NIfTI extension: '/data/sub-01_T2w.nii.gz' -> 'sub-01_T2w'."""
    name = os.path.basename(path)
    for suffix in NIFTI_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return os.path.splitext(name)[0]


def is_nifti(path: str) -> bool:
    return path.endswith(NIFTI_SUFFIXES)


def find_images(inputs, pattern: str = "*", skip_outputs: bool = True) -> list:
    """
    NIfTI files given as files, folders (all NIfTI files inside, not recursive) or glob patterns, sorted and without
    duplicates. ``pattern`` filters the file names found in folders (e.g. ``"*T2w*"``). SAMson's own outputs
    (``*_mask``, ``*_candidates``, ``*_brain``) are skipped in folders.
    """
    found = []
    for item in [inputs] if isinstance(inputs, str) else inputs:
        item = os.path.expanduser(str(item))
        if os.path.isdir(item):
            names = sorted(n for n in os.listdir(item) if is_nifti(n) and fnmatch.fnmatch(n, pattern))
            if skip_outputs:
                names = [n for n in names if not subject_stem(n).endswith(OUTPUT_SUFFIXES)]
            found += [os.path.join(item, n) for n in names]
        elif os.path.isfile(item):
            found.append(item)
        else:
            matches = sorted(m for m in glob.glob(item) if is_nifti(m))
            if not matches:
                raise FileNotFoundError(f"No NIfTI file found for {item!r}.")
            found += matches
    unique = []
    for path in found:
        path = os.path.abspath(path)
        if path not in unique:
            unique.append(path)
    return unique


@dataclass
class Subject:
    """An input image: the NIfTI image, the array used (3D) and the same array in SAMson's orientation."""
    path: str
    image: nib.Nifti1Image
    data: np.ndarray          # 3D array of the input (float64), in the input orientation
    volume: np.ndarray        # the same array in SAMson's orientation
    spec: DatasetSpec

    @property
    def stem(self) -> str:
        return subject_stem(self.path)

    @property
    def voxel_size(self) -> tuple:
        """Header voxel sizes along SAMson's axes (left-right, ventral-dorsal, caudal-rostral)."""
        return self.spec.orientation.voxel_size(self.image.header.get_zooms()[:3])

    @property
    def n_slices(self) -> int:
        return self.volume.shape[2]

    def to_input_space(self, volume_samson: np.ndarray) -> np.ndarray:
        """An array in SAMson's orientation (3D or 4D) -> the input orientation."""
        return self.spec.orientation.invert(volume_samson)


def load_volume_array(image: nib.Nifti1Image, volume_index: int = 0, path: str = "") -> np.ndarray:
    """3D float64 array of an image: the image itself, or volume ``volume_index`` of a 4D image."""
    data = image.get_fdata()
    if data.ndim == 3:
        return data
    if data.ndim > 3 and np.prod(data.shape[3:]) == 1:
        return data.reshape(data.shape[:3])   # Trailing singleton dimensions (as np.squeeze in the notebook)
    if data.ndim > 3:
        data = data.reshape(data.shape[:3] + (-1,))
        if volume_index >= data.shape[3]:
            raise SpecError(f"{os.path.basename(path)} has {data.shape[3]} volumes; volume {volume_index} does not exist.")
        return np.ascontiguousarray(data[..., volume_index])
    raise ValueError(f"{os.path.basename(path)} is not a 3D or 4D image (shape {data.shape}).")


def load_subject(path: str, spec: DatasetSpec) -> Subject:
    """Load an image and re-orient it to SAMson's orientation as the specification says."""
    image = nib.load(path)
    data = load_volume_array(image, spec.volume, path)
    if not np.all(np.isfinite(data)):
        warnings.warn(f"{os.path.basename(path)} contains NaN or infinite values; they are set to 0.", stacklevel=2)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    if data.min() < 0:
        warnings.warn(f"{os.path.basename(path)} has negative intensities; SAMson expects a magnitude image "
                      "(the notebook's 8-bit conversion assumes non-negative values).", stacklevel=2)
    volume = spec.orientation.apply(data)
    return Subject(path=os.path.abspath(path), image=image, data=data, volume=volume, spec=spec)


def load_mask(path: str, subject: Subject | None = None) -> np.ndarray:
    """A reference or SAMson mask as a boolean array in the input orientation (checked against the subject's grid)."""
    mask = np.asanyarray(nib.load(path).dataobj)
    if mask.ndim > 3:
        mask = mask.reshape(mask.shape[:3]) if np.prod(mask.shape[3:]) == 1 else mask[..., 0]
    if subject is not None and mask.shape != subject.data.shape:
        raise ValueError(f"The mask {os.path.basename(path)} {mask.shape} is not on the grid of {os.path.basename(subject.path)} {subject.data.shape}.")
    return mask > 0.5


def save_like(array: np.ndarray, image: nib.Nifti1Image, path: str, dtype=np.uint8) -> str:
    """
    Save an array on the grid of an input image, with the input header (affine and voxel sizes), as in the notebook's
    save_nifti_mask. 4D arrays (candidates) keep the spatial header of the input.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    nifti = nib.Nifti1Image(np.asarray(array).astype(dtype), image.affine, image.header)
    nifti.set_data_dtype(dtype)
    nib.save(nifti, path)
    return path
