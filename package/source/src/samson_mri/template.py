"""
SAMson's template (Mouse-X template, ``anat_template_FP``) bundled with the package.

The template has 40 coronal slices (index 0 = caudal end, 39 = rostral end) 0.4 mm apart. SAMson uses the brain box of
every template slice, relative to the box of its central slice, to build the box prompts of a subject. The template can
be replaced with ``SAMSON_TEMPLATE=/path/to/template.nii[.gz]`` or the ``template`` argument of the functions.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

import nibabel as nib
import numpy as np

from ._algorithm import template_relative_coordinates

TEMPLATE_FILENAME = "anat_template_FP.nii.gz"
TEMPLATE_HEADER_SCALE = 0.1   # The template header stores voxel sizes x10 (0.4 mm slices are stored as 4.0)


def default_template_path() -> str:
    """Path of the template used by default: $SAMSON_TEMPLATE if set, else the template bundled with the package."""
    override = os.environ.get("SAMSON_TEMPLATE")
    if override:
        return override
    return str(resources.files("samson_mri").joinpath("data", TEMPLATE_FILENAME))


@dataclass(frozen=True)
class Template:
    """The template volume (float64, as in the notebook), its brain mask and SAMson's relative box coordinates."""
    path: str
    volume: np.ndarray
    voxel_size_mm: tuple
    relative_coordinates: np.ndarray

    @property
    def n_slices(self) -> int:
        return self.volume.shape[2]

    @property
    def central_slice(self) -> int:
        """Template slice whose box is the reference of the relative coordinates (slice 20 of 40)."""
        return round(self.n_slices / 2)

    @property
    def brain(self) -> np.ndarray:
        return self.volume > 0.5


@lru_cache(maxsize=4)
def _load(path: str) -> Template:
    image = nib.load(path)
    volume = image.get_fdata()     # float64, as nib.load(template_path).get_fdata() in the notebook
    if volume.ndim > 3:
        volume = volume.reshape(volume.shape[:3])
    zooms = tuple(float(z) * TEMPLATE_HEADER_SCALE for z in image.header.get_zooms()[:3])
    return Template(path=path, volume=volume, voxel_size_mm=zooms, relative_coordinates=template_relative_coordinates(volume))


def load_template(path: str | None = None) -> Template:
    """Load (once) the template at ``path`` (default: :func:`default_template_path`)."""
    return _load(os.path.abspath(path or default_template_path()))
