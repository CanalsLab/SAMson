"""Shared test fixtures: a fake SAM predictor (no checkpoint needed) and the example data."""
import os
import sys

import nibabel as nib
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))   # reference_notebook.py

from samson_mri.example import example_path  # noqa: E402
from samson_mri.spec import DatasetSpec  # noqa: E402


class FakePredictor:
    """
    Deterministic stand-in for segment_anything.SamPredictor: the mask is the ellipse inscribed in the box, intersected
    with the bright pixels of the slice; with multimask_output=True, three versions (shrunk, as is, enlarged).
    """

    def __init__(self):
        self.image = None
        self.calls = 0

    def set_image(self, image):
        assert image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3
        self.image = image[:, :, 0]

    def predict(self, point_coords=None, point_labels=None, box=None, multimask_output=True):
        self.calls += 1
        x0, y0, x1, y1 = np.asarray(box, dtype=float).ravel()
        height, width = self.image.shape
        yy, xx = np.mgrid[:height, :width]
        bright = self.image > 0.15 * max(int(self.image.max()), 1)
        masks = []
        for scale in ((0.85, 1.0, 1.15) if multimask_output else (1.0,)):
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            rx, ry = max(scale * (x1 - x0) / 2, 1), max(scale * (y1 - y0) / 2, 1)
            masks.append((((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1) & bright)
        masks = np.array(masks)
        return masks, np.ones(len(masks)), np.zeros((len(masks), 256, 256))


@pytest.fixture
def fake_predictor():
    return FakePredictor()


@pytest.fixture
def example_image():
    return example_path("image")


@pytest.fixture
def example_mask():
    return example_path("manual_mask")


@pytest.fixture
def d1_spec():
    return DatasetSpec(template=(4, 33))


@pytest.fixture
def template_file():
    from samson_mri.template import default_template_path
    return default_template_path()


@pytest.fixture
def no_model(monkeypatch, fake_predictor):
    """Make SAMson (and the CLI and the app) use the fake predictor instead of loading SAM."""
    import samson_mri.pipeline as pipeline
    monkeypatch.setattr(pipeline, "load_predictor", lambda *args, **kwargs: fake_predictor)
    return fake_predictor


def write_scrambled(path_in, path_out, axes, flips):
    """Write a copy of an image with its axes permuted and flipped (to test the orientation handling)."""
    image = nib.load(path_in)
    data = np.asanyarray(image.dataobj)
    data = np.transpose(data, axes)
    for axis in flips:
        data = np.flip(data, axis)
    nib.save(nib.Nifti1Image(np.ascontiguousarray(data), np.eye(4)), path_out)
    return path_out


def read_json(path):
    import json
    with open(path) as f:
        return json.load(f)


def read_text(path):
    with open(path) as f:
        return f.read()
