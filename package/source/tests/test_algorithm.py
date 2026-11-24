"""The package's algorithm must give exactly the results of the notebook's functions (frozen in reference_notebook.py)."""
import gzip
import shutil

import nibabel as nib
import numpy as np
import pytest
import reference_notebook as nb

from samson_mri import _algorithm as alg
from samson_mri.metrics import evaluate, evaluate_slices, find_best_mask_indexes, subject_means
from samson_mri.template import load_template


@pytest.fixture(scope="module")
def notebook(tmp_path_factory):
    """The notebook's functions set up with the bundled template (uncompressed, as the notebook read it)."""
    from samson_mri.template import default_template_path
    folder = tmp_path_factory.mktemp("template")
    plain = folder / "anat_template_FP.nii"
    with gzip.open(default_template_path(), "rb") as f, open(plain, "wb") as g:
        shutil.copyfileobj(f, g)
    nb.setup(str(plain))
    return nb


def test_template_coordinates_match_notebook(notebook):
    assert np.array_equal(load_template().relative_coordinates, notebook.relative_coordinates)


def test_template_shape():
    template = load_template()
    assert template.volume.shape == (258, 214, 40)
    assert template.central_slice == 20
    assert np.allclose(template.voxel_size_mm, (0.0697674, 0.0700935, 0.4), atol=1e-6)


@pytest.mark.parametrize("ap_range", [(4, 33), (0, 39), (12, 22), (2, 39), (10, 10), (19, 21), (30, 38)])
def test_generate_boxes_match_notebook(notebook, example_image, ap_range):
    volume = nib.load(example_image).get_fdata()
    for otsu in ((2, 5), (1, 3)):
        expected = notebook.generate_boxes(volume, *otsu, *ap_range)
        result = alg.generate_boxes(volume, *otsu, *ap_range, load_template().relative_coordinates)
        assert np.array_equal(result, expected, equal_nan=True)


@pytest.mark.parametrize("part", [(0, 16), (3, 12), (7, 8)])
def test_generate_boxes_on_parts(notebook, example_image, part):
    volume = nib.load(example_image).get_fdata()[:, :, part[0]:part[1]]
    expected = notebook.generate_boxes(volume, 2, 5, 6, 30)
    assert np.array_equal(alg.generate_boxes(volume, 2, 5, 6, 30, load_template().relative_coordinates), expected, equal_nan=True)


def test_postprocessing_matches_notebook(notebook):
    rng = np.random.default_rng(0)
    for _ in range(30):
        noise = rng.random((90, 108))
        blob = np.zeros((90, 108), bool)
        cy, cx = rng.integers(20, 70), rng.integers(20, 90)
        yy, xx = np.mgrid[:90, :108]
        blob |= ((yy - cy) / rng.integers(5, 25)) ** 2 + ((xx - cx) / rng.integers(5, 30)) ** 2 < 1
        mask = blob ^ (noise > 0.97)
        assert np.array_equal(alg.postprocess(mask), notebook.postprocess(mask))
        assert np.array_equal(alg.remove_lobes_and_separate_pixels(mask, otsu=True), notebook.remove_lobes_and_separate_pixels(mask, otsu=True))


def test_segment_volume_matches_notebook(notebook, example_image, fake_predictor):
    volume = nib.load(example_image).get_fdata()
    boxes = alg.generate_boxes(volume, 2, 5, 4, 33, load_template().relative_coordinates)
    notebook.predictor = fake_predictor
    expected = notebook.segment_volume(volume, [(boxes, False), (boxes, True)], show_progress=False)
    result = alg.segment_volume(volume, [(boxes, False), (boxes, True)], fake_predictor)
    for a, b in zip(result, expected):
        assert np.array_equal(a, b)
    assert result[0].shape == (16, 1, 90, 108) and result[1].shape == (16, 3, 90, 108)


def test_to_rgb_matches_notebook(notebook, example_image):
    volume = nib.load(example_image).get_fdata()
    for k in range(volume.shape[2]):
        assert np.array_equal(alg.to_rgb(volume[:, :, k]), notebook.to_rgb(volume[:, :, k]))


def test_metrics_match_notebook(notebook, example_mask):
    reference = nib.load(example_mask).get_fdata() > 0
    shifted = np.roll(reference, 3, axis=0)
    for k in range(reference.shape[2]):
        assert evaluate(shifted[:, :, k], reference[:, :, k]) == notebook.evaluate(shifted[:, :, k], reference[:, :, k])
    masks = np.array([[np.rot90(shifted[:, :, k]), np.rot90(reference[:, :, k])] for k in range(reference.shape[2])])
    table = evaluate_slices(masks, reference)
    assert table.equals(notebook.evaluate_slices(masks, reference))
    assert find_best_mask_indexes(table) == notebook.find_best_mask_indexes(table) == [1] * reference.shape[2]
    single = evaluate_slices(masks[:, :1], reference)
    assert subject_means(single) == notebook.subject_means(single)


def test_guards():
    assert not alg.keep_largest_component(np.zeros((5, 5), bool)).any()
    assert alg.to_rgb(np.zeros((4, 6))).shape == (6, 4, 3)
    assert not alg.to_rgb(np.zeros((4, 6))).any()


def test_anchor_slice():
    # The anchor is the processed slice closest to the template centre (z = 0.5)
    assert alg.anchor_slice(16, 4, 33, 40) == 8
    assert alg.anchor_slice(6, 12, 22, 40) == 4
    assert alg.anchor_slice(1, 10, 10, 40) == 0


def test_default_otsu_rule():
    assert alg.default_otsu_parameters(np.zeros((108, 90, 16))) == (2, 5)
    assert alg.default_otsu_parameters(np.zeros((256, 140, 22))) == (1, 3)
