import os

import nibabel as nib
import numpy as np
import pytest
import reference_notebook as nb
from conftest import read_json, write_scrambled

from samson_mri import SAMson, apply_selection, load_result
from samson_mri.spec import DatasetSpec, Orientation
from samson_mri.template import default_template_path, load_template


def notebook_masks(volume, ap_range, otsu, predictor, semi_auto=False):
    """Masks as the notebook makes them (create_sam_candidates_list + save_nifti_mask), with the given predictor."""
    nb.relative_coordinates = load_template().relative_coordinates
    nb.predictor = predictor
    boxes = nb.generate_boxes(volume, *otsu, *ap_range)
    masks, = nb.segment_volume(volume, [(boxes, semi_auto)], show_progress=False)
    return masks, boxes


def test_full_auto_matches_notebook(example_image, d1_spec, fake_predictor, tmp_path):
    result = SAMson(predictor=fake_predictor).segment(example_image, d1_spec)
    volume = nib.load(example_image).get_fdata()
    masks, boxes = notebook_masks(volume, (4, 33), (2, 5), fake_predictor)
    assert np.array_equal(result.prompts.boxes, boxes)
    assert np.array_equal(result.full_auto, masks)
    # Saved mask = the notebook's save_nifti_mask
    paths = result.save(str(tmp_path))
    nb.save_nifti_mask(masks[:, 0], nib.load(example_image), str(tmp_path / "notebook_mask.nii.gz"))
    ours, theirs = nib.load(paths["mask"]), nib.load(str(tmp_path / "notebook_mask.nii.gz"))
    assert np.array_equal(np.asanyarray(ours.dataobj), np.asanyarray(theirs.dataobj))
    assert np.allclose(ours.affine, theirs.affine) and ours.get_data_dtype() == theirs.get_data_dtype() == np.uint8


def test_notebook_reorient_equals_flip_ap(example_image, fake_predictor, tmp_path):
    # The notebook's reorient=True (flip the z axis) corresponds to flip=(False, False, True)
    spec = DatasetSpec(template=(4, 33), orientation=Orientation((0, 1, 2), (False, False, True)))
    result = SAMson(predictor=fake_predictor).segment(example_image, spec)
    volume = nb.load_volume(example_image, reorient=True)
    masks, _ = notebook_masks(volume, (4, 33), (2, 5), fake_predictor)
    nb.save_nifti_mask(masks[:, 0], nib.load(example_image), str(tmp_path / "nb.nii.gz"), reorient=True)
    assert np.array_equal(result.mask, np.asanyarray(nib.load(str(tmp_path / "nb.nii.gz")).dataobj))


@pytest.mark.parametrize("axes, flips", [((0, 2, 1), (1,)), ((2, 1, 0), (0, 2)), ((1, 0, 2), ())])
def test_orientation_handling(example_image, d1_spec, fake_predictor, tmp_path, axes, flips):
    """A scrambled copy of the image, declared with the matching orientation, gives the same mask, scrambled alike."""
    reference = SAMson(predictor=fake_predictor).segment(example_image, d1_spec).mask
    scrambled = write_scrambled(example_image, str(tmp_path / "scrambled.nii.gz"), axes, flips)
    # Orientation that undoes the scrambling: SAMson axis i = scrambled axis axes.index(i); flips are on scrambled axes
    samson_axes = tuple(axes.index(i) for i in range(3))
    orientation = Orientation(samson_axes, tuple(samson_axes[i] in flips for i in range(3)))
    result = SAMson(predictor=fake_predictor).segment(scrambled, d1_spec.replace(orientation=orientation))
    expected = np.transpose(reference, axes)
    for axis in flips:
        expected = np.flip(expected, axis)
    assert np.array_equal(result.mask, expected)
    assert result.mask.shape == nib.load(scrambled).shape


def test_slice_range(example_image, d1_spec, fake_predictor):
    spec = d1_spec.replace(slices=(3, 12))
    result = SAMson(predictor=fake_predictor).segment(example_image, spec)
    mask = result.mask
    assert not mask[:, :, :3].any() and not mask[:, :, 13:].any() and mask[:, :, 3:13].any()
    volume = nib.load(example_image).get_fdata()[:, :, 3:13]
    masks, _ = notebook_masks(volume, (4, 33), (2, 5), fake_predictor)
    assert np.array_equal(result.full_auto, masks)


def test_semi_auto_reference_selection(example_image, example_mask, d1_spec, fake_predictor, tmp_path):
    result = SAMson(predictor=fake_predictor).segment(example_image, d1_spec, mode="semi-auto", reference=example_mask)
    volume = nib.load(example_image).get_fdata()
    candidates, _ = notebook_masks(volume, (4, 33), (2, 5), fake_predictor, semi_auto=True)
    reference = nib.load(example_mask).get_fdata()
    best = nb.find_best_mask_indexes(nb.evaluate_slices(candidates, reference))
    assert result.selection == best and result.selection_source == "reference"
    assert np.array_equal(result.selected_slices(), candidates[np.arange(len(candidates)), best])
    assert result.metrics is not None and len(result.metrics) == 16
    paths = result.save(str(tmp_path))
    stack = nib.load(paths["candidates"])
    assert stack.shape == (108, 90, 16, 4)
    data = np.asanyarray(stack.dataobj)
    assert np.array_equal(data[..., 0], result._to_volume(result.full_auto[:, 0]))


def test_selection_and_reload(example_image, d1_spec, fake_predictor, tmp_path):
    result = SAMson(predictor=fake_predictor).segment(example_image, d1_spec, mode="semi-auto")
    assert result.selection == [-1] * 16                     # starts with the Full-Auto masks
    assert np.array_equal(result.mask, SAMson(predictor=fake_predictor).segment(example_image, d1_spec).mask)
    paths = result.save(str(tmp_path))
    choices = [0, 1, 2, -1] * 4
    apply_selection(paths["sidecar"], choices=choices)
    reloaded = load_result(paths["sidecar"])
    assert reloaded.selection == choices and reloaded.selection_source == "manual"
    result.select(choices)
    assert np.array_equal(reloaded.mask, result.mask)
    assert np.array_equal(np.asanyarray(nib.load(paths["mask"]).dataobj), result.mask)
    with pytest.raises(ValueError):
        result.select([3] * 16)
    with pytest.raises(ValueError):
        result.select([0] * 5)


def test_sidecar_contents(example_image, d1_spec, fake_predictor, tmp_path):
    result = SAMson(predictor=fake_predictor).segment(example_image, d1_spec)
    paths = result.save(str(tmp_path), qc=True, save_brain=True)
    for key in ("mask", "sidecar", "qc", "brain"):
        assert os.path.exists(paths[key])
    sidecar = read_json(paths["sidecar"])
    assert sidecar["spec"]["template"] == [4, 33] and sidecar["processed_slices"] == [0, 15]
    assert sidecar["otsu"] == [2, 5] and sidecar["anchor_slice"] == 8 and len(sidecar["boxes"]) == 16
    assert sidecar["selection"] == [-1] * 16 and sidecar["mode"] == "full-auto"
    assert sidecar["outputs"]["mask"] == "d1_A110123_3mo_T2w_mask.nii.gz"
    assert sidecar["mask_voxels"] == int(result.mask.sum())
    brain = nib.load(paths["brain"]).get_fdata()
    assert np.allclose(brain, nib.load(example_image).get_fdata() * result.mask)


def test_full_auto_reload(example_image, d1_spec, fake_predictor, tmp_path):
    result = SAMson(predictor=fake_predictor).segment(example_image, d1_spec)
    paths = result.save(str(tmp_path), qc=False)
    reloaded = load_result(paths["sidecar"])
    assert reloaded.candidates is None and np.array_equal(reloaded.mask, result.mask)
    with pytest.raises(ValueError):
        apply_selection(paths["sidecar"], choices=0)


def test_batch(example_image, d1_spec, no_model, tmp_path):
    bad = tmp_path / "broken.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((5, 5)), np.eye(4)), str(bad))
    samson = SAMson()
    out = str(tmp_path / "out")
    rows = samson.segment_files([example_image, str(bad)], d1_spec, out, log=lambda *a: None)
    assert [r["status"][:5] for r in rows] == ["ok", "error"]
    assert os.path.exists(os.path.join(out, "samson_summary.csv"))
    rows = samson.segment_files([example_image], d1_spec, out, log=lambda *a: None)
    assert rows[0]["status"].startswith("skipped")
    rows = samson.segment_files([example_image], d1_spec, out, overwrite=True, log=lambda *a: None)
    assert rows[0]["status"] == "ok"


def test_progress_callback(example_image, d1_spec, fake_predictor):
    calls = []
    SAMson(predictor=fake_predictor).segment(example_image, d1_spec, progress=lambda done, total: calls.append((done, total)))
    assert calls == [(k, 16) for k in range(1, 17)]


def test_bundled_template_path():
    assert default_template_path().endswith("anat_template_FP.nii.gz") and os.path.exists(default_template_path())


def test_missing_reference_does_not_stop_the_batch(example_image, example_mask, d1_spec, no_model, tmp_path):
    import shutil
    images = []
    for name in ("m1", "m2", "m3"):
        images.append(str(tmp_path / f"{name}.nii.gz"))
        shutil.copy(example_image, images[-1])

    def references(path):
        if "m2" in path:
            raise FileNotFoundError("reference mask of m2 not found")
        return example_mask
    out = str(tmp_path / "out")
    rows = SAMson().segment_files(images, d1_spec, out, mode="semi-auto", references=references, qc=False, log=lambda *a: None)
    assert [r["status"][:5] for r in rows] == ["ok", "error", "ok"]
    assert os.path.exists(os.path.join(out, "samson_summary.csv"))


def test_brain_image_follows_the_selection(example_image, d1_spec, fake_predictor, tmp_path):
    result = SAMson(predictor=fake_predictor).segment(example_image, d1_spec, mode="semi-auto")
    paths = result.save(str(tmp_path), qc=False, save_brain=True)
    apply_selection(paths["sidecar"], choices=2, qc=False)
    brain = nib.load(paths["brain"]).get_fdata()
    mask = np.asanyarray(nib.load(paths["mask"]).dataobj)
    assert np.allclose(brain, nib.load(example_image).get_fdata() * mask)
    assert "brain" in read_json(paths["sidecar"])["outputs"]
