import os

import nibabel as nib
import numpy as np
import pytest

from samson_mri.io import find_images, load_mask, load_subject, save_like, subject_stem
from samson_mri.spec import DatasetSpec, SpecError


def test_subject_stem():
    assert subject_stem("/a/b/sub-01_T2w.nii.gz") == "sub-01_T2w"
    assert subject_stem("x.nii") == "x"
    assert subject_stem("T2Wmice_first_im_A110123_3mo_.nii") == "T2Wmice_first_im_A110123_3mo_"


def test_find_images(tmp_path):
    for name in ("a_T2w.nii.gz", "b_T2w.nii", "c_dwi.nii.gz", "a_T2w_mask.nii.gz", "notes.txt"):
        (tmp_path / name).write_bytes(b"")
    found = [os.path.basename(p) for p in find_images(str(tmp_path))]
    assert found == ["a_T2w.nii.gz", "b_T2w.nii", "c_dwi.nii.gz"]
    assert [os.path.basename(p) for p in find_images(str(tmp_path), pattern="*T2w*")] == ["a_T2w.nii.gz", "b_T2w.nii"]
    assert len(find_images([str(tmp_path / "c_dwi.nii.gz"), str(tmp_path / "c_dwi.nii.gz")])) == 1
    assert len(find_images(str(tmp_path / "*_T2w.nii*"))) == 2
    with pytest.raises(FileNotFoundError):
        find_images(str(tmp_path / "missing*.nii.gz"))


def test_4d_volume_selection(tmp_path):
    data = np.random.default_rng(0).random((8, 9, 5, 4)) + 1
    path = str(tmp_path / "dwi.nii.gz")
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)
    subject = load_subject(path, DatasetSpec(template=(4, 33), volume=2))
    assert np.allclose(subject.data, data[..., 2])
    with pytest.raises(SpecError):
        load_subject(path, DatasetSpec(template=(4, 33), volume=4))


def test_singleton_4d(tmp_path):
    data = np.random.default_rng(0).random((8, 9, 5, 1)) + 1
    path = str(tmp_path / "t2.nii.gz")
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)
    assert load_subject(path, DatasetSpec(template=(4, 33))).data.shape == (8, 9, 5)


def test_nan_values(tmp_path):
    data = np.ones((6, 6, 3))
    data[0, 0, 0] = np.nan
    path = str(tmp_path / "nan.nii.gz")
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)
    with pytest.warns(UserWarning, match="NaN"):
        subject = load_subject(path, DatasetSpec(template=(4, 33)))
    assert subject.data[0, 0, 0] == 0


def test_save_like_keeps_grid(tmp_path, example_image):
    image = nib.load(example_image)
    mask = (image.get_fdata() > 1000).astype(np.uint8)
    path = save_like(mask, image, str(tmp_path / "out" / "m_mask.nii.gz"))
    saved = nib.load(path)
    assert saved.shape == image.shape and np.allclose(saved.affine, image.affine)
    assert saved.get_data_dtype() == np.uint8
    assert np.array_equal(np.asanyarray(saved.dataobj), mask)
    assert np.allclose(saved.header.get_zooms(), image.header.get_zooms())


def test_load_mask_checks_grid(example_image, example_mask):
    subject = load_subject(example_image, DatasetSpec(template=(4, 33)))
    assert load_mask(example_mask, subject).dtype == bool
    other = load_subject(example_image, DatasetSpec(template=(4, 33), orientation="RAS"))
    with pytest.raises(ValueError):
        load_mask(example_mask, type(other)(other.path, other.image, other.volume, other.volume, other.spec))
