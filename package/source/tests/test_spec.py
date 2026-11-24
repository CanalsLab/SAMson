import json

import numpy as np
import pytest

from samson_mri.spec import PRESETS, DatasetSpec, Orientation, SpecError


@pytest.mark.parametrize("orientation", Orientation.all())
def test_orientation_round_trip(orientation):
    rng = np.random.default_rng(1)
    for shape in ((5, 6, 7), (5, 6, 7, 3)):
        array = rng.random(shape)
        oriented = orientation.apply(array)
        assert np.array_equal(orientation.invert(oriented), array)
        assert oriented.shape[:3] == tuple(shape[a] for a in orientation.axes)


def test_orientation_matches_manual_transform():
    # Same convention as the benchmark notebook: transpose by axes, then flip the flagged axes
    array = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    orientation = Orientation((0, 2, 1), (False, True, True))
    expected = np.flip(np.flip(np.transpose(array, (0, 2, 1)), 1), 2)
    assert np.array_equal(orientation.apply(array), expected)


@pytest.mark.parametrize("codes, axes, flip", [
    ("RIA", (0, 1, 2), (False, True, False)),
    ("RSA", (0, 1, 2), (False, False, False)),
    ("RAS", (0, 2, 1), (False, False, False)),
    ("LPI", (0, 2, 1), (False, True, True)),
    ("ASR", (2, 1, 0), (False, False, False)),
    ("SPL", (2, 0, 1), (False, False, True)),
])
def test_orientation_codes(codes, axes, flip):
    orientation = Orientation.from_codes(codes)
    assert orientation.axes == axes and orientation.flip == flip
    assert Orientation.from_codes(orientation.to_codes()) == orientation


@pytest.mark.parametrize("codes", ["RR", "RAX", "RLA", "AAS", ""])
def test_bad_orientation_codes(codes):
    with pytest.raises(SpecError):
        Orientation.from_codes(codes)


def test_bad_orientation():
    with pytest.raises(SpecError):
        Orientation((0, 0, 1))


def test_spec_requires_template():
    with pytest.raises(SpecError, match="template range is required"):
        DatasetSpec()


def test_spec_json_round_trip(tmp_path):
    spec = DatasetSpec(template=(12, 22), slices=(1, -2), orientation="RIA", otsu=(2, 5), volume=3, name="test")
    path = tmp_path / "spec.json"
    spec.save(str(path))
    data = json.loads(path.read_text())
    assert data["orientation"] == {"axes": [0, 1, 2], "flip": [False, True, False]}
    assert data["template"] == [12, 22] and data["slices"] == [1, -2] and data["otsu"] == [2, 5] and data["volume"] == 3
    assert DatasetSpec.load(str(path)) == spec


def test_spec_validation():
    spec = DatasetSpec(template=(4, 33))
    assert spec.slice_range(16) == (0, 15)
    assert spec.replace(slices=(2, -3)).slice_range(16) == (2, 13)
    with pytest.raises(SpecError):
        spec.replace(slices=(5, 20)).slice_range(16)
    with pytest.raises(SpecError):
        spec.replace(slices=(10, 3)).slice_range(16)
    for bad in ((33, 4), (-1, 10), (0, 40)):
        with pytest.raises(SpecError):
            DatasetSpec(template=bad).check_template()
    with pytest.raises(SpecError):
        DatasetSpec(template=(4, 33), otsu="sometimes")
    with pytest.raises(SpecError):
        DatasetSpec(template=(4, 33), otsu=(0, 5))
    with pytest.raises(SpecError):
        DatasetSpec.from_dict({"template": [4, 33], "colour": "blue"})
    with pytest.raises(SpecError):
        DatasetSpec.from_dict({"template": [4, 33], "samson_spec_version": 99})


def test_otsu_auto_rule():
    spec = DatasetSpec(template=(4, 33))
    assert spec.otsu_parameters(np.zeros((108, 90, 16))) == (2, 5)
    assert spec.otsu_parameters(np.zeros((256, 256, 22))) == (1, 3)
    assert spec.replace(otsu=(3, 4)).otsu_parameters(np.zeros((10, 10, 2))) == (3, 4)


def test_presets():
    assert PRESETS["article-d1"].template == (4, 33)
    assert PRESETS["article-d2"].template == (2, 39)
    assert PRESETS["article-d3"].template == (3, 35)
    for spec in PRESETS.values():
        spec.check_template()
        assert spec.orientation.is_identity and spec.slices == (0, -1)
