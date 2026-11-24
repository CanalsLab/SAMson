"""The app runs through its five steps without errors (Streamlit's headless app tester; fake SAM predictor)."""
import os

import pytest
from conftest import read_json

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

import samson_mri  # noqa: E402

APP = os.path.join(os.path.dirname(samson_mri.__file__), "app", "streamlit_app.py")
PAGES = ["1 · Images", "2 · Orientation", "3 · Anteroposterior range", "4 · Segment", "5 · Review"]


def button(at, label_start):
    matches = [b for b in at.button if b.label.startswith(label_start)]
    assert matches, f"no button starting with {label_start!r}: {[b.label for b in at.button]}"
    return matches[0]


def go(at, page):
    at.radio(key="w_page").set_value(page).run()
    assert not at.exception, at.exception


@pytest.fixture
def app(no_model, tmp_path, monkeypatch):
    dummy = tmp_path / "sam_vit_h_4b8939.pth"
    dummy.write_bytes(b"not a real checkpoint: the fake predictor is used")
    monkeypatch.setenv("SAMSON_CHECKPOINT", str(dummy))
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception, at.exception
    return at


def test_app_full_workflow(app, tmp_path):
    at = app
    button(at, "…or use the example subject").click().run()
    assert not at.exception, at.exception
    assert at.session_state.w_rep.endswith("d1_A110123_3mo_T2w.nii.gz")

    go(at, PAGES[1])
    at.button[[b.label for b in at.button].index("Guess from the image header")].click().run()
    assert not at.exception
    at.radio(key="w_ap_axis").set_value(2).run()
    at.radio(key="w_dv_axis").set_value(1).run()
    at.checkbox(key="w_flip_dv").uncheck().run()
    at.checkbox(key="w_flip_ap").uncheck().run()
    assert not at.exception

    go(at, PAGES[2])
    assert at.session_state.w_slices == (0, 15)
    assert not [w for w in at.warning]                       # D1 with 4-33: nothing to warn about
    at.slider(key="w_template").set_value((30, 38)).run()
    assert not at.exception and len(at.warning) >= 1         # a wrong range is flagged
    at.slider(key="w_template").set_value((4, 33)).run()
    at.toggle(key="w_link").set_value(True).run()
    assert not at.exception
    at.toggle(key="w_link").set_value(False).run()

    go(at, PAGES[3])
    at.radio(key="w_mode").set_value("Semi-Auto").run()
    at.text_input(key="w_output").set_value(str(tmp_path / "out")).run()
    button(at, "▶ Run SAMson").click().run()
    assert not at.exception, at.exception
    stem = "d1_A110123_3mo_T2w"
    for suffix in ("_mask.nii.gz", "_candidates.nii.gz", "_samson.json"):
        assert os.path.exists(tmp_path / "out" / (stem + suffix))
    assert os.path.exists(tmp_path / "out" / "samson_spec.json")

    go(at, PAGES[4])
    assert at.toggle(key="w_zoom").value is True
    at.toggle(key="w_zoom").set_value(False).run()
    assert not at.exception, at.exception
    at.toggle(key="w_zoom").set_value(True).run()
    radios = [r for r in at.radio if r.label == "Mask of this slice"]
    radios[0].set_value(2).run()
    assert not at.exception, at.exception
    button(at, "💾 Save the mask").click().run()
    assert not at.exception, at.exception
    assert read_json(tmp_path / "out" / (stem + "_samson.json"))["selection"][0] == 2

    # Reload the results from the folder
    at.radio(key="w_review_source").set_value("A results folder").run()
    at.text_input(key="w_review_folder").set_value(str(tmp_path / "out")).run()
    assert not at.exception, at.exception


def test_app_bad_folder(app):
    at = app
    at.text_input(key="w_folder").set_value("/this/folder/does/not/exist").run()
    assert not at.exception
    assert any("does not exist" in e.value for e in at.error)
    for page in PAGES[1:]:
        go(at, page)


def test_app_spec_with_negative_slices_and_single_slice(app, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))               # the output folder is given with ~ below
    at = app
    button(at, "…or use the example subject").click().run()
    at.session_state["w_slices"] = (3, -1)          # as stored by a loaded spec.json
    go(at, PAGES[2])
    assert at.session_state.w_slices == (3, 15)
    at.slider(key="w_slices").set_value((5, 5)).run()
    assert not at.exception, at.exception
    go(at, PAGES[3])
    at.radio(key="w_mode").set_value("Semi-Auto").run()
    at.text_input(key="w_output").set_value("~/one").run()
    button(at, "▶ Run SAMson").click().run()
    assert not at.exception, at.exception
    assert os.path.exists(tmp_path / "one" / "d1_A110123_3mo_T2w_mask.nii.gz")
    go(at, PAGES[4])
    assert not at.exception, at.exception
