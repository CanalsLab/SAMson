import os
import shutil
import subprocess
import sys

import nibabel as nib
import numpy as np
import pytest
from conftest import read_json, read_text

from samson_mri.cli import main


def test_help_and_version(capsys):
    assert main([]) == 0
    assert "segment" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0


@pytest.mark.skipif(shutil.which("samson") is None, reason="console script not installed")
def test_console_script():
    output = subprocess.run(["samson", "--help"], capture_output=True, text=True)
    assert output.returncode == 0 and "preview" in output.stdout


def test_module_entry_point():
    output = subprocess.run([sys.executable, "-m", "samson_mri", "--version"], capture_output=True, text=True)
    assert output.returncode == 0 and "samson-mri" in output.stdout


def test_info(capsys):
    assert main(["info"]) == 0
    assert "Template" in capsys.readouterr().out


def test_spec_command(tmp_path, capsys):
    path = str(tmp_path / "spec.json")
    assert main(["spec", "--template", "12", "22", "--orientation", "RIA", "--slices", "1", "-1", "--name", "collab", "-o", path]) == 0
    data = read_json(path)
    assert data["template"] == [12, 22] and data["orientation"]["flip"] == [False, True, False] and data["name"] == "collab"
    assert main(["spec", "--show", path]) == 0
    # options override the file
    assert main(["spec", "--spec", path, "--template", "10", "20", "--flip-ap", "-o", path]) == 0
    data = read_json(path)
    assert data["template"] == [10, 20] and data["orientation"]["flip"] == [False, True, True]


def test_missing_template_is_a_clear_error(example_image, capsys):
    assert main(["preview", example_image]) == 2
    assert "template range is required" in capsys.readouterr().err


def test_preview(example_image, tmp_path, capsys):
    output = str(tmp_path / "preview.png")
    assert main(["preview", example_image, "--preset", "article-d1", "--header-scale", "0.1", "-o", output]) == 0
    assert os.path.getsize(output) > 10000
    assert main(["preview", example_image, "--template", "30", "38", "--no-boxes", "-o", output]) == 0
    assert "WARNING" in capsys.readouterr().out


def test_segment_select_evaluate(example_image, example_mask, no_model, tmp_path, capsys):
    out = str(tmp_path / "out")
    assert main(["segment", example_image, "-o", out, "--template", "4", "33", "--mode", "semi-auto"]) == 0
    stem = "d1_A110123_3mo_T2w"
    for suffix in ("_mask.nii.gz", "_candidates.nii.gz", "_samson.json", "_qc.png"):
        assert os.path.exists(os.path.join(out, stem + suffix))
    assert os.path.exists(os.path.join(out, "samson_spec.json")) and os.path.exists(os.path.join(out, "samson_summary.csv"))
    sidecar = os.path.join(out, stem + "_samson.json")
    assert main(["select", sidecar, "--choices", "1"]) == 0
    assert read_json(sidecar)["selection"] == [1] * 16
    assert main(["select", sidecar, "--reference", example_mask]) == 0
    assert read_json(sidecar)["selection_source"] == "reference"
    csv_path = str(tmp_path / "eval.csv")
    assert main(["evaluate", "--masks", os.path.join(out, stem + "_mask.nii.gz"), "--references", example_mask, "-o", csv_path]) == 0
    assert "Jaccard" in capsys.readouterr().out and os.path.exists(csv_path)


def test_segment_folder_with_references(example_image, example_mask, no_model, tmp_path):
    images, references = tmp_path / "images", tmp_path / "references"
    images.mkdir(), references.mkdir()
    shutil.copy(example_image, images / "mouse1.nii.gz")
    shutil.copy(example_image, images / "mouse2.nii.gz")
    shutil.copy(example_mask, references / "mouse1_manual.nii.gz")
    shutil.copy(example_mask, references / "mouse2_manual.nii.gz")
    out = str(tmp_path / "out")
    assert main(["segment", str(images), "-o", out, "--preset", "article-d1", "--mode", "semi-auto", "--reference-dir", str(references),
                 "--reference-pattern", "{stem}_manual.nii.gz", "--no-qc"]) == 0
    summary = read_text(os.path.join(out, "samson_summary.csv"))
    assert summary.count(",ok,") == 2
    # Second run: nothing to do
    assert main(["segment", str(images), "-o", out, "--preset", "article-d1"]) == 0


def test_segment_with_spec_file_and_orientation(example_image, no_model, tmp_path):
    data = np.asanyarray(nib.load(example_image).dataobj)
    flipped = str(tmp_path / "flipped.nii.gz")
    nib.save(nib.Nifti1Image(np.ascontiguousarray(np.flip(data, 2)), np.eye(4)), flipped)
    spec = str(tmp_path / "spec.json")
    assert main(["spec", "--template", "4", "33", "--flip-ap", "-o", spec]) == 0
    assert main(["segment", flipped, "-o", str(tmp_path / "a"), "--spec", spec, "--no-qc"]) == 0
    assert main(["segment", example_image, "-o", str(tmp_path / "b"), "--template", "4", "33", "--no-qc"]) == 0
    a = np.asanyarray(nib.load(str(tmp_path / "a" / "flipped_mask.nii.gz")).dataobj)
    b = np.asanyarray(nib.load(str(tmp_path / "b" / "d1_A110123_3mo_T2w_mask.nii.gz")).dataobj)
    assert np.array_equal(a, np.flip(b, 2))


def test_suggest_range(example_mask, capsys):
    assert main(["suggest-range", example_mask, "--slice-mm", "0.8", "--top", "3"]) == 0
    assert "Suggested: --template" in capsys.readouterr().out


def test_example_command_with_fake_model(no_model, tmp_path, capsys):
    # With the fake model the masks differ from the published ones: the command must say so and return 1
    assert main(["example", "-o", str(tmp_path / "ex")]) == 1
    output = capsys.readouterr().out
    assert "identical to the article's published mask: NO" in output
    assert os.path.exists(tmp_path / "ex" / "example_ap_preview.png")


def test_choices_parsing():
    from samson_mri.cli import parse_choices
    assert parse_choices("F,0,2,-1") == [-1, 0, 2, -1]
    assert parse_choices("1") == 1 and parse_choices("f") == -1 and parse_choices(None) is None
    with pytest.raises(ValueError):
        parse_choices("3")


def test_choices_starting_with_minus_one(example_image, no_model, tmp_path):
    out = str(tmp_path / "out")
    assert main(["segment", example_image, "-o", out, "--preset", "article-d1", "--mode", "semi-auto", "--no-qc"]) == 0
    sidecar = os.path.join(out, "d1_A110123_3mo_T2w_samson.json")
    choices = ",".join(["-1"] * 8 + ["2"] * 8)
    assert main(["select", sidecar, "--choices", choices, "--no-qc"]) == 0
    assert read_json(sidecar)["selection"] == [-1] * 8 + [2] * 8
    assert main(["select", sidecar, "--choices", "F,F,F,F,F,F,F,F,1,1,1,1,1,1,1,1", "--no-qc"]) == 0
    assert read_json(sidecar)["selection"] == [-1] * 8 + [1] * 8


def test_orientation_options_override_spec(tmp_path):
    from samson_mri.cli import build_parser, spec_from_args
    from samson_mri.spec import SpecError
    path = str(tmp_path / "spec.json")
    assert main(["spec", "--template", "4", "33", "--axes", "0", "2", "1", "--flip-dv", "--flip-ap", "-o", path]) == 0

    def spec(*options):
        return spec_from_args(build_parser().parse_args(["preview", "x.nii.gz", *options]))
    assert spec("--spec", path).orientation.flip == (False, True, True)
    assert spec("--spec", path, "--flip-ap").orientation.flip == (False, True, True)          # sets, does not toggle
    assert spec("--spec", path, "--axes", "0", "1", "2").orientation.axes == (0, 1, 2)
    assert spec("--spec", path, "--axes", "0", "1", "2").orientation.flip == (False, True, True)   # flips kept
    assert spec("--spec", path, "--orientation", "RSA").orientation.flip == (False, False, False)
    assert spec("--template", "4", "33", "--orientation", "RIP", "--flip-ap").orientation.flip == (False, True, True)
    with pytest.raises(SpecError):
        spec("--template", "4", "33", "--orientation", "RIA", "--axes", "0", "2", "1")


def test_reference_names_cannot_collide_with_outputs(example_image, no_model, tmp_path, capsys):
    folder = tmp_path / "data"
    folder.mkdir()
    shutil.copy(example_image, folder / "mouse.nii.gz")
    assert main(["segment", str(folder), "-o", str(folder), "--preset", "article-d1", "--reference-dir", str(folder)]) == 1
    assert "names of SAMson's output masks" in capsys.readouterr().err


def test_app_command_listens_on_localhost(monkeypatch):
    pytest.importorskip("streamlit")
    import samson_mri.cli as cli
    calls = []
    monkeypatch.setattr(cli.subprocess, "call", lambda command: calls.append(command) or 0)
    assert main(["app", "--port", "8999", "--folder", "/data"]) == 0
    command = calls[0]
    assert command[command.index("--server.address") + 1] == "localhost"
    assert command[command.index("--server.port") + 1] == "8999" and command[-2:] == ["--folder", "/data"]
