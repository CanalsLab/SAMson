"""
SAMson app (Streamlit). Start it with ``samson app`` and open the address it prints in a web browser.

Five steps, in the sidebar: 1 Images -> 2 Orientation -> 3 Anteroposterior range -> 4 Segment -> 5 Review. The choices of
steps 2-3 form the dataset specification (spec.json), which the app saves with the results and which the command line
(`samson segment --spec spec.json ...`) can reuse for large batches.
"""
from __future__ import annotations

import argparse
import glob
import inspect
import json
import os
import shlex
import sys
import tempfile
import time

import nibabel as nib
import numpy as np
import streamlit as st

import samson_mri
from samson_mri.ap import TYPICAL_LENGTH_RATIO, check_range, length_ratio, linked_range, plausible_voxel_size, suggest_from_masks
from samson_mri.example import example_path
from samson_mri.io import find_images, load_mask, subject_stem
from samson_mri.model import CHECKPOINTS, CheckpointNotFound, cache_dir, download_checkpoint, find_checkpoint
from samson_mri.pipeline import FULL_AUTO, SAMson, compute_boxes, load_result
from samson_mri.spec import DatasetSpec, Orientation, SpecError
from samson_mri.viz import ap_figure, candidates_figure, orientation_figure, qc_figure

st.set_page_config(page_title="SAMson", page_icon="🧠", layout="wide")
_VERSION = tuple(int(v) for v in st.__version__.split(".")[:2] if v.isdigit())
STRETCH = {"width": "stretch"} if _VERSION >= (1, 46) else {"use_container_width": True}   # full-width elements
WRAP = {"wrap_lines": True} if "wrap_lines" in inspect.signature(st.code).parameters else {}   # wrap long code lines
ss = st.session_state
PAGES = ["1 · Images", "2 · Orientation", "3 · Anteroposterior range", "4 · Segment", "5 · Review"]


def app_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="")
    return parser.parse_known_args(sys.argv[1:])[0]


# ----------------------------------------------------------------------------------------------------------------------
# State: widget values are kept in st.session_state under keys "w_*" (re-assigned on every run so that Streamlit does
# not drop them when their page is not shown).
# ----------------------------------------------------------------------------------------------------------------------
DEFAULTS = dict(w_page=PAGES[0], w_folder=app_arguments().folder, w_pattern="*", w_rep=None, w_volume=0,
                w_ap_axis=2, w_dv_axis=1, w_flip_ap=False, w_flip_dv=False, w_flip_lr=False,
                w_slices=None, w_template=(4, 33), w_link=False, w_t0=4, w_ratio=1.1, w_header_scale=1.0, w_x=None,
                w_boxes=True, w_otsu_auto=True, w_otsu_radius=2, w_otsu_numpass=5,
                w_mode="Full-Auto", w_which="Only the representative image", w_selected=[], w_output="", w_qc=True,
                w_brain=False, w_overwrite=False, w_device="auto", w_model="vit_h", w_checkpoint="",
                w_mask_folder="", w_mask_pattern="*mask*", w_review_source="This session", w_review_folder="",
                w_reference="", w_zoom=True)
for key, value in DEFAULTS.items():
    if key not in ss:
        ss[key] = value
for key in [k for k in ss.keys() if str(k).startswith("w_")]:
    ss[key] = ss[key]
ss.setdefault("results", {})        # image path -> SegmentationResult (this session)
ss.setdefault("suggestions", None)


# ----------------------------------------------------------------------------------------------------------------------
# Cached helpers
# ----------------------------------------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=16)
def header_info(path: str) -> dict:
    image = nib.load(path)
    zooms = image.header.get_zooms()
    try:
        codes = "".join(nib.aff2axcodes(image.affine))
    except Exception:
        codes = "???"
    return dict(shape=tuple(int(s) for s in image.shape), zooms=tuple(round(float(z), 4) for z in zooms[:3]), codes=codes,
                volumes=int(np.prod(image.shape[3:])) if len(image.shape) > 3 else 1)


@st.cache_data(show_spinner="Loading the image...", max_entries=6)
def load_data(path: str, volume: int) -> np.ndarray:
    from samson_mri.io import load_volume_array
    data = load_volume_array(nib.load(path), volume, path)
    return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)


@st.cache_data(show_spinner="Computing SAMson's box prompts (median_otsu)...", max_entries=32)
def boxes_for(path: str, spec_json: str):
    from samson_mri.io import load_subject
    spec = DatasetSpec.from_dict(json.loads(spec_json))
    return compute_boxes(load_subject(path, spec), spec)


@st.cache_resource(show_spinner="Loading SAM (this takes a few seconds the first time)...")
def get_samson(model_type: str, checkpoint: str, device: str) -> SAMson:
    samson = SAMson(model_type=model_type, checkpoint=checkpoint or None, device=device)
    _ = samson.predictor   # load now (assigned: a bare expression would be displayed by Streamlit)
    return samson


def current_orientation() -> Orientation:
    ap, dv = ss.w_ap_axis, ss.w_dv_axis
    lr = ({0, 1, 2} - {ap, dv}).pop()
    return Orientation((lr, dv, ap), (ss.w_flip_lr, ss.w_flip_dv, ss.w_flip_ap))


def normalise_slices(n_slices: int) -> tuple:
    """The stored slice range as non-negative indices valid for an image with n_slices slices (stored back)."""
    first, last = ss.w_slices if ss.w_slices is not None else (0, -1)
    first, last = (first + n_slices if first < 0 else first), (last + n_slices if last < 0 else last)
    if not 0 <= first <= last < n_slices:
        first, last = 0, n_slices - 1
    if ss.w_slices != (int(first), int(last)):   # (only when it changes: the slider may already exist in this run)
        ss.w_slices = (int(first), int(last))
    return int(first), int(last)


def current_template(n_slices: int | None = None) -> tuple:
    if ss.w_link and n_slices is not None:
        first, last = normalise_slices(n_slices)
        size = current_orientation().voxel_size(header_info(ss.w_rep)["zooms"])
        return linked_range(ss.w_t0, last - first + 1, size[2] * ss.w_header_scale, ss.w_ratio)
    return tuple(ss.w_template)


def current_spec(n_slices: int | None = None) -> DatasetSpec:
    slices = normalise_slices(n_slices) if n_slices is not None else (ss.w_slices or (0, -1))
    if n_slices is not None and slices[1] == n_slices - 1:
        slices = (slices[0], -1)
    otsu = "auto" if ss.w_otsu_auto else (int(ss.w_otsu_radius), int(ss.w_otsu_numpass))
    return DatasetSpec(template=current_template(n_slices), slices=tuple(int(s) for s in slices), orientation=current_orientation(),
                       otsu=otsu, volume=int(ss.w_volume), name=os.path.basename(os.path.normpath(ss.w_folder)) if ss.w_folder else "")


def images() -> list:
    if not ss.w_folder:
        return []
    try:
        return find_images(ss.w_folder, pattern=ss.w_pattern or "*")
    except FileNotFoundError:
        return []


def representative_volume():
    """Representative image in SAMson's orientation, its voxel sizes and number of slices (or None)."""
    if not ss.w_rep or not os.path.exists(ss.w_rep):
        return None
    data = load_data(ss.w_rep, int(ss.w_volume))
    orientation = current_orientation()
    volume = orientation.apply(data)
    return volume, orientation.voxel_size(header_info(ss.w_rep)["zooms"])


def need_image() -> bool:
    if not ss.w_rep:
        st.info("First choose a folder with images and a representative image in step **1 · Images**.")
        return True
    return False


# ---- callbacks (they run before the next rerun, so they may change widget values) -------------------------------------
def use_example():
    ss.w_folder = os.path.dirname(example_path("image"))
    ss.w_pattern = "*T2w*"
    ss.w_rep = example_path("image")
    ss.w_ap_axis, ss.w_dv_axis, ss.w_flip_ap, ss.w_flip_dv, ss.w_flip_lr = 2, 1, False, False, False
    ss.w_slices, ss.w_template, ss.w_link, ss.w_header_scale = None, (4, 33), False, 0.1
    ss.w_t0 = 4
    ss.w_output = ""


def guess_orientation():
    codes = header_info(ss.w_rep)["codes"]
    try:
        orientation = Orientation.from_codes(codes)
    except SpecError:
        st.toast(f"The header codes {codes} could not be used.")
        return
    lr, dv, ap = orientation.axes
    ss.w_ap_axis, ss.w_dv_axis = ap, dv
    ss.w_flip_lr, ss.w_flip_dv, ss.w_flip_ap = orientation.flip
    ss.w_slices = None


def reset_slices():
    ss.w_slices = None


def folder_changed():
    ss.w_slices, ss.w_output = None, ""


def default_output_folder() -> str:
    """<image folder>/samson_output, or ./samson_output for the bundled example and folders that cannot be written."""
    folder = os.path.abspath(os.path.expanduser(ss.w_folder or "."))
    example = os.path.dirname(example_path("image"))
    if folder == os.path.abspath(example) or not os.access(folder, os.W_OK):
        return os.path.join(os.getcwd(), "samson_output")
    return os.path.join(folder, "samson_output")


def use_uploaded(folder, name, path):
    ss.w_folder, ss.w_pattern, ss.w_rep, ss.w_slices, ss.w_output = folder, name, path, None, ""


def apply_suggestion(t0, t1):
    ss.w_link = False
    ss.w_template = (int(t0), int(t1))


def load_spec_file():
    uploaded = ss.get("spec_upload")
    if uploaded is None:
        return
    try:
        spec = DatasetSpec.from_dict(json.loads(uploaded.getvalue()))
    except (SpecError, ValueError) as error:
        st.toast(f"Could not read the specification: {error}")
        return
    lr, dv, ap = spec.orientation.axes
    ss.w_ap_axis, ss.w_dv_axis = ap, dv
    ss.w_flip_lr, ss.w_flip_dv, ss.w_flip_ap = spec.orientation.flip
    ss.w_template, ss.w_link, ss.w_volume = spec.template, False, spec.volume
    ss.w_slices = None if tuple(spec.slices) == (0, -1) else tuple(spec.slices)
    ss.w_otsu_auto = spec.otsu == "auto"
    if spec.otsu != "auto":
        ss.w_otsu_radius, ss.w_otsu_numpass = spec.otsu


# ----------------------------------------------------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"## 🧠 SAMson `{samson_mri.__version__}`")
    st.caption("Brain extraction of rodent MRI with the Segment Anything Model")
    st.radio("Steps", PAGES, key="w_page", label_visibility="collapsed")
    st.divider()
    st.markdown("**Dataset specification**")
    try:
        info = representative_volume()
        spec = current_spec(info[0].shape[2] if info else None)
        st.code(spec.to_json(), language="json", **WRAP)
        st.download_button("Download spec.json", spec.to_json() + "\n", file_name="samson_spec.json", mime="application/json")
    except SpecError as error:
        st.error(str(error))
    st.file_uploader("Load a spec.json", type=["json"], key="spec_upload", on_change=load_spec_file)
    st.caption("Help: README.md and docs/user_guide.md of the package.")


# ----------------------------------------------------------------------------------------------------------------------
# 1 Images
# ----------------------------------------------------------------------------------------------------------------------
def page_images():
    st.title("1 · Images")
    st.markdown("Choose the folder with the NIfTI images (`.nii` or `.nii.gz`) of **one dataset** (same acquisition and "
                "orientation), and one representative image: the next steps use it to set the orientation and the "
                "anteroposterior range for the whole dataset.")
    left, right = st.columns([3, 1])
    left.text_input("Folder with the images (a path on the computer that runs the app)", key="w_folder",
                    placeholder="/data/my_study/T2w", on_change=folder_changed)
    right.text_input("File name filter", key="w_pattern", help="e.g. *T2w* to use only the files whose name contains T2w")
    st.button("…or use the example subject (a mouse of the article's dataset D1)", on_click=use_example)
    with st.expander("…or upload an image (for a quick test on this computer)"):
        uploaded = st.file_uploader("NIfTI image", type=["nii", "gz"])
        if uploaded is not None:
            folder = os.path.join(tempfile.gettempdir(), "samson_uploads")
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, uploaded.name)
            with open(path, "wb") as f:
                f.write(uploaded.getbuffer())
            st.button(f"Use {uploaded.name}", on_click=use_uploaded, args=(folder, uploaded.name, path))

    files = images()
    if ss.w_folder and not os.path.isdir(os.path.expanduser(ss.w_folder)):
        st.error(f"The folder {ss.w_folder} does not exist.")
        return
    if not files:
        if ss.w_folder:
            st.warning("No NIfTI images found in this folder (with this filter).")
        return
    rows = []
    for path in files:
        try:
            info = header_info(path)
            rows.append(dict(image=os.path.basename(path), shape=" × ".join(map(str, info["shape"])),
                             voxel_size=" × ".join(f"{z:g}" for z in info["zooms"]), header_axes=info["codes"]))
        except Exception as error:   # unreadable file
            rows.append(dict(image=os.path.basename(path), shape=f"unreadable: {error}", voxel_size="", header_axes=""))
    st.markdown(f"**{len(files)} image(s)**")
    st.dataframe(rows, hide_index=True, **STRETCH, height=min(38 * (len(rows) + 1), 300))
    if ss.w_rep not in files:
        ss.w_rep = files[0]
    st.selectbox("Representative image", files, key="w_rep", format_func=os.path.basename, on_change=reset_slices)
    info = header_info(ss.w_rep)
    if info["volumes"] > 1:
        st.number_input(f"This is a 4D image with {info['volumes']} volumes: volume to use (0 = first; e.g. the first echo "
                        "or the b = 0 image)", min_value=0, max_value=info["volumes"] - 1, key="w_volume", on_change=reset_slices)
    zooms = info["zooms"]
    if min(zooms[:2]) > 0.5:
        st.info(f"The header voxel size ({zooms[0]:g} × {zooms[1]:g} mm in plane) looks 10 times too large for a rodent: "
                "some datasets store voxel sizes ×10. This does not change SAMson's masks; in step 3, set the header scale to 0.1 "
                "so that lengths in mm are right.")
    st.success("Next: **2 · Orientation** (sidebar).")


# ----------------------------------------------------------------------------------------------------------------------
# 2 Orientation
# ----------------------------------------------------------------------------------------------------------------------
def page_orientation():
    st.title("2 · Orientation")
    if need_image():
        return
    st.markdown("SAMson works on **coronal slices** with the **dorsal side up**, stacked from the **caudal** end (cerebellum) "
                "to the **rostral** end (olfactory bulbs). Tell it how your images are stored: compare your image (top) with "
                "the template (bottom) and change the options until they look alike. The image is only re-oriented in memory; "
                "the masks are saved on the grid of your images.")
    controls, figure = st.columns([1.45, 3])
    with controls:
        st.radio("Image axis that runs from the back to the front of the head (anteroposterior)", [0, 1, 2], key="w_ap_axis",
                 horizontal=True, on_change=reset_slices)
        options = [a for a in (0, 1, 2) if a != ss.w_ap_axis]
        if ss.w_dv_axis not in options:
            ss.w_dv_axis = options[-1]
        st.radio("Image axis that runs from the bottom to the top of the head (dorsoventral)", options, key="w_dv_axis", horizontal=True)
        st.checkbox("Reverse anteroposterior axis", key="w_flip_ap", on_change=reset_slices,
                    help="The side view must show the caudal end (cerebellum) on the LEFT, as in the template.")
        st.checkbox("Reverse dorsoventral axis", key="w_flip_dv",
                    help="The coronal slice must show the top of the head UP, as in the template.")
        st.checkbox("Reverse left–right (optional)", key="w_flip_lr", help="SAMson's masks do not depend on it.")
        st.button("Guess from the image header", on_click=guess_orientation,
                  help="Uses the axis labels of the NIfTI header. Rodent headers are often wrong, so check the views.")
        orientation = current_orientation()
        header, codes = header_info(ss.w_rep)["codes"], orientation.to_codes()
        same = codes == header.replace("L", "R")     # (SAMson does not depend on left and right)
        st.markdown(f"Image header labels: **`{header}`**  \nCurrent setting: **`{codes}`**" + (" ✓ same as the header" if same else ""))
        st.caption("Labels: the direction each image axis points to (R right, A anterior, P posterior, S superior, I inferior). "
                   f"Command line: `--axes {' '.join(map(str, orientation.axes))}`" + (" `--flip-dv`" if orientation.flip[1] else "")
                   + (" `--flip-ap`" if orientation.flip[2] else ""))
    with figure:
        volume, size = representative_volume()
        st.pyplot(orientation_figure(volume, size), **STRETCH)
    st.success("When your image looks like the template, go to **3 · Anteroposterior range**.")


# ----------------------------------------------------------------------------------------------------------------------
# 3 Anteroposterior range
# ----------------------------------------------------------------------------------------------------------------------
def page_range():
    st.title("3 · Anteroposterior range")
    if need_image():
        return
    volume, size = representative_volume()
    n = volume.shape[2]
    _ = normalise_slices(n)
    st.markdown("SAMson builds the box prompt of every slice from the template slice that matches it. Tell it which **template "
                "slices** (0 = caudal end, 39 = rostral end) match your **first and last processed slice**; the slices in between "
                "are matched linearly. Check the coloured pairs below the side views: **each pair should show the same anatomy** "
                "(the green box should fit the brain).")
    controls, figure = st.columns([1.3, 2.6])
    with controls:
        st.slider("Slices to process (0 = caudal)", 0, n - 1, key="w_slices",
                  help="Slices outside this range are left empty in the mask (e.g. slices without brain).")
        st.toggle("Link the two range lengths", key="w_link",
                  help="Keeps the template range as long (in mm) as the processed slices, divided by the length ratio; you then "
                       "only move the first template slice.")
        if ss.w_link:
            st.slider("First template slice", 0, 38, key="w_t0")
            st.slider("Length ratio (subject range / template range; 1.0-1.25 in the article and benchmark data)", 0.7, 1.6,
                      step=0.05, key="w_ratio")
            t0, t1 = current_template(n)
            st.markdown(f"Template slices: **{t0}–{t1}**")
        else:
            st.slider("Template slices matching the first and last processed slice", 0, 39, key="w_template")
        st.number_input("Header scale (0.1 if the header stores voxel sizes ×10)", min_value=0.001, max_value=10.0, step=0.1,
                        key="w_header_scale", format="%.3g")
        if ss.w_x is None or ss.w_x >= volume.shape[0]:
            ss.w_x = volume.shape[0] // 2
        st.slider("Side view position (left-right)", 0, volume.shape[0] - 1, key="w_x")
        st.checkbox("Show the box prompts", key="w_boxes", help="SAMson's box prompts (computes median_otsu)")
        with st.expander("median_otsu settings (advanced)"):
            st.checkbox("Automatic (as in the article: 2 and 5 for images with fewer than 140 rows, else 1 and 3)", key="w_otsu_auto")
            if not ss.w_otsu_auto:
                st.number_input("median radius", 1, 10, key="w_otsu_radius")
                st.number_input("number of passes", 1, 20, key="w_otsu_numpass")
    t0, t1 = current_template(n)
    spec = current_spec(n)
    first, last = ss.w_slices
    with figure:
        # the length check and the warnings first, where they are seen without scrolling
        scaled = [v * ss.w_header_scale for v in size]
        if plausible_voxel_size(scaled) and last > first and t1 > t0:
            st.caption(f"Processed range {(last - first) * scaled[2]:.1f} mm; template range {(t1 - t0) * 0.4:.1f} mm; length ratio "
                       f"{length_ratio(last - first + 1, scaled[2], t0, t1):.2f} (usually {TYPICAL_LENGTH_RATIO[0]}-{TYPICAL_LENGTH_RATIO[1]}).")
        for warning in check_range(last - first + 1, t0, t1, size, ss.w_header_scale):
            st.warning(warning["detail"])
        prompts = None
        if ss.w_boxes:
            try:
                prompts = boxes_for(ss.w_rep, json.dumps(spec.to_dict()))
            except Exception as error:
                st.error(f"The box prompts could not be computed: {error}")
        fig = ap_figure(volume, size, t0, t1, first, last, x=ss.w_x, prompts=prompts, header_scale=ss.w_header_scale,
                        title=os.path.basename(ss.w_rep))
        st.pyplot(fig, **STRETCH)

    with st.expander("Suggest the range from manual masks of a few subjects (optional)"):
        st.markdown("If you have manual brain masks of some subjects of this dataset (same grid as their images), SAMson can rank "
                    "the template ranges by how well its template boxes fit the boxes of your masks. Check the best ones above.")
        c1, c2 = st.columns([3, 1])
        c1.text_input("Folder with the manual masks", key="w_mask_folder")
        c2.text_input("File name filter", key="w_mask_pattern")
        if st.button("Suggest"):
            try:
                paths = find_images(ss.w_mask_folder, pattern=ss.w_mask_pattern, skip_outputs=False)
                masks = [current_orientation().apply(load_mask(p)) for p in paths]
                masks = [m for m in masks if m.shape[2] == n]
                if not masks:
                    st.error(f"No masks with {n} slices found.")
                else:
                    slice_mm = size[2] * ss.w_header_scale if plausible_voxel_size([v * ss.w_header_scale for v in size]) else None
                    ss.suggestions = suggest_from_masks(masks, first, last, top=8, slice_mm=slice_mm)
                    ss.suggestions_n = len(masks)
            except Exception as error:
                st.error(str(error))
        if ss.suggestions is not None:
            st.caption(f"Ranked by box agreement (IoU) with {ss.get('suggestions_n', '?')} mask(s):")
            for _, row in ss.suggestions.iterrows():
                a, b = int(row.template_first), int(row.template_last)
                st.button(f"Use template slices {a}–{b}  (box IoU {row.box_iou:.3f})", key=f"use_{a}_{b}", on_click=apply_suggestion, args=(a, b))
    with st.expander("How to read this page"):
        st.markdown("""
* **Top left**: side view of your image (dorsal up, caudal end on the left). Vertical lines: the processed slices.
* **Top right**: side view of the template. The lines show where each processed slice falls in the template.
* **Bottom**: the coloured pairs: a slice of your image (with SAMson's box prompt in green) above the template slice it is
  matched to. Adjust the template slices until every pair shows the same anatomy and the boxes fit the brain.
* The dashed line is the **anchor slice**: SAMson measures the brain box on it (median_otsu) and scales the template boxes
  on it. It should show the brain clearly.
* Images that do not cover the whole brain (e.g. a few slices around the hippocampus) need a template range that covers
  only the matching part of the template.
""")
    st.success("When the pairs match, go to **4 · Segment**.")


# ----------------------------------------------------------------------------------------------------------------------
# 4 Segment
# ----------------------------------------------------------------------------------------------------------------------
def page_segment():
    st.title("4 · Segment")
    if need_image():
        return
    volume, _ = representative_volume()
    try:
        spec = current_spec(volume.shape[2])
        spec.check_template()
    except SpecError as error:
        st.error(str(error))
        return
    files = images()
    with st.container(border=True):
        st.markdown("**SAM model**")
        c1, c2, c3 = st.columns([1, 1, 3])
        c1.selectbox("Model", list(CHECKPOINTS), key="w_model", help="vit_h is the model of the article; vit_l and vit_b are smaller")
        c2.selectbox("Device", ["auto", "cuda", "cpu", "mps"], key="w_device", help="auto: the GPU if there is one")
        c3.text_input("Checkpoint file (empty: the downloaded one)", key="w_checkpoint")
        try:
            checkpoint = find_checkpoint(ss.w_model, ss.w_checkpoint or None)
            st.caption(f"Checkpoint: `{checkpoint}`")
        except CheckpointNotFound:
            checkpoint = None
            st.warning(f"SAM's {ss.w_model} checkpoint was not found. Download it once (it is saved in `{cache_dir()}`).")
            if st.button(f"Download {CHECKPOINTS[ss.w_model]['file']}"):
                bar = st.progress(0.0, text="Downloading...")
                _ = download_checkpoint(ss.w_model, progress=False,
                                    callback=lambda done, total: bar.progress(done / total, text=f"Downloading... {done / 1e9:.2f} / {total / 1e9:.2f} GB"))
                st.rerun()
    with st.container(border=True):
        st.markdown("**What to segment**")
        st.radio("Mode", ["Full-Auto", "Semi-Auto"], key="w_mode", horizontal=True,
                 help="Full-Auto: one mask per slice. Semi-Auto: SAM's three candidates per slice are also saved, and you can "
                      "choose the best one of any slice in step 5.")
        st.radio(f"Images ({len(files)} in the folder)", ["Only the representative image", "All images of the folder", "Choose"],
                 key="w_which", horizontal=True)
        if ss.w_which == "Choose":
            ss.w_selected = [f for f in ss.w_selected if f in files]
            st.multiselect("Images to segment", files, key="w_selected", format_func=os.path.basename)
        if not ss.w_output:
            ss.w_output = default_output_folder()
        st.text_input("Output folder", key="w_output")
        c1, c2, c3 = st.columns(3)
        c1.checkbox("Quality-control figures", key="w_qc")
        c2.checkbox("Also save the brain-extracted images", key="w_brain")
        c3.checkbox("Redo images that already have a mask", key="w_overwrite")
    output = os.path.abspath(os.path.expanduser(ss.w_output))
    todo = [ss.w_rep] if ss.w_which.startswith("Only") else files if ss.w_which.startswith("All") else list(ss.w_selected)
    mode = "semi-auto" if ss.w_mode == "Semi-Auto" else "full-auto"
    command = ["samson", "segment"] + ([ss.w_folder] if ss.w_which.startswith("All") else todo) + [
        "-o", output, "--spec", os.path.join(output, "samson_spec.json"), "--mode", mode]
    if ss.w_which.startswith("All") and ss.w_pattern not in ("", "*"):
        command += ["--pattern", ss.w_pattern]
    st.caption("The same run from the command line (for large batches; the app saves samson_spec.json in the output folder when it runs):")
    st.code(shlex.join(command), language="bash", **WRAP)

    if st.button(f"▶ Run SAMson on {len(todo)} image(s)", type="primary", disabled=checkpoint is None or not todo):
        samson = get_samson(ss.w_model, checkpoint, ss.w_device)
        ss["loaded"] = {}   # results reloaded from folders may be out of date now
        os.makedirs(output, exist_ok=True)
        spec.save(os.path.join(output, "samson_spec.json"))
        bar = st.progress(0.0)
        log = st.empty()
        rows = []
        start = time.time()
        for i, path in enumerate(todo):
            stem = subject_stem(path)
            if os.path.exists(os.path.join(output, stem + "_mask.nii.gz")) and not ss.w_overwrite:
                rows.append(dict(image=stem, status="skipped (mask exists)"))
                continue

            def progress(done, total, i=i, stem=stem):
                bar.progress((i + done / total) / len(todo), text=f"{stem}: slice {done}/{total} (image {i + 1}/{len(todo)})")
            try:
                result = samson.segment(path, spec, mode=mode, progress=progress)
                _ = result.save(output, qc=ss.w_qc, save_brain=ss.w_brain)
                ss.results[path] = (result, output)
                rows.append(dict(image=stem, status="ok", slices=f"{result.prompts.first}-{result.prompts.last}",
                                 seconds=round(sum(result.timings.values()), 1)))
            except Exception as error:
                rows.append(dict(image=stem, status=f"error: {error}"))
            log.dataframe(rows, hide_index=True, **STRETCH)
        elapsed = time.time() - start
        bar.progress(1.0, text=f"Done in {elapsed:.0f} s" if elapsed < 90 else f"Done in {elapsed / 60:.1f} min")
        st.success(f"Masks saved in `{output}`. Check them in **5 · Review**.")


# ----------------------------------------------------------------------------------------------------------------------
# 5 Review
# ----------------------------------------------------------------------------------------------------------------------
def page_review():
    st.title("5 · Review")
    st.radio("Results", ["This session", "A results folder"], key="w_review_source", horizontal=True)
    result = None
    if ss.w_review_source == "This session":
        if not ss.results:
            st.info("No image segmented in this session yet (step 4), or choose 'A results folder'.")
            return
        key = st.selectbox("Image", list(ss.results), format_func=os.path.basename)
        result, folder = ss.results[key]
    else:
        if not ss.w_review_folder:
            ss.w_review_folder = ss.w_output
        st.text_input("Results folder", key="w_review_folder")
        sidecars = sorted(glob.glob(os.path.join(os.path.abspath(os.path.expanduser(ss.w_review_folder)), "*_samson.json")))
        if not sidecars:
            st.info("No SAMson results (*_samson.json) in this folder.")
            return
        path = st.selectbox("Image", sidecars, format_func=lambda p: os.path.basename(p)[:-len("_samson.json")])
        cache = ss.setdefault("loaded", {})
        if path not in cache:
            try:
                cache[path] = load_result(path)
            except Exception as error:
                st.error(str(error))
                return
        result = cache[path]
        folder = os.path.dirname(path)

    info, zoom = st.columns([3, 1])
    info.caption(f"{result.mode}; slices {result.prompts.first}-{result.prompts.last}; template {result.spec.template[0]}-{result.spec.template[1]}")
    zoom.toggle("Zoom on the brain", key="w_zoom", help="Show only the region around the brain in the figures below (the saved "
                "quality-control figure shows whole slices).")
    reference = None
    with st.expander("Compare with a reference mask (optional)"):
        st.text_input("Reference mask file (same grid as the image)", key="w_reference")
        if ss.w_reference:
            try:
                reference = load_mask(os.path.expanduser(ss.w_reference), result.subject)
                summary = result.compare(reference)
                st.markdown(f"Mean slice Jaccard **{summary['jaccard']:.4f}**, Dice {summary['dice']:.4f}, accuracy "
                            f"{summary['accuracy']:.2f} % over {summary['n_slices']} slices; 3D Jaccard {summary['jaccard_3d']:.4f}.")
                if result.candidates is not None and st.button("Select the best candidate of every slice (the article's Semi-Auto)"):
                    _ = result.select_by_reference(reference)
                    st.rerun()
            except Exception as error:
                st.error(str(error))

    if result.candidates is not None:
        st.markdown("**Semi-Auto: choose the mask of each slice.** The selection starts with the Full-Auto mask of every slice; "
                    "change only the slices that need it, then save.")
        n = result.n_processed
        if n > 1:
            index = st.slider("Slice", result.prompts.first, result.prompts.last, key=f"slice_{id(result)}") - result.prompts.first
        else:
            index = 0
            st.caption(f"One processed slice: {result.prompts.first}")
        st.pyplot(candidates_figure(result, index, zoom=ss.w_zoom), **STRETCH)
        labels = {FULL_AUTO: "Full-Auto"} | {c: f"candidate {c}" for c in range(result.candidates.shape[1])}
        choice = st.radio("Mask of this slice", list(labels), index=list(labels).index(result.selection[index]),
                          format_func=labels.get, horizontal=True, key=f"choice_{id(result)}_{index}_{result.selection[index]}")
        if choice != result.selection[index]:
            selection = list(result.selection)
            selection[index] = choice
            _ = result.select(selection)
            st.rerun()
        c1, c2 = st.columns([1, 3])
        everywhere = c1.selectbox("Set every slice to", list(labels), format_func=labels.get, key=f"all_{id(result)}")
        if c2.button("Apply to all slices"):
            _ = result.select([everywhere] * n)
            st.rerun()
        st.caption("Selection per slice (F = Full-Auto; 0, 1, 2 = candidates): " + " ".join(f"{result.prompts.first + i}:{'F' if c == FULL_AUTO else c}" for i, c in enumerate(result.selection)))

    st.pyplot(qc_figure(result, reference=reference, zoom=ss.w_zoom), **STRETCH)
    c1, c2 = st.columns(2)
    if c1.button("💾 Save the mask", type="primary"):
        written = result.save(folder, qc=True)
        st.success("Saved: " + ", ".join(f"`{os.path.basename(p)}`" for p in written.values()))
    mask_path = os.path.join(folder, result.subject.stem + "_mask.nii.gz")
    if os.path.exists(mask_path):
        with open(mask_path, "rb") as f:
            c2.download_button("Download the saved mask", f.read(), file_name=os.path.basename(mask_path))


{PAGES[0]: page_images, PAGES[1]: page_orientation, PAGES[2]: page_range, PAGES[3]: page_segment, PAGES[4]: page_review}[ss.w_page]()
