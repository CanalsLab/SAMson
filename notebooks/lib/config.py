"""
Paths, environments, tool registry and tool parameters of the mask generation (`run_all.py`, `SAMson_experiments` and `SAMson_benchmark`).

Every path inside the release is derived from the location of this file (`<release>/notebooks/lib/config.py`). The
external software (model weights, conda environments, FSL, ANTs, the BEN and RBM repositories) is located with
environment variables; their defaults are the installation of the publication run (`notebooks/REPRODUCIBILITY.md`,
section "External software").
"""
import os
import sys
import json
import shutil
from os.path import join as pjoin

# ------------------------------------------------------------------ paths of the release
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # release folder
P = ROOT                                                      # (name used by the notebooks and the library)
DATASETS = pjoin(P, "datasets")                               # harmonized datasets
DOCS = pjoin(P, "docs")
NOTEBOOKS = pjoin(P, "notebooks")
RUNNERS = pjoin(NOTEBOOKS, "runners")
FULL_RUN_FLAG = pjoin(P, ".FULL_RUN_ACTIVE")                  # while it exists the full run owns the allocation (cores, GPU)


# ------------------------------------------------------------------ external software (environment variables)
def _external(variable, default):
    return os.path.expanduser(os.environ.get(variable, default))


# Model weights: SAM ViT-H (sam_vit_h_4b8939.pth; `samson download-model` puts it in ~/.cache/samson_mri), MedSAM ViT-B
# (medsam_vit_b.pth) and SAM2.1 hiera-large (sam2.1_hiera_large.pt), all in one folder
CHECKPOINTS = _external("SAMSON_CHECKPOINTS", "~/.cache/samson_mri")
SAM_CHECKPOINT = pjoin(CHECKPOINTS, "sam_vit_h_4b8939.pth")
MEDSAM_CHECKPOINT = pjoin(CHECKPOINTS, "medsam_vit_b.pth")
SAM2_CHECKPOINT = pjoin(CHECKPOINTS, "sam2.1_hiera_large.pt")
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
_ENVS = os.path.dirname(sys.prefix) if os.path.basename(os.path.dirname(sys.prefix)) == "envs" else pjoin(sys.prefix, "envs")
CONDA = _external("SAMSON_CONDA_ENVS", _ENVS)   # folder of the conda environments below (default: the envs folder of the running conda)
PY_SAMSON = _external("SAMSON_PY_SAMSON", f"{CONDA}/samson/bin/python")           # SAMson, MedSAM, this library (torch 2.4.1)
PY_SAM2 = _external("SAMSON_PY_SAM2", f"{CONDA}/samson_sam2/bin/python")          # SAM2 (torch 2.5.1)
PY_BEN = _external("SAMSON_PY_BEN", f"{CONDA}/ben/bin/python")                    # BEN and RBM (Python 3.7, TensorFlow 1.15, Keras 2.2.4)
PY_DEEPBET = _external("SAMSON_PY_DEEPBET", f"{CONDA}/deepbet_1_env/bin/python")  # deepbet 1.0.2
FSL_6074 = _external("SAMSON_FSL_6074", f"{CONDA}/fsl_6.0.7.4")                   # BET (fsl-bet2 2111.4)
FSL_60717 = _external("SAMSON_FSL_60717", f"{CONDA}/fsl_6.0.7.17")                # BET4Animal (fsl-bet2 2111.8)
ANTS_BIN = _external("SAMSON_ANTS_BIN", os.environ.get("ANTSPATH") or os.path.dirname(shutil.which("antsBrainExtraction.sh") or ""))   # ANTs 2.5.0
THIRD_PARTY = _external("SAMSON_THIRD_PARTY", pjoin(P, "third_party"))   # BEN and RBM repositories (not distributed)
BEN_REPO = _external("SAMSON_BEN_REPO", pjoin(THIRD_PARTY, "BEN"))               # github.com/yu02019/BEN, commit f23ed39
BEN_WEIGHT = "weight/Mouse_T2WI_7T_2022_07191047/"                                 # relative to BEN_REPO
RBM_MODEL = _external("SAMSON_RBM_MODEL", pjoin(THIRD_PARTY, "RodentMRISkullStripping", "rbm", "scripts", "rat_brain-2d_unet.hdf5"))   # commit dfa1a12
TEMPLATE = pjoin(DATASETS, "template", "anat_template_FP.nii.gz")                  # SAMson template (x10 header)
TEMPLATE_MASK = pjoin(DATASETS, "template", "anat_template_FP_mask.nii.gz")        # template > 0.5 (ANTs)

# ------------------------------------------------------------------ tools
TOOLS = {   # key: (display name, resource kind)
    "samson_full_auto":    ("SAMson Full-Auto", "gpu"),
    "samson_semi_auto":    ("SAMson Semi-Auto", "gpu"),
    "samson_oracle":       ("SAMson Oracle", "gpu"),
    "sam2_samson_boxes":   ("SAM2 (SAMson boxes)", "gpu"),
    "medsam_samson_boxes": ("MedSAM (SAMson boxes)", "gpu"),
    "bet":                 ("BET", "cpu"),
    "bet4animal":          ("BET4Animal", "cpu"),
    "deepbet":             ("deepbet", "gpu"),
    "ben":                 ("BEN", "cpu"),
    "rbm":                 ("RBM", "cpu"),
    "ants":                ("ANTs", "cpu"),
    "sherm":               ("SHERM", "external"),   # MATLAB, run on other computers (runners/sherm); masks read from outputs/masks/sherm
}
TOOL_ORDER = list(TOOLS)
EXTERNAL_TOOLS = [k for k, v in TOOLS.items() if v[1] == "external"]   # not run by run_all.py
DISPLAY = {k: v[0] for k, v in TOOLS.items()}

# Dataset-specific settings of five (tool, dataset-contrast) pairs, for which the common settings of
# docs/tool_parameters.json gave less accurate masks: per-dataset start centre -c and thresholds (BET, BET4Animal),
# BET4Animal on the real voxel size with its mouse scaling (-z 6), RBM without the reversal of axis 1. The published
# masks of these pairs (outputs/masks) were made with these settings.
DATASET_SPECIFIC_SETTINGS = {
    ("bet", "D1_T2w"): dict(options="-m -r 40 -c 55 42 7 -f 0.44 -g 0.027"),
    ("bet", "D2_T2w"): dict(options="-m -r 40 -c 126 173 10 -f 0.2 -g 0.07"),
    ("bet4animal", "D2_T2w"): dict(input="true", options="-m -z 6 -r 6 -c 126 173 10 -f 0.44 -g 0.07"),
    ("bet4animal", "D3_T2w"): dict(input="true", options="-m -z 6 -r 4 -c 87 87 23 -f 0.4 -g 0"),
    ("rbm", "D3_T2w"): dict(flip_axis1=False),
}

# Header orientation labels given to each tool (docs/tools_and_conventions.md, section 2): "stored" = the canonical affine
# of the dataset; "anatomical" = axis 1 labelled S and axis 2 labelled A (direction-cosine columns 1 and 2 swapped for
# the datasets labelled in the template convention: D1, D2, D5); "template" = the template's convention, axis 1 A and
# axis 2 S (columns swapped for D3, D4, D6, D7). Voxel data, voxel sizes, origin and axis 0 unchanged.
HEADER_LABELS = {"deepbet": "anatomical"}   # ANTs and all other tools: "stored"
EXPECTED_LABELS = {"D1": "template", "D2": "template", "D5": "template", "D3": "anatomical", "D4": "anatomical",
                   "D6": "anatomical", "D7": "anatomical"}   # stored labels of the canonical affines (checked before relabelling)
# Stored (x10) affine of the images as given to the tools, where it differs from the dataset header.
# D2: the tools received a header voxel size of 0.1302 x 0.1302 x 0.8 mm (x10: 1.302 x 1.302 x 8); stage_input gives
# them this affine, so that a re-run reproduces the published D2 masks. It matters only for the tools that use the voxel
# size (BET, BET4Animal, RBM, ANTs). SHERM (MATLAB runner) and every metric use the voxel size of the dataset files.
TOOL_INPUT_AFFINE_X10 = {"D2": ((-1.3020000457763672, 0.0, 0.0, 0.0), (0.0, 1.2968246936798096, 0.7125833034515381, 0.0), (0.0, -0.11597292870283127, 7.968201160430908, 0.0), (0.0, 0.0, 0.0, 1.0))}   # (float32 values)
REFERENCE_TOOLS = ("samson_semi_auto", "samson_oracle")   # use the reference mask (Semi-Auto: candidate selection; Oracle: boxes)
DC_ORDER = ["D1_T2w", "D2_T2w", "D3_T2w", "D4_T2w", "D4_T2starw", "D4_DWI", "D5_DWI", "D6_T2w", "D6_BOLD", "D7_T2w"]

# Base settings of every tool; `input` = header given to the tool: "x10" = the canonical image with the x10 header of the
# dataset, "true" = the same image with the real voxel size. tool_parameters() completes them from docs/tool_parameters.json.
BASE_PARAMETERS = {
    "samson_full_auto":    dict(input="x10", checkpoint=SAM_CHECKPOINT, model="vit_h"),
    "samson_semi_auto":    dict(input="x10", checkpoint=SAM_CHECKPOINT, model="vit_h"),
    "samson_oracle":       dict(input="x10", checkpoint=SAM_CHECKPOINT, model="vit_h"),
    "medsam_samson_boxes": dict(input="x10", checkpoint=MEDSAM_CHECKPOINT, model="vit_b"),
    "sam2_samson_boxes":   dict(input="x10", checkpoint=SAM2_CHECKPOINT, config=SAM2_CONFIG),
    "bet":                 dict(input="x10", options="-m -r 40 -f 0.2 -g 0.07", fsldir=FSL_6074),
    "bet4animal":          dict(input="x10", fsldir=FSL_60717),
    "deepbet":             dict(input="x10", threshold=0.95, n_dilate=0),
    "ben":                 dict(input="x10", weight=BEN_WEIGHT, check=None),
    "rbm":                 dict(input="x10", voxsize=1.0, model=RBM_MODEL),
    "ants":                dict(input="x10", template=TEMPLATE, template_mask=TEMPLATE_MASK, seed=0, threads=1, min_slices=8),
}


def _orientation_option_b(spec):
    """True if option B of the BEN/RBM orientation decision applies (`orientation_decision_ben_rbm` of
    tool_parameters.json): BEN and RBM receive the canonical image with axis 1 reversed (dorsal-up coronal slices)."""
    decision = spec.get("orientation_decision_ben_rbm") or {}
    text = str(decision.get("decision", decision.get("author_decision", ""))).strip().lower()
    return text.startswith("option b") or text in ("b", "option_b")


def tool_parameters():
    """
    Parameters of every tool: the single x10 input rule with the parameters of `docs/tool_parameters.json` (parsed
    explicitly below; the JSON is the documented record of the run). Model weights, FSL folders and the BEN/RBM files are
    located with the external-software settings above (the JSON keeps the paths of the publication run).
    """
    params = {k: dict(v, header_labels="stored") for k, v in BASE_PARAMETERS.items()}
    spec = json.load(open(pjoin(DOCS, "tool_parameters.json")))
    tools = spec["tools"]
    for k in params:                                    # single x10 input rule
        params[k]["input"] = "x10"
        params[k]["spec"] = tools.get(k, {})
    for k in ("samson_full_auto", "medsam_samson_boxes"):
        params[k].update(checkpoint=pjoin(CHECKPOINTS, os.path.basename(tools[k]["checkpoint"]["path"])), model=tools[k]["checkpoint"]["model"])
    for k in ("samson_semi_auto", "samson_oracle"):
        params[k].update(checkpoint=params["samson_full_auto"]["checkpoint"], model=params["samson_full_auto"]["model"])
    params["sam2_samson_boxes"].update(checkpoint=pjoin(CHECKPOINTS, os.path.basename(tools["sam2_samson_boxes"]["checkpoint"]["path"])),
                                       config=tools["sam2_samson_boxes"]["checkpoint"]["config"])
    params["bet"].update(options=" ".join(tools["bet"]["command"][3:]), fsldir=FSL_6074)
    params["bet4animal"].update(options=tools["bet4animal"]["command"][3], fsldir=FSL_60717)
    params["deepbet"].update(threshold=tools["deepbet"]["parameters"]["threshold"], n_dilate=tools["deepbet"]["parameters"]["n_dilate"])
    ben = tools["ben"]["command"]
    params["ben"].update(weight=ben[ben.index("-weight") + 1], check=tools["ben"]["parameters"].get("check_orientation"))
    params["rbm"].update(voxsize=float(tools["rbm"]["parameters"]["voxsize"]))
    ants = tools["ants"]
    params["ants"].update(seed=int(ants["command"][ants["command"].index("-u") + 1]), threads=int(ants["parameters"]["threads"]))
    flip = _orientation_option_b(spec)
    for k in ("ben", "rbm"):
        params[k]["flip_axis1"] = flip
    for k in params:   # header labels (per-tool setting `header_labels` of tool_parameters.json; default: HEADER_LABELS)
        params[k]["header_labels"] = tools.get(k, {}).get("header_labels") or HEADER_LABELS.get(k, "stored")
    return params
