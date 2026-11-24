"""
Tool runners. Every tool receives the same input (the canonical image of the scan with its x10 header; BET4Animal on D2
and D3 the real voxel size, config.DATASET_SPECIFIC_SETTINGS), and every output mask is mapped back to
the raw grid of the dataset image and saved at outputs/masks/<tool>/<Dk>/<image folder relative to <Dk>>/<image stem>_mask.nii.gz
(e.g. D1/anat/3_month/, D6/func/).

Masks that exist are not computed again (resumable). `run_batch(tool, scans, ctx)` processes the scans of one
dataset-contrast (or of several, pooled) with one tool on the allocation and records the wall time of every scan and of
the batch. Tool keys: config.TOOLS and the pseudo-tool
FULL_SEMI (SAMson Full-Auto and Semi-Auto in one embedding pass).
"""
import os
import csv
import json
import time
import threading
import subprocess
from dataclasses import dataclass, field
from os.path import join as pjoin

import numpy as np
import nibabel as nib

from . import config, datasets as ds, resources as rs
from . import samson_core as core

SAMSON_TOOLS = ("samson_full_auto", "samson_semi_auto", "samson_oracle", "medsam_samson_boxes")
FULL_SEMI = "samson_full_semi"   # one embedding pass for Full-Auto and Semi-Auto (resource protocol); writes both tools' masks
GROUPS = {FULL_SEMI: ("samson_full_auto", "samson_semi_auto")}


SELECTION_COLUMNS = ["dataset", "contrast", "subject", "session", "stem", "slice", "selected_candidate",
                     "jaccard_candidate_0", "jaccard_candidate_1", "jaccard_candidate_2"]


def members(tool):
    """Tool keys whose masks a (pseudo-)tool writes."""
    return GROUPS.get(tool, (tool,))


@dataclass
class Context:
    root: str                       # outputs/ and results/ are written under root
    work: str                       # staged inputs, tool working files and per-scan logs
    allocation: rs.Allocation
    params: dict = field(default_factory=dict)
    full_run: bool = False          # True only for the full run (allowed while <release>/.FULL_RUN_ACTIVE exists)
    template: str = None            # SAMson template (box shapes)
    verbose: bool = True

    def __post_init__(self):
        self.params = self.params or config.tool_parameters()
        if self.template is None:
            self.template = config.TEMPLATE
        self.runtime = rs.RuntimeLog(pjoin(self.root, "results", "runtimes.csv"), "new")   # column `inputs` of runtimes.csv
        self.lock = threading.Lock()

    def runtime_log(self, tool):
        return self.runtime

    def on_scan(self, row):   # replaced by run_batch while a batch runs: records a scan's run time as soon as it is done
        pass

    def stop(self):
        return (not self.full_run) and rs.full_run_active()

    def log(self, tool, scan):
        return pjoin(self.work, "logs", tool, scan.dataset, scan.rel_dir, scan.stem + ".log")

    def tool_dir(self, tool, scan):
        return pjoin(self.work, "tools", tool, scan.dataset, scan.rel_dir)

    def params_for(self, tool, scan=None):
        """Parameters of a tool for one scan: the tool's parameters, with the dataset-specific settings of five
        (tool, dataset-contrast) pairs (config.DATASET_SPECIFIC_SETTINGS)."""
        p = dict(self.params[tool])
        if scan is not None:
            p.update(config.DATASET_SPECIFIC_SETTINGS.get((tool, scan.dc), {}))
        return p

    def header(self, tool, scan=None):
        p = self.params_for(tool, scan)
        labels = {"stored": "", "anatomical": "_anat", "template": "_tmpl"}[p.get("header_labels", "stored")]
        return p["input"] + labels + ("_flip1" if p.get("flip_axis1") else "")

    def staged(self, tool, scan):
        return ds.stage_input(scan, pjoin(self.work, "inputs"), self.header(tool, scan))

    def gpu_name(self):
        if self.allocation.gpu is None:
            return ""
        try:
            name = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", str(self.allocation.gpu)],
                                  capture_output=True, text=True).stdout.strip()
        except Exception:
            name = ""
        return f"{self.allocation.gpu}: {name}"


def mask_path(ctx, tool, scan):
    return ds.output_mask_path(ctx.root, tool, scan)


def write_log(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(text.rstrip("\n") + "\n")


def convert_output(tool_mask_file, scan, out_path, reference_input, flip_axis1=False):
    """Canonical tool mask (on the staged input grid) -> raw grid of the dataset image, saved with its header."""
    image = nib.load(tool_mask_file)
    data = np.squeeze(np.asanyarray(image.dataobj)) > 0.5
    if flip_axis1:   # undo the axis-1 reversal of the input (option B for BEN/RBM)
        data = data[:, ::-1, :]
    expected = nib.load(reference_input).shape[:3]
    if data.shape != expected:
        raise ValueError(f"{tool_mask_file}: shape {data.shape}, expected {expected}")
    ds.save_mask(data, scan, out_path)


# ------------------------------------------------------------------ SAMson and SAM models (GPU, in this process)

_models = {}


def sam_model(ctx, tool):
    """SamPredictor of SAMson (ViT-H) or MedSAM (ViT-B) on the allocation's GPU (loaded once per batch)."""
    import torch
    from segment_anything import sam_model_registry, SamPredictor
    p = ctx.params[members(tool)[0]]
    device = f"cuda:0" if (ctx.allocation.gpu is not None and torch.cuda.is_available()) else "cpu"
    return SamPredictor(sam_model_registry[p["model"]](checkpoint=p["checkpoint"]).to(device=device))


def release_models():
    _models.clear()
    try:
        import torch, gc
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass


def set_template(ctx):
    core.relative_coordinates = core.template_relative_coordinates(ctx.template)   # template box shapes (template > 0.5)


def samson_boxes(scan, part, ctx):
    """SAMson's Full-Auto box prompts of the processed slices."""
    t0, t1 = scan.ap[2], scan.ap[3]
    median_radius, numpass = scan.otsu or ((2, 5) if part.shape[1] < 140 else (1, 3))
    return core.generate_boxes(part, median_radius, numpass, t0, t1)


def save_boxes(ctx, scan, boxes, z0):
    path = ds.boxes_path(ctx.root, scan)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, boxes)
    with open(ds.boxes_path(ctx.root, scan, ".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["slice", "x_left", "y_up", "x_right", "y_down"])
        for i, b in enumerate(boxes):
            w.writerow([z0 + i + 1, *[repr(float(v)) for v in b]])


def load_boxes(ctx, scan, part=None):
    path = ds.boxes_path(ctx.root, scan)
    if os.path.exists(path):
        return np.load(path)
    if part is None:
        raise FileNotFoundError(path)
    return samson_boxes(scan, part, ctx)


def stack_masks(masks, shape, z0):
    """Masks of the processed (rotated) slices -> canonical volume; other slices empty."""
    out = np.zeros(shape, dtype=np.uint8)
    out[:, :, z0:z0 + len(masks)] = np.stack([np.rot90(m, k=-1) for m in masks[:, 0]], axis=2)   # undo the slice rotation
    return out


def append_selection(ctx, scan, z0, best, metrics):
    path = pjoin(ctx.root, "outputs", "semi_auto_selection.csv")
    table = metrics.pivot(index="slice", columns="candidate", values="jaccard")
    with ctx.lock:
        new = not os.path.exists(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(SELECTION_COLUMNS)
            for i, b in enumerate(best):
                j = table.iloc[i].tolist()
                w.writerow([scan.dataset, scan.contrast, scan.subject, scan.session, scan.stem, z0 + i + 1, b, *j])


def samson_scan(tool, scan, ctx, model):
    """One scan with SAMson Full-Auto + Semi-Auto (one pass), Full-Auto, Semi-Auto, Oracle or MedSAM (SAMson boxes);
    returns seconds (processing of the scan, before the masks are saved)."""
    t = time.perf_counter()
    volume, _ = ds.load_canonical(scan)                     # float64, as SAMson's load_volume
    z0, z1, _, _ = scan.ap_slices(volume.shape[2])
    part = volume[:, :, z0:z1 + 1]
    if tool == FULL_SEMI:   # each slice embedded once for both modes
        boxes = samson_boxes(scan, part, ctx)
        full, candidates = core.segment_volume(part, [(boxes, False), (boxes, True)], model=model, show_progress=False)
        reference = ds.load_reference(scan)
        metrics = core.evaluate_slices(candidates, reference[:, :, z0:z1 + 1])
        best = core.find_best_mask_indexes(metrics)     # best of the 3 candidates according to the reference mask
        semi = candidates[np.arange(len(candidates)), best][:, None]
        seconds = time.perf_counter() - t
        save_boxes(ctx, scan, boxes, z0)
        append_selection(ctx, scan, z0, best, metrics)
        ds.save_mask(stack_masks(semi, volume.shape, z0), scan, mask_path(ctx, "samson_semi_auto", scan))
        ds.save_mask(stack_masks(full, volume.shape, z0), scan, mask_path(ctx, "samson_full_auto", scan))
        return seconds
    elif tool == "samson_full_auto":
        boxes = samson_boxes(scan, part, ctx)
        masks, = core.segment_volume(part, [(boxes, False)], model=model, show_progress=False)
        save_boxes(ctx, scan, boxes, z0)
    elif tool == "samson_semi_auto":
        boxes = samson_boxes(scan, part, ctx)
        candidates, = core.segment_volume(part, [(boxes, True)], model=model, show_progress=False)
        reference = ds.load_reference(scan)
        metrics = core.evaluate_slices(candidates, reference[:, :, z0:z1 + 1])
        best = core.find_best_mask_indexes(metrics)     # best of the 3 candidates according to the reference mask
        masks = candidates[np.arange(len(candidates)), best][:, None]
        append_selection(ctx, scan, z0, best, metrics)
    elif tool == "samson_oracle":
        reference = ds.load_reference(scan)
        boxes = core.bounding_boxes(reference[:, :, z0:z1 + 1].astype(float))   # box prompts of the reference mask
        masks, = core.segment_volume(part, [(boxes, False)], model=model, show_progress=False)
    elif tool == "medsam_samson_boxes":
        boxes = load_boxes(ctx, scan, part)
        masks, = core.segment_volume(part, [(boxes, False)], model=model, show_progress=False)
    else:
        raise ValueError(tool)
    out = stack_masks(masks, volume.shape, z0)
    seconds = time.perf_counter() - t
    ds.save_mask(out, scan, mask_path(ctx, tool, scan))
    return seconds


def run_samson_batch(tool, scans, ctx):
    set_template(ctx)
    t = time.perf_counter()
    model = sam_model(ctx, tool)
    load = time.perf_counter() - t
    rows = []
    for scan in scans:
        if ctx.stop():
            break
        start = rs.now()
        seconds = samson_scan(tool, scan, ctx, model)
        rows.append(dict(scan=scan, seconds=seconds, start=start, end=rs.now(), note=""))
        ctx.on_scan(rows[-1])
        write_log(ctx.log(tool, scan), f"{start} {tool} {scan.stem}: {seconds:.1f} s")
    del model
    release_models()
    return rows, f"model load {load:.1f} s"


# ------------------------------------------------------------------ subprocess GPU tools (deepbet, SAM2)

def _json_lines(text):
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def run_deepbet_batch(scans, ctx, tool="deepbet"):
    p = ctx.params[tool]
    jobs, by_mask = [], {}
    for scan in scans:
        folder = ctx.tool_dir(tool, scan)
        os.makedirs(folder, exist_ok=True)
        job = dict(input=ctx.staged(tool, scan), mask=pjoin(folder, scan.stem + "_mask.nii.gz"),
                   brain=pjoin(folder, scan.stem + "_brain.nii.gz"), tiv=pjoin(folder, scan.stem + "_tiv.csv"))
        jobs.append(job)
        by_mask[job["mask"]] = (scan, job)
    config_path = pjoin(ctx.work, "jobs", f"{tool}_{scans[0].dc}.json")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    json.dump(dict(threshold=p["threshold"], n_dilate=p["n_dilate"], no_gpu=ctx.allocation.gpu is None, jobs=jobs), open(config_path, "w"), indent=1)
    env = rs.tool_env(ctx.allocation.n, drop_mkl=True, gpu=ctx.allocation.gpu)
    log = pjoin(ctx.work, "logs", tool, f"{scans[0].dc}_batch_{time.strftime('%Y%m%d-%H%M%S')}.log")   # one log per execution
    code, start, end, seconds = rs.run_command([config.PY_DEEPBET, pjoin(config.RUNNERS, "deepbet_run.py"), config_path], env=env, log=log)
    rows = []
    text = open(log).read()
    for event in _json_lines(text):
        if event.get("event") == "job" and event["mask"] in by_mask:
            scan, job = by_mask[event["mask"]]
            note = ""
            try:
                convert_output(job["mask"], scan, mask_path(ctx, tool, scan), job["input"])
            except Exception as error:
                note = f"no usable output: {error}"
            rows.append(dict(scan=scan, seconds=event["seconds"], start=event["start"], end=event["end"], note=note))
            write_log(ctx.log(tool, scan), f"{event['start']} {tool} {scan.stem}: {event['seconds']:.1f} s {note}")
    load = next((e["seconds"] for e in _json_lines(text) if e.get("event") == "model_loaded"), float("nan"))
    return rows, f"exit code {code}; model load {load:.1f} s"


def run_sam2_batch(scans, ctx):
    p = ctx.params["sam2_samson_boxes"]
    jobs, by_output = [], {}
    for scan in scans:
        folder = ctx.tool_dir("sam2_samson_boxes", scan)
        os.makedirs(folder, exist_ok=True)
        staged = ctx.staged("sam2_samson_boxes", scan)
        n = nib.load(staged).shape[2]
        z0, z1, _, _ = scan.ap_slices(n)
        boxes = ds.boxes_path(ctx.root, scan)
        if not os.path.exists(boxes):   # Full-Auto boxes not saved yet: compute them as Full-Auto does
            set_template(ctx)
            volume, _ = ds.load_canonical(scan)
            save_boxes(ctx, scan, samson_boxes(scan, volume[:, :, z0:z1 + 1], ctx), z0)
        job = dict(input=staged, boxes=boxes, z0=z0, z1=z1, output=pjoin(folder, scan.stem + "_mask.nii.gz"))
        jobs.append(job)
        by_output[job["output"]] = (scan, job)
    config_path = pjoin(ctx.work, "jobs", f"sam2_{scans[0].dc}.json")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    json.dump(dict(checkpoint=p["checkpoint"], config=p["config"], jobs=jobs), open(config_path, "w"), indent=1)
    env = rs.tool_env(ctx.allocation.n, gpu=ctx.allocation.gpu)
    log = pjoin(ctx.work, "logs", "sam2_samson_boxes", f"{scans[0].dc}_batch_{time.strftime('%Y%m%d-%H%M%S')}.log")   # one log per execution
    code, start, end, seconds = rs.run_command([config.PY_SAM2, pjoin(config.RUNNERS, "sam2_run.py"), config_path], env=env, log=log)
    rows = []
    text = open(log).read()
    for event in _json_lines(text):
        if event.get("event") == "job" and event["output"] in by_output:
            scan, job = by_output[event["output"]]
            convert_output(job["output"], scan, mask_path(ctx, "sam2_samson_boxes", scan), job["input"])
            rows.append(dict(scan=scan, seconds=event["seconds"], start=event["start"], end=event["end"], note=""))
            write_log(ctx.log("sam2_samson_boxes", scan), f"{event['start']} sam2 {scan.stem}: {event['seconds']:.1f} s")
    load = next((e["seconds"] for e in _json_lines(text) if e.get("event") == "model_loaded"), float("nan"))
    return rows, f"exit code {code}; model load {load:.1f} s"


# ------------------------------------------------------------------ CPU tools (one pinned single-thread process per scan)

def fsl_env(fsldir):
    return rs.tool_env(1, extra=dict(FSLDIR=fsldir, FSLOUTPUTTYPE="NIFTI_GZ", PATH=f"{fsldir}/bin:/usr/bin:/bin"))


def ants_input(scan, staged, min_slices, folder):
    """x10 input of ANTs; images with fewer than `min_slices` slices are zero-padded along the slice axis (one empty
    slice at each end for D4's 6 slices, origin shifted accordingly)."""
    image = nib.load(staged)
    pad = max(0, -(-(min_slices - image.shape[2]) // 2))
    if not pad:
        return staged, 0
    padded = pjoin(folder, scan.stem + "_padded_input.nii.gz")
    if not os.path.exists(padded):
        shift = np.eye(4)
        shift[2, 3] = -pad
        affine = image.affine @ shift
        ds.write_nifti(np.pad(image.get_fdata(dtype=np.float32), ((0, 0), (0, 0), (pad, pad))), affine, padded)
    return padded, pad


def cpu_command(tool, scan, ctx):
    """(command, env, cwd, canonical output mask path, postprocess function or None) of a CPU tool for one scan."""
    p = ctx.params_for(tool, scan)
    folder = ctx.tool_dir(tool, scan)
    os.makedirs(folder, exist_ok=True)
    staged = ctx.staged(tool, scan)
    prefix = pjoin(folder, scan.stem)
    if tool == "bet":
        return (["bet", staged, prefix] + p["options"].split(), fsl_env(p["fsldir"]), None, prefix + "_mask.nii.gz", None)
    if tool == "bet4animal":
        return (["bet4animal", staged, prefix + ".nii.gz", p["options"]], fsl_env(p["fsldir"]), None, prefix + "_mask.nii.gz", None)
    if tool == "ben":
        folder_in, folder_out = pjoin(folder, scan.stem, "input"), pjoin(folder, scan.stem, "output")
        os.makedirs(folder_in, exist_ok=True)
        link = pjoin(folder_in, scan.stem + ".nii.gz")
        if not os.path.lexists(link):
            os.symlink(staged, link)
        cmd = [config.PY_BEN, "BEN_infer.py", "-i", folder_in, "-o", folder_out, "-weight", p["weight"]]
        if p.get("check"):
            cmd += ["-check", p["check"]]
        env = rs.tool_env(1, extra=dict(PYTHONDONTWRITEBYTECODE="1"))   # nothing written into the (read-only) BEN repository
        return (cmd, env, config.BEN_REPO, pjoin(folder_out, scan.stem + ".nii.gz"), None)
    if tool == "rbm":
        job = pjoin(ctx.work, "jobs", "rbm", scan.dataset, scan.rel_dir, scan.stem + ".json")
        os.makedirs(os.path.dirname(job), exist_ok=True)
        json.dump(dict(model_path=p["model"], jobs=[dict(input=staged, output=prefix + "_mask.nii.gz", voxsize=p["voxsize"])]), open(job, "w"), indent=1)
        env = rs.tool_env(1, extra=dict(PYTHONDONTWRITEBYTECODE="1"))
        return ([config.PY_BEN, pjoin(config.RUNNERS, "rbm_unet_run_padded.py"), job], env, None, prefix + "_mask.nii.gz", None)
    if tool == "ants":
        image, pad = ants_input(scan, staged, p["min_slices"], folder)
        output = prefix + ("_padded_" if pad else "_")
        env = rs.tool_env(p["threads"], extra=dict(ANTSPATH=config.ANTS_BIN + "/", PATH=f"{config.ANTS_BIN}:/usr/bin:/bin"))
        cmd = ["antsBrainExtraction.sh", "-d", "3", "-a", image, "-e", p["template"], "-m", p["template_mask"],
               "-u", str(p["seed"]), "-o", output]
        final = prefix + "_BrainExtractionMask.nii.gz"

        def crop():   # crop a padded mask back to the image (header of the staged input)
            if pad and os.path.exists(output + "BrainExtractionMask.nii.gz"):
                x10 = nib.load(staged)
                mask = np.asanyarray(nib.load(output + "BrainExtractionMask.nii.gz").dataobj)[:, :, pad:-pad]
                ds.write_nifti(mask.astype(np.uint8), x10.affine, final)
        return (cmd, env, None, final, crop)
    raise ValueError(tool)


ANTS_EXPECTED_HOURS = {"D6_T2w": 2.83, "D6_BOLD": 2.43, "D4_T2w": 2.21, "D4_T2starw": 2.21, "D4_DWI": 2.21, "D5_DWI": 2.08,
                       "D7_T2w": 2.01, "D1_T2w": 1.9, "D2_T2w": 1.9, "D3_T2w": 1.9}   # expected hours per scan (test runs), to order the pool


def run_cpu_batch(tool, scans, ctx):
    workers = min(ctx.allocation.n, len(scans))

    def one(scan, core):
        cmd, env, cwd, output, post = cpu_command(tool, scan, ctx)
        code, start, end, seconds = rs.run_command(cmd, env=env, cwd=cwd, log=ctx.log(tool, scan), core=core)
        note = f"exit code {code}" if code else ""
        try:
            if post:
                post()
            convert_output(output, scan, mask_path(ctx, tool, scan), ctx.staged(tool, scan), ctx.params_for(tool, scan).get("flip_axis1", False))
        except Exception as error:
            note = (note + "; " if note else "") + f"no usable output: {type(error).__name__}: {error}"
        row = dict(scan=scan, seconds=seconds, start=start, end=end, note=note)
        ctx.on_scan(row)   # recorded at once: a long pooled phase keeps its times if it is interrupted
        return row

    # longest runs first, so that the pool is not left waiting for them at the end: ANTs by the expected per-scan time of
    # each dataset-contrast (ANTS_EXPECTED_HOURS), then (and for the other tools) by image size
    expected = ANTS_EXPECTED_HOURS if tool == "ants" else {}
    order = sorted(scans, key=lambda s: (-expected.get(s.dc, 0), -int(np.prod(nib.load(s.image).shape[:3]))))
    rows = [r for r in rs.pinned_map(one, order, ctx.allocation.cores[:workers], stop=ctx.stop) if r is not None]
    return rows, f"{workers} single-thread workers"


# ------------------------------------------------------------------ batch driver

def _missing(ctx, tool, scan):
    return any(not os.path.exists(mask_path(ctx, t, scan)) for t in members(tool))


def run_batch(tool, scans, ctx):
    """Process the scans (one dataset-contrast, or several when pooled) with one tool (or the joint Full-Auto + Semi-Auto
    pass); returns a summary dict. Masks that exist are skipped. Every scan's time is appended to the run-time log as soon
    as the scan is done (subprocess runners: after their batch); the batch rows at the end."""
    todo = [s for s in scans if _missing(ctx, tool, s)]
    summary = dict(tool=tool, dc=scans[0].dc if scans else "", scans=len(scans), done_before=len(scans) - len(todo), ran=0)
    if not todo:
        return summary
    rs.check_compute_allowed(ctx.full_run)
    is_samson = tool in SAMSON_TOOLS or tool == FULL_SEMI
    for scan in todo:                       # inputs staged before the timing (re-orientation is not part of the tool)
        if not is_samson:
            ctx.staged(tool, scan)
    kind = config.TOOLS[members(tool)[0]][1]
    gpu = ctx.gpu_name() if kind == "gpu" else ""
    workers, threads = (1, ctx.allocation.n) if kind == "gpu" else (min(ctx.allocation.n, len(todo)), 1)
    shared = f"one pass for {' and '.join(config.DISPLAY[m] for m in members(tool))} (shared embedding): same time recorded for both" if tool in GROUPS else ""
    join = lambda *x: "; ".join(v for v in x if v)
    recorded = set()

    def record(r):
        s = r["scan"]
        with ctx.lock:
            if s.key in recorded:
                return
            recorded.add(s.key)
        for member in members(tool):
            ctx.runtime_log(member).add(tool=member, dataset=s.dataset, contrast=s.contrast, subject=s.subject, session=s.session,
                                        seconds=round(r["seconds"], 3), start=r["start"], end=r["end"], workers=workers,
                                        threads_per_worker=threads, gpu=gpu, level="scan", n_scans=1, note=join(r["note"], shared))

    ctx.on_scan = record
    rs.snapshot(ctx, tool, "start", todo)
    start, t = rs.now(), time.perf_counter()
    try:
        if is_samson:
            rows, note = run_samson_batch(tool, todo, ctx)
        elif tool == "deepbet":
            rows, note = run_deepbet_batch(todo, ctx, tool)
        elif tool == "sam2_samson_boxes":
            rows, note = run_sam2_batch(todo, ctx)
        else:
            rows, note = run_cpu_batch(tool, todo, ctx)
    finally:
        del ctx.on_scan   # back to the class default
    wall, end = time.perf_counter() - t, rs.now()
    for r in rows:   # scans of the subprocess runners (deepbet, SAM2): recorded after their batch
        record(r)
    rs.snapshot(ctx, tool, "end", todo)
    dcs = sorted({s.dc for s in todo})
    for member in members(tool):
        if len(dcs) == 1:   # one dataset-contrast alone on the allocation: batch wall time
            ctx.runtime_log(member).add(tool=member, dataset=todo[0].dataset, contrast=todo[0].contrast, seconds=round(wall, 3), start=start,
                                        end=end, workers=workers, threads_per_worker=threads, gpu=gpu, level="dataset", n_scans=len(rows),
                                        note=join(note, shared, f"cores {rs.cores_text(ctx.allocation.cores)}",
                                                  "interrupted" if len(rows) < len(todo) else ""))
        else:               # pooled run of several dataset-contrasts: span of each one's scans within the pool
            from datetime import datetime
            for dc in dcs:
                r = [x for x in rows if x["scan"].dc == dc]
                if not r:
                    continue
                first, last = min(x["start"] for x in r), max(x["end"] for x in r)
                span = (datetime.fromisoformat(last) - datetime.fromisoformat(first)).total_seconds()
                ctx.runtime_log(member).add(tool=member, dataset=r[0]["scan"].dataset, contrast=r[0]["scan"].contrast, seconds=span, start=first,
                                            end=last, workers=workers, threads_per_worker=threads, gpu=gpu, level="dataset", n_scans=len(r),
                                            note=join(note, f"pooled with {len(dcs)} dataset-contrasts (span of this one's scans; pool wall {wall:.0f} s)",
                                                      f"cores {rs.cores_text(ctx.allocation.cores)}"))
    failed = sum(any(not os.path.exists(mask_path(ctx, m, r["scan"])) for m in members(tool)) for r in rows)
    summary.update(ran=len(rows), failed=failed, wall_s=wall, note=note, start=start, end=end, workers=workers, threads=threads, gpu=gpu)
    return summary
