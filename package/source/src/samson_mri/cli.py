"""
Command line interface: ``samson <command> ...`` (run ``samson --help`` or ``samson <command> --help``).

Commands
--------
segment         segment images (files or folders) and save the masks
preview         save a figure to check the anteroposterior range (subject and template side views; no SAM needed)
app             open the interactive app in the web browser
spec            write or show a dataset specification file (spec.json)
suggest-range   rank template ranges from a few reference masks
select          choose the Semi-Auto mask of every slice after a run
evaluate        compare masks with reference masks (Jaccard, Dice, accuracy)
download-model  download SAM's checkpoint (once)
example         run SAMson on the example subject bundled with the package
info            versions, device, checkpoint and template
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time

from ._version import __version__
from .spec import PRESETS, DatasetSpec, Orientation, SpecError

# ----------------------------------------------------------------------------------------------------------------------
# Shared options
# ----------------------------------------------------------------------------------------------------------------------

def add_spec_options(parser: argparse.ArgumentParser, required: bool = False) -> None:
    group = parser.add_argument_group("dataset specification (a --spec file, a --preset, and/or the options below, which override them)")
    group.add_argument("--spec", metavar="FILE", help="dataset specification file (JSON), e.g. written by the app or `samson spec`")
    group.add_argument("--preset", choices=sorted(PRESETS), help="specification of one of the article's datasets")
    group.add_argument("--template", nargs=2, type=int, metavar=("FIRST", "LAST"),
                       help="template slices (0-39, caudal to rostral) matching the first and last processed slice")
    group.add_argument("--slices", nargs=2, type=int, metavar=("FIRST", "LAST"),
                       help="first and last slice to process (0-based, SAMson orientation, -1 = last); default: all")
    group.add_argument("--orientation", metavar="CODES",
                       help="anatomical direction of the input axes, e.g. RIA (axis 0 right, 1 ventral, 2 rostral); see the docs")
    group.add_argument("--axes", nargs=3, type=int, metavar=("LR", "VD", "CR"),
                       help="input axis that becomes SAMson's left-right, ventral-dorsal and caudal-rostral axis "
                            "(keeps the flips of the --spec/--preset; not with --orientation)")
    group.add_argument("--flip-ap", action="store_true", help="reverse the anteroposterior axis: the input slices go from rostral to "
                       "caudal (the notebook's reorient=True). Sets the flip (it does not toggle a flip of the --spec file)")
    group.add_argument("--flip-dv", action="store_true", help="reverse the dorsoventral axis: the brain is upside down in the coronal "
                       "view. Sets the flip")
    group.add_argument("--otsu", nargs="+", metavar="V", help="'auto' (default) or median_radius numpass, e.g. --otsu 2 5")
    group.add_argument("--volume", type=int, help="volume of 4D images to use (default 0, the first)")


def spec_from_args(args) -> DatasetSpec:
    """Dataset specification from --spec/--preset and the command line options."""
    base = None
    if args.spec:
        base = DatasetSpec.load(args.spec)
    elif args.preset:
        base = PRESETS[args.preset]
    orientation = None
    if args.orientation and args.axes:
        raise SpecError("give the orientation either as --orientation letters or as --axes (with --flip-ap/--flip-dv), not both.")
    if args.orientation or args.axes or args.flip_ap or args.flip_dv:
        start = base.orientation if base else Orientation()
        if args.orientation:                      # the letters give the axes and the flips
            start = Orientation.from_codes(args.orientation)
        elif args.axes:                           # new axes, flips of the spec file kept
            start = Orientation(tuple(args.axes), start.flip)
        flip = list(start.flip)
        if args.flip_dv:
            flip[1] = True
        if args.flip_ap:
            flip[2] = True
        orientation = Orientation(start.axes, tuple(flip))
    otsu = None
    if args.otsu:
        otsu = "auto" if args.otsu == ["auto"] else [int(v) for v in args.otsu]
    if base is None:
        if not args.template:
            raise SpecError("the template range is required: give --template FIRST LAST, a --spec file or a --preset. "
                            "Use `samson preview` or `samson app` to choose it.")
        return DatasetSpec(template=tuple(args.template), slices=tuple(args.slices) if args.slices else (0, -1),
                           orientation=orientation or Orientation(), otsu=otsu or "auto",
                           volume=args.volume if args.volume is not None else 0)
    return base.replace(template=tuple(args.template) if args.template else None, slices=tuple(args.slices) if args.slices else None,
                        orientation=orientation, otsu=otsu, volume=args.volume)


def add_model_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("model")
    group.add_argument("--checkpoint", metavar="FILE", help="SAM checkpoint (default: $SAMSON_CHECKPOINT or the downloaded one)")
    group.add_argument("--model-type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"], help="SAM model (default vit_h, as in the article)")
    group.add_argument("--device", default="auto", help="auto (default), cuda, cuda:1, cpu or mps")


def parse_choices(text):
    """Semi-Auto choices: 'F' (or -1) = Full-Auto mask, 0-2 = candidate; one value for every slice or a comma-separated list."""
    if text is None:
        return None
    values = []
    for token in text.replace(" ", "").split(","):
        if token.upper() in ("F", "FULL", "-1"):
            values.append(-1)
        elif token in ("0", "1", "2"):
            values.append(int(token))
        else:
            raise ValueError(f"invalid choice {token!r}: use F (Full-Auto mask) or 0, 1, 2 (candidates), separated by commas.")
    return values[0] if len(values) == 1 else values


def progress_bar(i, n, stem, done, total):
    width = 24
    filled = int(width * done / total)
    sys.stdout.write(f"\r  [{i + 1}/{n}] {stem[:40]:<40} [{'#' * filled}{'.' * (width - filled)}] {done}/{total} slices")
    sys.stdout.flush()
    if done == total:
        sys.stdout.write("\r" + " " * 100 + "\r")


# ----------------------------------------------------------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------------------------------------------------------

def cmd_segment(args) -> int:
    from .io import find_images, subject_stem
    from .pipeline import SAMson

    spec = spec_from_args(args)
    images = find_images(args.inputs, pattern=args.pattern)
    if not images:
        print("No NIfTI images found.", file=sys.stderr)
        return 1
    references = None
    if args.reference:
        if len(images) != 1:
            print("--reference is for one image; use --reference-dir (and --reference-pattern) for several.", file=sys.stderr)
            return 1
        references = {images[0]: args.reference}
    elif args.reference_dir:
        if os.path.abspath(args.reference_dir) == os.path.abspath(args.output) and args.reference_pattern.format(stem="x") == "x_mask.nii.gz":
            print("The reference masks would have the names of SAMson's output masks (<image>_mask.nii.gz) in the output folder: "
                  "use another output folder (or rename the reference masks and give --reference-pattern).", file=sys.stderr)
            return 1

        def references(path):
            candidate = os.path.join(args.reference_dir, args.reference_pattern.format(stem=subject_stem(path)))
            if not os.path.exists(candidate):
                raise FileNotFoundError(f"reference mask {candidate} not found")
            return candidate
    choices = parse_choices(args.choices)
    print(f"SAMson {__version__}: {len(images)} image(s), mode {args.mode}, template range {spec.template[0]}-{spec.template[1]}, "
          f"slices {spec.slices[0]}-{spec.slices[1]}, orientation: {spec.orientation.describe()}")
    os.makedirs(args.output, exist_ok=True)
    spec.save(os.path.join(args.output, "samson_spec.json"))
    samson = SAMson(model_type=args.model_type, checkpoint=args.checkpoint, device=args.device)
    start = time.time()
    rows = samson.segment_files(images, spec, args.output, mode=args.mode, references=references, choices=choices,
                                qc=not args.no_qc, save_brain=True if args.save_brain else None, overwrite=args.overwrite,
                                progress=progress_bar if sys.stdout.isatty() else None)
    errors = [r for r in rows if r["status"].startswith("error")]
    print(f"Finished in {(time.time() - start) / 60:.1f} min: {sum(r['status'] == 'ok' for r in rows)} segmented, "
          f"{sum(r['status'].startswith('skipped') for r in rows)} skipped, {len(errors)} errors. Outputs in {os.path.abspath(args.output)}")
    return 1 if errors else 0


def cmd_preview(args) -> int:
    from .io import load_subject
    from .pipeline import compute_boxes
    from .viz import ap_figure, save_figure

    spec = spec_from_args(args)
    subject = load_subject(args.input, spec)
    first, last = spec.slice_range(subject.n_slices)
    prompts = None if args.no_boxes else compute_boxes(subject, spec)
    fig = ap_figure(subject.volume, subject.voxel_size, spec.template[0], spec.template[1], first, last, x=args.x,
                    prompts=prompts, header_scale=args.header_scale, title=subject.stem)
    output = args.output or f"{subject.stem}_ap_preview.png"
    save_figure(fig, output)
    print(f"Saved {os.path.abspath(output)}")
    if prompts is not None:
        print(f"Anchor slice {first + prompts.anchor}; median_otsu {prompts.otsu}; central box {prompts.central_box}")
    from .ap import check_range
    for warning in check_range(last - first + 1, *spec.template, subject.voxel_size, args.header_scale):
        print("WARNING:", warning["detail"])
    return 0


def cmd_app(args) -> int:
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print("The app needs Streamlit: pip install \"samson-mri[app]\"", file=sys.stderr)
        return 1
    app = os.path.join(os.path.dirname(__file__), "app", "streamlit_app.py")
    command = [sys.executable, "-m", "streamlit", "run", app, "--server.port", str(args.port),
               "--server.headless", "true", "--browser.gatherUsageStats", "false", "--client.toolbarMode", "minimal",
               "--runner.magicEnabled", "false"]
    command += ["--server.address", args.address or "localhost"]   # not reachable from other computers unless --address is given
    command += ["--"] + (["--folder", args.folder] if args.folder else [])
    print(f"SAMson app: open http://localhost:{args.port} in your web browser (Ctrl+C here to stop it).\n"
          f"On a remote server, first run on your computer: ssh -L {args.port}:localhost:{args.port} <user>@<server>")
    return subprocess.call(command)


def cmd_spec(args) -> int:
    if args.show:
        spec = DatasetSpec.load(args.show)
        print(json.dumps(spec.to_dict(), indent=2))
        print(f"Orientation: {spec.orientation.describe()}")
        return 0
    spec = spec_from_args(args)
    if args.name:
        spec = spec.replace(name=args.name)
    spec.save(args.output)
    print(f"Saved {os.path.abspath(args.output)}:\n{json.dumps(spec.to_dict(), indent=2)}")
    return 0


def cmd_suggest_range(args) -> int:
    from .ap import suggest_from_masks
    from .io import load_mask
    if not args.template:
        args.template = [0, 39]   # Not used for the suggestion (only orientation and slices are)
    spec = spec_from_args(args)
    masks = [spec.orientation.apply(load_mask(path)) for path in args.masks]
    n = masks[0].shape[2]
    if any(m.shape[2] != n for m in masks):
        print("All reference masks need the same number of slices.", file=sys.stderr)
        return 1
    first, last = spec.slice_range(n)
    table = suggest_from_masks(masks, first, last, top=args.top, slice_mm=args.slice_mm)
    print(f"Template ranges for slices {first}-{last}, ranked by the agreement (box IoU) between SAMson's template boxes and the "
          f"boxes of {len(masks)} reference mask(s):")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    best = table.iloc[0]
    print(f"\nSuggested: --template {int(best.template_first)} {int(best.template_last)}  (check it with `samson preview`)")
    return 0


def cmd_select(args) -> int:
    from .pipeline import apply_selection
    choices = parse_choices(args.choices)
    written = apply_selection(args.sidecar, choices=choices, reference=args.reference, qc=not args.no_qc)
    print("Written:", *written.values(), sep="\n  ")
    return 0


def cmd_evaluate(args) -> int:
    from .io import load_mask
    from .metrics import compare_volumes
    if len(args.masks) != len(args.references):
        print("Give one reference per mask.", file=sys.stderr)
        return 1
    ap_axis = DatasetSpec.load(args.spec).orientation.axes[2] if args.spec else args.ap_axis
    rows = []
    for mask, reference in zip(args.masks, args.references):
        _, summary = compare_volumes(load_mask(mask), load_mask(reference), ap_axis=ap_axis)
        rows.append(dict(mask=mask, reference=reference, **summary))
        print(f"{os.path.basename(mask)}: Jaccard {summary['jaccard']:.4f}, Dice {summary['dice']:.4f}, accuracy {summary['accuracy']:.2f} % "
              f"(mean of {summary['n_slices']} slices); 3D Jaccard {summary['jaccard_3d']:.4f}")
    if args.output:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {os.path.abspath(args.output)}")
    return 0


def cmd_download_model(args) -> int:
    from .model import download_checkpoint
    download_checkpoint(args.model_type, args.folder)
    return 0


def cmd_example(args) -> int:
    from .example import run_example
    return run_example(args.output, device=args.device, checkpoint=args.checkpoint)


def cmd_info(args) -> int:
    from .model import CHECKPOINTS, CheckpointNotFound, cache_dir, find_checkpoint
    from .template import default_template_path, load_template
    print(f"samson-mri {__version__}  (Python {sys.version.split()[0]})")
    for module in ("numpy", "torch", "torchvision", "segment_anything", "dipy", "nibabel", "skimage", "cv2", "streamlit"):
        try:
            mod = __import__(module)
            print(f"  {module:<17} {getattr(mod, '__version__', 'installed')}")
        except ImportError:
            print(f"  {module:<17} not installed")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"Device: cuda ({torch.cuda.get_device_name(0)})")
        elif shutil.which("nvidia-smi"):
            print(f"Device: cpu. An NVIDIA GPU seems to be present, but this PyTorch build ({torch.__version__}) cannot use it, often "
                  "because the GPU driver is older than the CUDA version of PyTorch. Install a PyTorch build for your driver (see "
                  "https://pytorch.org/get-started/locally, e.g. pip install torch torchvision --index-url "
                  "https://download.pytorch.org/whl/cu121) or the tested versions (pip install -r requirements-lock.txt).")
        else:
            print("Device: cpu (no GPU found: SAMson works, but slowly)")
    except Exception as error:   # pragma: no cover
        print(f"Device: unknown ({error})")
    template = load_template()
    print(f"Template: {default_template_path()} ({template.n_slices} slices)")
    print(f"Checkpoint folder: {cache_dir()}")
    for model_type in CHECKPOINTS:
        try:
            print(f"  {model_type}: {find_checkpoint(model_type)}")
        except CheckpointNotFound:
            print(f"  {model_type}: not downloaded" + ("  ->  samson download-model" if model_type == "vit_h" else ""))
    return 0


# ----------------------------------------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="samson", description="SAMson: brain extraction of rodent MRI with the Segment Anything Model.",
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog="Run `samson <command> --help` for the options of a command. Documentation: README.md")
    parser.add_argument("--version", action="version", version=f"samson-mri {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    p = sub.add_parser("segment", help="segment images and save the masks", description="Segment NIfTI images with SAMson.")
    p.add_argument("inputs", nargs="+", help="NIfTI files, folders or glob patterns")
    p.add_argument("-o", "--output", required=True, help="output folder")
    p.add_argument("--pattern", default="*", help="file name pattern for folders (default: all .nii/.nii.gz), e.g. '*T2w*'")
    p.add_argument("--mode", default="full-auto", choices=["full-auto", "semi-auto"],
                   help="full-auto (default) or semi-auto (also saves SAM's 3 candidates per slice to choose from)")
    p.add_argument("--reference", help="reference mask of the (single) input image: Semi-Auto selection and evaluation")
    p.add_argument("--reference-dir", help="folder with reference masks of the inputs")
    p.add_argument("--reference-pattern", default="{stem}_mask.nii.gz",
                   help="reference file name in --reference-dir; {stem} = image name without extension (default '{stem}_mask.nii.gz')")
    p.add_argument("--choices", help="Semi-Auto selection: one value for all slices or a comma-separated list, one per processed slice "
                   "(F = Full-Auto mask, 0-2 = candidate), e.g. F,F,2,1,F")
    p.add_argument("--no-qc", action="store_true", help="do not save the quality-control figures")
    p.add_argument("--save-brain", action="store_true", help="also save the brain-extracted images (<image>_brain.nii.gz)")
    p.add_argument("--overwrite", action="store_true", help="redo images whose mask exists")
    add_spec_options(p)
    add_model_options(p)
    p.set_defaults(func=cmd_segment)

    p = sub.add_parser("preview", help="figure to check the anteroposterior range (no SAM needed)")
    p.add_argument("input", help="a representative NIfTI image of the dataset")
    p.add_argument("-o", "--output", help="figure file (default: <image>_ap_preview.png)")
    p.add_argument("--x", type=int, help="left-right index of the side view (default: middle)")
    p.add_argument("--no-boxes", action="store_true", help="do not compute the box prompts (faster)")
    p.add_argument("--header-scale", type=float, default=1.0, help="factor from header voxel sizes to mm (0.1 for x10 headers)")
    add_spec_options(p)
    p.set_defaults(func=cmd_preview)

    p = sub.add_parser("app", help="open the interactive app in the web browser")
    p.add_argument("--port", type=int, default=8501)
    p.add_argument("--address", help="network address to listen on (default: localhost, i.e. only this computer; the app can read and "
                   "write files on this computer, so do not open it to a network you do not trust)")
    p.add_argument("--folder", help="folder with the images to show first")
    p.set_defaults(func=cmd_app)

    p = sub.add_parser("spec", help="write (or --show) a dataset specification file")
    p.add_argument("-o", "--output", default="samson_spec.json", help="file to write (default samson_spec.json)")
    p.add_argument("--name", help="a name for the dataset")
    p.add_argument("--show", metavar="FILE", help="show a specification file instead of writing one")
    add_spec_options(p)
    p.set_defaults(func=cmd_spec)

    p = sub.add_parser("suggest-range", help="rank template ranges from reference masks")
    p.add_argument("masks", nargs="+", help="reference masks of a few subjects of the dataset")
    p.add_argument("--slice-mm", type=float, help="true slice spacing in mm (leaves out implausible range lengths)")
    p.add_argument("--top", type=int, default=10)
    add_spec_options(p)
    p.set_defaults(func=cmd_suggest_range)

    p = sub.add_parser("select", help="choose the Semi-Auto mask of every slice after a run")
    p.add_argument("sidecar", help="the <image>_samson.json file written by `samson segment --mode semi-auto`")
    p.add_argument("--choices", help="one value or a comma-separated list, one per processed slice (F = Full-Auto mask, 0-2 = candidate), "
                   "e.g. F,F,2,1,F")
    p.add_argument("--reference", help="reference mask: the best candidate of every slice (the article's Semi-Auto)")
    p.add_argument("--no-qc", action="store_true")
    p.set_defaults(func=cmd_select)

    p = sub.add_parser("evaluate", help="compare masks with reference masks")
    p.add_argument("--masks", nargs="+", required=True)
    p.add_argument("--references", nargs="+", required=True)
    p.add_argument("--ap-axis", type=int, default=2, help="input axis along which the coronal slices are stacked (default 2)")
    p.add_argument("--spec", help="take the anteroposterior axis from a specification file")
    p.add_argument("-o", "--output", help="CSV file with the results")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("download-model", help="download SAM's checkpoint (2.4 GB for vit_h)")
    p.add_argument("--model-type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    p.add_argument("--folder", help="destination folder (default: the SAMson cache folder)")
    p.set_defaults(func=cmd_download_model)

    p = sub.add_parser("example", help="segment the example subject bundled with the package")
    p.add_argument("-o", "--output", default="samson_example", help="output folder (default ./samson_example)")
    p.add_argument("--device", default="auto")
    p.add_argument("--checkpoint")
    p.set_defaults(func=cmd_example)

    p = sub.add_parser("info", help="versions, device, checkpoint and template")
    p.set_defaults(func=cmd_info)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    given, argv = list(sys.argv[1:] if argv is None else argv), []
    while given:   # "--choices -1,..." would be read as an option by argparse: pass it as "--choices=-1,..."
        item = given.pop(0)
        if item == "--choices" and given and given[0].startswith("-"):
            item = f"--choices={given.pop(0)}"
        argv.append(item)
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    try:
        return args.func(args) or 0
    except (SpecError, FileNotFoundError, ValueError) as error:
        print(f"samson {args.command}: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":   # pragma: no cover
    sys.exit(main())
