#!/usr/bin/env python
"""
Command-line driver of the brain extraction of all tools on all datasets (the masks of outputs/masks and outputs/boxes;
they are evaluated by `SAMson_experiments` and `SAMson_benchmark`). Run it with the `samson` environment (SAMson, MedSAM and this library); the
other tools run in their own environments (lib/config.py, notebooks/REPRODUCIBILITY.md).

Every tool runs alone on the allocation (default hardware threads 16-95 and GPU 0; docs/tools_and_conventions.md): tools one
after another in the notebook's order (SAMson Full-Auto and Semi-Auto in one embedding pass), and for each tool the
dataset-contrasts one after another, so that every (tool, dataset-contrast) batch has its own wall time; ANTs (hours
per scan) processes the scans of all dataset-contrasts in one pool of single-thread workers, longest first (`--pool`).
Masks that exist are skipped (resumable); every scan's wall time, every batch's and every tool phase's wall time are
appended to results/runtimes.csv, load snapshots to results/load_snapshots.jsonl.

Examples
  Full run (writes only outputs/, results/ and the work folder; masks that exist are skipped):
    touch ../.FULL_RUN_ACTIVE
    python run_all.py --cores 16-95 --gpu 0 --full-run --environment --evaluate --work ../work/full_run
  Test into another folder (1 scan per dataset-contrast, two tools, cores 56-79):
    python run_all.py --root ../work/test --cores 56-79 --limit 1 --tools samson_full_auto,bet --evaluate
"""
import os
import sys
import time
import json
import argparse

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.dirname(HERE), help="outputs/ and results/ are written here (default: the release folder)")
    ap.add_argument("--work", default=None, help="staged inputs, tool working files and per-scan logs (default: <root>/work/full_run)")
    ap.add_argument("--cores", default="16-95", help="hardware threads of the allocation")
    ap.add_argument("--gpu", default="0", help="GPU index ('none' for CPU only)")
    ap.add_argument("--tools", default="all", help="comma-separated tool keys (default: all, in the notebook's order)")
    ap.add_argument("--dcs", default="all", help="comma-separated dataset-contrast keys (default: all)")
    ap.add_argument("--limit", type=int, default=0, help="first N scans of every dataset-contrast (tests)")
    ap.add_argument("--stems", default="", help="comma-separated image stems to process (tests)")
    ap.add_argument("--pool", default="ants", help="CPU tools whose scans of all dataset-contrasts share one pool of workers: comma-separated tool keys, "
                    "'all' or 'dataset' (none). Default: ants (longest scans first; the others run one dataset-contrast at a time)")
    ap.add_argument("--separate-samson", action="store_true", help="Full-Auto and Semi-Auto in separate passes (default: one embedding pass for both)")
    ap.add_argument("--full-run", action="store_true", help="this is the full run (allowed while <release>/.FULL_RUN_ACTIVE exists)")
    ap.add_argument("--environment", action="store_true", help="write <root>/results/environment.json")
    ap.add_argument("--evaluate", action="store_true", help="compute <root>/results/{subjects,slices,boxes}.csv after the runs")
    ap.add_argument("--eval-workers", type=int, default=0, help="processes for the evaluation (default: allocation size, max 32)")
    ap.add_argument("--dry-run", action="store_true", help="list the work to do and exit")
    args = ap.parse_args()

    gpu = None if args.gpu.lower() == "none" else int(args.gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if gpu is None else str(gpu)   # before torch is imported
    from lib import config, datasets as ds, resources as rs, tools as tl, evaluation as ev

    allocation = rs.Allocation(args.cores, gpu)
    rs.check_compute_allowed(args.full_run)
    rs.set_process_allocation(allocation)
    root = os.path.abspath(args.root)
    work = os.path.abspath(args.work or os.path.join(root, "work", "full_run"))
    default = [t for t in config.TOOL_ORDER if t not in config.EXTERNAL_TOOLS]   # every tool except SHERM (MATLAB), ANTs (the long phase) last
    tools = default if args.tools == "all" else args.tools.split(",")
    unknown = [t for t in tools if t not in config.TOOLS]
    assert not unknown, f"unknown tools {unknown}"
    external = [t for t in tools if t in config.EXTERNAL_TOOLS]
    assert not external, f"{external} cannot be run here (MATLAB, another computer): see runners/sherm/README.md"
    dcs = None if args.dcs == "all" else args.dcs.split(",")
    scans = ds.list_scans(dcs)
    if args.stems:
        keep = set(args.stems.split(","))
        scans = [s for s in scans if s.stem in keep]
    if args.limit:
        per = {}
        scans = [s for s in scans if per.setdefault(s.dc, []).append(s) or len(per[s.dc]) <= args.limit]
    by_dc = {}
    for s in scans:
        by_dc.setdefault(s.dc, []).append(s)
    print(f"{time.strftime('%F %T')} root={root} work={work} {allocation}")
    print("scans per dataset-contrast:", {k: len(v) for k, v in by_dc.items()}, flush=True)

    params = config.tool_parameters()
    ctx = tl.Context(root=root, work=work, allocation=allocation, params=params, full_run=args.full_run)
    relabelled = sorted({ctx.params[t]["header_labels"] for t in tools if t in ctx.params} - {"stored"})
    if relabelled:   # header labels: stop if any scan deviates from the expected convention of its dataset
        labels = ds.check_header_labels(scans)
        print("header labels checked:", labels.groupby(["dc", "labels"]).size().to_dict(), flush=True)
    if args.dry_run:
        for tool in tools:
            todo = sum(not os.path.exists(tl.mask_path(ctx, tool, s)) for s in scans)
            print(f"{tool:<22} {todo} masks to compute")
        return
    if args.environment:
        os.makedirs(os.path.join(root, "results"), exist_ok=True)
        info = rs.environment_info(allocation)
        info["tool_parameters"] = {k: {kk: vv for kk, vv in v.items() if kk != "spec"} for k, v in ctx.params.items()}
        info["inputs"] = "new"
        json.dump(info, open(os.path.join(root, "results", "environment.json"), "w"), indent=1, default=str)
        print("environment.json written", flush=True)

    # Full-Auto and Semi-Auto in one embedding pass when both are requested (resource protocol)
    phases = []
    for tool in tools:
        if tool in tl.GROUPS[tl.FULL_SEMI] and all(t in tools for t in tl.GROUPS[tl.FULL_SEMI]) and not args.separate_samson:
            if tl.FULL_SEMI not in phases:
                phases.append(tl.FULL_SEMI)
        else:
            phases.append(tool)
    pooled = config.TOOL_ORDER if args.pool == "all" else ([] if args.pool == "dataset" else args.pool.split(","))
    for tool in phases:
        kind = config.TOOLS[tl.members(tool)[0]][1]
        groups = [scans] if (tool in pooled and kind == "cpu") else [by_dc[k] for k in config.DC_ORDER if k in by_dc]
        phase_start, phase_t, ran = rs.now(), time.time(), 0
        for group in groups:
            if not group:
                continue
            summary = tl.run_batch(tool, group, ctx)
            ran += summary["ran"]
            if not summary["scans"]:
                continue
            label = summary["dc"] if len({s.dc for s in group}) == 1 else "pooled"
            print(f"{time.strftime('%F %T')} {tool:<20} {label:<11} "
                  f"{summary['scans']} scans, {summary['done_before']} done before, {summary['ran']} run"
                  + (f", {summary.get('failed', 0)} without usable output, wall {summary['wall_s'] / 60:.1f} min ({summary['note']})" if summary['ran'] else ""),
                  flush=True)
            if ctx.stop():
                print(".FULL_RUN_ACTIVE appeared: test compute stopped", flush=True)
                return
        if ran:   # wall time of the tool's whole phase (all dataset-contrasts of this execution)
            for member in tl.members(tool):
                ctx.runtime_log(member).add(tool=member, dataset="all", contrast="all", seconds=round(time.time() - phase_t, 3), start=phase_start,
                                end=rs.now(), level="tool", n_scans=ran,
                                note=("one pass with Semi-Auto; " if tool in tl.GROUPS else "") + f"phase of this execution; cores {args.cores}, gpu {args.gpu}")

    if args.evaluate:
        workers = args.eval_workers or min(allocation.n, 32)
        t = time.time()
        ev.finalize_selection(root)
        subjects, slices, boxes = ev.evaluate_all(scans, config.TOOL_ORDER, root, workers=workers)
        print(f"{time.strftime('%F %T')} evaluation: {len(subjects)} scan rows, {len(slices)} slice rows, {len(boxes)} box rows "
              f"({time.time() - t:.0f} s)", flush=True)
        print(subjects.groupby(["tool", "dataset", "contrast"]).jaccard.mean().unstack([1, 2]).round(3).to_string())


if __name__ == "__main__":
    main()
