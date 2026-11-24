"""
SHERM records (`outputs/sherm/`). SHERM (MATLAB, `runners/sherm/run_sherm_samson.m`) was run outside `run_all.py`, on two
other computers: run 1 gave the masks of D1 and D3-D6, run 2 those of D2 and D7. The record
folder holds one JSON log per scan used (`logs/<Dk>_<stem>.json`, as written by the runner), the environment of each
run (`environment_run1.json`, `environment_run2.json`) and the scan-level run times (`sherm_runtimes.csv`, written by
`scan_runtimes`).
"""
import glob
import json
import os
from os.path import join as pjoin

import pandas as pd

COLUMNS = ["tool", "level", "dataset", "contrast", "subject", "session", "stem", "sherm_run", "run", "run_id", "status",
           "seconds", "cpu_seconds", "peak_memory_gb", "start", "end", "workers", "threads_per_worker", "gpu", "load_1min",
           "available_memory_gb", "voxel_mm", "animal", "isotropic", "matlab", "computer", "message"]


def environments(folder):
    """{run number: environment JSON} of the SHERM runs (environment_run<k>.json)."""
    return {int(os.path.basename(f)[len("environment_run"):-len(".json")]): json.load(open(f))
            for f in sorted(glob.glob(pjoin(folder, "environment_run*.json")))}


def scan_runtimes(folder, participants_tsv):
    """
    One row per scan of `participants.tsv` from the SHERM logs of `folder`/logs: wall time (`seconds`), CPU time, workers
    running at the same time, the run that produced the mask (`sherm_run` 1 or 2, from the run_id of the environment
    files; `run` = the runner's label of that run) and the computer (CPU model).
    """
    runs = {env["run_id"]: (k, env["machine"]["cpu_model"]) for k, env in environments(folder).items()}
    participants = pd.read_csv(participants_tsv, sep="\t", dtype=str, keep_default_na=False)
    rows = []
    for _, p in participants.iterrows():
        stem = os.path.basename(p.image)[:-len(".nii.gz")]
        log = json.load(open(pjoin(folder, "logs", f"{p.dataset}_{stem}.json")))
        assert log["level"] == "scan" and log["stem"] == stem, (p.dataset, stem)
        k, cpu = runs[log["run_id"]]
        rows.append({**log, "session": log.get("session") or "", "start": log.get("start_time"), "end": log.get("end_time"),
                     "sherm_run": k, "computer": cpu})
    return pd.DataFrame(rows).reindex(columns=COLUMNS)
