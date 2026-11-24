"""
Computational resources: core allocation, pinned single-thread workers for CPU tools, run-time records
(`results/runtimes.csv`) and the characterisation of hardware and software (`results/environment.json`).

Protocol (author): every tool runs alone on the same allocation (by default hardware threads 16-95 = 80 threads, and
GPU 0). CPU tools run as parallel single-thread workers, each pinned to one hardware thread of the allocation (one scan
per process). GPU tools run sequentially on the GPU, in one process pinned to the whole allocation. Every scan's wall
time is recorded, and every (tool, dataset-contrast) batch's total wall time.
"""
import os
import re
import csv
import json
import time
import queue
import socket
import platform
import threading
import subprocess
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from os.path import join as pjoin

from . import config

RUNTIME_COLUMNS = ["tool", "dataset", "contrast", "subject", "session", "seconds", "start", "end", "workers",
                   "threads_per_worker", "gpu", "host", "level", "n_scans", "inputs", "note"]
HOST_LABEL = "lab-server"   # neutral label of the machine (resource protocol: not the server name)
THREAD_VARIABLES = ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"]


def parse_cores(spec):
    """'16-95' or '56-79,90' -> sorted list of hardware-thread indices."""
    cores = set()
    for part in str(spec).split(","):
        a, _, b = part.partition("-")
        cores.update(range(int(a), int(b or a) + 1))
    return sorted(cores)


def cores_text(cores):
    cores, out, start = sorted(cores), [], None
    for i, c in enumerate(cores):
        if start is None:
            start = c
        if i + 1 == len(cores) or cores[i + 1] != c + 1:
            out.append(f"{start}-{c}" if c != start else str(c))
            start = None
    return ",".join(out)


class Allocation:
    def __init__(self, cores="16-95", gpu=0):
        self.cores = parse_cores(cores) if isinstance(cores, str) else sorted(cores)
        self.gpu = gpu   # index or None

    @property
    def n(self):
        return len(self.cores)

    def __repr__(self):
        return f"Allocation(cores={cores_text(self.cores)} [{self.n} threads], gpu={self.gpu})"


def full_run_active():
    return os.path.exists(config.FULL_RUN_FLAG)


def check_compute_allowed(full_run=False):
    """Tests must stop when the full run is active (it owns cores 16-95 and the GPU)."""
    if full_run_active() and not full_run:
        raise RuntimeError(f"{config.FULL_RUN_FLAG} exists: the full run owns the compute resources; test compute is not allowed")


def tool_env(threads, extra=None, drop_mkl=False, gpu=None, base=None):
    """Environment for a tool process: thread variables set to `threads`; GPU visibility; no MKL_* variables if drop_mkl
    (they make PyTorch abort in deepbet's environment)."""
    env = dict(os.environ if base is None else base)
    if drop_mkl:
        env = {k: v for k, v in env.items() if not k.startswith("MKL_")}
    for var in THREAD_VARIABLES:
        if drop_mkl and var.startswith("MKL_"):
            continue
        env[var] = str(threads)
    env["CUDA_VISIBLE_DEVICES"] = "" if gpu is None else str(gpu)
    env.update(extra or {})
    return env


def set_process_allocation(allocation, threads=None):
    """Pin the current process (and its future children) to the allocation; set its thread counts."""
    os.sched_setaffinity(0, allocation.cores)
    threads = threads or allocation.n
    for var in THREAD_VARIABLES:
        os.environ[var] = str(threads)
    try:
        import torch
        torch.set_num_threads(threads)
    except Exception:
        pass
    try:
        import cv2
        cv2.setNumThreads(threads)
    except Exception:
        pass


def now():
    return datetime.now().isoformat(timespec="seconds")


def run_command(cmd, env=None, cwd=None, log=None, core=None):
    """Run a command (pinned to one hardware thread if `core` is given); output appended to `log`.
    Returns (return code, start, end, seconds)."""
    if core is not None:
        cmd = ["taskset", "-c", str(core)] + list(cmd)
    start, t0 = now(), time.perf_counter()
    result = subprocess.run(cmd, env=env, cwd=cwd, capture_output=True, text=True)
    seconds, end = time.perf_counter() - t0, now()
    if log:
        os.makedirs(os.path.dirname(log), exist_ok=True)
        with open(log, "a") as f:
            f.write(f"# {start} -> {end} ({seconds:.1f} s), exit code {result.returncode}\n$ {' '.join(map(str, cmd))}\n"
                    f"{result.stdout}{result.stderr}\n")
    return result.returncode, start, end, seconds


def pinned_map(function, jobs, cores, stop=None):
    """Run function(job, core) for every job on a pool of single-thread workers, one per hardware thread in `cores`.
    `stop()` is checked before each job starts (e.g. the full-run flag during tests)."""
    free = queue.Queue()
    for c in cores:
        free.put(c)

    def work(job):
        if stop is not None and stop():
            return None
        core = free.get()
        try:
            return function(job, core)
        finally:
            free.put(core)

    with ThreadPoolExecutor(len(cores)) as pool:
        return list(pool.map(work, jobs))


class RuntimeLog:
    """Append-only `runtimes.csv` (thread-safe). Re-runs of a scan append a new row; readers keep the last one."""

    def __init__(self, path, inputs):
        self.path, self.inputs, self.lock = path, inputs, threading.Lock()
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def add(self, **row):
        row.setdefault("host", HOST_LABEL)
        row.setdefault("inputs", self.inputs)
        with self.lock:
            new = not os.path.exists(self.path)
            with open(self.path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=RUNTIME_COLUMNS, extrasaction="ignore")
                if new:
                    writer.writeheader()
                writer.writerow({k: row.get(k, "") for k in RUNTIME_COLUMNS})


def read_runtimes(path):
    import pandas as pd
    if not os.path.exists(path):
        return pd.DataFrame(columns=RUNTIME_COLUMNS)
    table = pd.read_csv(path, dtype={"session": str, "subject": str}, keep_default_na=False)
    scans = table[table.level == "scan"].drop_duplicates(["tool", "dataset", "contrast", "subject", "session"], keep="last")
    return pd.concat([scans, table[table.level != "scan"]], ignore_index=True)


# ------------------------------------------------------------------ hardware and software characterisation

def _run(cmd, **kwargs):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=120, **kwargs).stdout.strip()
    except Exception as error:
        return f"not available ({type(error).__name__})"


def _python_versions(python, modules, env=None):
    code = ("import json, sys, importlib\nout = {'python': sys.version.split()[0]}\n"
            f"for m in {modules!r}:\n"
            "    try:\n        mod = importlib.import_module(m)\n        v = getattr(mod, '__version__', None)\n"
            "        if v is None:\n            import importlib.metadata as md\n"
            "            for dist in (m, m.replace('_', '-'), {'sam2': 'SAM-2'}.get(m, m)):\n"
            "                try:\n                    v = md.version(dist); break\n                except Exception:\n                    pass\n"
            "        out[m] = v or 'installed (version not reported)'\n"
            "    except Exception as e:\n        out[m] = 'not importable: ' + type(e).__name__\n"
            "try:\n    import torch\n    out['torch.version.cuda'] = torch.version.cuda\n    out['cudnn'] = torch.backends.cudnn.version()\nexcept Exception:\n    pass\n"
            "print(json.dumps(out))")
    text = _run([python, "-c", code], env=env)
    try:
        return json.loads(text.splitlines()[-1])
    except Exception:
        return {"error": text[-300:]}


def _conda_package(env_dir, name):
    for path in sorted(os.listdir(pjoin(env_dir, "conda-meta"))) if os.path.isdir(pjoin(env_dir, "conda-meta")) else []:
        if re.match(rf"{re.escape(name)}-\d", path):
            return path[len(name) + 1:-5].rsplit("-", 1)[0]
    return "not installed"


def environment_info(allocation):
    """Hardware and software of the run (written to results/environment.json)."""
    lscpu = _run(["lscpu"])
    field = lambda name: next((l.split(":", 1)[1].strip() for l in lscpu.splitlines() if l.startswith(name + ":")), "not reported")
    meminfo = open("/proc/meminfo").read()
    mem_kb = int(re.search(r"MemTotal:\s+(\d+)", meminfo).group(1))
    gpu = _run(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader"])
    smi = _run(["nvidia-smi"])
    cuda_driver = re.search(r"CUDA Version:\s*([\d.]+)", smi)
    os_release = dict(l.split("=", 1) for l in open("/etc/os-release").read().splitlines() if "=" in l)
    siblings = {}
    for c in allocation.cores:
        try:
            siblings[c] = open(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list").read().strip()
        except OSError:
            pass
    physical = sorted({s for s in siblings.values()})
    env_ben = tool_env(1, drop_mkl=False)
    info = dict(
        date=now(), host=HOST_LABEL,
        allocation=dict(hardware_threads=cores_text(allocation.cores), n_hardware_threads=allocation.n,
                        physical_cores_touched=len(physical), gpu_index=allocation.gpu,
                        note=("Hardware threads 16-47 and 64-95 are both SMT threads of physical cores 16-47; threads 48-63 "
                              "are the SMT siblings of physical cores 0-15 (threads 0-15), which stayed available to other users."
                              if cores_text(allocation.cores) == "16-95" else "")),
        cpu=dict(model=field("Model name"), sockets=field("Socket(s)"), cores_per_socket=field("Core(s) per socket"),
                 threads_per_core=field("Thread(s) per core"), logical_cpus=field("CPU(s)"), max_mhz=field("CPU max MHz"),
                 l3_cache=field("L3 cache")),
        ram_gb=round(mem_kb / 1024 ** 2, 1),
        gpu=dict(nvidia_smi=gpu, driver_cuda_version=cuda_driver.group(1) if cuda_driver else "not reported"),
        os=dict(name=os_release.get("PRETTY_NAME", "").strip('"'), kernel=platform.release(), shared_machine=True),
        software=dict(
            samson_env=_python_versions(config.PY_SAMSON, ["numpy", "scipy", "nibabel", "dipy", "skimage", "cv2", "pandas", "torch", "torchvision", "segment_anything"]),
            samson_sam2_env=_python_versions(config.PY_SAM2, ["numpy", "nibabel", "torch", "sam2"]),
            deepbet_env=_python_versions(config.PY_DEEPBET, ["deepbet", "torch", "nibabel", "numpy"], env=tool_env(1, drop_mkl=True)),
            ben_env=_python_versions(config.PY_BEN, ["tensorflow", "keras", "SimpleITK", "nibabel", "numpy"], env=env_ben),
            fsl_bet=dict(fsl_6_0_7_4=dict(fsl_bet2=_conda_package(config.FSL_6074, "fsl-bet2"), fsl_newimage=_conda_package(config.FSL_6074, "fsl-newimage")),
                         fsl_6_0_7_17=dict(fsl_bet2=_conda_package(config.FSL_60717, "fsl-bet2"), fsl_newimage=_conda_package(config.FSL_60717, "fsl-newimage"))),
            ants=_run([pjoin(config.ANTS_BIN, "antsRegistration"), "--version"]).replace("\n", "; "),
        ),
    )
    return info


# ------------------------------------------------------------------ load snapshots (resource protocol)

def _cpu_busy(groups, interval=0.5):
    """Busy fraction of each group of logical CPUs over `interval` seconds (/proc/stat), one sampling for all groups."""
    def read():
        out = {}
        for line in open("/proc/stat"):
            if line.startswith("cpu") and line[3].isdigit():
                f = line.split()
                values = list(map(int, f[1:]))
                out[int(f[0][3:])] = (sum(values) - values[3] - values[4], sum(values))
        return out
    a = read(); time.sleep(interval); b = read()
    result = []
    for cores in groups:
        busy = [(b[c][0] - a[c][0]) / max(1, b[c][1] - a[c][1]) for c in cores if c in a and c in b]
        result.append(round(float(sum(busy) / len(busy)), 3) if busy else None)
    return result


def snapshot(ctx, tool, phase, scans):
    """Append a load snapshot (GPU compute processes, CPU load on CPUs 0-15 and on the allocation) to
    results/load_snapshots.jsonl at the start and end of every batch."""
    apps = _run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"])
    own = {os.getpid()}
    rows = []
    for line in apps.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 3 and parts[0].isdigit():
            rows.append(dict(pid=int(parts[0]), process=os.path.basename(parts[1]), memory=parts[2], this_run=int(parts[0]) in own))
    record = dict(time=now(), tool=tool, phase=phase, dcs=sorted({s.dc for s in scans}), n_scans=len(scans),
                  gpu_compute_apps=rows, **dict(zip(("cpu_busy_0_15", "cpu_busy_allocation"), _cpu_busy([range(16), ctx.allocation.cores]))),
                  loadavg=open("/proc/loadavg").read().split()[:3])
    path = pjoin(ctx.root, "results", "load_snapshots.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with ctx.lock, open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
