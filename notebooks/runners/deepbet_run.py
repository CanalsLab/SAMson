"""
deepbet on a list of images, one model load for all of them, with the wall time of every image.
Same calls as deepbet.run_bet (deepbet 1.0.2) with skip_broken=False: BrainExtraction(no_gpu) once, then check_file and
.run(input, brain, mask, tiv, threshold, n_dilate) per image. Prints one JSON line per image.
Usage: python deepbet_run.py jobs.json   (threshold, n_dilate, no_gpu, jobs: input/brain/mask/tiv)
"""
import sys
import json
import time
import warnings
from datetime import datetime
from deepbet.bet import BrainExtraction
from deepbet.utils import check_file, is_file_broken

cfg = json.load(open(sys.argv[1]))
t0 = time.perf_counter()
bet = BrainExtraction(no_gpu=cfg["no_gpu"])
print(json.dumps(dict(event="model_loaded", seconds=time.perf_counter() - t0)), flush=True)
for job in cfg["jobs"]:
    start, t = datetime.now().isoformat(timespec="seconds"), time.perf_counter()
    status = "ok"
    if cfg.get("skip_broken", False) and is_file_broken(job["input"]):   # run_bet's default skips files < 100 kB; not used here
        warnings.warn(f"Skipped file {job['input']}", Warning)
        status = "broken input"
    else:
        check_file(job["input"])
        bet.run(job["input"], job["brain"], job["mask"], job["tiv"], cfg["threshold"], cfg["n_dilate"])
    print(json.dumps(dict(event="job", mask=job["mask"], start=start, end=datetime.now().isoformat(timespec="seconds"),
                          seconds=time.perf_counter() - t, status=status)), flush=True)
