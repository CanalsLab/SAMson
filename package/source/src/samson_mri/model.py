"""
The Segment Anything Model (SAM): checkpoints and predictor.

SAMson uses SAM's ViT-H model (``sam_vit_h_4b8939.pth``, 2.4 GB), as in the article. The checkpoint is looked for, in
this order, at: the ``checkpoint`` argument (``--checkpoint``), the ``SAMSON_CHECKPOINT`` environment variable (for vit_h), and the
SAMson cache folder (``$SAMSON_HOME``, default ``~/.cache/samson_mri``), where ``samson download-model`` puts it.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import urllib.request

CHECKPOINTS = {
    # model type: file name, official URL (Meta AI), MD5 prefix in the file name, full SHA-256 when known, size in bytes
    "vit_h": dict(file="sam_vit_h_4b8939.pth", url="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                  md5_prefix="4b8939", sha256="a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e", size=2564550879),
    "vit_l": dict(file="sam_vit_l_0b3195.pth", url="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
                  md5_prefix="0b3195", sha256=None, size=None),
    "vit_b": dict(file="sam_vit_b_01ec64.pth", url="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                  md5_prefix="01ec64", sha256=None, size=None),
}
DEFAULT_MODEL = "vit_h"


class CheckpointNotFound(FileNotFoundError):
    pass


def cache_dir() -> str:
    """Folder for downloaded checkpoints: $SAMSON_HOME or ~/.cache/samson_mri."""
    return os.path.expanduser(os.environ.get("SAMSON_HOME", os.path.join("~", ".cache", "samson_mri")))


def find_checkpoint(model_type: str = DEFAULT_MODEL, checkpoint: str | None = None) -> str:
    """Path of the checkpoint to use (see the module documentation); raises CheckpointNotFound with instructions."""
    if model_type not in CHECKPOINTS:
        raise ValueError(f"Unknown model type {model_type!r}; choose one of {sorted(CHECKPOINTS)}.")
    environment = os.environ.get("SAMSON_CHECKPOINT")   # For the default model (or a file named like this model's checkpoint)
    if environment and model_type != DEFAULT_MODEL and os.path.basename(environment) != CHECKPOINTS[model_type]["file"]:
        environment = None
    candidates = [checkpoint, environment, os.path.join(cache_dir(), CHECKPOINTS[model_type]["file"])]
    for path in candidates:
        if path and os.path.isfile(os.path.expanduser(path)):
            return os.path.abspath(os.path.expanduser(path))
    if checkpoint:
        raise CheckpointNotFound(f"The checkpoint {checkpoint} does not exist.")
    raise CheckpointNotFound(
        f"SAM's {model_type} checkpoint was not found. Download it once with\n\n    samson download-model --model-type {model_type}\n\n"
        f"(it goes to {cache_dir()}), or give its path with --checkpoint or the SAMSON_CHECKPOINT environment variable.")


def file_md5(path: str, chunk: int = 1 << 24) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def download_checkpoint(model_type: str = DEFAULT_MODEL, folder: str | None = None, progress: bool = True, callback=None) -> str:
    """
    Download a SAM checkpoint from Meta AI's server into ``folder`` (default: the cache folder) and check its MD5.
    callback, if given, is called as callback(downloaded_bytes, total_bytes) during the download.
    """
    info = CHECKPOINTS[model_type]
    folder = os.path.expanduser(folder or cache_dir())
    os.makedirs(folder, exist_ok=True)
    destination = os.path.join(folder, info["file"])
    if os.path.isfile(destination):
        if progress:
            print(f"{destination} already exists.")
        return destination
    partial = destination + ".part"

    def report(blocks, block_size, total):
        if callback is not None and total > 0:
            callback(min(blocks * block_size, total), total)
        if progress and total > 0:
            done = min(blocks * block_size, total)
            sys.stdout.write(f"\rDownloading {info['file']}: {done / 1e9:5.2f} / {total / 1e9:.2f} GB ({100 * done / total:5.1f} %)")
            sys.stdout.flush()

    urllib.request.urlretrieve(info["url"], partial, reporthook=report)
    if progress:
        sys.stdout.write("\nChecking the file... ")
    md5 = file_md5(partial)
    if not md5.startswith(info["md5_prefix"]):
        os.remove(partial)
        raise IOError(f"The downloaded file is corrupted (MD5 {md5} does not start with {info['md5_prefix']}); please try again.")
    shutil.move(partial, destination)
    if progress:
        print(f"ok.\nSaved {destination}")
    return destination


def resolve_device(device: str = "auto") -> str:
    """'auto' -> 'cuda' if an NVIDIA GPU is available, else 'mps' (Apple GPU) if available, else 'cpu'."""
    if device != "auto":
        return device
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_predictor(model_type: str = DEFAULT_MODEL, checkpoint: str | None = None, device: str = "auto"):
    """SAM predictor (segment_anything.SamPredictor) of the given model type, on the given device."""
    import warnings

    import torch  # noqa: F401  (imported here so that the rest of the package works without loading torch)
    from segment_anything import SamPredictor, sam_model_registry

    path = find_checkpoint(model_type, checkpoint)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)   # torch.load notice about weights_only
        model = sam_model_registry[model_type](checkpoint=path)
    predictor = SamPredictor(model.to(device=resolve_device(device)))
    predictor.samson_checkpoint = path
    return predictor
