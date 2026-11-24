"""
SAMson: brain extraction of rodent MRI with the Segment Anything Model (SAM).

Python API (see README.md and docs/)::

    from samson_mri import SAMson, DatasetSpec
    spec = DatasetSpec(template=(4, 33))            # template slices matching your first and last slice
    result = SAMson().segment("image.nii.gz", spec)  # Full-Auto
    result.save("output_folder")

Command line: ``samson --help``. Interactive app: ``samson app``.
"""
from ._version import __version__
from .io import Subject, find_images, load_subject
from .metrics import compare_volumes, evaluate
from .pipeline import FULL_AUTO, BoxPrompts, SAMson, SegmentationResult, apply_selection, compute_boxes, load_result
from .spec import PRESETS, DatasetSpec, Orientation, SpecError
from .template import Template, load_template

__all__ = ["__version__", "SAMson", "SegmentationResult", "BoxPrompts", "DatasetSpec", "Orientation", "SpecError", "PRESETS",
           "Subject", "load_subject", "find_images", "compute_boxes", "apply_selection", "load_result", "FULL_AUTO", "Template", "load_template",
           "evaluate", "compare_volumes"]
