"""
Dataset specification: everything SAMson needs to know about a dataset besides the images.

A specification is saved as a small JSON file (see ``docs/io_conventions.md``) so that the choices made once (in the
app, or by hand) are reused for every subject of the dataset and recorded with the results::

    {
      "samson_spec_version": 1,
      "name": "my T2w dataset",
      "orientation": {"axes": [0, 1, 2], "flip": [false, false, false]},
      "slices": [0, -1],
      "template": [4, 33],
      "otsu": "auto",
      "volume": 0
    }

* ``orientation``: how to turn the input array into SAMson's orientation (see :class:`Orientation`).
* ``slices``: first and last slice processed, 0-based and inclusive, counted in SAMson's orientation (slice 0 = caudal);
  ``-1`` means the last slice. Slices outside the range are left empty in the mask.
* ``template``: the template slices (0-39, caudal to rostral) that match the first and the last processed slice.
* ``otsu``: ``"auto"`` (the rule of the notebook: median radius 2 and 5 passes for images with fewer than 140 rows,
  otherwise 1 and 3) or ``[median_radius, numpass]``.
* ``volume``: the volume used from 4D images (0 = first; e.g. the first echo or the b=0 image).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from itertools import permutations

import numpy as np

SPEC_VERSION = 1
AXIS_NAMES = ("left-right", "ventral-dorsal", "caudal-rostral")
_CODE_AXIS = {"R": 0, "L": 0, "S": 1, "I": 1, "D": 1, "V": 1, "A": 2, "P": 2}
_CODE_FLIP = {"R": False, "L": False, "S": False, "D": False, "I": True, "V": True, "A": False, "P": True}


class SpecError(ValueError):
    """An invalid dataset specification (the message says what to change)."""


@dataclass(frozen=True)
class Orientation:
    """
    Re-orientation of an input array to SAMson's orientation, without interpolation.

    SAMson's orientation: axis 0 = left-right, axis 1 = ventral to dorsal (so that ``np.rot90`` of a coronal slice shows
    the dorsal side up), axis 2 = caudal to rostral (slice 0 = caudal end). ``axes[i]`` is the input axis that becomes
    SAMson's axis ``i``; ``flip[i]`` reverses SAMson's axis ``i`` afterwards. The default (no change) is the convention of
    the notebook; the notebook's ``reorient=True`` corresponds to ``flip=(False, False, True)``.
    """
    axes: tuple = (0, 1, 2)
    flip: tuple = (False, False, False)

    def __post_init__(self):
        axes, flip = tuple(int(a) for a in self.axes), tuple(bool(f) for f in self.flip)
        if sorted(axes) != [0, 1, 2]:
            raise SpecError(f"orientation axes must be a permutation of 0, 1, 2 (got {list(axes)}).")
        if len(flip) != 3:
            raise SpecError(f"orientation flip needs three values (got {list(flip)}).")
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "flip", flip)

    @property
    def is_identity(self) -> bool:
        return self.axes == (0, 1, 2) and not any(self.flip)

    def apply(self, array: np.ndarray) -> np.ndarray:
        """Input array (3D, or 4D with the extra axes last) -> SAMson's orientation."""
        out = np.transpose(array, self.axes + tuple(range(3, array.ndim)))
        for axis, flip in enumerate(self.flip):
            if flip:
                out = np.flip(out, axis)
        return out

    def invert(self, array: np.ndarray) -> np.ndarray:
        """SAMson's orientation -> input array orientation (exact inverse of :meth:`apply`)."""
        for axis, flip in enumerate(self.flip):
            if flip:
                array = np.flip(array, axis)
        inverse = tuple(int(i) for i in np.argsort(self.axes))
        return np.transpose(array, inverse + tuple(range(3, array.ndim)))

    def voxel_size(self, zooms) -> tuple:
        """Voxel sizes of the input axes -> voxel sizes along SAMson's axes."""
        return tuple(float(zooms[a]) for a in self.axes)

    @classmethod
    def from_codes(cls, codes: str) -> "Orientation":
        """
        Orientation from the anatomical direction of each input axis (the direction in which its index increases), e.g.
        ``"RIA"``: axis 0 to the right, axis 1 ventral (inferior), axis 2 rostral (anterior). Letters: R/L, A (rostral)/P
        (caudal), S or D (dorsal)/I or V (ventral). Left and right are not swapped (SAMson does not depend on them).
        """
        codes = codes.strip().upper()
        if len(codes) != 3 or any(c not in _CODE_AXIS for c in codes):
            raise SpecError(f"orientation codes must be three letters among R, L, A, P, S, I, D, V (got {codes!r}).")
        samson_axes = [_CODE_AXIS[c] for c in codes]
        if sorted(samson_axes) != [0, 1, 2]:
            raise SpecError(f"orientation codes {codes!r} must name each anatomical axis once.")
        axes = tuple(samson_axes.index(i) for i in range(3))
        flip = tuple(_CODE_FLIP[codes[axes[i]]] for i in range(3))
        return cls(axes, flip)

    def to_codes(self) -> str:
        """Anatomical direction of each input axis, as accepted by :meth:`from_codes` (left-right given as R)."""
        names = [("R", "R"), ("S", "I"), ("A", "P")]
        codes = [""] * 3
        for samson_axis, input_axis in enumerate(self.axes):
            codes[input_axis] = names[samson_axis][int(self.flip[samson_axis])]
        return "".join(codes)

    @staticmethod
    def all() -> list:
        """The 48 orientations (6 axis orders x 8 flip combinations)."""
        return [Orientation(axes, (bool(f & 1), bool(f & 2), bool(f & 4))) for axes in permutations(range(3)) for f in range(8)]

    def to_dict(self) -> dict:
        return {"axes": list(self.axes), "flip": list(self.flip)}

    def describe(self) -> str:
        if self.is_identity:
            return "input already in SAMson orientation"
        parts = [f"SAMson {AXIS_NAMES[i]} = input axis {a}" + (" (reversed)" if f else "") for i, (a, f) in enumerate(zip(self.axes, self.flip))]
        return "; ".join(parts)


@dataclass(frozen=True)
class DatasetSpec:
    """SAMson's settings for one dataset (see the module documentation)."""
    template: tuple = None
    slices: tuple = (0, -1)
    orientation: Orientation = field(default_factory=Orientation)
    otsu: object = "auto"
    volume: int = 0
    name: str = ""

    def __post_init__(self):
        if self.template is None:
            raise SpecError("the template range is required: the template slices (0-39, caudal to rostral) that match your first "
                            "and last processed slice, e.g. template=(4, 33). Use `samson preview` or `samson app` to choose it.")
        template = tuple(int(t) for t in self.template)
        slices = tuple(int(s) for s in self.slices)
        if len(template) != 2 or len(slices) != 2:
            raise SpecError("template and slices need two values each: first and last.")
        orientation = self.orientation
        if isinstance(orientation, str):
            orientation = Orientation.from_codes(orientation)
        elif isinstance(orientation, dict):
            orientation = Orientation(tuple(orientation.get("axes", (0, 1, 2))), tuple(orientation.get("flip", (False,) * 3)))
        otsu = self.otsu
        if isinstance(otsu, str):
            if otsu.lower() != "auto":
                raise SpecError(f"otsu must be 'auto' or [median_radius, numpass] (got {otsu!r}).")
            otsu = "auto"
        else:
            otsu = tuple(int(v) for v in otsu)
            if len(otsu) != 2 or otsu[0] < 1 or otsu[1] < 1:
                raise SpecError(f"otsu must be 'auto' or two positive integers [median_radius, numpass] (got {list(otsu)}).")
        if int(self.volume) < 0:
            raise SpecError("volume must be 0 or a positive index.")
        object.__setattr__(self, "template", template)
        object.__setattr__(self, "slices", slices)
        object.__setattr__(self, "orientation", orientation)
        object.__setattr__(self, "otsu", otsu)
        object.__setattr__(self, "volume", int(self.volume))

    # ------------------------------------------------------------------------------------------------------------------
    def check_template(self, n_template_slices: int = 40) -> None:
        first, last = self.template
        if not (0 <= first <= last < n_template_slices):
            raise SpecError(f"template range {first}-{last} is not valid: it needs 0 <= first <= last <= {n_template_slices - 1} "
                            "(template slice 0 is the caudal end). For images whose slices go from rostral to caudal, reverse the "
                            "anteroposterior axis in the orientation instead of the template range.")

    def slice_range(self, n_slices: int) -> tuple:
        """First and last processed slice (0-based, inclusive) of an image with n_slices slices in SAMson's orientation."""
        first, last = self.slices
        first = first + n_slices if first < 0 else first
        last = last + n_slices if last < 0 else last
        if not (0 <= first <= last < n_slices):
            raise SpecError(f"slice range {self.slices[0]}-{self.slices[1]} is not valid for an image with {n_slices} slices "
                            f"(valid indices: 0-{n_slices - 1}, or -1 for the last slice).")
        return first, last

    def otsu_parameters(self, part: np.ndarray) -> tuple:
        """median_otsu (median_radius, numpass) for the processed part of a volume."""
        if self.otsu == "auto":
            from ._algorithm import default_otsu_parameters
            return default_otsu_parameters(part)
        return self.otsu

    # ------------------------------------------------------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"samson_spec_version": SPEC_VERSION, "name": self.name, "orientation": self.orientation.to_dict(),
                "slices": list(self.slices), "template": list(self.template),
                "otsu": self.otsu if self.otsu == "auto" else list(self.otsu), "volume": self.volume}

    @classmethod
    def from_dict(cls, data: dict) -> "DatasetSpec":
        version = data.get("samson_spec_version", SPEC_VERSION)
        if version > SPEC_VERSION:
            raise SpecError(f"this specification was written by a newer SAMson (spec version {version}); please update samson-mri.")
        unknown = set(data) - {"samson_spec_version", "name", "orientation", "slices", "template", "otsu", "volume", "notes"}
        if unknown:
            raise SpecError(f"unknown keys in the specification: {sorted(unknown)}.")
        return cls(template=data.get("template"), slices=data.get("slices", (0, -1)), orientation=data.get("orientation", {}),
                   otsu=data.get("otsu", "auto"), volume=data.get("volume", 0), name=data.get("name", ""))

    def to_json(self) -> str:
        """JSON text with the short lists on one line."""
        text = json.dumps(self.to_dict(), indent=2)
        return re.sub(r"\[\s+([^\[\]{}]*?)\s+\]", lambda m: "[" + ", ".join(v.strip() for v in m.group(1).split(",")) + "]", text)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_json() + "\n")

    @classmethod
    def load(cls, path: str) -> "DatasetSpec":
        with open(path) as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as error:
                raise SpecError(f"{path} is not a valid JSON file: {error}") from None
        return cls.from_dict(data)

    def replace(self, **changes) -> "DatasetSpec":
        values = dict(template=self.template, slices=self.slices, orientation=self.orientation, otsu=self.otsu,
                      volume=self.volume, name=self.name)
        values.update({k: v for k, v in changes.items() if v is not None})
        return DatasetSpec(**values)


# Specifications of the article's datasets (public data: https://doi.org/10.20350/digitalCSIC/17000)
PRESETS = {
    "article-d1": DatasetSpec(name="Article D1 (T2w, 16 slices of 0.8 mm; whole brain)", template=(4, 33)),
    "article-d2": DatasetSpec(name="Article D2 (T2w *_flip.nii.gz images)", template=(2, 39)),
    "article-d3": DatasetSpec(name="Article D3 (T2w *_flip.nii.gz images)", template=(3, 35)),
}
