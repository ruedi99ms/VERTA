# Geometry helpers

from dataclasses import dataclass

import numpy as np


@dataclass
class Circle:
    cx: float
    cz: float
    r: float

    def contains(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        return (x - self.cx) ** 2 + (z - self.cz) ** 2 <= self.r ** 2


@dataclass
class Rect:
    xmin: float
    xmax: float
    zmin: float
    zmax: float

    def contains(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        return (x >= self.xmin) & (x <= self.xmax) & (z >= self.zmin) & (z <= self.zmax)


def entered_junction_idx(x: np.ndarray, z: np.ndarray, junction: "Circle") -> tuple[bool, int]:
    """Return (entered, index). If entered==True: index is first inside sample.
       Otherwise: index of nearest approach (used for plotting)."""
    r = np.hypot(x - junction.cx, z - junction.cz)
    inside = r <= junction.r
    if inside.any():
        return True, int(np.argmax(inside))
    return False, int(np.argmin(r))


