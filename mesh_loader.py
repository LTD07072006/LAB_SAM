"""Load the small JSON triangle mesh used by the localisation pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from locator import TriangleMesh


def load_triangle_mesh(path: Path | str) -> TriangleMesh:
    """Load ``{vertices: [[x,y,z], ...], faces: [[i,j,k], ...]}`` JSON."""
    mesh_path = Path(path)
    data = json.loads(mesh_path.read_text(encoding="utf-8"))
    return TriangleMesh(
        np.asarray(data["vertices"], dtype=np.float64),
        np.asarray(data["faces"], dtype=np.int64),
    )
