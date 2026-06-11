"""Tests for trajectory map plotting helpers."""

import matplotlib
matplotlib.use("Agg")

import os
import tempfile

import numpy as np
import pytest

from verta.verta_data_loader import Trajectory
from verta.verta_geometry import Circle
from verta.verta_plotting import (
    coordinate_labels,
    plot_branch_counts,
    plot_branch_directions,
    plot_branch_trajectories_map,
    plot_sample_trajectories_map,
)


@pytest.fixture
def sample_trajectories():
    trajs = []
    for i in range(5):
        t = np.linspace(0, 10, 50)
        x = 100 + i * 20 + t * 5 * (1 if i % 2 == 0 else -1)
        z = 200 + t * 3
        trajs.append(Trajectory(tid=i, x=x, z=z, t=t))
    return trajs


def test_coordinate_labels_with_unit():
    labels = coordinate_labels(scale=0.2, unit="m")
    assert "m" in labels["x"]
    assert "m" in labels["z"]
    assert "scale=0.2" in labels["caption"]


def test_coordinate_labels_scene_units():
    labels = coordinate_labels(scale=1.0)
    assert "scene units" in labels["x"]


def test_plot_sample_trajectories_map(sample_trajectories):
    junction = Circle(cx=150, cz=220, r=15)
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "sample_map.png")
        plot_sample_trajectories_map(
            sample_trajectories,
            n_samples=3,
            junctions=[junction],
            out_path=out,
        )
        assert os.path.isfile(out)
        assert os.path.getsize(out) > 1000


def test_plot_branch_counts(sample_trajectories):
    import pandas as pd

    df = pd.DataFrame({"trajectory": [0, 1, 2, 3], "branch": [0, 0, 1, 1]})
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "counts.png")
        plot_branch_counts(df, out_path=out)
        assert os.path.isfile(out)


def test_plot_branch_trajectories_map(sample_trajectories):
    import pandas as pd

    junctions = [
        Circle(cx=150, cz=220, r=15),
        Circle(cx=250, cz=280, r=15),
    ]
    df = pd.DataFrame({
        "trajectory": [str(t.tid) for t in sample_trajectories],
        "branch": [0, 0, 1, 1, 2],
    })
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "branch_map.png")
        plot_branch_trajectories_map(
            sample_trajectories,
            df,
            junctions=junctions,
            junction_number=0,
            out_path=out,
        )
        assert os.path.isfile(out)
        assert os.path.getsize(out) > 1000


def test_plot_branch_directions():
    centers = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "directions.png")
        plot_branch_directions(centers, (100.0, 200.0), out_path=out)
        assert os.path.isfile(out)
