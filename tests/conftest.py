"""
Shared pytest fixtures for route_analyzer tests.
"""
import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import sys
from pathlib import Path

# Import from package
from route_analyzer_ruedi99ms.ra_data_loader import Trajectory, ColumnMapping
from route_analyzer_ruedi99ms.ra_geometry import Circle, Rect


@pytest.fixture
def sample_trajectory():
    """Create a sample trajectory for testing."""
    return Trajectory(
        tid="test_traj_1",
        x=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        z=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        t=np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    )


@pytest.fixture
def sample_trajectory_no_time():
    """Create a sample trajectory without time data."""
    return Trajectory(
        tid="test_traj_2",
        x=np.array([0.0, 1.0, 2.0, 3.0]),
        z=np.array([0.0, 1.0, 2.0, 3.0]),
        t=None
    )


@pytest.fixture
def sample_trajectory_with_gaze():
    """Create a sample trajectory with gaze data."""
    return Trajectory(
        tid="test_traj_gaze",
        x=np.array([0.0, 1.0, 2.0, 3.0]),
        z=np.array([0.0, 1.0, 2.0, 3.0]),
        t=np.array([0.0, 0.1, 0.2, 0.3]),
        head_forward_x=np.array([1.0, 1.0, 1.0, 1.0]),
        head_forward_y=np.array([0.0, 0.0, 0.0, 0.0]),
        head_forward_z=np.array([0.0, 0.0, 0.0, 0.0]),
        gaze_x=np.array([0.5, 0.6, 0.7, 0.8]),
        gaze_y=np.array([0.1, 0.2, 0.3, 0.4])
    )


@pytest.fixture
def sample_circle():
    """Create a sample circle (junction) for testing."""
    return Circle(cx=2.0, cz=2.0, r=1.0)


@pytest.fixture
def sample_rect():
    """Create a sample rectangle for testing."""
    return Rect(xmin=0.0, xmax=2.0, zmin=0.0, zmax=2.0)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_csv_file(temp_dir):
    """Create a sample CSV file for testing data loading."""
    csv_path = os.path.join(temp_dir, "test_trajectory.csv")
    df = pd.DataFrame({
        'x': [0.0, 1.0, 2.0, 3.0, 4.0],
        'z': [0.0, 1.0, 2.0, 3.0, 4.0],
        't': [0.0, 0.1, 0.2, 0.3, 0.4]
    })
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_csv_folder(temp_dir):
    """Create a folder with multiple sample CSV files."""
    folder = os.path.join(temp_dir, "trajectories")
    os.makedirs(folder, exist_ok=True)

    for i in range(3):
        csv_path = os.path.join(folder, f"traj_{i}.csv")
        df = pd.DataFrame({
            'x': np.linspace(0, 4, 5) + i * 0.1,
            'z': np.linspace(0, 4, 5) + i * 0.1,
            't': np.linspace(0, 0.4, 5)
        })
        df.to_csv(csv_path, index=False)

    return folder


@pytest.fixture
def mock_args():
    """Create mock argparse.Namespace for testing commands."""
    class MockArgs:
        def __init__(self):
            self.input = "/test/input"
            self.out = "/test/output"
            self.glob = "*.csv"
            self.columns = None
            self.scale = 1.0
            self.motion_threshold = 0.001
            self.junction = [2.0, 2.0]
            self.radius = 1.0
            self.distance = 100.0
            self.epsilon = 0.015
            self.decision_mode = "hybrid"
            self.r_outer = None
            self.linger_delta = 5.0

    return MockArgs()


@pytest.fixture
def column_mapping():
    """Create a default column mapping."""
    return ColumnMapping()
