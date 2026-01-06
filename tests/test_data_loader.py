"""
Tests for data loading module (Trajectory, ColumnMapping, load functions).
"""
import pytest
import numpy as np
import pandas as pd
import os
import tempfile
from pathlib import Path

from verta.verta_data_loader import (
    Trajectory, ColumnMapping,
    has_gaze_data, has_physio_data
)


class TestTrajectory:
    """Test cases for Trajectory dataclass."""

    def test_trajectory_creation_basic(self):
        """Test creating a basic trajectory."""
        traj = Trajectory(
            tid="test_1",
            x=np.array([0.0, 1.0, 2.0]),
            z=np.array([0.0, 1.0, 2.0])
        )
        assert traj.tid == "test_1"
        assert len(traj.x) == 3
        assert len(traj.z) == 3
        assert traj.t is None

    def test_trajectory_creation_with_time(self):
        """Test creating a trajectory with time data."""
        traj = Trajectory(
            tid="test_2",
            x=np.array([0.0, 1.0, 2.0]),
            z=np.array([0.0, 1.0, 2.0]),
            t=np.array([0.0, 0.1, 0.2])
        )
        assert traj.t is not None
        assert len(traj.t) == 3

    def test_trajectory_creation_with_gaze(self):
        """Test creating a trajectory with gaze data."""
        traj = Trajectory(
            tid="test_gaze",
            x=np.array([0.0, 1.0, 2.0]),
            z=np.array([0.0, 1.0, 2.0]),
            head_forward_x=np.array([1.0, 1.0, 1.0]),
            head_forward_z=np.array([0.0, 0.0, 0.0]),
            gaze_x=np.array([0.5, 0.6, 0.7]),
            gaze_y=np.array([0.1, 0.2, 0.3])
        )
        assert traj.head_forward_x is not None
        assert traj.gaze_x is not None
        assert traj.gaze_y is not None


class TestColumnMapping:
    """Test cases for ColumnMapping dataclass."""

    def test_column_mapping_defaults(self):
        """Test ColumnMapping with default values."""
        mapping = ColumnMapping()
        assert mapping.x == "x"
        assert mapping.z == "z"
        assert mapping.t == "t"

    def test_column_mapping_vr_defaults(self):
        """Test VR default column mapping."""
        mapping = ColumnMapping.vr_defaults()
        assert mapping.headset_x == "Headset.Head.Position.X"
        assert mapping.headset_z == "Headset.Head.Position.Z"
        assert mapping.gaze_x == "Headset.Gaze.X"

    def test_column_mapping_from_dict(self):
        """Test creating ColumnMapping from dictionary."""
        custom_mapping = {
            "x": "pos_x",
            "z": "pos_z",
            "t": "time"
        }
        mapping = ColumnMapping.from_dict(custom_mapping)
        assert mapping.x == "pos_x"
        assert mapping.z == "pos_z"
        assert mapping.t == "time"
        # Other fields should still have defaults
        assert mapping.headset_x == "Headset.Head.Position.X"


class TestHasGazeData:
    """Test cases for has_gaze_data function."""

    def test_has_gaze_data_true(self, sample_trajectory_with_gaze):
        """Test has_gaze_data returns True when gaze data is present."""
        assert has_gaze_data(sample_trajectory_with_gaze) == True

    def test_has_gaze_data_false(self, sample_trajectory):
        """Test has_gaze_data returns False when gaze data is missing."""
        assert has_gaze_data(sample_trajectory) == False

    def test_has_gaze_data_partial(self):
        """Test has_gaze_data returns False when only partial gaze data exists."""
        traj = Trajectory(
            tid="partial",
            x=np.array([0.0, 1.0]),
            z=np.array([0.0, 1.0]),
            head_forward_x=np.array([1.0, 1.0]),
            head_forward_z=np.array([0.0, 0.0]),
            # Missing gaze_x and gaze_y
        )
        assert has_gaze_data(traj) == False


class TestHasPhysioData:
    """Test cases for has_physio_data function."""

    def test_has_physio_data_true(self):
        """Test has_physio_data returns True when physio data is present."""
        traj = Trajectory(
            tid="physio",
            x=np.array([0.0, 1.0]),
            z=np.array([0.0, 1.0]),
            heart_rate=np.array([70.0, 72.0]),
            pupil_l=np.array([3.5, 3.6]),
            pupil_r=np.array([3.5, 3.6])
        )
        assert has_physio_data(traj) == True

    def test_has_physio_data_false(self, sample_trajectory):
        """Test has_physio_data returns False when physio data is missing."""
        assert has_physio_data(sample_trajectory) == False

    def test_has_physio_data_partial(self):
        """Test has_physio_data returns False when only partial physio data exists."""
        traj = Trajectory(
            tid="partial_physio",
            x=np.array([0.0, 1.0]),
            z=np.array([0.0, 1.0]),
            heart_rate=np.array([70.0, 72.0]),
            pupil_l=np.array([3.5, 3.6])
            # Missing pupil_r - requires all three fields
        )
        assert has_physio_data(traj) == False
