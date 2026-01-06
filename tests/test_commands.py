"""
Tests for command classes (BaseCommand, DiscoverCommand, AssignCommand, etc.).
"""
import pytest
import os
import tempfile
import numpy as np
import pandas as pd
import argparse
from unittest.mock import Mock, patch, MagicMock

"""
Tests for command classes (BaseCommand, DiscoverCommand, AssignCommand, etc.).

These tests verify that command classes can be instantiated and their argument
parsers are correctly configured. The tests will skip if the verta package
is not installed (e.g., run `pip install -e .` from the repository root).
"""
import sys
from pathlib import Path

# Import from package
try:
    from verta.verta_commands import (
        BaseCommand, CommandConfig, DiscoverCommand, 
        AssignCommand, MetricsCommand, GazeCommand
    )
    from verta.verta_geometry import Circle
    COMMANDS_AVAILABLE = True
except (ImportError, AttributeError) as e:
    # If package is not installed, mark as unavailable
    COMMANDS_AVAILABLE = False
    BaseCommand = None
    CommandConfig = None
    DiscoverCommand = None
    AssignCommand = None
    MetricsCommand = None
    GazeCommand = None
    Circle = None


# Skip all command tests if verta package is not available
pytestmark = pytest.mark.skipif(
    not COMMANDS_AVAILABLE, 
    reason="verta package not available - install with 'pip install -e .' from repository root"
)


class TestCommandConfig:
    """Test cases for CommandConfig dataclass."""
    
    @pytest.mark.skipif(not COMMANDS_AVAILABLE, reason="verta_commands not available")
    def test_command_config_defaults(self):
        """Test CommandConfig with default values."""
        config = CommandConfig(input="/test/input")
        assert config.input == "/test/input"
        assert config.glob == "*.csv"
        assert config.columns is None
        assert config.scale == 1.0
        assert config.motion_threshold == 0.001
        assert config.out is None
        assert config.config is None
    
    def test_command_config_custom_values(self):
        """Test CommandConfig with custom values."""
        config = CommandConfig(
            input="/test/input",
            glob="*.txt",
            scale=2.0,
            motion_threshold=0.01,
            out="/test/output"
        )
        assert config.input == "/test/input"
        assert config.glob == "*.txt"
        assert config.scale == 2.0
        assert config.motion_threshold == 0.01
        assert config.out == "/test/output"


class TestBaseCommand:
    """Test cases for BaseCommand abstract base class."""
    
    def test_base_command_initialization(self):
        """Test BaseCommand initialization."""
        # Create a concrete implementation for testing
        class TestCommand(BaseCommand):
            def add_arguments(self, parser):
                pass
            
            def execute(self, args):
                pass
        
        cmd = TestCommand()
        assert cmd.logger is not None
    
    def test_base_command_create_output_dir(self, temp_dir):
        """Test _create_output_dir method."""
        class TestCommand(BaseCommand):
            def add_arguments(self, parser):
                pass
            
            def execute(self, args):
                pass
        
        cmd = TestCommand()
        output_path = os.path.join(temp_dir, "test_output")
        cmd._create_output_dir(output_path)
        assert os.path.exists(output_path)
        assert os.path.isdir(output_path)
    
    def test_base_command_save_run_args(self, temp_dir):
        """Test _save_run_args method."""
        class TestCommand(BaseCommand):
            def add_arguments(self, parser):
                pass
            
            def execute(self, args):
                pass
        
        cmd = TestCommand()
        args = argparse.Namespace(
            input="/test/input",
            out=temp_dir,
            glob="*.csv",
            scale=1.0
        )
        cmd._save_run_args(args, temp_dir)
        
        args_path = os.path.join(temp_dir, "run_args.json")
        assert os.path.exists(args_path)
        
        import json
        with open(args_path, 'r') as f:
            saved_args = json.load(f)
        assert saved_args['input'] == "/test/input"
        assert saved_args['glob'] == "*.csv"


class TestDiscoverCommand:
    """Test cases for DiscoverCommand."""
    
    def test_discover_command_add_arguments(self):
        """Test that DiscoverCommand adds all required arguments."""
        cmd = DiscoverCommand()
        parser = argparse.ArgumentParser()
        cmd.add_arguments(parser)
        
        # Check that key arguments are present
        args = parser.parse_args([
            "--input", "/test/input",
            "--out", "/test/output",
            "--junction", "2.0", "2.0",
            "--radius", "1.0"
        ])
        assert args.input == "/test/input"
        assert args.out == "/test/output"
        assert args.junction == [2.0, 2.0]
        assert args.radius == 1.0
    
    def test_discover_command_defaults(self):
        """Test DiscoverCommand default argument values."""
        cmd = DiscoverCommand()
        parser = argparse.ArgumentParser()
        cmd.add_arguments(parser)
        
        args = parser.parse_args([
            "--input", "/test/input",
            "--out", "/test/output",
            "--junction", "2.0", "2.0",
            "--radius", "1.0"
        ])
        assert args.glob == "*.csv"
        assert args.scale == 1.0
        assert args.motion_threshold == 0.001
        assert args.distance == 100.0
        assert args.epsilon == 0.015
        assert args.k == 3
        assert args.decision_mode == "hybrid"
        assert args.cluster_method == "kmeans"


class TestAssignCommand:
    """Test cases for AssignCommand."""
    
    def test_assign_command_add_arguments(self):
        """Test that AssignCommand adds all required arguments."""
        cmd = AssignCommand()
        parser = argparse.ArgumentParser()
        cmd.add_arguments(parser)
        
        args = parser.parse_args([
            "--input", "/test/input",
            "--out", "/test/output",
            "--junction", "2.0", "2.0",
            "--radius", "1.0",
            "--centers", "/test/centers.npy"
        ])
        assert args.input == "/test/input"
        assert args.out == "/test/output"
        assert args.centers == "/test/centers.npy"
    
    def test_assign_command_defaults(self):
        """Test AssignCommand default argument values."""
        cmd = AssignCommand()
        parser = argparse.ArgumentParser()
        cmd.add_arguments(parser)
        
        args = parser.parse_args([
            "--input", "/test/input",
            "--out", "/test/output",
            "--junction", "2.0", "2.0",
            "--radius", "1.0",
            "--centers", "/test/centers.npy"
        ])
        assert args.decision_mode == "pathlen"
        assert args.distance == 100.0
        assert args.epsilon == 0.015


class TestMetricsCommand:
    """Test cases for MetricsCommand."""
    
    def test_metrics_command_add_arguments(self):
        """Test that MetricsCommand adds all required arguments."""
        cmd = MetricsCommand()
        parser = argparse.ArgumentParser()
        cmd.add_arguments(parser)
        
        args = parser.parse_args([
            "--input", "/test/input",
            "--out", "/test/output",
            "--junction", "2.0", "2.0",
            "--radius", "1.0"
        ])
        assert args.input == "/test/input"
        assert args.out == "/test/output"
    
    def test_metrics_command_defaults(self):
        """Test MetricsCommand default argument values."""
        cmd = MetricsCommand()
        parser = argparse.ArgumentParser()
        cmd.add_arguments(parser)
        
        args = parser.parse_args([
            "--input", "/test/input",
            "--out", "/test/output",
            "--junction", "2.0", "2.0",
            "--radius", "1.0"
        ])
        assert args.decision_mode == "pathlen"
        assert args.trend_window == 5
        assert args.min_outward == 0.0


class TestGazeCommand:
    """Test cases for GazeCommand."""
    
    def test_gaze_command_add_arguments_single_junction(self):
        """Test GazeCommand with single junction."""
        cmd = GazeCommand()
        parser = argparse.ArgumentParser()
        cmd.add_arguments(parser)
        
        args = parser.parse_args([
            "--input", "/test/input",
            "--out", "/test/output",
            "--junction", "2.0", "2.0",
            "--radius", "1.0"
        ])
        assert args.input == "/test/input"
        assert args.junction == [2.0, 2.0]
        assert args.radius == 1.0
    
    def test_gaze_command_add_arguments_multiple_junctions(self):
        """Test GazeCommand with multiple junctions."""
        cmd = GazeCommand()
        parser = argparse.ArgumentParser()
        cmd.add_arguments(parser)
        
        args = parser.parse_args([
            "--input", "/test/input",
            "--out", "/test/output",
            "--junctions", "2.0", "2.0", "1.0", "5.0", "5.0", "1.5"
        ])
        assert args.junctions == [2.0, 2.0, 1.0, 5.0, 5.0, 1.5]
    
    def test_gaze_command_defaults(self):
        """Test GazeCommand default argument values."""
        cmd = GazeCommand()
        parser = argparse.ArgumentParser()
        cmd.add_arguments(parser)
        
        args = parser.parse_args([
            "--input", "/test/input",
            "--out", "/test/output",
            "--junction", "2.0", "2.0",
            "--radius", "1.0"
        ])
        assert args.decision_mode == "hybrid"
        assert args.physio_window == 3.0
        assert args.cluster_method == "kmeans"
        assert args.k == 3

