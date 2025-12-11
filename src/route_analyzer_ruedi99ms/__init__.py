"""
VERTA - Virtual Environment Route and Trajectory Analyzer

A comprehensive toolkit for analyzing route choices and trajectories in virtual environments.
"""

__version__ = "0.1.0"

# Import main classes and functions for convenient access
from route_analyzer_ruedi99ms.ra_data_loader import (
    Trajectory,
    ColumnMapping,
    load_folder,
    load_folder_with_gaze,
)

from route_analyzer_ruedi99ms.ra_geometry import Circle, Rect

from route_analyzer_ruedi99ms.ra_decisions import (
    discover_branches,
    assign_branches,
    discover_decision_chain,
)

from route_analyzer_ruedi99ms.ra_commands import (
    BaseCommand,
    CommandConfig,
    DiscoverCommand,
    AssignCommand,
    MetricsCommand,
    GazeCommand,
    COMMANDS,
)

__all__ = [
    "Trajectory",
    "ColumnMapping",
    "load_folder",
    "load_folder_with_gaze",
    "Circle",
    "Rect",
    "discover_branches",
    "assign_branches",
    "discover_decision_chain",
    "BaseCommand",
    "CommandConfig",
    "DiscoverCommand",
    "AssignCommand",
    "MetricsCommand",
    "GazeCommand",
    "COMMANDS",
]
