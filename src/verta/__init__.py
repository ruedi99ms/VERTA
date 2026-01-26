"""
VERTA - Virtual Environment Route and Trajectory Analyzer

A Python tool for route and trajectory analysis in virtual environments.
"""

__version__ = "0.1.0"

# Import main classes and functions for convenient access
from verta.verta_data_loader import (
    Trajectory,
    ColumnMapping,
    load_folder,
    load_folder_with_gaze,
)

from verta.verta_geometry import Circle, Rect

from verta.verta_decisions import (
    discover_branches,
    assign_branches,
    discover_decision_chain,
)

from verta.verta_commands import (
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
