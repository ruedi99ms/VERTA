"""
route_analysis.py — Standalone route-decision analysis on x–z trajectories

This module provides a small, dependency-light toolkit to analyze route choices
from movement trajectories on a Cartesian x–z plane. It is **generic**: no
study-specific names or parameters are baked in. You can:

- Load one or many "movement files" (CSV/TSV/parquet). Only coordinates are
  required; time is optional but enables timing metrics.
- Detect an initial movement direction after passing near a junction and
  cluster these directions into k branches.
- Assign branches to trajectories using learned/global centers.
- Compute simple timing metrics: time to travel a certain path length after a
  junction; time between entering two generic regions (e.g., reorientation).
- Summarize branch distribution and Shannon entropy.

The analysis is **solely based on the imported coordinates** (and optional time
column). No participant codes or metadata are required; trajectory IDs default
to the input filenames or provided keys.

Examples (CLI):

  # Discover 3 branches from CSVs in a folder and write results
  python route_analysis.py \
      --input ./data \
      --glob "*.csv" \
      --columns x=X,z=Z,t=time  \
      --junction 700 150 --radius 17.5 \
      --distance 100 \
      --k 3 \
      --out ./outputs

  # Assign branches to a second dataset using previously learned centers
  python route_analysis.py assign \
      --input ./new_data \
      --glob "*.csv" \
      --columns x=X,z=Z,t=time \
      --centers ./outputs/branch_centers.npy \
      --out ./outputs/new_assignments

"""


import argparse
import sys
from typing import Optional, Sequence

from verta.verta_commands import COMMANDS
from verta.verta_config import load_config_file, overlay_config_on_namespace, parse_columns
from verta.verta_validation import validate_args
from verta.verta_logging import get_logger


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Main entry point for VERTA"""
    logger = get_logger()

    # Create main parser
    parser = argparse.ArgumentParser(description="Standalone route-decision analysis (x–z)")
    parser.add_argument("--config", help="Path to YAML/JSON config with defaults", default=None)

    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest="cmd", help="Available commands")

    # Register all commands (including chain-enhanced which is in COMMANDS)
    for cmd_name, cmd_class in COMMANDS.items():
        cmd_instance = cmd_class()
        subparser = subparsers.add_parser(cmd_name, help=f"{cmd_name} command")
        cmd_instance.add_arguments(subparser)

    # Parse arguments
    args = parser.parse_args(argv)

    # Determine command (default to discover)
    cmd = args.cmd or "discover"

    # Load configuration if provided
    if args.config:
        try:
            cfg = load_config_file(args.config)
            # Apply configuration overlay
            overlay_config_on_namespace(args, cfg, subcommand=cmd, provided_keys=set())
        except Exception as e:
            logger.error(f"Failed to load config file: {e}")
            return

    # Convert columns if needed
        # Convert columns if needed
    if hasattr(args, 'columns') and args.columns:
        if isinstance(args.columns, dict):
            cols = args.columns
        else:
            cols = parse_columns(args.columns)
        args.columns = cols

    # Validate arguments
    try:
        args = validate_args(args, parser, strict=False)
    except Exception as e:
        logger.error(f"Argument validation failed: {e}")
        return

    # Execute command
    try:
        cmd_instance = COMMANDS[cmd]()
        cmd_instance.execute(args)
    except Exception as e:
        logger.error(f"Command execution failed: {e}")
        raise


if __name__ == "__main__":
    main()
