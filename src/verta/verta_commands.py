# ------------------------------
# Command Handler Architecture
# ------------------------------

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, Sequence, List
import argparse
import numpy as np
import pandas as pd
import os
import json

from verta.verta_clustering import split_small_branches
from verta.verta_decisions import assign_branches, discover_branches, discover_decision_chain
from verta.verta_gaze import (
    compute_head_yaw_at_decisions,
    analyze_physiological_at_junctions, plot_gaze_directions_at_junctions,
    plot_physiological_by_branch, gaze_movement_consistency_report,
    analyze_pupil_dilation_trajectory, plot_pupil_trajectory_analysis
)
from verta.verta_consistency import normalize_assignments, validate_trajectories_unique
from verta.verta_geometry import Circle
from verta.verta_data_loader import load_folder, load_folder_with_gaze, save_assignments, save_centers, save_centers_json, save_summary
from verta.verta_metrics import _timing_for_traj, time_between_regions, speed_through_junction, junction_transit_speed
from verta.verta_plotting import (
    plot_decision_intercepts,
    plot_chain_overview, plot_chain_small_multiples,
    plot_flow_graph_map, plot_per_junction_flow_graph
)
from verta.verta_logging import VERTALogger, get_logger
from verta.verta_data_loader import Trajectory
from verta.verta_prediction import analyze_junction_choice_patterns, JunctionChoiceAnalyzer
from verta.verta_intent_recognition import analyze_intent_recognition
from collections import Counter


@dataclass
class CommandConfig:
    """Shared configuration for all commands"""
    input: str
    glob: str = "*.csv"
    columns: Optional[Dict[str, str]] = None
    scale: float = 1.0
    motion_threshold: float = 0.001
    out: Optional[str] = None
    config: Optional[str] = None # Added for consistency, though handled by main


class BaseCommand(ABC):
    """Abstract base class for all commands"""

    def __init__(self):
        self.logger = VERTALogger()

    @abstractmethod
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add command-specific arguments"""
        pass

    @abstractmethod
    def execute(self, args: argparse.Namespace) -> None:
        """Execute the command"""
        pass

    def _create_output_dir(self, out_path: str) -> None:
        """Create output directory if it doesn't exist"""
        os.makedirs(out_path, exist_ok=True)

    def _save_run_args(self, args: argparse.Namespace, out_path: str) -> None:
        """Save run arguments to JSON file"""
        args_path = os.path.join(out_path, "run_args.json")
        with open(args_path, "w") as f:
            json.dump(vars(args), f, indent=2)


class DiscoverCommand(BaseCommand):
    """Command handler for branch discovery"""

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--input", required=True, help="Input folder path")
        parser.add_argument("--out", required=True, help="Output folder path")
        parser.add_argument("--glob", default="*.csv", help="File pattern")
        parser.add_argument("--columns", default=None, help="Column mapping")
        parser.add_argument("--scale", type=float, default=1.0, help="Coordinate scaling factor")
        parser.add_argument("--motion_threshold", type=float, default=0.001, help="Motion detection threshold")
        parser.add_argument("--junction", nargs=2, type=float, required=True, metavar=("X", "Z"), help="Junction coordinates")
        parser.add_argument("--radius", type=float, required=True, help="Junction radius")
        parser.add_argument("--distance", type=float, default=100.0, help="Path length for decision")
        parser.add_argument("--epsilon", type=float, default=0.015, help="Minimum step size")
        parser.add_argument("--k", type=int, default=3, help="Number of clusters")
        parser.add_argument("--decision_mode", choices=["pathlen", "radial", "hybrid"], default="hybrid", help="Decision mode")
        parser.add_argument("--r_outer", type=float, default=None, help="Outer radius for radial mode")
        parser.add_argument("--linger_delta", type=float, default=5.0, help="Linger distance beyond junction")
        parser.add_argument("--cluster_method", choices=["kmeans", "auto", "dbscan"], default="kmeans", help="Clustering method")
        parser.add_argument("--k_min", type=int, default=2, help="Minimum k for auto clustering")
        parser.add_argument("--k_max", type=int, default=6, help="Maximum k for auto clustering")
        parser.add_argument("--min_sep_deg", type=float, default=12.0, help="Minimum separation in degrees")
        parser.add_argument("--angle_eps", type=float, default=15.0, help="Angle epsilon for DBSCAN")
        parser.add_argument("--min_samples", type=int, default=5, help="Minimum samples for DBSCAN")
        parser.add_argument("--seed", type=int, default=0, help="Random seed")
        parser.add_argument("--plot_intercepts", action="store_true", default=True, help="Plot decision intercepts")
        parser.add_argument("--show_paths", action="store_true", default=True, help="Show trajectory paths")
        parser.add_argument("--outlier_frac", type=float, default=0.05, help="Outlier fraction threshold")
        parser.add_argument("--outlier_min", type=int, default=3, help="Minimum outlier count")
        parser.add_argument("--plot_outliers", action="store_true", help="Plot outlier trajectories")
        parser.add_argument("--plot_noenter_paths", action="store_true", help="Plot no-entry paths")
        parser.add_argument("--legend_noenter_as_line", action="store_true", help="Legend style for no-entry")
        parser.add_argument("--include_noenter_in_assignments", action="store_true", help="Include no-entry in assignments")

    def execute(self, args: argparse.Namespace) -> None:
        self._create_output_dir(args.out)

        with self.logger.operation("Loading trajectories"):
            trajectories = load_folder(
                args.input, args.glob,
                columns=args.columns,
                require_time=False,
                scale=args.scale,
                motion_threshold=args.motion_threshold
            )

        if len(trajectories) == 0:
            self.logger.error("No trajectories loaded. Check your input path, file pattern, and column mappings.")
            return

        junction = Circle(cx=float(args.junction[0]), cz=float(args.junction[1]), r=float(args.radius))

        with self.logger.operation("Discovering branches"):
            assignments, summary, centers = discover_branches(
                trajectories, junction,
                k=int(args.k),
                path_length=float(args.distance),
                epsilon=float(args.epsilon),
                seed=int(args.seed),
                decision_mode=args.decision_mode,
                r_outer=float(args.r_outer) if args.r_outer is not None else None,
                linger_delta=float(args.linger_delta),
                out_dir=args.out,
                cluster_method=args.cluster_method,
                k_min=int(args.k_min),
                k_max=int(args.k_max),
                min_sep_deg=float(args.min_sep_deg),
                angle_eps=float(args.angle_eps),
                min_samples=int(args.min_samples),
                junction_number=0,  # CLI discover command is always for junction 0
                all_junctions=[junction]
            )

        with self.logger.operation("Processing assignments"):
            self._process_assignments(assignments, centers, args)

        with self.logger.operation("Generating plots"):
            self._generate_plots(trajectories, assignments, centers, junction, args)

        self._save_run_args(args, args.out)
        self.logger.info(f"Discovery completed. Results saved to {args.out}")

    def _process_assignments(self, assignments: pd.DataFrame, centers: np.ndarray, args: argparse.Namespace) -> None:
        """Process and save assignment results"""
        # Save all assignments
        save_assignments(assignments, os.path.join(args.out, "branch_assignments_main.csv"))

        # Create summary
        summary_all = (assignments["branch"]
                    .value_counts()
                    .sort_index()
                    .rename_axis("branch")
                    .to_frame("count"))
        summary_all["percent"] = summary_all["count"] / max(1, int(summary_all["count"].sum())) * 100.0
        save_summary(summary_all.reset_index(), os.path.join(args.out, "branch_summary_all.csv"), with_entropy=True)

        # Split small branches
        min_needed = max(int(np.ceil(float(args.outlier_frac) * len(assignments))), int(args.outlier_min))
        main_assign, minor_assign, counts = split_small_branches(assignments, min_frac=float(args.outlier_frac))

        if len(minor_assign):
            small_branches_abs = set(counts[counts < min_needed].index)
            if small_branches_abs:
                keep_mask = ~main_assign["branch"].isin(small_branches_abs)
                extra_minor = main_assign[~keep_mask]
                main_assign = main_assign[keep_mask]
                minor_assign = pd.concat([minor_assign, extra_minor], ignore_index=True)

        # Save main assignments
        save_assignments(main_assign, os.path.join(args.out, "branch_assignments.csv"))

        # Include no-entry if requested
        if args.include_noenter_in_assignments:
            all_path = os.path.join(args.out, "branch_assignments_all.csv")
            if os.path.exists(all_path):
                df_all = pd.read_csv(all_path)
                noenter = df_all[df_all["branch"] == -2]
                combined = pd.concat([main_assign, noenter], ignore_index=True)
                save_assignments(combined, os.path.join(args.out, "branch_assignments.csv"))

        # Create main summary
        summary_main = (main_assign["branch"]
                        .value_counts()
                        .sort_index()
                        .rename_axis("branch")
                        .to_frame("count"))
        summary_main["percent"] = summary_main["count"] / max(1, int(summary_main["count"].sum())) * 100.0
        save_summary(summary_main.reset_index(), os.path.join(args.out, "branch_summary.csv"), with_entropy=True)

        # Log outlier info
        if len(minor_assign):
            self.logger.info(f"Flagged outlier branches: {len(minor_assign)} trajectories "
                            f"(threshold = max({args.outlier_frac*100:.1f}% of N, {args.outlier_min}))")
        else:
            self.logger.info("No outlier branches flagged")

        # Save centers
        save_centers(centers, os.path.join(args.out, "branch_centers.npy"))
        save_centers_json(centers, os.path.join(args.out, "branch_centers.json"))

    def _generate_plots(self, trajectories, assignments, centers, junction, args):
        """Generate visualization plots"""
        main_assignments = pd.read_csv(os.path.join(args.out, "branch_assignments.csv"))

        # Branch directions plot (optional - function may not exist)
        try:
            from .verta_plotting import plot_branch_directions
            plot_branch_directions(centers, (junction.cx, junction.cz),
                                 os.path.join(args.out, "Branch_Directions.png"))
        except (ImportError, AttributeError):
            self.logger.warning("plot_branch_directions not available, skipping")

        # Branch counts plot (optional - function may not exist)
        try:
            from .verta_plotting import plot_branch_counts
            plot_branch_counts(main_assignments, os.path.join(args.out, "Branch_Counts.png"))
        except (ImportError, AttributeError):
            self.logger.warning("plot_branch_counts not available, skipping")

        # Decision intercepts plot
        if args.plot_intercepts:
            try:
                assign_all_path = os.path.join(args.out, "branch_assignments_all.csv")
                if args.plot_outliers and os.path.exists(assign_all_path):
                    assign_for_plot = pd.read_csv(assign_all_path)
                else:
                    assign_for_plot = main_assignments

                mode_log_path = os.path.join(args.out, "decision_mode_used.csv")
                mode_log_df = pd.read_csv(mode_log_path) if os.path.exists(mode_log_path) else None

                # Load decision points data for plotting
                decision_points_df = None
                try:
                    decision_points_path = os.path.join(args.out, "decision_points.csv")
                    if os.path.exists(decision_points_path):
                        decision_points_df = pd.read_csv(decision_points_path)
                except Exception:
                    pass

                plot_decision_intercepts(
                    trajectories=trajectories,
                    assignments_df=assign_for_plot,
                    mode_log_df=mode_log_df,
                    centers=centers,
                    junction=junction,
                    r_outer=float(args.r_outer) if args.r_outer is not None else None,
                    path_length=float(args.distance),
                    epsilon=float(args.epsilon),
                    linger_delta=float(args.linger_delta),
                    out_path=os.path.join(args.out, "Decision_Intercepts.png"),
                    show_paths=False,
                    legend_noenter_as_line=args.legend_noenter_as_line,
                    decision_points_df=decision_points_df
                )
                self.logger.info("Decision intercepts plot generated")
            except Exception as e:
                self.logger.error(f"Intercept plot failed: {e}")

        # Decision map plot (optional - function may not exist)
        try:
            from .verta_plotting import plot_discover_map
            plot_discover_map(
                trajectories=trajectories,
                assignments_df=main_assignments,
                junction=junction,
                centers=centers,
                r_outer=float(args.r_outer) if args.r_outer is not None else None,
                out_path=os.path.join(args.out, "Decision_Map.png")
            )
            self.logger.info("Decision map plot generated")
        except (ImportError, AttributeError, NameError) as e:
            self.logger.warning(f"plot_discover_map not available, skipping: {e}")


class AssignCommand(BaseCommand):
    """Command handler for branch assignment using precomputed centers"""

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--input", required=True, help="Input folder path")
        parser.add_argument("--out", required=True, help="Output folder path")
        parser.add_argument("--glob", default="*.csv", help="File pattern")
        parser.add_argument("--columns", default=None, help="Column mapping")
        parser.add_argument("--scale", type=float, default=1.0, help="Coordinate scaling factor")
        parser.add_argument("--motion_threshold", type=float, default=0.001, help="Motion detection threshold")
        parser.add_argument("--junction", nargs=2, type=float, required=True, metavar=("X", "Z"), help="Junction coordinates")
        parser.add_argument("--radius", type=float, required=True, help="Junction radius")
        parser.add_argument("--distance", type=float, default=100.0, help="Path length for decision")
        parser.add_argument("--epsilon", type=float, default=0.015, help="Minimum step size")
        parser.add_argument("--decision_mode", choices=["pathlen", "radial", "hybrid"], default="pathlen", help="Decision mode")
        parser.add_argument("--r_outer", type=float, default=None, help="Outer radius for radial mode")
        parser.add_argument("--linger_delta", type=float, default=5.0, help="Linger distance beyond junction")
        parser.add_argument("--centers", required=True, help="Path to precomputed centers")

    def execute(self, args: argparse.Namespace) -> None:
        self._create_output_dir(args.out)

        with self.logger.operation("Loading trajectories"):
            trajectories = load_folder(
                args.input, args.glob,
                columns=args.columns,
                require_time=False,
                scale=args.scale,
                motion_threshold=args.motion_threshold
            )

        if len(trajectories) == 0:
            self.logger.error("No trajectories loaded. Check your input path, file pattern, and column mappings.")
            return

        junction = Circle(cx=float(args.junction[0]), cz=float(args.junction[1]), r=float(args.radius))
        centers = np.load(args.centers)

        with self.logger.operation("Assigning branches"):
            assignments = assign_branches(
                trajectories, centers, junction,
                path_length=float(args.distance),
                epsilon=float(args.epsilon),
                decision_mode=args.decision_mode,
                r_outer=float(args.r_outer) if args.r_outer is not None else None,
                linger_delta=float(args.linger_delta),
                out_dir=args.out
            )
        # Normalize/validate and add branch_j0 for single-junction compatibility
        try:
            validate_trajectories_unique(trajectories)
        except Exception:
            pass
        if "branch_j0" not in assignments.columns:
            assignments = assignments.copy()
            assignments["branch_j0"] = assignments["branch"]

        # Consistency warnings (after enriching columns)
        try:
            from .verta_consistency import validate_consistency
            validate_consistency(assignments, trajectories, [junction])
        except Exception:
            pass

        save_assignments(assignments, os.path.join(args.out, "branch_assignments.csv"))
        self._save_run_args(args, args.out)
        self.logger.info(f"Assignment completed. Results saved to {args.out}")


class MetricsCommand(BaseCommand):
    """Command handler for timing metrics computation"""

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--input", required=True, help="Input folder path")
        parser.add_argument("--out", required=True, help="Output folder path")
        parser.add_argument("--glob", default="*.csv", help="File pattern")
        parser.add_argument("--columns", default=None, help="Column mapping")
        parser.add_argument("--scale", type=float, default=1.0, help="Coordinate scaling factor")
        parser.add_argument("--motion_threshold", type=float, default=0.001, help="Motion detection threshold")
        parser.add_argument("--junction", nargs=2, type=float, required=True, metavar=("X", "Z"), help="Junction coordinates")
        parser.add_argument("--radius", type=float, required=True, help="Junction radius")
        parser.add_argument("--distance", type=float, default=100.0, help="Path length for decision")
        parser.add_argument("--decision_mode", choices=["pathlen", "radial", "hybrid"], default="pathlen", help="Decision mode")
        parser.add_argument("--r_outer", type=float, default=None, help="Outer radius for radial mode")
        parser.add_argument("--trend_window", type=int, default=5, help="Trend window for radial mode")
        parser.add_argument("--min_outward", type=float, default=0.0, help="Minimum outward movement")
        parser.add_argument("--linger_delta", type=float, default=5.0, help="Linger distance beyond junction")
        parser.add_argument("--epsilon", type=float, default=0.015, help="Minimum step size")
        parser.add_argument("--regions", default=None, help="JSON regions specification")

    def execute(self, args: argparse.Namespace) -> None:
        self._create_output_dir(args.out)

        with self.logger.operation("Loading trajectories"):
            trajectories = load_folder(
                args.input, args.glob,
                columns=args.columns,
                require_time=True,
                scale=args.scale,
                motion_threshold=args.motion_threshold
            )

        if len(trajectories) == 0:
            self.logger.error("No trajectories loaded. Check your input path, file pattern, and column mappings.")
            return

        junction = Circle(cx=float(args.junction[0]), cz=float(args.junction[1]), r=float(args.radius))

        # Basic consistency check on loaded trajectories
        try:
            validate_trajectories_unique(trajectories)
        except Exception:
            pass

        with self.logger.operation("Computing timing and speed metrics"):
            rows = []
            for tr in trajectories:
                # Compute timing metrics
                t_val, mode_used = _timing_for_traj(
                    tr=tr,
                    junction=junction,
                    decision_mode=str(args.decision_mode),
                    distance=float(args.distance),
                    r_outer=float(args.r_outer) if args.r_outer is not None else None,
                    trend_window=int(args.trend_window),
                    min_outward=float(args.min_outward),
                )

                # Compute speed metrics
                speed_val, speed_mode = speed_through_junction(
                    tr=tr,
                    junction=junction,
                    decision_mode=str(args.decision_mode),
                    path_length=float(args.distance),
                    r_outer=float(args.r_outer) if args.r_outer is not None else None,
                    window=int(args.trend_window),
                    min_outward=float(args.min_outward),
                )

                # Compute junction transit speeds
                entry_speed, exit_speed, avg_transit_speed = junction_transit_speed(tr, junction)

                row = {
                    "trajectory": tr.tid,
                    "time_value": t_val,
                    "decision_mode_requested": str(args.decision_mode),
                    "decision_mode_used": mode_used,
                    "distance": float(args.distance) if mode_used == "pathlen" else None,
                    "r_outer": float(args.r_outer) if (mode_used == "radial" and args.r_outer is not None) else None,
                    "trend_window": int(args.trend_window) if mode_used == "radial" else None,
                    "min_outward": float(args.min_outward) if mode_used == "radial" else None,
                    # Speed analysis columns
                    "speed_through_junction": speed_val,
                    "speed_mode_used": speed_mode,
                    "entry_speed": entry_speed,
                    "exit_speed": exit_speed,
                    "average_transit_speed": avg_transit_speed,
                }

                if args.regions:
                    spec = json.loads(args.regions)
                    def parse_region(obj):
                        if "rect" in obj:
                            a,b,c,d = obj["rect"]
                            from .verta_geometry import Rect
                            return Rect(float(a), float(b), float(c), float(d))
                        if "circle" in obj:
                            a,b,r = obj["circle"]
                            return Circle(float(a), float(b), float(r))
                    A = parse_region(spec["A"]) if "A" in spec else None
                    B = parse_region(spec["B"]) if "B" in spec else None
                    if A is not None and B is not None:
                        tA, tB, dt = time_between_regions(tr, A, B)
                        row.update({"t_A": tA, "t_B": tB, "dt_AB": dt})

                rows.append(row)

            df = pd.DataFrame(rows)
            df.to_csv(os.path.join(args.out, "timing_and_speed_metrics.csv"), index=False)

        self._save_run_args(args, args.out)
        self.logger.info(f"Metrics computation completed. Results saved to {args.out}")


class GazeCommand(BaseCommand):
    """Command handler for gaze and physiological analysis"""

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--input", required=True, help="Input folder path")
        parser.add_argument("--out", required=True, help="Output folder path")
        parser.add_argument("--glob", default="*.csv", help="File pattern")
        parser.add_argument("--columns", default=None, help="Column mapping")
        parser.add_argument("--scale", type=float, default=1.0, help="Coordinate scaling factor")
        parser.add_argument("--motion_threshold", type=float, default=0.001, help="Motion detection threshold")

        junction_group = parser.add_mutually_exclusive_group(required=True)
        junction_group.add_argument("--junction", nargs=2, type=float, metavar=("X", "Z"), help="Single junction coordinates")
        junction_group.add_argument("--junctions", nargs="+", type=float, help="Multiple junction coordinates (x z r ...)")

        parser.add_argument("--radius", type=float, default=None, help="Junction radius")
        parser.add_argument("--r_outer", type=float, default=None, help="Outer radius for radial mode")
        parser.add_argument("--r_outer_list", nargs="*", type=float, default=None, help="Outer radii for each junction")
        parser.add_argument("--distance", type=float, default=100.0, help="Path length for decision")
        parser.add_argument("--epsilon", type=float, default=0.015, help="Minimum step size")
        parser.add_argument("--decision_mode", choices=["pathlen", "radial", "hybrid"], default="hybrid", help="Decision mode")
        parser.add_argument("--linger_delta", type=float, default=5.0, help="Linger distance beyond junction")
        parser.add_argument("--cluster_method", choices=["kmeans", "auto", "dbscan"], default="kmeans", help="Clustering method")
        parser.add_argument("--k", type=int, default=3, help="Number of clusters")
        parser.add_argument("--k_min", type=int, default=2, help="Minimum k for auto clustering")
        parser.add_argument("--k_max", type=int, default=6, help="Maximum k for auto clustering")
        parser.add_argument("--min_sep_deg", type=float, default=12.0, help="Minimum separation in degrees")
        parser.add_argument("--angle_eps", type=float, default=15.0, help="Angle epsilon for DBSCAN")
        parser.add_argument("--min_samples", type=int, default=5, help="Minimum samples for DBSCAN")
        parser.add_argument("--seed", type=int, default=0, help="Random seed")
        parser.add_argument("--centers", help="Path to pre-computed branch centers (.npy file)")
        parser.add_argument("--physio_window", type=float, default=3.0, help="Physiological analysis window")
        parser.add_argument("--plot_outliers", action="store_true", help="Include outlier branches in gaze plots (gray)")

    def execute(self, args: argparse.Namespace) -> None:
        self._create_output_dir(args.out)

        with self.logger.operation("Loading gaze trajectories"):
            gaze_trajectories = load_folder_with_gaze(
                args.input, args.glob,
                columns=args.columns,
                require_time=True,
                scale=args.scale,
                motion_threshold=args.motion_threshold
            )

        if len(gaze_trajectories) == 0:
            self.logger.error("No gaze trajectories loaded. Check your input path, file pattern, and column mappings.")
            return

        # Parse junctions
        if hasattr(args, 'junction') and args.junction is not None:
            junctions = [Circle(cx=float(args.junction[0]), cz=float(args.junction[1]), r=float(args.radius))]
            rlist = [float(args.r_outer)] if args.r_outer is not None else None
        else:
            vals = list(map(float, args.junctions))
            if len(vals) % 3 != 0:
                raise ValueError("--junctions must be triples: x z r ...")
            triples = [vals[i:i+3] for i in range(0, len(vals), 3)]
            junctions = [Circle(cx=a, cz=b, r=c) for a, b, c in triples]
            rlist = list(args.r_outer_list) if args.r_outer_list is not None and len(args.r_outer_list) > 0 else None

        with self.logger.operation("Discovering branches for gaze analysis"):
            chain_df, centers_list = discover_decision_chain(
                trajectories=gaze_trajectories,
                junctions=junctions,
                path_length=float(args.distance),
                epsilon=float(args.epsilon),
                seed=int(args.seed),
                decision_mode=str(args.decision_mode),
                r_outer_list=rlist,
                linger_delta=float(args.linger_delta),
                out_dir=str(args.out),
                cluster_method=str(args.cluster_method),
                k=int(args.k),
                k_min=int(args.k_min),
                k_max=int(args.k_max),
                min_sep_deg=float(args.min_sep_deg),
                angle_eps=float(args.angle_eps),
                min_samples=int(args.min_samples),
            )

        with self.logger.operation("Computing gaze analysis"):
            # Normalize assignments and merge decisions when available
            decisions_path = os.path.join(str(args.out), "branch_decisions_chain.csv")
            decisions_df = pd.read_csv(decisions_path) if os.path.exists(decisions_path) else None
            norm_df, report = normalize_assignments(
                chain_df,
                trajectories=gaze_trajectories,
                junctions=junctions,
                decisions_df=decisions_df,
                prefer_decisions=True,
                include_outliers=False,
            )
            self.logger.info(f"Assignments normalized: in={int(report['input_rows'])} kept={int(report['kept_after_tid_map'])} dropped={int(report['dropped_unmapped_ids'])} decisions={'yes' if report['has_decisions'] else 'no'}")
            # Extra consistency warnings
            try:
                from .verta_consistency import validate_consistency
                validate_consistency(norm_df, gaze_trajectories, junctions)
            except Exception:
                pass

            gaze_df = compute_head_yaw_at_decisions(
                trajectories=gaze_trajectories,
                junctions=junctions,
                assignments_df=norm_df,
                decision_mode=str(args.decision_mode),
                r_outer_list=rlist,
                path_length=float(args.distance),
                epsilon=float(args.epsilon),
                linger_delta=float(args.linger_delta),
            )

        with self.logger.operation("Computing physiological analysis"):
            physio_df = analyze_physiological_at_junctions(
                trajectories=gaze_trajectories,
                junctions=junctions,
                assignments_df=chain_df,
                decision_mode=str(args.decision_mode),
                r_outer_list=rlist,
                path_length=float(args.distance),
                epsilon=float(args.epsilon),
                linger_delta=float(args.linger_delta),
                physio_window=float(args.physio_window),
            )

        # Save results
        gaze_df.to_csv(os.path.join(args.out, "gaze_analysis.csv"), index=False)
        physio_df.to_csv(os.path.join(args.out, "physiological_analysis.csv"), index=False)

        # Generate consistency report
        consistency = gaze_movement_consistency_report(gaze_df)
        with open(os.path.join(args.out, "gaze_consistency_report.json"), "w") as f:
            json.dump(consistency, f, indent=2)

        with self.logger.operation("Generating gaze plots"):
            self._generate_gaze_plots(gaze_trajectories, junctions, gaze_df, physio_df, chain_df, rlist, args)

        self._save_run_args(args, args.out)
        self.logger.info(f"Gaze analysis completed. Results saved to {args.out}")
        self.logger.info(f"Found {len(gaze_df)} valid gaze-decision pairs")
        if "mean_absolute_yaw_difference" in consistency:
            self.logger.info(f"Mean head-movement alignment: {consistency['mean_absolute_yaw_difference']:.1f}°")
            self.logger.info(f"Well-aligned decisions: {consistency['aligned_percentage']:.1f}%")

    def _generate_gaze_plots(self, gaze_trajectories, junctions, gaze_df, physio_df, chain_df, rlist, args):
        """Generate gaze visualization plots"""
        try:
            plot_gaze_directions_at_junctions(
                trajectories=gaze_trajectories,
                junctions=junctions,
                gaze_df=gaze_df,
                out_path=os.path.join(args.out, "Gaze_Directions.png"),
                r_outer_list=rlist,
                junction_labels=[f"Junction {i}" for i in range(len(junctions))],
                centers_list=None,  # Optional parameter - not available in this context
            )
            self.logger.info("Gaze directions plot generated")
        except Exception as e:
            self.logger.error(f"Gaze directions plot failed: {e}")

        try:
            plot_physiological_by_branch(
                physio_df=physio_df,
                out_path=os.path.join(args.out, "Physiological_Analysis.png"),
            )
            self.logger.info("Physiological analysis plot generated")
        except Exception as e:
            self.logger.error(f"Physiological analysis plot failed: {e}")

        # Pupil trajectory analysis
        pupil_traj_df = analyze_pupil_dilation_trajectory(
            trajectories=gaze_trajectories,
            junctions=junctions,
            assignments_df=chain_df,
            decision_mode=str(args.decision_mode),
            r_outer_list=rlist,
            path_length=float(args.distance),
            epsilon=float(args.epsilon),
            linger_delta=float(args.linger_delta),
        )

        pupil_traj_df.to_csv(os.path.join(args.out, "pupil_trajectory_analysis.csv"), index=False)

        try:
            plot_pupil_trajectory_analysis(
                pupil_traj_df=pupil_traj_df,
                out_path=os.path.join(args.out, "Pupil_Trajectory_Analysis.png"),
            )
            self.logger.info("Pupil trajectory analysis plot generated")
        except Exception as e:
            self.logger.error(f"Pupil trajectory plot failed: {e}")


class PredictCommand(BaseCommand):
    """Command handler for junction-based choice prediction analysis"""

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--input", required=True, help="Input folder path")
        parser.add_argument("--out", required=True, help="Output folder path")
        parser.add_argument("--glob", default="*.csv", help="File pattern")
        parser.add_argument("--columns", default=None, help="Column mapping")
        parser.add_argument("--scale", type=float, default=1.0, help="Coordinate scaling factor")
        parser.add_argument("--motion_threshold", type=float, default=0.001, help="Motion detection threshold")
        parser.add_argument("--junctions", nargs="+", type=float, required=True, help="Junction coordinates (x z r ...)")
        parser.add_argument("--r_outer_list", nargs="*", type=float, default=None, help="Outer radii for each junction")
        parser.add_argument("--distance", type=float, default=100.0, help="Path length for decision")
        parser.add_argument("--epsilon", type=float, default=0.015, help="Minimum step size")
        parser.add_argument("--k", type=int, default=3, help="Number of clusters")
        parser.add_argument("--decision_mode", choices=["pathlen", "radial", "hybrid"], default="hybrid", help="Decision mode")
        parser.add_argument("--linger_delta", type=float, default=5.0, help="Linger distance beyond junction")
        parser.add_argument("--cluster_method", choices=["kmeans", "auto", "dbscan"], default="kmeans", help="Clustering method")
        parser.add_argument("--k_min", type=int, default=2, help="Minimum k for auto clustering")
        parser.add_argument("--k_max", type=int, default=6, help="Maximum k for auto clustering")
        parser.add_argument("--min_sep_deg", type=float, default=12.0, help="Minimum separation in degrees")
        parser.add_argument("--angle_eps", type=float, default=15.0, help="Angle epsilon for DBSCAN")
        parser.add_argument("--min_samples", type=int, default=5, help="Minimum samples for DBSCAN")
        parser.add_argument("--seed", type=int, default=0, help="Random seed")

        # Prediction-specific arguments
        parser.add_argument("--min_pattern_samples", type=int, default=3, help="Minimum samples for pattern recognition")
        parser.add_argument("--pattern_threshold", type=float, default=0.3, help="Minimum probability threshold for patterns")
        parser.add_argument("--confidence_threshold", type=float, default=0.5, help="Minimum confidence for predictions")
        parser.add_argument("--analyze_sequences", action="store_true", help="Analyze complete route sequences")
        parser.add_argument("--predict_examples", type=int, default=10, help="Number of prediction examples to generate")

    def execute(self, args: argparse.Namespace) -> None:
        """Execute the prediction analysis"""
        logger = get_logger()
        logger.info("Starting junction-based choice prediction analysis...")

        # Create output directory
        os.makedirs(args.out, exist_ok=True)

        # Load trajectories
        logger.info(f"Loading trajectories from {args.input}")
        trajectories = load_folder(
            folder=args.input,
            pattern=args.glob,
            columns=args.columns,
            scale=args.scale,
            motion_threshold=args.motion_threshold
        )

        if not trajectories:
            logger.error("No trajectories loaded!")
            return

        logger.info(f"Loaded {len(trajectories)} trajectories")

        # Parse junctions
        junctions = self._parse_junctions(args.junctions)
        logger.info(f"Analyzing {len(junctions)} junctions")

        # Discover decision chain
        logger.info("Discovering decision chains...")
        chain_df, branch_centers_list = discover_decision_chain(
            trajectories=trajectories,
            junctions=junctions,
            r_outer_list=args.r_outer_list,
            path_length=args.distance,
            epsilon=args.epsilon,
            k=args.k,
            decision_mode=args.decision_mode,
            linger_delta=args.linger_delta,
            cluster_method=args.cluster_method,
            k_min=args.k_min,
            k_max=args.k_max,
            min_sep_deg=args.min_sep_deg,
            angle_eps=args.angle_eps,
            min_samples=args.min_samples,
            seed=args.seed,
            out_dir=args.out
        )

        if chain_df.empty:
            logger.error("No decision chains discovered!")
            return

        logger.info(f"Discovered decision chains for {len(chain_df)} trajectories")

        # Normalize chain for downstream prediction (ID dtype, columns)
        norm_df, _rep = normalize_assignments(
            chain_df,
            trajectories=trajectories,
            junctions=junctions,
            prefer_decisions=False,
            include_outliers=False,
        )
        # Consistency warnings (optional)
        try:
            from verta.verta_consistency import validate_consistency
            validate_consistency(norm_df, trajectories, junctions)
        except Exception:
            pass
        # Save normalized chain
        norm_df.to_csv(os.path.join(args.out, "decision_chains.csv"), index=False)

        # Run prediction analysis
        logger.info("Analyzing junction choice patterns...")
        analysis_results = analyze_junction_choice_patterns(
            trajectories=trajectories,
            chain_df=norm_df,
            junctions=junctions,
            output_dir=args.out,
            r_outer_list=args.r_outer_list,
            gui_mode=False  # Terminal mode
        )

        # Generate additional analysis if requested
        if args.analyze_sequences:
            self._analyze_route_sequences(norm_df, junctions, args.out)

        # Generate prediction examples
        if args.predict_examples > 0:
            self._generate_prediction_examples(trajectories, norm_df, junctions, args.out, args.predict_examples)

        # Save summary
        self._save_analysis_summary(analysis_results, args.out)

        logger.info(f"Prediction analysis complete! Results saved to {args.out}")

    def _parse_junctions(self, junction_args: List[float]) -> List[Circle]:
        """Parse junction arguments into Circle objects"""
        if len(junction_args) % 3 != 0:
            raise ValueError("Junctions must be specified as x z r triplets")

        junctions = []
        for i in range(0, len(junction_args), 3):
            x, z, r = junction_args[i:i+3]
            junctions.append(Circle(cx=x, cz=z, r=r))

        return junctions

    def _analyze_route_sequences(self, chain_df: pd.DataFrame, junctions: List[Circle], output_dir: str):
        """Analyze complete route sequences for behavioral patterns"""
        logger = get_logger()
        logger.info("Analyzing route sequences...")

        # Extract sequences
        sequences = []
        for _, row in chain_df.iterrows():
            sequence = []
            for i in range(len(junctions)):
                branch_col = f"branch_j{i}"
                if branch_col in row and pd.notna(row[branch_col]):
                    sequence.append(int(row[branch_col]))
                else:
                    break

            if len(sequence) > 1:
                sequences.append(sequence)

        # Analyze sequence patterns
        sequence_analysis = {
            'total_sequences': len(sequences),
            'sequence_lengths': [len(seq) for seq in sequences],
            'common_patterns': self._find_common_sequence_patterns(sequences),
            'sequence_diversity': len(set(tuple(seq) for seq in sequences)),
            'average_length': np.mean([len(seq) for seq in sequences]) if sequences else 0
        }

        # Save sequence analysis
        with open(os.path.join(output_dir, "sequence_analysis.json"), "w") as f:
            json.dump(sequence_analysis, f, indent=2)

        logger.info(f"Found {len(sequences)} route sequences")

    def _find_common_sequence_patterns(self, sequences: List[List[int]]) -> List[Dict[str, Any]]:
        """Find common patterns in route sequences"""
        # Count sequence patterns
        sequence_counts = Counter(tuple(seq) for seq in sequences)

        # Get most common patterns
        common_patterns = []
        for pattern, count in sequence_counts.most_common(10):
            if count > 1:  # Only patterns that occur more than once
                common_patterns.append({
                    'pattern': list(pattern),
                    'count': count,
                    'frequency': count / len(sequences)
                })

        return common_patterns

    def _generate_prediction_examples(self, trajectories: List[Trajectory], chain_df: pd.DataFrame, junctions: List[Circle],
                                    output_dir: str, num_examples: int):
        """Generate prediction examples for sample trajectories"""
        logger = get_logger()
        logger.info(f"Generating {num_examples} prediction examples...")

        # Initialize analyzer
        analyzer = JunctionChoiceAnalyzer(trajectories, chain_df, junctions)

        # Get sample trajectories
        sample_trajectories = chain_df.head(num_examples)

        predictions = []
        for _, row in sample_trajectories.iterrows():
            trajectory_id = row['trajectory']

            # Find first junction with a valid branch
            for i in range(len(junctions)):
                branch_col = f"branch_j{i}"
                if branch_col in row and pd.notna(row[branch_col]):
                    current_junction = i
                    current_branch = int(row[branch_col])

                    # Make prediction
                    prediction = analyzer.predict_next_choice(trajectory_id, current_junction, current_branch)
                    predictions.append(prediction)
                    break

        # Save prediction examples
        prediction_data = []
        for pred in predictions:
            prediction_data.append({
                'trajectory_id': pred.trajectory_id,
                'current_junction': pred.current_junction,
                'current_branch': pred.current_branch,
                'predicted_next_junction': pred.predicted_next_junction,
                'predicted_next_branch': pred.predicted_next_branch,
                'confidence': pred.confidence,
                'pattern_used': pred.pattern_used,
                'alternatives': [
                    {'junction': j, 'branch': b, 'probability': p}
                    for j, b, p in pred.alternative_predictions
                ]
            })

        with open(os.path.join(output_dir, "prediction_examples.json"), "w") as f:
            json.dump(prediction_data, f, indent=2)

        logger.info(f"Generated {len(predictions)} prediction examples")

    def _save_analysis_summary(self, analysis_results: Dict[str, Any], output_dir: str):
        """Save a summary of the analysis results"""
        summary = {
            'analysis_type': 'Junction-Based Choice Prediction',
            'summary': analysis_results['summary'],
            'key_findings': {
                'total_patterns': analysis_results['summary']['unique_patterns'],
                'pattern_distribution': analysis_results['pattern_types'],
                'top_pattern': analysis_results['top_patterns'][0] if analysis_results['top_patterns'] else None
            },
            'recommendations': self._generate_recommendations(analysis_results)
        }

        with open(os.path.join(output_dir, "analysis_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    def _generate_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis results"""
        recommendations = []

        # Check for strong patterns
        preferred_patterns = [p for p in analysis_results['top_patterns'] if p['pattern_type'] == 'preferred']
        if preferred_patterns:
            recommendations.append(f"Found {len(preferred_patterns)} strong behavioral patterns that could be used for route prediction")

        # Check for junction-specific insights
        junction_analysis = analysis_results['junction_analysis']
        for junction_idx, analysis in junction_analysis.items():
            if analysis['pattern_diversity'] == 1:
                recommendations.append(f"Junction {junction_idx} shows very predictable behavior - consider this for traffic optimization")
            elif analysis['pattern_diversity'] > 3:
                recommendations.append(f"Junction {junction_idx} shows high variability - may need better signage or design")

        # Check for learning opportunities
        learned_patterns = [p for p in analysis_results['top_patterns'] if p['pattern_type'] == 'learned']
        if learned_patterns:
            recommendations.append(f"Found {len(learned_patterns)} learned patterns - participants are adapting to the environment")

        return recommendations


class IntentRecognitionCommand(BaseCommand):
    """Command handler for Intent Recognition (ML-based route prediction)"""

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--input", required=True, help="Input folder path")
        parser.add_argument("--out", required=True, help="Output folder path")
        parser.add_argument("--glob", default="*.csv", help="File pattern")
        parser.add_argument("--columns", default=None, help="Column mapping")
        parser.add_argument("--scale", type=float, default=1.0, help="Coordinate scaling factor")
        parser.add_argument("--motion_threshold", type=float, default=0.001, help="Motion detection threshold")

        # Junction specification
        junction_group = parser.add_mutually_exclusive_group(required=True)
        junction_group.add_argument("--junction", nargs=3, type=float, metavar=("X", "Z", "R"),
                                   help="Single junction (x z r)")
        junction_group.add_argument("--junctions", nargs="+", type=float,
                                   help="Multiple junctions (x z r x z r ...)")

        # Branch discovery/assignment
        parser.add_argument("--distance", type=float, default=100.0, help="Path length for decision")
        parser.add_argument("--epsilon", type=float, default=0.015, help="Minimum step size")
        parser.add_argument("--k", type=int, default=3, help="Number of clusters")
        parser.add_argument("--decision_mode", choices=["pathlen", "radial", "hybrid"], default="hybrid",
                           help="Decision mode")
        parser.add_argument("--linger_delta", type=float, default=5.0, help="Linger distance beyond junction")
        parser.add_argument("--cluster_method", choices=["kmeans", "auto", "dbscan"], default="kmeans",
                           help="Clustering method")
        parser.add_argument("--k_min", type=int, default=2, help="Minimum k for auto clustering")
        parser.add_argument("--k_max", type=int, default=6, help="Maximum k for auto clustering")
        parser.add_argument("--min_sep_deg", type=float, default=12.0, help="Minimum separation in degrees")
        parser.add_argument("--angle_eps", type=float, default=15.0, help="Angle epsilon for DBSCAN")
        parser.add_argument("--min_samples", type=int, default=5, help="Minimum samples for DBSCAN")
        parser.add_argument("--seed", type=int, default=0, help="Random seed")
        parser.add_argument("--centers", help="Path to pre-computed branch centers (.npy file)")
        parser.add_argument("--assignments", help="Path to pre-computed branch assignments CSV file")

        # Intent recognition specific
        parser.add_argument("--prediction_distances", nargs="+", type=float,
                           default=[100.0, 75.0, 50.0, 25.0],
                           help="Distances before junction to make predictions (units)")
        parser.add_argument("--model_type", choices=["random_forest", "gradient_boosting"],
                           default="random_forest", help="ML model type")
        parser.add_argument("--cv_folds", type=int, default=5, help="Cross-validation folds")
        parser.add_argument("--test_split", type=float, default=0.2, help="Test set fraction")

        # Gaze/physiological data (optional)
        parser.add_argument("--with_gaze", action="store_true",
                           help="Load gaze and physiological data if available")

    def execute(self, args: argparse.Namespace) -> None:
        """Execute the intent recognition analysis"""
        logger = get_logger()
        logger.info("Starting Intent Recognition analysis...")

        # Check if scikit-learn is available
        try:
            import sklearn
        except ImportError:
            logger.error("scikit-learn is required for Intent Recognition. Install with: pip install scikit-learn")
            return

        # Create output directory
        os.makedirs(args.out, exist_ok=True)

        # Load trajectories
        logger.info(f"Loading trajectories from {args.input}")
        if args.with_gaze:
            trajectories = load_folder_with_gaze(
                folder=args.input,
                pattern=args.glob,
                columns=args.columns,
                scale=args.scale,
                motion_threshold=args.motion_threshold
            )
        else:
            trajectories = load_folder(
                folder=args.input,
                pattern=args.glob,
                columns=args.columns,
                scale=args.scale,
                motion_threshold=args.motion_threshold
            )

        if not trajectories:
            logger.error("No trajectories loaded!")
            return

        logger.info(f"Loaded {len(trajectories)} trajectories")

        # Parse junctions
        junctions = self._parse_junctions(args)
        logger.info(f"Analyzing {len(junctions)} junction(s)")

        # Process each junction
        all_results = {}
        summary_data = []

        for junction_idx, junction in enumerate(junctions):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing Junction {junction_idx}")
            logger.info(f"{'='*60}")

            junction_output = os.path.join(args.out, f"junction_{junction_idx}")
            os.makedirs(junction_output, exist_ok=True)

            # Get branch assignments
            assignments_df = self._get_branch_assignments(
                trajectories, junction, junction_idx, args, junction_output
            )

            if assignments_df is None or assignments_df.empty:
                logger.warning(f"Junction {junction_idx}: No valid branch assignments found. Skipping.")
                continue

            # Filter valid assignments
            valid_assignments = assignments_df[assignments_df['branch'] >= 0]
            if len(valid_assignments) < 10:
                logger.warning(f"Junction {junction_idx}: Insufficient valid trajectories ({len(valid_assignments)} < 10). Skipping.")
                continue

            logger.info(f"Junction {junction_idx}: Found {len(valid_assignments)} valid trajectories")

            # Run intent recognition
            logger.info(f"Training intent recognition models for Junction {junction_idx}...")
            results = analyze_intent_recognition(
                trajectories=trajectories,
                junction=junction,
                actual_branches=valid_assignments,
                output_dir=junction_output,
                prediction_distances=args.prediction_distances,
                previous_choices=None  # Could be extended for multi-junction support
            )

            if 'error' in results:
                logger.error(f"Junction {junction_idx} failed: {results['error']}")
                all_results[f"junction_{junction_idx}"] = results
                continue

            # Extract summary statistics
            training_results = results.get('training_results', {})
            models_trained = training_results.get('models_trained', {})

            for dist, model_info in models_trained.items():
                summary_data.append({
                    'junction': f"J{junction_idx}",
                    'distance': dist,
                    'accuracy': model_info.get('cv_mean_accuracy', 0.0) * 100,
                    'std_dev': model_info.get('cv_std_accuracy', 0.0) * 100,
                    'samples': model_info.get('n_samples', 0)
                })

            all_results[f"junction_{junction_idx}"] = results
            logger.info(f"Junction {junction_idx}: Intent recognition complete!")

        # Save overall summary
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(os.path.join(args.out, "intent_recognition_summary.csv"), index=False)

            # Calculate average accuracy per junction
            junction_summary = summary_df.groupby('junction').agg({
                'accuracy': 'mean',
                'samples': 'first'
            }).reset_index()
            junction_summary.columns = ['Junction', 'Avg Accuracy (%)', 'Samples']
            junction_summary.to_csv(os.path.join(args.out, "intent_recognition_junction_summary.csv"), index=False)

            logger.info("\n" + "="*60)
            logger.info("Intent Recognition Summary")
            logger.info("="*60)
            logger.info(junction_summary.to_string(index=False))

        # Save run arguments
        self._save_run_args(args, args.out)

        # Save full results
        results_path = os.path.join(args.out, "intent_recognition_results.json")
        # Convert results to JSON-serializable format
        json_results = {}
        for key, value in all_results.items():
            if 'analyzer' in value:
                # Remove analyzer object (not JSON serializable)
                json_results[key] = {
                    'training_results': value.get('training_results', {}),
                    'test_predictions': value.get('test_predictions', {})
                }
            else:
                json_results[key] = value

        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)

        logger.info(f"\nIntent Recognition analysis complete! Results saved to {args.out}")
        logger.info(f"   - Models saved in: junction_*/models/")
        logger.info(f"   - Summary: intent_recognition_summary.csv")
        logger.info(f"   - Full results: intent_recognition_results.json")

    def _parse_junctions(self, args: argparse.Namespace) -> List[Circle]:
        """Parse junction arguments into Circle objects"""
        if args.junction:
            # Single junction: x z r
            x, z, r = args.junction
            return [Circle(cx=x, cz=z, r=r)]
        elif args.junctions:
            # Multiple junctions: x z r x z r ...
            if len(args.junctions) % 3 != 0:
                raise ValueError("Junctions must be specified as x z r triplets")

            junctions = []
            for i in range(0, len(args.junctions), 3):
                x, z, r = args.junctions[i:i+3]
                junctions.append(Circle(cx=x, cz=z, r=r))

            return junctions
        else:
            raise ValueError("Must specify either --junction or --junctions")

    def _get_branch_assignments(self, trajectories: List, junction: Circle,
                                junction_idx: int, args: argparse.Namespace,
                                output_dir: str) -> Optional[pd.DataFrame]:
        """Get branch assignments either from file or by discovery"""
        logger = get_logger()

        # Option 1: Load from assignments file
        if args.assignments:
            logger.info(f"Loading branch assignments from {args.assignments}")
            try:
                assignments_df = pd.read_csv(args.assignments)
                # Check if it has the right columns
                if 'trajectory' in assignments_df.columns and 'branch' in assignments_df.columns:
                    return assignments_df
                else:
                    logger.warning("Assignments file missing required columns. Discovering branches instead.")
            except Exception as e:
                logger.warning(f"Failed to load assignments file: {e}. Discovering branches instead.")

        # Option 2: Use pre-computed centers
        if args.centers:
            logger.info(f"Using pre-computed branch centers from {args.centers}")
            try:
                centers = np.load(args.centers)
                assignments_df = assign_branches(
                    trajectories=trajectories,
                    junction=junction,
                    centers=centers,
                    path_length=args.distance,
                    epsilon=args.epsilon,
                    decision_mode=args.decision_mode,
                    linger_delta=args.linger_delta
                )
                # Save assignments for future use
                assignments_df.to_csv(os.path.join(output_dir, "branch_assignments.csv"), index=False)
                return assignments_df
            except Exception as e:
                logger.warning(f"Failed to load centers: {e}. Discovering branches instead.")

        # Option 3: Discover branches
        logger.info(f"Discovering branches for Junction {junction_idx}...")
        try:
            centers, assignments_df = discover_branches(
                trajectories=trajectories,
                junction=junction,
                path_length=args.distance,
                epsilon=args.epsilon,
                k=args.k,
                decision_mode=args.decision_mode,
                linger_delta=args.linger_delta,
                cluster_method=args.cluster_method,
                k_min=args.k_min,
                k_max=args.k_max,
                min_sep_deg=args.min_sep_deg,
                angle_eps=args.angle_eps,
                min_samples=args.min_samples,
                seed=args.seed
            )

            # Save discovered centers and assignments
            np.save(os.path.join(output_dir, "branch_centers.npy"), centers)
            assignments_df.to_csv(os.path.join(output_dir, "branch_assignments.csv"), index=False)

            logger.info(f"Discovered {len(centers)} branches with {len(assignments_df[assignments_df['branch'] >= 0])} valid assignments")
            return assignments_df

        except Exception as e:
            logger.error(f"Failed to discover branches: {e}")
            return None


class EnhancedChainCommand(BaseCommand):
    """Enhanced command handler for multi-junction decision chain analysis with flow graph features"""

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--input", required=True, help="Input folder path")
        parser.add_argument("--out", required=True, help="Output folder path")
        parser.add_argument("--glob", default="*.csv", help="File pattern")
        parser.add_argument("--columns", default=None, help="Column mapping")
        parser.add_argument("--scale", type=float, default=1.0, help="Coordinate scaling factor")
        parser.add_argument("--motion_threshold", type=float, default=0.001, help="Motion detection threshold")
        parser.add_argument("--junctions", nargs="+", type=float, required=True, help="Junction coordinates (x z r ...)")
        parser.add_argument("--r_outer_list", nargs="*", type=float, default=None, help="Outer radii for each junction")
        parser.add_argument("--distance", type=float, default=100.0, help="Path length for decision")
        parser.add_argument("--epsilon", type=float, default=0.015, help="Minimum step size")
        parser.add_argument("--k", type=int, default=3, help="Number of clusters")
        parser.add_argument("--decision_mode", choices=["pathlen", "radial", "hybrid"], default="hybrid", help="Decision mode")
        parser.add_argument("--linger_delta", type=float, default=5.0, help="Linger distance beyond junction")
        parser.add_argument("--cluster_method", choices=["kmeans", "auto", "dbscan"], default="kmeans", help="Clustering method")
        parser.add_argument("--k_min", type=int, default=2, help="Minimum k for auto clustering")
        parser.add_argument("--k_max", type=int, default=6, help="Maximum k for auto clustering")
        parser.add_argument("--min_sep_deg", type=float, default=12.0, help="Minimum separation in degrees")
        parser.add_argument("--angle_eps", type=float, default=15.0, help="Angle epsilon for DBSCAN")
        parser.add_argument("--min_samples", type=int, default=5, help="Minimum samples for DBSCAN")
        parser.add_argument("--seed", type=int, default=0, help="Random seed")
        parser.add_argument("--small_multiples", action="store_true", help="Generate small multiples plot")
        parser.add_argument("--small_multiples_window", type=float, default=80.0, help="Small multiples window size")

        # Enhanced analysis flags
        parser.add_argument("--evacuation_analysis", action="store_true", help="Run enhanced flow analysis")
        parser.add_argument("--generate_recommendations", action="store_true", help="Generate flow recommendations")
        parser.add_argument("--risk_assessment", action="store_true", help="Generate risk assessment")
        parser.add_argument("--efficiency_metrics", action="store_true", help="Generate efficiency metrics")

        # Junction naming
        parser.add_argument("--junction_names", nargs="*", type=str, default=None, help="Names for junctions")

    def execute(self, args: argparse.Namespace) -> None:
        """Execute enhanced chain analysis"""
        self.logger.info("Starting Loading trajectories")

        # Load trajectories
        trajectories = load_folder(
            folder=args.input,
            pattern=args.glob,
            columns=args.columns,
            scale=args.scale,
            motion_threshold=args.motion_threshold
        )

        if len(trajectories) == 0:
            self.logger.warning("No trajectories loaded. Exiting.")
            return

        self.logger.info(f"Loaded {len(trajectories)} trajectories")

        # Parse junctions
        junctions = self._parse_junctions(args.junctions)

        # Parse r_outer_list
        rlist = args.r_outer_list or [None] * len(junctions)

        self.logger.info("Starting Discovering decision chain")

        # Discover decision chain
        chain_df, centers_list = discover_decision_chain(
            trajectories=trajectories,
            junctions=junctions,
            r_outer_list=rlist,
            path_length=args.distance,
            epsilon=args.epsilon,
            k=args.k,
            decision_mode=args.decision_mode,
            linger_delta=args.linger_delta,
            cluster_method=args.cluster_method,
            k_min=args.k_min,
            k_max=args.k_max,
            min_sep_deg=args.min_sep_deg,
            angle_eps=args.angle_eps,
            min_samples=args.min_samples,
            seed=args.seed,
            out_dir=args.out
        )

        self.logger.info("Completed Discovering decision chain")

        # Generate chain plots
        self.logger.info("Starting Generating chain plots")
        self._generate_chain_plots(trajectories, chain_df, junctions, rlist, args)
        self.logger.info("Completed Generating chain plots")

        # Run enhanced analysis if requested
        if args.evacuation_analysis:
            self.logger.info("Starting Running enhanced flow analysis")
            self._run_enhanced_analysis(chain_df, junctions, trajectories, args)
            self.logger.info("Completed Running enhanced flow analysis")

        self.logger.info("Enhanced chain analysis completed. Results saved to " + args.out)

    def _parse_junctions(self, junction_coords: list) -> list:
        """Parse junction coordinates into Circle objects"""
        if len(junction_coords) % 3 != 0:
            raise ValueError("Junction coordinates must be triples: x z r ...")

        junctions = []
        for i in range(0, len(junction_coords), 3):
            x, z, r = junction_coords[i:i+3]
            junctions.append(Circle(cx=x, cz=z, r=r))

        return junctions

    def _generate_chain_plots(self, trajectories, chain_df, junctions, rlist, args):
        """Generate chain visualization plots"""
        try:
            plot_chain_overview(
                trajectories=trajectories,
                chain_df=chain_df,
                junctions=junctions,
                r_outer_list=rlist,
                path_length=float(args.distance),
                epsilon=float(args.epsilon),
                linger_delta=float(args.linger_delta),
                decision_mode=str(args.decision_mode),
                out_path=os.path.join(args.out, "Chain_Overview.png"),
                show_paths=True,
                show_centers=False,
                centers_list=None,
                annotate_counts=False,
            )
            self.logger.info("Chain overview plot generated")
        except Exception as e:
            self.logger.error(f"Chain overview plot failed: {e}")

        # Always generate small multiples for enhanced chain analysis
        try:
            plot_chain_small_multiples(
                trajectories=trajectories,
                chain_df=chain_df,
                junctions=junctions,
                r_outer_list=rlist,
                window_radius=float(args.small_multiples_window),
                path_length=float(args.distance),
                epsilon=float(args.epsilon),
                linger_delta=float(args.linger_delta),
                decision_mode=str(args.decision_mode),
                out_path=os.path.join(args.out, "Chain_SmallMultiples.png"),
            )
            self.logger.info("Chain small multiples plot generated")
        except Exception as e:
            self.logger.error(f"Chain small multiples plot failed: {e}")

        # Generate flow graph map
        try:
            plot_flow_graph_map(
                trajectories=trajectories,
                chain_df=chain_df,
                junctions=junctions,
                r_outer_list=rlist,
                out_path=os.path.join(args.out, "Flow_Graph_Map.png"),
                junction_names=getattr(args, 'junction_names', None),
                show_junction_names=True,
                min_flow_threshold=0.01,  # Show flows >= 1%
                arrow_scale=1.0,
                start_zones=getattr(args, 'start_zones', None),
                end_zones=getattr(args, 'end_zones', None),
            )
            self.logger.info("Flow graph map generated")
        except Exception as e:
            self.logger.error(f"Flow graph map failed: {e}")

    def _run_enhanced_analysis(self, chain_df, junctions, trajectories, args):
        """Run enhanced flow analysis - simplified to only generate flow maps"""
        try:
            # Generate per-junction flow graph map
            plot_per_junction_flow_graph(
                trajectories=trajectories,
                chain_df=chain_df,
                junctions=junctions,
                r_outer_list=getattr(args, 'r_outer_list', None),
                out_path=os.path.join(args.out, "Per_Junction_Flow_Graph.png"),
                junction_names=getattr(args, 'junction_names', None),
                show_junction_names=True,
                min_flow_threshold=0.01,  # Show flows >= 1%
                arrow_scale=1.0,
                start_zones=getattr(args, 'start_zones', None),
                end_zones=getattr(args, 'end_zones', None),
            )
            self.logger.info("Per-junction flow graph map generated")

        except Exception as e:
            self.logger.error(f"Enhanced analysis failed: {e}")
            import traceback
            traceback.print_exc()


class GUICommand(BaseCommand):
    """Command handler for launching the web GUI"""

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--port", type=int, default=8501, help="Port to run the GUI on (default: 8501)")
        parser.add_argument("--host", type=str, default="localhost", help="Host to run the GUI on (default: localhost)")

    def execute(self, args: argparse.Namespace) -> None:
        """Launch the Streamlit GUI"""
        try:
            import streamlit
        except ImportError:
            self.logger.error("Streamlit is not installed. Install GUI dependencies with: pip install verta[gui]")
            return

        import subprocess
        import sys
        from pathlib import Path

        # Get the path to verta_gui.py
        gui_path = Path(__file__).parent / "verta_gui.py"
        
        if not gui_path.exists():
            self.logger.error(f"GUI file not found at: {gui_path}")
            return

        # Build streamlit command
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(gui_path),
            "--server.port", str(args.port),
            "--server.address", args.host
        ]

        self.logger.info(f"Launching VERTA GUI on http://{args.host}:{args.port}")
        self.logger.info("Press Ctrl+C to stop the server")

        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            self.logger.info("GUI stopped by user")
        except Exception as e:
            self.logger.error(f"Failed to launch GUI: {e}")
            raise


# Command registry
COMMANDS = {
    "discover": DiscoverCommand,
    "assign": AssignCommand,
    "metrics": MetricsCommand,
    "gaze": GazeCommand,
    "predict": PredictCommand,
    "intent": IntentRecognitionCommand,
    "chain-enhanced": EnhancedChainCommand,
    "gui": GUICommand,
}
