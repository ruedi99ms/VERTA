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

from route_analyzer.ra_clustering import split_small_branches
from route_analyzer.ra_decisions import assign_branches, discover_branches, discover_decision_chain
from route_analyzer.ra_gaze import (
    compute_head_yaw_at_decisions, 
    analyze_physiological_at_junctions, plot_gaze_directions_at_junctions,
    plot_physiological_by_branch, gaze_movement_consistency_report,
    analyze_pupil_dilation_trajectory, plot_pupil_trajectory_analysis
)
from route_analyzer.ra_consistency import normalize_assignments, validate_trajectories_unique
from route_analyzer.ra_geometry import Circle
from route_analyzer.ra_data_loader import load_folder, load_folder_with_gaze, save_assignments, save_centers, save_centers_json, save_summary
from route_analyzer.ra_metrics import _timing_for_traj, time_between_regions, speed_through_junction, junction_transit_speed
from route_analyzer.ra_plotting import (
    plot_decision_intercepts,
    plot_chain_overview, plot_chain_small_multiples
)
from route_analyzer.ra_logging import RouteAnalyzerLogger


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
        self.logger = RouteAnalyzerLogger()
    
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
            from .ra_plotting import plot_branch_directions
            plot_branch_directions(centers, (junction.cx, junction.cz), 
                                 os.path.join(args.out, "Branch_Directions.png"))
        except (ImportError, AttributeError):
            self.logger.warning("plot_branch_directions not available, skipping")
        
        # Branch counts plot (optional - function may not exist)
        try:
            from .ra_plotting import plot_branch_counts
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
            from .ra_plotting import plot_discover_map
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
            from .ra_consistency import validate_consistency
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
                            from .ra_geometry import Rect
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
                from .ra_consistency import validate_consistency
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


# Import predict command
from route_analyzer.ra_commands_predict import PredictCommand

# Import intent recognition command
from route_analyzer.ra_commands_intent import IntentRecognitionCommand

# Command registry
COMMANDS = {
    "discover": DiscoverCommand,
    "assign": AssignCommand,
    "metrics": MetricsCommand,
    "gaze": GazeCommand,
    "predict": PredictCommand,
    "intent": IntentRecognitionCommand,
}