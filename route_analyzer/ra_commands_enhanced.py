import argparse
import os
import json
import numpy as np
import pandas as pd

from route_analyzer.ra_plotting import (
    plot_flow_graph_map, 
    plot_per_junction_flow_graph,
    plot_chain_overview, 
    plot_chain_small_multiples
)
from route_analyzer.ra_commands import BaseCommand
from route_analyzer.ra_data_loader import load_folder
from route_analyzer.ra_decisions import discover_decision_chain
from route_analyzer.ra_geometry import Circle


class EnhancedChainCommand(BaseCommand):
    """Enhanced command handler for multi-junction decision chain analysis with flow graph features"""
    
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        # Inherit all arguments from ChainCommand
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
