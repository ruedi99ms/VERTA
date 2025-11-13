# ------------------------------
# Intent Recognition Command Implementation
# ------------------------------

import argparse
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

from route_analyzer.ra_commands import BaseCommand
from route_analyzer.ra_data_loader import load_folder, load_folder_with_gaze
from route_analyzer.ra_decisions import discover_branches, assign_branches
from route_analyzer.ra_geometry import Circle
from route_analyzer.ra_intent_recognition import analyze_intent_recognition
from route_analyzer.ra_logging import get_logger


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
        
        logger.info(f"\n✅ Intent Recognition analysis complete! Results saved to {args.out}")
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

