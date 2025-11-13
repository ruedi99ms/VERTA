# ------------------------------
# Predict Command Implementation
# ------------------------------

import argparse
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

from route_analyzer.ra_commands import BaseCommand
from route_analyzer.ra_data_loader import load_folder, Trajectory
from route_analyzer.ra_decisions import discover_decision_chain
from route_analyzer.ra_geometry import Circle
from route_analyzer.ra_prediction import analyze_junction_choice_patterns, JunctionChoiceAnalyzer
from route_analyzer.ra_logging import get_logger
from route_analyzer.ra_consistency import normalize_assignments


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
            from route_analyzer.ra_consistency import validate_consistency
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
        from collections import Counter
        
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
