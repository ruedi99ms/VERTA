"""
Junction-based choice prediction and analysis module.

This module provides functionality for analyzing trajectory patterns at junctions,
predicting user choices, and generating behavioral insights for XR spatial studies.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass

# Try relative imports first (when used as package)
try:
    from .verta_data_loader import Trajectory
    from .verta_geometry import Circle
    from .verta_plotting import plot_flow_graph_map, plot_per_junction_flow_graph, _track_trajectory_junction_sequence
    from .verta_progress import AnalysisProgressManager, show_analysis_estimate
except ImportError:
    # Fall back to absolute imports (when used standalone)
    from .verta_data_loader import Trajectory
    from .verta_geometry import Circle
    from .verta_plotting import plot_flow_graph_map, plot_per_junction_flow_graph, _track_trajectory_junction_sequence
    from .verta_progress import AnalysisProgressManager, show_analysis_estimate

# Get logger
try:
    from .verta_logging import get_logger
    logger = get_logger()
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class JunctionChoiceAnalyzer:
    """Analyzer for junction-based choice patterns."""

    trajectories: List[Trajectory]
    junctions: List[Circle]
    r_outer_list: List[float]

    def __init__(self, trajectories: List[Trajectory], junctions: List[Circle], r_outer_list: List[float]):
        self.trajectories = trajectories
        self.junctions = junctions
        self.r_outer_list = r_outer_list


def analyze_junction_choice_patterns(
    trajectories: List[Trajectory],
    chain_df: pd.DataFrame,
    junctions: List[Circle],
    output_dir: str,
    r_outer_list: List[float],
    gui_mode: bool = False
) -> Dict[str, Any]:
    """
    Analyze junction-based choice patterns and generate predictions.

    Args:
        trajectories: List of trajectory objects
        chain_df: DataFrame with decision chain data
        junctions: List of junction circles
        output_dir: Output directory for results
        r_outer_list: List of r_outer values for each junction

    Returns:
        Dictionary containing analysis results
    """
    logger.info("Starting junction choice pattern analysis...")

    # Show analysis estimate
    show_analysis_estimate(len(trajectories), len(junctions), "predict", gui_mode)

    # Initialize progress manager
    progress_manager = AnalysisProgressManager(gui_mode)
    progress_manager.add_step("Conditional Probabilities", weight=2.0)
    progress_manager.add_step("Visualizations", weight=1.5)

    progress_manager.start_analysis(len(trajectories))

    # Analyze conditional probabilities
    step_tracker = progress_manager.start_step("Conditional Probabilities", len(trajectories))
    conditional_probs, cached_sequences = _analyze_conditional_probabilities(trajectories, junctions, r_outer_list, step_tracker)
    step_tracker.close()
    progress_manager.update_overall_progress(1.0)

    # Create visualizations
    step_tracker = progress_manager.start_step("Visualizations", 1)
    _create_conditional_probability_visualization(conditional_probs, output_dir)
    _create_flow_graph_visualizations(trajectories, junctions, r_outer_list, output_dir, chain_df, cached_sequences)
    _create_pattern_visualizations(conditional_probs, output_dir)
    step_tracker.close()
    progress_manager.update_overall_progress(1.0)

    # Compile results
    results = {
        'conditional_probabilities': conditional_probs,
        'cached_sequences': cached_sequences,  # Cached for interactive prediction tool
        'summary': {
            'total_trajectories': len(trajectories),
            'total_junctions': len(junctions),
            'total_transitions': sum(len(probs) for probs in conditional_probs.values()),
            'unique_patterns': len(conditional_probs)
        }
    }

    # Save results to JSON
    _save_results_to_json(results, output_dir)

    progress_manager.finish_analysis()
    logger.info("Junction choice pattern analysis completed successfully")
    return results


def _analyze_conditional_probabilities(
    trajectories: List[Trajectory],
    junctions: List[Circle],
    r_outer_list: List[float],
    progress_tracker=None
) -> Tuple[Dict[str, Dict[str, float]], Dict[int, List[int]]]:
    """
    Analyze conditional probabilities based on actual trajectory sequences.

    This function tracks each trajectory step-by-step to identify the temporal
    order of junction visits, then calculates conditional probabilities based
    on these actual sequences.

    Returns:
        Tuple of (conditional_probs, cached_sequences) where cached_sequences
        is a dict mapping trajectory index to node sequence for reuse.
    """
    logger.info("Analyzing conditional probabilities from trajectory sequences...")

    # Track trajectory-based flows and conditional probabilities in one pass
    flow_matrix, cached_sequences = _calculate_trajectory_based_flows(trajectories, junctions, r_outer_list, progress_tracker)
    conditional_probs = {}

    # Calculate conditional probabilities for ALL origin-destination pairs
    for origin_idx in range(len(junctions)):
        origin_probs = {}

        # Calculate total exits from this origin junction
        total_exits = np.sum(flow_matrix[origin_idx, :])

        if total_exits > 0:
            # Calculate destination probabilities for this origin
            for dest_idx in range(len(junctions)):
                if origin_idx != dest_idx and flow_matrix[origin_idx, dest_idx] > 0:
                    percentage = (flow_matrix[origin_idx, dest_idx] / total_exits) * 100
                    origin_probs[f"J{dest_idx}"] = percentage

            if origin_probs:  # Only add if there are actual destinations
                conditional_probs[f"from_J{origin_idx}"] = origin_probs

    logger.info(f"Calculated conditional probabilities for {len(conditional_probs)} origin junctions")
    return conditional_probs, cached_sequences


def _calculate_trajectory_based_flows(
    trajectories: List[Trajectory],
    junctions: List[Circle],
    r_outer_list: List[float],
    progress_tracker=None
) -> Tuple[np.ndarray, Dict[int, List[int]]]:
    """
    Calculate flow matrix based on actual trajectory movements.

    This function tracks actual trajectory paths through space, accounting for
    multiple entries/exits into junctions and not relying heavily on spatial proximity.

    Returns:
        Tuple of (flow_matrix, cached_sequences) where cached_sequences
        maps trajectory index to junction sequence for reuse.
    """
    logger.info("Calculating trajectory-based flow matrix...")

    flow_matrix = np.zeros((len(junctions), len(junctions)))
    cached_sequences = {}

    for traj_idx, trajectory in enumerate(trajectories):
        if traj_idx < 5 or traj_idx % 50 == 0:
            logger.debug(f"Processing trajectory {traj_idx + 1}/{len(trajectories)}")

        # Update progress if tracker is provided
        if progress_tracker:
            progress_tracker.update(1, f"Processing trajectory {traj_idx + 1}/{len(trajectories)}")

        # Track junction sequence for this trajectory (cache for reuse)
        junction_sequence = _track_trajectory_junction_sequence(trajectory, junctions, r_outer_list)
        cached_sequences[traj_idx] = junction_sequence

        # Extract transitions from sequence (only count real transitions between different junctions)
        transitions = []
        for i in range(len(junction_sequence) - 1):
            origin_idx = junction_sequence[i]
            dest_idx = junction_sequence[i + 1]
            # Only count transitions between different junctions
            if origin_idx != dest_idx:
                transitions.append((origin_idx, dest_idx))

        # Update flow matrix
        for origin_idx, dest_idx in transitions:
            flow_matrix[origin_idx][dest_idx] += 1

    logger.info(f"Flow matrix calculated with {np.sum(flow_matrix)} total transitions")
    return flow_matrix, cached_sequences


def _track_junction_transitions(
    trajectory: Trajectory,
    junctions: List[Circle],
    r_outer_list: List[float]
) -> List[Tuple[int, int]]:
    """
    Track junction transitions for a single trajectory.

    Returns list of (origin_junction_idx, destination_junction_idx) tuples.
    """
    transitions = []

    # Find all junction visits in temporal order
    junction_sequence = _track_trajectory_junction_sequence(trajectory, junctions, r_outer_list)

    # Extract transitions from sequence (only count real transitions between different junctions)
    for i in range(len(junction_sequence) - 1):
        origin_idx = junction_sequence[i]
        dest_idx = junction_sequence[i + 1]
        # Only count transitions between different junctions
        if origin_idx != dest_idx:
            transitions.append((origin_idx, dest_idx))

    return transitions


def _analyze_start_end_patterns(
    trajectories: List[Trajectory],
    start_zones: Optional[List[Dict]] = None,
    end_zones: Optional[List[Dict]] = None,
    progress_tracker=None
) -> Dict[str, Any]:
    """
    Analyze start and end zone patterns.

    Args:
        trajectories: List of trajectory objects
        start_zones: Optional list of start zone definitions
        end_zones: Optional list of end zone definitions

    Returns:
        Dictionary containing start/end analysis results
    """
    logger.info("Analyzing start and end zone patterns...")

    # Use default zones if not provided
    if start_zones is None:
        start_zones = [{
            "type": "circle",
            "center_x": 180,
            "center_z": -250,
            "radius": 50,
            "proximity_tolerance": 0
        }]

    if end_zones is None:
        end_zones = [{
            "type": "rectangle",
            "x1": 600,
            "z1": 520,
            "x2": 720,
            "z2": 450,
            "tilt": 0,
            "proximity_tolerance": 0
        }]

    # Track zone visits
    start_zone_visits = 0
    end_zone_visits = 0

    for traj_idx, trajectory in enumerate(trajectories):
        # Update progress if tracker is provided
        if progress_tracker:
            progress_tracker.update(1, f"Analyzing trajectory {traj_idx + 1}/{len(trajectories)}")

        for point_idx in range(len(trajectory.x)):
            x, z = trajectory.x[point_idx], trajectory.z[point_idx]

            # Check start zones
            for zone in start_zones:
                if _identify_zone(x, z, zone):
                    start_zone_visits += 1
                    break

            # Check end zones
            for zone in end_zones:
                if _identify_zone(x, z, zone):
                    end_zone_visits += 1
                    break

    results = {
        'start_zone_visits': start_zone_visits,
        'end_zone_visits': end_zone_visits,
        'total_trajectories': len(trajectories),
        'start_zones': start_zones,
        'end_zones': end_zones
    }

    logger.info(f"Start zone visits: {start_zone_visits}, End zone visits: {end_zone_visits}")
    return results


def _identify_zone(x: float, z: float, zone: Dict[str, Any]) -> bool:
    """
    Check if a point is within a zone (circular or rectangular).

    Args:
        x, z: Point coordinates
        zone: Zone definition dictionary

    Returns:
        True if point is within zone (including proximity tolerance)
    """
    proximity_tolerance = zone.get('proximity_tolerance', 0)

    if zone['type'] == 'circle':
        center_x = zone['center_x']
        center_z = zone['center_z']
        radius = zone['radius']

        distance = np.sqrt((x - center_x)**2 + (z - center_z)**2)
        return distance <= (radius + proximity_tolerance)

    elif zone['type'] == 'rectangle':
        # Handle GUI format (x_min, x_max, z_min, z_max)
        x_min, x_max = zone['x_min'], zone['x_max']
        z_min, z_max = zone['z_min'], zone['z_max']

        tilt = zone.get('tilt', 0)

        # Apply inverse rotation if tilted
        if tilt != 0:
            cos_tilt = np.cos(-np.radians(tilt))
            sin_tilt = np.sin(-np.radians(tilt))

            # Translate to origin
            x_rel = x - (x_min + x_max) / 2
            z_rel = z - (z_min + z_max) / 2

            # Apply inverse rotation
            x_rot = x_rel * cos_tilt - z_rel * sin_tilt
            z_rot = x_rel * sin_tilt + z_rel * cos_tilt

            # Translate back
            x = x_rot + (x_min + x_max) / 2
            z = z_rot + (z_min + z_max) / 2

        # Check if point is within rectangle bounds (with proximity tolerance)
        return (x_min - proximity_tolerance <= x <= x_max + proximity_tolerance and
                z_min - proximity_tolerance <= z <= z_max + proximity_tolerance)

    return False


def _generate_prediction_examples(
    analyzer: JunctionChoiceAnalyzer,
    conditional_probs: Dict[str, Dict[str, float]]
) -> List[Dict[str, Any]]:
    """
    Generate prediction examples based on conditional probabilities.
    """
    examples = []

    for origin_key, destinations in conditional_probs.items():
        # Extract origin junction number from "from_J0" format
        origin_idx = int(origin_key.split('_')[1][1:])  # Extract origin number from "from_J0"

        # Find most likely destination
        if destinations:
            best_dest = max(destinations.items(), key=lambda x: x[1])
            confidence = min(best_dest[1] / 100.0, 0.9)  # Cap at 90%

            example = {
                'origin': origin_key,
                'predicted_destination': best_dest[0],
                'confidence': confidence,
                'all_probabilities': destinations
            }
            examples.append(example)

    return examples[:20]  # Return top 20 examples


def _create_conditional_probability_visualization(
    conditional_probs: Dict[str, Dict[str, float]],
    output_dir: str
):
    """Create conditional probability heatmap visualization."""
    if not conditional_probs:
        logger.warning("No conditional probabilities to visualize")
        return

    # Prepare data for heatmap - new structure: {from_J0: {J1: 100.0, J2: 0.0}, ...}
    origins = sorted(conditional_probs.keys(), key=lambda x: int(x.split('_')[1][1:]))
    destinations = set()

    # Collect all possible destinations
    for origin_probs in conditional_probs.values():
        destinations.update(origin_probs.keys())

    destinations = sorted(destinations, key=lambda x: int(x[1:]))

    # Create heatmap data
    heatmap_data = np.zeros((len(origins), len(destinations)))

    for i, origin in enumerate(origins):
        for j, dest in enumerate(destinations):
            # Get the probability for this origin-destination pair
            if origin in conditional_probs and dest in conditional_probs[origin]:
                heatmap_data[i, j] = conditional_probs[origin][dest]

    # Create visualization
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        heatmap_data,
        xticklabels=destinations,
        yticklabels=origins,
        annot=True,
        fmt='.1f',
        cmap='YlOrRd',
        cbar_kws={'label': 'Probability (%)'}
    )

    plt.title('Conditional Probability Heatmap\n(Origin → Destination)', fontsize=14, fontweight='bold')
    plt.xlabel('Destination Junction', fontsize=12)
    plt.ylabel('Origin Junction', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save plot
    output_path = os.path.join(output_dir, 'conditional_probability_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Conditional probability heatmap saved to {output_path}")


def _create_flow_graph_visualizations(
    trajectories: List[Trajectory],
    junctions: List[Circle],
    r_outer_list: List[float],
    output_dir: str,
    chain_df: pd.DataFrame = None,
    cached_sequences: Optional[Dict[int, List[int]]] = None
):
    """Create flow graph visualizations."""
    try:
        # Use provided chain_df or create empty one
        if chain_df is None:
            chain_df = pd.DataFrame()

        # Create flow graph map
        plot_flow_graph_map(
            trajectories=trajectories,
            chain_df=chain_df,  # Use the actual chain_df
            junctions=junctions,
            r_outer_list=r_outer_list,
            out_path=os.path.join(output_dir, "Flow_Graph_Map.png"),
            cached_sequences=cached_sequences
        )

        # Create per-junction flow graph
        plot_per_junction_flow_graph(
            trajectories=trajectories,
            chain_df=chain_df,  # Use the actual chain_df
            junctions=junctions,
            r_outer_list=r_outer_list,
            out_path=os.path.join(output_dir, "Per_Junction_Flow_Graph.png"),
            cached_sequences=cached_sequences
        )

        logger.info("Flow graph visualizations created successfully")
    except Exception as e:
        logger.error(f"Error creating flow graph visualizations: {e}")


def _create_pattern_visualizations(
    conditional_probs: Dict[str, Dict[str, float]],
    output_dir: str
):
    """Create behavioral pattern visualization based on actual trajectory routing."""
    if not conditional_probs:
        logger.warning("No patterns to visualize")
        return

    # Count behavioral pattern types based on actual routing
    pattern_types = defaultdict(int)

    for origin_key, dest_probs in conditional_probs.items():
        num_destinations = len(dest_probs)
        if num_destinations == 1:
            pattern_types['Single Route'] += 1
        elif num_destinations == 2:
            pattern_types['Two Routes'] += 1
        else:
            pattern_types['Multiple Routes'] += 1

    # Filter out zero values
    pattern_types = {k: v for k, v in pattern_types.items() if v > 0}

    if not pattern_types:
        logger.warning("No patterns found for visualization")
        return

    # Create pie chart
    plt.figure(figsize=(8, 6))
    plt.pie(pattern_types.values(), labels=pattern_types.keys(), autopct='%1.1f%%', startangle=90)
    plt.title('Behavioral Pattern Distribution\n(Actual Trajectory Routing)', fontsize=14, fontweight='bold')
    plt.axis('equal')
    plt.tight_layout()

    # Save plot
    output_path = os.path.join(output_dir, 'behavioral_patterns.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Behavioral pattern visualization saved to {output_path}")


def _create_start_end_visualization(
    start_end_results: Dict[str, Any],
    output_dir: str
):
    """Create start/end zone visualization."""
    # Create summary visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Zone visit statistics
    visits_data = {
        'Start Zone': start_end_results['start_zone_visits'],
        'End Zone': start_end_results['end_zone_visits']
    }

    # Filter out zero values
    visits_data = {k: v for k, v in visits_data.items() if v > 0}

    if visits_data:
        ax1.bar(visits_data.keys(), visits_data.values(), color=['green', 'red'])
        ax1.set_title('Zone Visit Statistics', fontweight='bold')
        ax1.set_ylabel('Number of Visits')
        ax1.tick_params(axis='x', rotation=45)

    # Zone definitions
    ax2.text(0.1, 0.8, 'Start Zones:', fontsize=12, fontweight='bold', transform=ax2.transAxes)
    y_pos = 0.7
    for i, zone in enumerate(start_end_results['start_zones']):
        if zone['type'] == 'circle':
            text = f"Zone {i+1}: Circle at ({zone['center_x']}, {zone['center_z']}), r={zone['radius']}"
        else:
            text = f"Zone {i+1}: Rectangle ({zone['x_min']}, {zone['z_min']}) to ({zone['x_max']}, {zone['z_max']})"
        ax2.text(0.1, y_pos, text, fontsize=10, transform=ax2.transAxes)
        y_pos -= 0.1

    ax2.text(0.1, y_pos-0.1, 'End Zones:', fontsize=12, fontweight='bold', transform=ax2.transAxes)
    y_pos -= 0.2
    for i, zone in enumerate(start_end_results['end_zones']):
        if zone['type'] == 'circle':
            text = f"Zone {i+1}: Circle at ({zone['center_x']}, {zone['center_z']}), r={zone['radius']}"
        else:
            text = f"Zone {i+1}: Rectangle ({zone['x_min']}, {zone['z_min']}) to ({zone['x_max']}, {zone['z_max']})"
        ax2.text(0.1, y_pos, text, fontsize=10, transform=ax2.transAxes)
        y_pos -= 0.1

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.set_title('Zone Definitions', fontweight='bold')

    plt.tight_layout()

    # Save plot
    output_path = os.path.join(output_dir, 'start_end_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Start/end analysis visualization saved to {output_path}")


def _save_results_to_json(results: Dict[str, Any], output_dir: str):
    """Save analysis results to JSON file."""
    def _convert_numpy_to_list(obj):
        """Recursively convert numpy arrays to lists for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: _convert_numpy_to_list(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [_convert_numpy_to_list(item) for item in obj]
        else:
            return obj

    # Convert numpy arrays to lists
    json_results = _convert_numpy_to_list(results)

    # Save to file
    output_path = os.path.join(output_dir, 'analysis_results.json')
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)

    logger.info(f"Analysis results saved to {output_path}")


def predict_next_choice(
    current_junction: int,
    origin_junction: int,
    conditional_probs: Dict[str, Dict[str, float]]
) -> Tuple[str, float]:
    """
    Predict the next junction choice based on current and origin junctions.

    Args:
        current_junction: Index of current junction
        origin_junction: Index of origin junction
        conditional_probs: Conditional probability data

    Returns:
        Tuple of (predicted_destination, confidence)
    """
    junction_key = f"J{current_junction}"
    origin_key = f"from_J{origin_junction}"

    if junction_key in conditional_probs and origin_key in conditional_probs[junction_key]:
        destinations = conditional_probs[junction_key][origin_key]
        if destinations:
            best_dest = max(destinations.items(), key=lambda x: x[1])
            confidence = min(best_dest[1] / 100.0, 0.9)
            return best_dest[0], confidence

    return "Unknown", 0.0
