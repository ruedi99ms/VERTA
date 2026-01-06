"""
VERTA Plotting Module

This module provides plotting functions for trajectory analysis and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Optional, Dict, Any
import os
from dataclasses import dataclass
from typing import Tuple

try:
    from .verta_data_loader import Trajectory
    from .verta_geometry import Circle
except ImportError:
    from verta.verta_data_loader import Trajectory
    from verta.verta_geometry import Circle


@dataclass
class PlotConfig:
    """Centralized plot configuration settings."""
    
    # Figure settings
    dpi: int = 100
    figsize: Tuple[float, float] = (12, 8)
    tight_layout: bool = True
    
    # Color settings - colors are used again if more than 10 branches
    branch_colors: Tuple[str, ...] = ('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf')
    
    # Font settings
    fontsize: int = 12
    title_fontsize: int = 14
    label_fontsize: int = 12
    legend_fontsize: int = 10
    
    # Arrow settings
    arrow_scale: float = 0.8
    arrow_width: float = 0.002
    arrow_head_width: float = 0.01
    arrow_head_length: float = 0.01
    
    # Junction settings
    junction_alpha: float = 0.3
    junction_edge_width: float = 2.0
    
    # Trajectory settings
    trajectory_alpha: float = 0.6
    trajectory_linewidth: float = 1.0
    
    # Grid settings
    grid_alpha: float = 0.3
    grid_linewidth: float = 0.5
    
    def apply_to_figure(self, fig) -> None:
        """Apply configuration to a matplotlib figure."""
        if self.tight_layout:
            fig.tight_layout()
    
    def get_branch_color(self, branch_idx: int) -> str:
        """Get color for a branch index."""
        return self.branch_colors[branch_idx % len(self.branch_colors)]


# Default plot configuration instance
DEFAULT_PLOT_CONFIG = PlotConfig()


def _draw_tilted_rectangle(ax, zone, color, label, zorder=2):
    """Helper function to draw a tilted rectangle zone."""
    if 'x_min' in zone:
        x_min, x_max = zone['x_min'], zone['x_max']
        z_min, z_max = zone['z_min'], zone['z_max']
    else:
        x_min, x_max = zone['x1'], zone['x2']
        z_min, z_max = zone['z1'], zone['z2']
    
    tilt = zone.get('tilt', 0)
    
    # Create rectangle corners
    corners_x = [x_min, x_max, x_max, x_min, x_min]
    corners_z = [z_min, z_min, z_max, z_max, z_min]
    
    # Apply rotation if tilted
    if tilt != 0:
        center_x = (x_min + x_max) / 2
        center_z = (z_min + z_max) / 2
        
        cos_tilt = np.cos(np.radians(tilt))
        sin_tilt = np.sin(np.radians(tilt))
        
        # Rotate each corner
        rotated_x = []
        rotated_z = []
        for x, z in zip(corners_x, corners_z):
            # Translate to origin
            x_rel = x - center_x
            z_rel = z - center_z
            
            # Apply rotation
            x_rot = x_rel * cos_tilt - z_rel * sin_tilt
            z_rot = x_rel * sin_tilt + z_rel * cos_tilt
            
            # Translate back
            rotated_x.append(x_rot + center_x)
            rotated_z.append(z_rot + center_z)
        
        corners_x = rotated_x
        corners_z = rotated_z
    
    # Draw rectangle
    ax.plot(corners_x, corners_z, color=color, linewidth=2, alpha=0.7, zorder=zorder)
    
    # Draw center
    center_x = (x_min + x_max) / 2
    center_z = (z_min + z_max) / 2
    ax.scatter([center_x], [center_z], color=color, s=50, marker='s', zorder=zorder+1)
    
    return center_x, center_z


def plot_decision_intercepts(
    trajectories: List[Trajectory],
    assignments_df: pd.DataFrame,
    mode_log_df: pd.DataFrame,
    centers: np.ndarray,
    junction: Circle,
    r_outer: float,
    path_length: float,
    epsilon: float,
    linger_delta: float,
    *,
    out_path: str = "Decision_Intercepts.png",
    show_paths: bool = True,
    legend_noenter_as_line: bool = False,
    junction_number: int = 0,
    all_junctions: List[Circle] = None,
    decision_points_df: pd.DataFrame = None
) -> None:
    """Plot decision intercepts for a junction with branch analysis.
    
    Args:
        trajectories: List of Trajectory objects
        assignments_df: DataFrame with trajectory assignments
        mode_log_df: DataFrame with mode information
        centers: Array of cluster centers
        junction: Junction circle
        r_outer: Outer radius for analysis
        path_length: Path length threshold
        epsilon: Epsilon parameter
        linger_delta: Linger delta parameter
        out_path: Output path for the plot
        show_paths: Whether to show trajectory paths
        legend_noenter_as_line: Whether to show no-entry as line in legend
        junction_number: Junction number for title
        all_junctions: List of all junctions for mini-map
        decision_points_df: DataFrame with actual decision point coordinates
    """
    # Create a more comprehensive layout with heading and mini-map
    fig = plt.figure(figsize=(16, 12))
    
    # Create a grid layout: heading at top, main plot in center, mini-map at bottom
    gs = fig.add_gridspec(3, 1, height_ratios=[0.5, 3, 1], hspace=0.3)
    
    # Add heading
    ax_title = fig.add_subplot(gs[0])
    ax_title.text(0.5, 0.5, f'Junction {junction_number} - Decision Intercepts Analysis', 
                  ha='center', va='center', fontsize=16, fontweight='bold')
    ax_title.set_xlim(0, 1)
    ax_title.set_ylim(0, 1)
    ax_title.axis('off')
    
    # Main plot
    ax = fig.add_subplot(gs[1])
    
    # Plot trajectories if requested
    if show_paths:
        for tr in trajectories:
            ax.plot(tr.x, tr.z, color="0.8", linewidth=0.5, alpha=0.3)
    
    # Plot junction
    theta = np.linspace(0, 2*np.pi, 100)
    junction_x = junction.cx + junction.r * np.cos(theta)
    junction_z = junction.cz + junction.r * np.sin(theta)
    ax.plot(junction_x, junction_z, 'k-', linewidth=2, label='Junction')
    
    # Plot outer radius
    outer_x = junction.cx + r_outer * np.cos(theta)
    outer_z = junction.cz + r_outer * np.sin(theta)
    ax.plot(outer_x, outer_z, 'orange', linewidth=1, alpha=0.7, label='Analysis Radius')
    
    # Plot junction center
    ax.scatter([junction.cx], [junction.cz], c='black', s=50, marker='o', 
              label='Junction Center')
    
    # Plot branch analysis with triangular markers and connections
    if len(centers) > 0 and assignments_df is not None:
        # Define colors and markers for different branches
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']
        markers = ['^', '>', '<', 'v', 's', 'D', 'o', 'h']
        
        # Calculate branch statistics
        branch_counts = assignments_df['branch'].value_counts().sort_index()
        total_trajectories = len(assignments_df)
        
        # Plot branch markers and directions
        for i, center in enumerate(centers):
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            
            # Get assignments for this branch
            branch_assignments = assignments_df[assignments_df['branch'] == i]
            branch_count = len(branch_assignments)
            
            if branch_count > 0:
                # Calculate branch direction from actual decision points if available
                if decision_points_df is not None and len(decision_points_df) > 0:
                    # Merge decision points with assignments to get actual coordinates
                    assignments_df_copy = assignments_df.copy()
                    decision_points_df_copy = decision_points_df.copy()
                    
                    # Convert trajectory columns to string to ensure compatibility
                    if 'trajectory' in assignments_df_copy.columns:
                        assignments_df_copy['trajectory'] = assignments_df_copy['trajectory'].astype(str)
                    if 'trajectory' in decision_points_df_copy.columns:
                        decision_points_df_copy['trajectory'] = decision_points_df_copy['trajectory'].astype(str)
                    
                    try:
                        merged_df = assignments_df_copy.merge(decision_points_df_copy, on='trajectory', how='inner')
                        branch_points = merged_df[merged_df['branch'] == i]
                        
                        if len(branch_points) > 0:
                            # Calculate average direction from junction center to actual decision points
                            intercept_x = branch_points['intercept_x'].values
                            intercept_z = branch_points['intercept_z'].values
                            
                            # Calculate direction vectors from junction center to decision points
                            dx = intercept_x - junction.cx
                            dz = intercept_z - junction.cz
                            
                            # Calculate average direction (normalize to unit vector)
                            avg_dx = np.mean(dx)
                            avg_dz = np.mean(dz)
                            norm = np.sqrt(avg_dx**2 + avg_dz**2)
                            
                            if norm > 0:
                                branch_direction = np.array([avg_dx/norm, avg_dz/norm])
                            else:
                                # Fallback to cluster center if no valid direction
                                branch_direction = center
                        else:
                            # Fallback to cluster center if no decision points
                            branch_direction = center
                    except Exception:
                        # Fallback to cluster center if merge fails
                        branch_direction = center
                else:
                    # Use cluster center if no decision points available
                    branch_direction = center
                
                # Plot branch marker using calculated direction
                branch_x = junction.cx + r_outer * branch_direction[0]
                branch_z = junction.cz + r_outer * branch_direction[1]
                
                # Plot triangular markers for this branch
                ax.scatter([branch_x], [branch_z], c=color, s=100, marker=marker, 
                          label=f'branch {i} (radial) - {branch_count} trajectories', 
                          edgecolors='black', linewidth=1)
                
                # Draw dashed line from branch to junction center
                ax.plot([branch_x, junction.cx], [branch_z, junction.cz], 
                       color=color, linestyle='--', linewidth=2, alpha=0.7)
        
        # Plot all individual intercept points using ACTUAL decision point coordinates
        if decision_points_df is not None and len(decision_points_df) > 0:
            # Ensure trajectory columns have consistent data types for merging
            assignments_df_copy = assignments_df.copy()
            decision_points_df_copy = decision_points_df.copy()
            
            # Convert trajectory columns to string to ensure compatibility
            if 'trajectory' in assignments_df_copy.columns:
                assignments_df_copy['trajectory'] = assignments_df_copy['trajectory'].astype(str)
            if 'trajectory' in decision_points_df_copy.columns:
                decision_points_df_copy['trajectory'] = decision_points_df_copy['trajectory'].astype(str)
            
            # Merge decision points with assignments to get branch information
            try:
                merged_df = assignments_df_copy.merge(decision_points_df_copy, on='trajectory', how='inner')
            except Exception as e:
                # If merge still fails, try alternative approach
                print(f"Warning: Merge failed with error: {e}")
                print(f"Assignments trajectory type: {assignments_df_copy['trajectory'].dtype}")
                print(f"Decision points trajectory type: {decision_points_df_copy['trajectory'].dtype}")
                print(f"Sample assignments trajectory values: {assignments_df_copy['trajectory'].head().tolist()}")
                print(f"Sample decision points trajectory values: {decision_points_df_copy['trajectory'].head().tolist()}")
                # Fall back to theoretical positions
                merged_df = None
            
            if merged_df is not None and len(merged_df) > 0:
                for i, center in enumerate(centers):
                    color = colors[i % len(colors)]
                    branch_assignments = merged_df[merged_df['branch'] == i]
                    
                    if len(branch_assignments) > 0:
                        # Plot actual decision intercept points
                        intercept_x = branch_assignments['intercept_x'].values
                        intercept_z = branch_assignments['intercept_z'].values
                        
                        ax.scatter(intercept_x, intercept_z, 
                                 c=color, s=40, alpha=0.8, marker='o', 
                                 edgecolors='white', linewidth=1.5,
                                 label=f'branch {i} (actual) - {len(branch_assignments)} trajectories')
            else:
                # Fall back to theoretical positions if merge failed or no data
                print("Falling back to theoretical positions due to merge issues")
                for i, center in enumerate(centers):
                    color = colors[i % len(colors)]
                    branch_assignments = assignments_df[assignments_df['branch'] == i]
                    
                    if len(branch_assignments) > 0:
                        # Calculate base intercept point position using unit vector directions
                        base_x = junction.cx + r_outer * center[0]
                        base_z = junction.cz + r_outer * center[1]
                        
                        # Plot theoretical positions (larger, more visible)
                        ax.scatter([base_x], [base_z], 
                                 c=color, s=80, alpha=0.9, marker='s', 
                                 edgecolors='black', linewidth=2,
                                 label=f'branch {i} (theoretical) - {len(branch_assignments)} trajectories')
        else:
            # Fallback: Plot theoretical positions if no decision points data available
            for i, center in enumerate(centers):
                color = colors[i % len(colors)]
                branch_assignments = assignments_df[assignments_df['branch'] == i]
                
                if len(branch_assignments) > 0:
                    # Calculate base intercept point position using unit vector directions
                    base_x = junction.cx + r_outer * center[0]
                    base_z = junction.cz + r_outer * center[1]
                    
                    # Plot theoretical positions (larger, more visible)
                    ax.scatter([base_x], [base_z], 
                             c=color, s=80, alpha=0.9, marker='s', 
                             edgecolors='black', linewidth=2,
                             label=f'branch {i} (theoretical) - {len(branch_assignments)} trajectories')
        
        # Add branch statistics text box
        stats_text = "Branch Statistics:\n"
        for i in range(len(centers)):
            count = branch_counts.get(i, 0)
            percentage = (count / total_trajectories * 100) if total_trajectories > 0 else 0
            stats_text += f"Branch {i}: {count} ({percentage:.1f}%)\n"
        
        # Position text box in upper left corner
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_title('Decision intercepts by branch')
    
    # Move legend outside the plot area to the right
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Add mini-map showing overall view
    if all_junctions is not None and len(all_junctions) > 1:
        ax_mini = fig.add_subplot(gs[2])
        
        # Plot all trajectories in mini-map
        for tr in trajectories:
            ax_mini.plot(tr.x, tr.z, color="0.7", linewidth=0.3, alpha=0.2)
        
        # Plot all junctions
        for i, junc in enumerate(all_junctions):
            if i == junction_number:
                # Highlight current junction
                circle = plt.Circle((junc.cx, junc.cz), junc.r, fill=False, 
                                  color='red', linewidth=3, label=f'Junction {i} (Current)')
                ax_mini.add_patch(circle)
                ax_mini.scatter([junc.cx], [junc.cz], c='red', s=50, marker='o', zorder=5)
            else:
                # Other junctions
                circle = plt.Circle((junc.cx, junc.cz), junc.r, fill=False, 
                                  color='gray', linewidth=1, alpha=0.7)
                ax_mini.add_patch(circle)
                ax_mini.scatter([junc.cx], [junc.cz], c='gray', s=30, marker='o', alpha=0.7)
        
        # Add rectangle showing the area shown in main plot
        main_xlim = ax.get_xlim()
        main_ylim = ax.get_ylim()
        rect = plt.Rectangle((main_xlim[0], main_ylim[0]), 
                           main_xlim[1] - main_xlim[0], 
                           main_ylim[1] - main_ylim[0],
                           fill=False, color='blue', linewidth=2, linestyle='--', alpha=0.8)
        ax_mini.add_patch(rect)
        
        ax_mini.set_aspect('equal')
        ax_mini.set_xlabel('X (Overall View)')
        ax_mini.set_ylabel('Z (Overall View)')
        ax_mini.set_title('Mini-map: All Junctions and Trajectories')
        ax_mini.grid(True, alpha=0.3)
        
        # Set reasonable limits for mini-map
        all_x = [tr.x for tr in trajectories]
        all_z = [tr.z for tr in trajectories]
        if all_x and all_z:
            x_min, x_max = min([min(x) for x in all_x]), max([max(x) for x in all_x])
            z_min, z_max = min([min(z) for z in all_z]), max([max(z) for z in all_z])
            margin = 50
            ax_mini.set_xlim(x_min - margin, x_max + margin)
            ax_mini.set_ylim(z_min - margin, z_max + margin)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_flow_graph_map(
    trajectories: List[Trajectory],
    chain_df: pd.DataFrame,
    junctions: List[Circle],
    *,
    r_outer_list: Optional[List[float]] = None,
    out_path: str = "Flow_Graph_Map.png",
    junction_names: Optional[List[str]] = None,
    show_junction_names: bool = True,
    min_flow_threshold: float = 0.001,
    arrow_scale: float = 1.0,
    cached_sequences: Optional[Dict[int, List[int]]] = None,
) -> None:
    """Map-style flow graph showing flow percentages between junctions.
    
    Shows arrows between junctions indicating what percentage of trajectories go directly from each source to each destination.
    
    Args:
        trajectories: List of Trajectory objects
        chain_df: DataFrame with trajectory assignments for each junction
        junctions: List of Circle objects representing junctions
        r_outer_list: List of outer radii for each junction
        out_path: Path to save the plot
        junction_names: Optional list of names for junctions
        show_junction_names: Whether to show junction names
        min_flow_threshold: Minimum flow percentage to show arrow (0.001 = 0.1%)
        arrow_scale: Scale factor for arrow sizes
        cached_sequences: Optional pre-computed node sequences for trajectories
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    
    if r_outer_list is None:
        r_outer_list = [None] * len(junctions)
    
    # Generate default junction names if not provided
    if junction_names is None:
        junction_names = [f"J{i}" for i in range(len(junctions))]
    elif len(junction_names) != len(junctions):
        junction_names = junction_names[:len(junctions)]
        while len(junction_names) < len(junctions):
            junction_names.append(f"J{len(junction_names)}")
    
    # Calculate overall flow matrix (total percentages across all trajectories)
    flow_matrix = _calculate_overall_flow_matrix(trajectories, chain_df, junctions, r_outer_list, cached_sequences)
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    
    # Plot all trajectories in light gray background
    for tr in trajectories:
        ax.plot(tr.x, tr.z, color="0.8", linewidth=0.8, alpha=0.3, zorder=0)
    
    # Draw junction circles
    theta = np.linspace(0, 2*np.pi, 512)
    for i, (junc, r_out) in enumerate(zip(junctions, r_outer_list)):
        # Inner circle
        jr_x = junc.cx + junc.r*np.cos(theta)
        jr_z = junc.cz + junc.r*np.sin(theta)
        ax.plot(jr_x, jr_z, color="black", linewidth=1.5, zorder=2)
        
        # Outer circle if provided
        if r_out is not None and float(r_out) > float(junc.r):
            ox = junc.cx + float(r_out)*np.cos(theta)
            oz = junc.cz + float(r_out)*np.sin(theta)
            ax.plot(ox, oz, color="orange", linewidth=1.0, alpha=0.7, zorder=2)
        
        # Junction center
        ax.scatter([junc.cx], [junc.cz], color="black", s=30, zorder=3)
        
        # Junction name label
        if show_junction_names:
            ax.annotate(junction_names[i], 
                       (junc.cx, junc.cz), 
                       xytext=(8, 8), 
                       textcoords='offset points',
                       fontsize=12, 
                       fontweight='bold',
                       color='red',
                       bbox=dict(boxstyle='round,pad=0.4', 
                                facecolor='white', 
                                edgecolor='red',
                                alpha=0.9),
                       zorder=5)
    
    # Draw flow arrows
    max_flow = np.max(flow_matrix) if flow_matrix.size > 0 else 1.0
    min_arrow_size = 15
    max_arrow_size = 100
    
    # Create node positions for arrow drawing
    node_positions = []
    node_names = []
    
    # Junction positions
    for i, junc in enumerate(junctions):
        node_positions.append((junc.cx, junc.cz))
        node_names.append(f"J{i}")
    
    for i in range(len(junctions)):
        for j in range(len(junctions)):
            if i != j and flow_matrix[i, j] >= min_flow_threshold:
                # Calculate arrow properties
                flow_percentage = flow_matrix[i, j]
                arrow_size = min_arrow_size + (max_arrow_size - min_arrow_size) * (flow_percentage / max_flow)
                arrow_size *= arrow_scale
                
                # Arrow color based on flow intensity
                color_intensity = flow_percentage / max_flow
                arrow_color = plt.cm.Blues(0.3 + 0.7 * color_intensity)
                
                # Calculate start and end points with appropriate offsets
                start_junc = junctions[i]
                end_junc = junctions[j]
                
                # Calculate arrow direction
                dx = end_junc.cx - start_junc.cx
                dy = end_junc.cz - start_junc.cz
                distance = np.sqrt(dx**2 + dy**2)
                
                if distance > 0:
                    # Normalize direction
                    dx_norm = dx / distance
                    dy_norm = dy / distance
                    
                    # Calculate start and end points
                    start_x = start_junc.cx + dx_norm * start_junc.r
                    start_y = start_junc.cz + dy_norm * start_junc.r
                    end_x = end_junc.cx - dx_norm * end_junc.r
                    end_y = end_junc.cz - dy_norm * end_junc.r
                    
                    # Create arrow
                    arrow = FancyArrowPatch(
                        (start_x, start_y), (end_x, end_y),
                        arrowstyle='->', 
                        mutation_scale=arrow_size,
                        color=arrow_color,
                        linewidth=max(1, int(2 + 3 * flow_percentage / max_flow)),
                        alpha=0.8,
                        zorder=4
                    )
                    ax.add_patch(arrow)
                    
                    # Add flow percentage label
                    mid_x = (start_x + end_x) / 2
                    mid_y = (start_y + end_y) / 2
                    
                    # Offset label slightly to avoid overlap
                    label_offset_x = -dy_norm * 10
                    label_offset_y = dx_norm * 10
                    
                    ax.annotate(f'{flow_percentage:.1%}', 
                               (mid_x + label_offset_x, mid_y + label_offset_y),
                               ha='center', va='center',
                               fontsize=9,
                               fontweight='bold',
                               color='darkblue',
                               bbox=dict(boxstyle='round,pad=0.3', 
                                        facecolor='white', 
                                        edgecolor='darkblue',
                                        alpha=0.9),
                               zorder=6)
    
    # Set equal aspect and labels
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_title("Flow Graph Map", fontsize=14, fontweight='bold')
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color='black', linewidth=2, label='Junction'),
        plt.Line2D([0], [0], color='orange', linewidth=2, label='Analysis Radius'),
        plt.Line2D([0], [0], color='blue', linewidth=2, label='Flow Direction (%)'),
        plt.Line2D([0], [0], color='0.8', linewidth=1, alpha=0.3, label='Trajectories')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_per_junction_flow_graph(
    trajectories: List[Trajectory],
    chain_df: pd.DataFrame,
    junctions: List[Circle],
    *,
    r_outer_list: Optional[List[float]] = None,
    out_path: str = "Per_Junction_Flow_Graph.png",
    junction_names: Optional[List[str]] = None,
    show_junction_names: bool = True,
    min_flow_threshold: float = 0.001,  # Reduced from 0.01 to show flows as low as 0.1%
    arrow_scale: float = 1.0,
    cached_sequences: Optional[Dict[int, List[int]]] = None,
) -> None:
    """Map-style flow graph showing percentages per junction including zones.
    
    Shows arrows between junctions and zones indicating what percentage of trajectories leaving each junction go directly to each destination.
    
    Args:
        trajectories: List of Trajectory objects
        chain_df: DataFrame with trajectory assignments for each junction
        junctions: List of Circle objects representing junctions
        r_outer_list: List of outer radii for each junction
        out_path: Path to save the plot
        junction_names: Optional list of names for junctions
        show_junction_names: Whether to show junction names
        min_flow_threshold: Minimum flow percentage to show arrow (0.001 = 0.1%)
        arrow_scale: Scale factor for arrow sizes
        start_zones: Optional list of start zone definitions
        end_zones: Optional list of end zone definitions
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    
    if r_outer_list is None:
        r_outer_list = [None] * len(junctions)
    
    # Generate default junction names if not provided
    if junction_names is None:
        junction_names = [f"J{i}" for i in range(len(junctions))]
    elif len(junction_names) != len(junctions):
        junction_names = junction_names[:len(junctions)]
        while len(junction_names) < len(junctions):
            junction_names.append(f"J{len(junction_names)}")
    
    # Calculate per-junction flows
    flow_matrix = _calculate_per_junction_flows(trajectories, chain_df, junctions, r_outer_list, cached_sequences)
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    
    # Plot all trajectories in light gray background
    for tr in trajectories:
        ax.plot(tr.x, tr.z, color="0.8", linewidth=0.8, alpha=0.3, zorder=0)
    
    # Draw junction circles
    theta = np.linspace(0, 2*np.pi, 512)
    for i, (junc, r_out) in enumerate(zip(junctions, r_outer_list)):
        # Inner circle
        jr_x = junc.cx + junc.r*np.cos(theta)
        jr_z = junc.cz + junc.r*np.sin(theta)
        ax.plot(jr_x, jr_z, color="black", linewidth=1.5, zorder=2)
        
        # Outer circle if provided
        if r_out is not None and float(r_out) > float(junc.r):
            ox = junc.cx + float(r_out)*np.cos(theta)
            oz = junc.cz + float(r_out)*np.sin(theta)
            ax.plot(ox, oz, color="orange", linewidth=1.0, alpha=0.7, zorder=2)
        
        # Junction center
        ax.scatter([junc.cx], [junc.cz], color="black", s=30, zorder=3)
        
        # Junction name label
        if show_junction_names:
            ax.annotate(junction_names[i], 
                       (junc.cx, junc.cz), 
                       xytext=(8, 8), 
                       textcoords='offset points',
                       fontsize=12, 
                       fontweight='bold',
                       color='red',
                       bbox=dict(boxstyle='round,pad=0.4', 
                                facecolor='white', 
                                edgecolor='red',
                                alpha=0.9),
                       zorder=5)
    
    # Draw flow arrows with improved labeling
    max_flow = np.max(flow_matrix) if flow_matrix.size > 0 else 1.0
    min_arrow_size = 15
    max_arrow_size = 100
    
    print(f"DEBUG: Per-junction flow matrix:\n{flow_matrix}")
    print(f"DEBUG: Max flow: {max_flow}")
    
    # Create node positions for arrow drawing
    node_positions = []
    node_names = []
    
    # Junction positions
    for i, junc in enumerate(junctions):
        node_positions.append((junc.cx, junc.cz))
        node_names.append(f"J{i}")
    
    for i in range(len(junctions)):
        for j in range(len(junctions)):
            if i != j and flow_matrix[i, j] >= min_flow_threshold:
                # Calculate arrow properties
                flow_percentage = flow_matrix[i, j]
                arrow_size = min_arrow_size + (max_arrow_size - min_arrow_size) * (flow_percentage / max_flow)
                arrow_size *= arrow_scale
                
                # Arrow color based on flow intensity
                color_intensity = flow_percentage / max_flow
                arrow_color = plt.cm.Greens(0.3 + 0.7 * color_intensity)  # Use green for per-junction flows
                
                # Calculate start and end points with appropriate offsets
                start_junc = junctions[i]
                end_junc = junctions[j]
                
                # Calculate arrow direction
                dx = end_junc.cx - start_junc.cx
                dy = end_junc.cz - start_junc.cz
                distance = np.sqrt(dx**2 + dy**2)
                
                if distance > 0:
                    # Normalize direction
                    dx_norm = dx / distance
                    dy_norm = dy / distance
                    
                    # Calculate start and end points
                    start_x = start_junc.cx + dx_norm * start_junc.r
                    start_y = start_junc.cz + dy_norm * start_junc.r
                    end_x = end_junc.cx - dx_norm * end_junc.r
                    end_y = end_junc.cz - dy_norm * end_junc.r
                    
                    # Create arrow
                    arrow = FancyArrowPatch(
                        (start_x, start_y), (end_x, end_y),
                        arrowstyle='->', 
                        mutation_scale=arrow_size,
                        color=arrow_color,
                        linewidth=max(1, int(2 + 3 * flow_percentage / max_flow)),
                        alpha=0.8,
                        zorder=4
                    )
                    ax.add_patch(arrow)
                    
                    # Add flow percentage label with better positioning
                    mid_x = (start_x + end_x) / 2
                    mid_y = (start_y + end_y) / 2
                    
                    # Offset label slightly to avoid overlap
                    label_offset_x = -dy_norm * 10  # Perpendicular offset
                    label_offset_y = dx_norm * 10
                    
                    ax.annotate(f'{flow_percentage:.1%}', 
                               (mid_x + label_offset_x, mid_y + label_offset_y),
                               ha='center', va='center',
                               fontsize=9,
                               fontweight='bold',
                               color='darkgreen',
                               bbox=dict(boxstyle='round,pad=0.3', 
                                        facecolor='white', 
                                        edgecolor='darkgreen',
                                        alpha=0.9),
                               zorder=6)
    
    # Set equal aspect and labels
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_title("Flow Graph: Per-Junction Percentages", fontsize=14, fontweight='bold')
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color='black', linewidth=2, label='Junction'),
        plt.Line2D([0], [0], color='orange', linewidth=2, label='Decision Radius'),
        plt.Line2D([0], [0], color='green', linewidth=2, label='Flow Direction (per-junction %)'),
        plt.Line2D([0], [0], color='0.8', linewidth=1, alpha=0.3, label='Trajectories')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _calculate_flow_matrix(
    trajectories: List[Trajectory],
    chain_df: pd.DataFrame,
    junctions: List[Circle],
    r_outer_list: Optional[List[float]] = None
) -> np.ndarray:
    """Calculate flow matrix showing flow percentages between junctions."""
    n_junctions = len(junctions)
    flow_counts = np.zeros((n_junctions, n_junctions))
    
    # Ensure chain_df is properly indexed
    if 'trajectory' in chain_df.columns:
        chain_df = chain_df.set_index('trajectory')
    
    branch_cols = [f"branch_j{i}" for i in range(n_junctions)]
    
    for tr in trajectories:
        tid = tr.tid
        if tid not in chain_df.index:
            continue
            
        # Get junction assignments for this trajectory
        assignments = []
        for i in range(n_junctions):
            b = chain_df.loc[tid].get(branch_cols[i], None)
            if b is not None and not (isinstance(b, float) and np.isnan(b)) and int(b) >= 0:
                assignments.append(i)
        
        # Only consider trajectories that pass through at least 2 junctions
        if len(assignments) >= 2:
            # Find all pairs of junctions this trajectory connects
            for i in range(len(assignments)):
                for j in range(len(assignments)):
                    if i != j:
                        from_junc = assignments[i]
                        to_junc = assignments[j]
                        
                        # Check if this is a direct spatial connection
                        if _is_direct_spatial_connection(tr, junctions, from_junc, to_junc):
                            flow_counts[from_junc, to_junc] += 1
    
    # Convert counts to percentages
    total_flows = np.sum(flow_counts)
    if total_flows > 0:
        flow_matrix = flow_counts / total_flows
    else:
        flow_matrix = flow_counts
    
    return flow_matrix


def _track_trajectory_junction_sequence(
    trajectory: Trajectory, 
    junctions: List[Circle], 
    r_outer_list: List[float]
) -> List[int]:
    """
    Track the temporal sequence of junction visits for a trajectory.
    
    Returns list of junction indices in the order they were visited.
    """
    sequence = []
    current_junction = None
    
    # Debug: Log trajectory info (only once per trajectory)
    traj_id = getattr(trajectory, 'tid', 'unknown')
    print(f"🔍 DEBUG: Tracking trajectory {traj_id}, length: {len(trajectory.x)} points")
    print(f"🔍 DEBUG: Trajectory range: X={min(trajectory.x):.1f}-{max(trajectory.x):.1f}, Z={min(trajectory.z):.1f}-{max(trajectory.z):.1f}")
    
    junction_detections = 0
    points_processed = 0
    
    for point_idx in range(len(trajectory.x)):
        x, z = trajectory.x[point_idx], trajectory.z[point_idx]
        points_processed += 1
        
        # Check if we're at a new junction
        for junction_idx, (junction, r_outer) in enumerate(zip(junctions, r_outer_list)):
            distance = np.sqrt((x - junction.cx)**2 + (z - junction.cz)**2)
            
            if distance <= r_outer:
                junction_detections += 1
                if current_junction != junction_idx:
                    sequence.append(junction_idx)
                    current_junction = junction_idx
                    print(f"🔍 DEBUG: Added junction {junction_idx} to sequence at point {point_idx}")
                break
        else:
            # Not at any junction
            current_junction = None
    
    print(f"DEBUG: Trajectory {traj_id} final sequence: {sequence}")
    print(f"DEBUG: Total junction detections: {junction_detections}, Points processed: {points_processed}")
    
    return sequence


def _calculate_per_junction_flows(
    trajectories: List[Trajectory],
    chain_df: pd.DataFrame,
    junctions: List[Circle],
    r_outer_list: Optional[List[float]] = None,
    cached_sequences: Optional[Dict[int, List[int]]] = None
) -> np.ndarray:
    """Calculate per-junction flow matrix showing percentages of trajectories leaving each junction."""
    n_junctions = len(junctions)
    flow_counts = np.zeros((n_junctions, n_junctions))
    junction_exits = np.zeros(n_junctions)  # Count trajectories leaving each junction
    
    # Use cached sequences if provided, otherwise do spatial tracking
    if cached_sequences is not None:
        print(f"🔍 DEBUG: Using cached sequences for {len(cached_sequences)} trajectories")
        for tr_idx, tr in enumerate(trajectories):
            junction_sequence = cached_sequences.get(tr_idx, [])
            
            # Extract transitions from sequence (only count real transitions between different junctions)
            if len(junction_sequence) >= 2:
                for i in range(len(junction_sequence) - 1):
                    from_junc = junction_sequence[i]
                    to_junc = junction_sequence[i + 1]
                    # Only count transitions between different junctions
                    if from_junc != to_junc:
                        flow_counts[from_junc, to_junc] += 1
                        junction_exits[from_junc] += 1
    else:
        print(f"DEBUG: No cached sequences provided, doing spatial tracking for {len(trajectories)} trajectories")
        # Use spatial tracking to determine junction visit sequences
        for tr_idx, tr in enumerate(trajectories):
            # Track junction sequence for this trajectory using spatial tracking
            junction_sequence = _track_trajectory_junction_sequence(tr, junctions, r_outer_list)
            
            # Extract transitions from sequence (only count real transitions between different junctions)
            if len(junction_sequence) >= 2:
                for i in range(len(junction_sequence) - 1):
                    from_junc = junction_sequence[i]
                    to_junc = junction_sequence[i + 1]
                    # Only count transitions between different junctions
                    if from_junc != to_junc:
                        flow_counts[from_junc, to_junc] += 1
                        junction_exits[from_junc] += 1
    
    # Convert counts to percentages per junction
    flow_matrix = np.zeros((n_junctions, n_junctions))
    for i in range(n_junctions):
        if junction_exits[i] > 0:
            flow_matrix[i, :] = flow_counts[i, :] / junction_exits[i]
    
    return flow_matrix


def _calculate_overall_flow_matrix(
    trajectories: List[Trajectory],
    chain_df: pd.DataFrame,
    junctions: List[Circle],
    r_outer_list: Optional[List[float]] = None,
    cached_sequences: Optional[Dict[int, List[int]]] = None
) -> np.ndarray:
    """Calculate overall flow matrix showing percentage of all trajectories making each transition."""
    n_junctions = len(junctions)
    flow_counts = np.zeros((n_junctions, n_junctions))
    total_trajectories = len(trajectories)
    
    # Use cached sequences if provided, otherwise do spatial tracking
    if cached_sequences is not None:
        print(f"🔍 DEBUG: Using cached sequences for overall flow calculation")
        for tr_idx, tr in enumerate(trajectories):
            junction_sequence = cached_sequences.get(tr_idx, [])
            
            # Extract transitions from sequence (only count real transitions between different junctions)
            if len(junction_sequence) >= 2:
                for i in range(len(junction_sequence) - 1):
                    from_junc = junction_sequence[i]
                    to_junc = junction_sequence[i + 1]
                    # Only count transitions between different junctions
                    if from_junc != to_junc:
                        flow_counts[from_junc, to_junc] += 1
    else:
        print(f"🔍 DEBUG: No cached sequences provided, doing spatial tracking for overall flow calculation")
        # Use spatial tracking to determine junction visit sequences
        for tr_idx, tr in enumerate(trajectories):
            # Track junction sequence for this trajectory using spatial tracking
            junction_sequence = _track_trajectory_junction_sequence(tr, junctions, r_outer_list)
            
            # Extract transitions from sequence (only count real transitions between different junctions)
            if len(junction_sequence) >= 2:
                for i in range(len(junction_sequence) - 1):
                    from_junc = junction_sequence[i]
                    to_junc = junction_sequence[i + 1]
                    # Only count transitions between different junctions
                    if from_junc != to_junc:
                        flow_counts[from_junc, to_junc] += 1
    
    # Convert counts to percentages of all trajectories (not per-junction percentages)
    flow_matrix = np.zeros((n_junctions, n_junctions))
    if total_trajectories > 0:
        flow_matrix = flow_counts / total_trajectories
    
    print(f"🔍 DEBUG: Overall flow matrix - total trajectories: {total_trajectories}")
    print(f"🔍 DEBUG: Overall flow matrix:\n{flow_matrix}")
    
    return flow_matrix


def _is_direct_spatial_connection(
    trajectory: Trajectory,
    junctions: List[Circle],
    from_junc_idx: int,
    to_junc_idx: int
) -> bool:
    """Check if trajectory goes directly from one junction to another spatially."""
    from_junc = junctions[from_junc_idx]
    to_junc = junctions[to_junc_idx]
    
    # Find points where trajectory is near each junction
    from_distances = np.sqrt((trajectory.x - from_junc.cx)**2 + (trajectory.z - from_junc.cz)**2)
    to_distances = np.sqrt((trajectory.x - to_junc.cx)**2 + (trajectory.z - to_junc.cz)**2)
    
    # Find indices where trajectory is near each junction (within 2x radius)
    from_near_indices = np.where(from_distances <= from_junc.r * 2)[0]
    to_near_indices = np.where(to_distances <= to_junc.r * 2)[0]
    
    if len(from_near_indices) == 0 or len(to_near_indices) == 0:
        return False
    
    # Check if trajectory goes from first junction to second junction
    # (first junction appears before second junction in the trajectory)
    from_max_idx = np.max(from_near_indices)
    to_min_idx = np.min(to_near_indices)
    
    # Direct connection if trajectory goes from first junction to second
    if from_max_idx < to_min_idx:
        # Check if trajectory doesn't pass through other junctions in between
        for i, junc in enumerate(junctions):
            if i == from_junc_idx or i == to_junc_idx:
                continue
                
            # Check if trajectory intersects this junction between the two points
            segment_distances = np.sqrt((trajectory.x[from_max_idx:to_min_idx] - junc.cx)**2 + 
                                       (trajectory.z[from_max_idx:to_min_idx] - junc.cz)**2)
            if np.any(segment_distances <= junc.r):
                return False  # Trajectory passes through another junction
        
        return True  # Direct connection
    
    return False


def _calculate_flow_matrix_with_zones(
    trajectories: List[Trajectory],
    chain_df: pd.DataFrame,
    junctions: List[Circle],
    r_outer_list: Optional[List[float]] = None,
    start_zones: Optional[List[Dict]] = None,
    end_zones: Optional[List[Dict]] = None,
    cached_sequences: Optional[Dict[int, List[int]]] = None
) -> np.ndarray:
    """Calculate overall flow matrix including start/end zones (total percentages)."""
    n_junctions = len(junctions)
    n_start_zones = len(start_zones) if start_zones else 0
    n_end_zones = len(end_zones) if end_zones else 0
    total_nodes = n_junctions + n_start_zones + n_end_zones
    
    flow_counts = np.zeros((total_nodes, total_nodes))
    total_trajectories = len(trajectories)
    trajectories_with_sequences = 0
    
    # Compute node sequences for flow calculations
    cached_sequences = {}
    for tr_idx, tr in enumerate(trajectories):
        cached_sequences[tr_idx] = _track_trajectory_node_sequence(tr, junctions, r_outer_list, start_zones, end_zones)
    
    for tr_idx, tr in enumerate(trajectories):
        # Use cached sequence
        node_sequence = cached_sequences.get(tr_idx, [])
        
        if len(node_sequence) > 0:
            trajectories_with_sequences += 1
            
            # Count flows between consecutive nodes (ensure unique transitions only)
            unique_transitions = set()
            for i in range(len(node_sequence) - 1):
                from_node = node_sequence[i]
                to_node = node_sequence[i + 1]
                transition = (from_node, to_node)
                
                # Only count each unique transition once per trajectory
                if transition not in unique_transitions:
                    unique_transitions.add(transition)
                    flow_counts[from_node, to_node] += 1
    
    print(f"DEBUG: {trajectories_with_sequences}/{total_trajectories} trajectories have node sequences")
    if start_zones:
        start_zone_idx = len(junctions)
        start_flows = np.sum(flow_counts[start_zone_idx, :])
        print(f"DEBUG: Total flows from start zone: {start_flows}")
    if end_zones:
        end_zone_idx = len(junctions) + len(start_zones)
        end_flows = np.sum(flow_counts[:, end_zone_idx])
        print(f"DEBUG: Total flows to end zone: {end_flows}")
    
    # Convert counts to percentages
    total_flows = np.sum(flow_counts)
    if total_flows > 0:
        flow_matrix = flow_counts / total_flows
    else:
        flow_matrix = flow_counts
    
    return flow_matrix


def _calculate_per_junction_flows_with_zones(
    trajectories: List[Trajectory],
    chain_df: pd.DataFrame,
    junctions: List[Circle],
    r_outer_list: Optional[List[float]] = None,
    start_zones: Optional[List[Dict]] = None,
    end_zones: Optional[List[Dict]] = None,
    gui_mode: bool = False,
    cached_sequences: Optional[Dict[int, List[int]]] = None
) -> np.ndarray:
    """Calculate per-junction flow matrix including zones.
    
    Args:
        cached_sequences: Optional pre-computed node sequences for trajectories
                         (key: trajectory index, value: node sequence)
    """
    n_junctions = len(junctions)
    n_start_zones = len(start_zones) if start_zones else 0
    n_end_zones = len(end_zones) if end_zones else 0
    total_nodes = n_junctions + n_start_zones + n_end_zones
    
    flow_counts = np.zeros((total_nodes, total_nodes))
    node_exits = np.zeros(total_nodes)  # Count trajectories leaving each node
    node_entries = np.zeros(total_nodes)  # Count trajectories entering each node
    total_trajectories = len(trajectories)
    trajectories_with_sequences = 0
    
    # Compute node sequences for flow calculations
    cached_sequences = {}
    for tr_idx, tr in enumerate(trajectories):
        cached_sequences[tr_idx] = _track_trajectory_node_sequence(tr, junctions, r_outer_list, start_zones, end_zones)
    
    # Only print debug stats in terminal mode (not GUI)
    if not gui_mode:
        print(f"\n=== TRAJECTORY FLOW DEBUG STATISTICS ===")
        print(f"Total trajectories: {total_trajectories}")
        print(f"Junctions: {n_junctions}, Start zones: {n_start_zones}, End zones: {n_end_zones}")
        print(f"Total nodes: {total_nodes}")
    
    for tr_idx, tr in enumerate(trajectories):
        # Use cached sequence
        node_sequence = cached_sequences.get(tr_idx, [])
        
        if len(node_sequence) > 0:
            trajectories_with_sequences += 1
            
            # Count flows between consecutive nodes (ensure unique transitions only)
            unique_transitions = set()
            for i in range(len(node_sequence) - 1):
                from_node = node_sequence[i]
                to_node = node_sequence[i + 1]
                transition = (from_node, to_node)
                
                # Only count each unique transition once per trajectory
                if transition not in unique_transitions:
                    unique_transitions.add(transition)
                    flow_counts[from_node, to_node] += 1
                    node_exits[from_node] += 1  # Count exits from source node
                    node_entries[to_node] += 1   # Count entries to destination node
    
    # Only print debug stats in terminal mode (not GUI)
    if not gui_mode:
        print(f"\nTrajectories with sequences: {trajectories_with_sequences}/{total_trajectories} ({trajectories_with_sequences/total_trajectories*100:.1f}%)")
        
        # Print statistics for each node
        print(f"\n=== NODE STATISTICS ===")
        for i in range(total_nodes):
            if i < n_junctions:
                node_name = f"J{i}"
                node_type = "Junction"
            elif i < n_junctions + n_start_zones:
                zone_idx = i - n_junctions
                node_name = f"S{zone_idx + 1}"
                node_type = "Start Zone"
            else:
                zone_idx = i - n_junctions - n_start_zones
                node_name = f"E{zone_idx + 1}"
                node_type = "End Zone"
            
            entries = int(node_entries[i])
            exits = int(node_exits[i])
            print(f"{node_name} ({node_type}): {entries} entries, {exits} exits")
            
            # Show outgoing flows
            outgoing_flows = []
            for j in range(total_nodes):
                if flow_counts[i, j] > 0:
                    if j < n_junctions:
                        dest_name = f"J{j}"
                    elif j < n_junctions + n_start_zones:
                        dest_name = f"S{j - n_junctions + 1}"
                    else:
                        dest_name = f"E{j - n_junctions - n_start_zones + 1}"
                    outgoing_flows.append(f"{dest_name}({int(flow_counts[i, j])})")
            
            if outgoing_flows:
                print(f"  → Outgoing flows: {', '.join(outgoing_flows)}")
    
    # Convert counts to percentages per node
    flow_matrix = np.zeros((total_nodes, total_nodes))
    for i in range(total_nodes):
        if node_exits[i] > 0:
            flow_matrix[i, :] = flow_counts[i, :] / node_exits[i]
    
    # Only print flow percentages in terminal mode (not GUI)
    if not gui_mode:
        print(f"\n=== FLOW PERCENTAGES ===")
        for i in range(total_nodes):
            if node_exits[i] > 0:
                if i < n_junctions:
                    node_name = f"J{i}"
                elif i < n_junctions + n_start_zones:
                    node_name = f"S{i - n_junctions + 1}"
                else:
                    node_name = f"E{i - n_junctions - n_start_zones + 1}"
                
                print(f"\n{node_name} outgoing percentages:")
                for j in range(total_nodes):
                    if flow_matrix[i, j] > 0.01:  # Only show flows > 1%
                        if j < n_junctions:
                            dest_name = f"J{j}"
                        elif j < n_junctions + n_start_zones:
                            dest_name = f"S{j - n_junctions + 1}"
                        else:
                            dest_name = f"E{j - n_junctions - n_start_zones + 1}"
                        print(f"  → {dest_name}: {flow_matrix[i, j]*100:.1f}%")
        
        print(f"\n=== END DEBUG STATISTICS ===\n")
    
    return flow_matrix


def _track_trajectory_node_sequence(
    trajectory: Trajectory,
    junctions: List[Circle],
    r_outer_list: Optional[List[float]] = None,
    start_zones: Optional[List[Dict]] = None,
    end_zones: Optional[List[Dict]] = None
) -> List[int]:
    """
    Track the temporal sequence of node visits for a trajectory.
    
    Returns list of node indices in the order they were visited.
    Node indices: 0 to n_junctions-1 for junctions, n_junctions to n_junctions+n_start_zones-1 for start zones, n_junctions+n_start_zones to n_junctions+n_start_zones+n_end_zones-1 for end zones.
    
    This function prioritizes nodes by temporal order (which is reached first along the trajectory path), not by distance or type priority.
    """
    sequence = []
    current_node = None
    
    if r_outer_list is None:
        r_outer_list = [None] * len(junctions)
    
    # Track the trajectory point by point to find temporal order
    for point_idx in range(len(trajectory.x)):
        x, z = trajectory.x[point_idx], trajectory.z[point_idx]
        
        # Check all possible nodes and find which one this point is in
        nodes_at_this_point = []
        
        # Check junctions
        for junction_idx, (junction, r_outer) in enumerate(zip(junctions, r_outer_list)):
            distance = np.sqrt((x - junction.cx)**2 + (z - junction.cz)**2)
            radius_to_check = r_outer if r_outer is not None else junction.r
            
            if distance <= radius_to_check:
                nodes_at_this_point.append(junction_idx)
        
        # Check start zones
        if start_zones:
            for zone_idx, zone in enumerate(start_zones):
                if _identify_zone(x, z, zone):
                    node_idx = len(junctions) + zone_idx
                    nodes_at_this_point.append(node_idx)
        
        # Check end zones
        if end_zones:
            for zone_idx, zone in enumerate(end_zones):
                if _identify_zone(x, z, zone):
                    node_idx = len(junctions) + len(start_zones) + zone_idx
                    nodes_at_this_point.append(node_idx)
        
        # If we're in any nodes at this point, select the first one we encounter
        # (this maintains temporal order - first node reached along the path)
        if nodes_at_this_point:
            # Use the first node in the list (maintains order of checking)
            closest_node = nodes_at_this_point[0]
            
            # Add node to sequence if it's different from current node
            if closest_node != current_node:
                sequence.append(closest_node)
                current_node = closest_node
        else:
            # Not in any node
            current_node = None
    
    # Special handling for start zones: if trajectory starts in a start zone,
    # ensure it's the first node in the sequence
    if start_zones and len(trajectory.x) > 0:
        first_x, first_z = trajectory.x[0], trajectory.z[0]
        for zone_idx, zone in enumerate(start_zones):
            if _identify_zone(first_x, first_z, zone):
                node_idx = len(junctions) + zone_idx
                # If sequence doesn't start with this start zone, prepend it
                if not sequence or sequence[0] != node_idx:
                    sequence.insert(0, node_idx)
                break
    
    # Special handling for end zones: if trajectory ends in an end zone,
    # ensure it's the last node in the sequence
    if end_zones and len(trajectory.x) > 0:
        last_x, last_z = trajectory.x[-1], trajectory.z[-1]
        for zone_idx, zone in enumerate(end_zones):
            if _identify_zone(last_x, last_z, zone):
                node_idx = len(junctions) + len(start_zones) + zone_idx
                # If sequence doesn't end with this end zone, append it
                if not sequence or sequence[-1] != node_idx:
                    sequence.append(node_idx)
                break
    
    return sequence


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
        
def plot_chain_overview(trajectories: List[Trajectory], chain_df: pd.DataFrame, junctions: List[Circle], 
                       r_outer_list: Optional[List[float]] = None, path_length: float = 100.0, 
                       epsilon: float = 0.015, linger_delta: float = 5.0, decision_mode: str = "hybrid",
                       out_path: str = "Chain_Overview.png", show_paths: bool = True, 
                       show_centers: bool = False, centers_list: Optional[List[np.ndarray]] = None,
                       annotate_counts: bool = False):
    """
    Plot an overview of the decision chain analysis showing all junctions and trajectories.
    
    Args:
        trajectories: List of trajectory objects
        chain_df: DataFrame with chain analysis results
        junctions: List of junction Circle objects
        r_outer_list: Optional list of outer radii for each junction
        path_length: Path length for decision analysis
        epsilon: Minimum step size
        linger_delta: Linger distance beyond junction
        decision_mode: Decision mode ("pathlen", "radial", "hybrid")
        out_path: Output file path
        show_paths: Whether to show trajectory paths
        show_centers: Whether to show branch centers
        centers_list: Optional list of branch centers for each junction
        annotate_counts: Whether to annotate branch counts
    """
    fig, ax = plt.subplots(figsize=DEFAULT_PLOT_CONFIG.figsize, dpi=DEFAULT_PLOT_CONFIG.dpi)
    
    # Plot trajectories
    if show_paths:
        for traj in trajectories:
            ax.plot(traj.x, traj.z, 'b-', alpha=0.3, linewidth=0.5)
    
    # Plot junctions
    for i, junction in enumerate(junctions):
        # Main junction circle
        circle = plt.Circle((junction.cx, junction.cz), junction.r, 
                          color='red', alpha=0.3, edgecolor='red', linewidth=2)
        ax.add_patch(circle)
        
        # Outer radius if available
        if r_outer_list and i < len(r_outer_list) and r_outer_list[i] is not None:
            outer_circle = plt.Circle((junction.cx, junction.cz), r_outer_list[i], 
                                    color='red', alpha=0.1, edgecolor='red', linewidth=1, linestyle='--')
            ax.add_patch(outer_circle)
        
        # Junction label
        ax.text(junction.cx, junction.cz, f'J{i}', ha='center', va='center', 
                fontsize=DEFAULT_PLOT_CONFIG.fontsize, fontweight='bold', color='red')
    
    # Plot branch centers if available
    if show_centers and centers_list:
        for i, centers in enumerate(centers_list):
            if centers is not None and len(centers) > 0:
                junction = junctions[i]
                for j, center in enumerate(centers):
                    ax.plot(center[0], center[1], 'o', color=DEFAULT_PLOT_CONFIG.get_branch_color(j), 
                           markersize=8, markeredgecolor='black', markeredgewidth=1)
                    if annotate_counts:
                        ax.text(center[0], center[1], f'B{j}', ha='center', va='center', 
                               fontsize=8, fontweight='bold')
    
    ax.set_xlabel('X Position', fontsize=DEFAULT_PLOT_CONFIG.label_fontsize)
    ax.set_ylabel('Z Position', fontsize=DEFAULT_PLOT_CONFIG.label_fontsize)
    ax.set_title('Decision Chain Overview', fontsize=DEFAULT_PLOT_CONFIG.title_fontsize)
    ax.grid(True, alpha=DEFAULT_PLOT_CONFIG.grid_alpha)
    ax.set_aspect('equal')
    
    DEFAULT_PLOT_CONFIG.apply_to_figure(fig)
    plt.savefig(out_path, dpi=DEFAULT_PLOT_CONFIG.dpi, bbox_inches='tight')
    plt.close()


def plot_chain_small_multiples(trajectories: List[Trajectory], chain_df: pd.DataFrame, junctions: List[Circle],
                              r_outer_list: Optional[List[float]] = None, window_radius: float = 80.0,
                              path_length: float = 100.0, epsilon: float = 0.015, linger_delta: float = 5.0,
                              decision_mode: str = "hybrid", out_path: str = "Chain_SmallMultiples.png",
                              centers_list: Optional[List[np.ndarray]] = None, decisions_df: Optional[pd.DataFrame] = None):
    """
    Plot small multiples showing trajectory patterns around each junction with branch coloring and intercepts.
    
    Args:
        trajectories: List of trajectory objects
        chain_df: DataFrame with chain analysis results
        junctions: List of junction Circle objects
        r_outer_list: Optional list of outer radii for each junction
        window_radius: Radius of the window around each junction
        path_length: Path length for decision analysis
        epsilon: Minimum step size
        linger_delta: Linger distance beyond junction
        decision_mode: Decision mode ("pathlen", "radial", "hybrid")
        out_path: Output file path
        centers_list: Optional list of branch centers for each junction
        decisions_df: DataFrame with decision points (intercept_x, intercept_z, trajectory, junction_idx)
    """
    n_junctions = len(junctions)
    if n_junctions == 0:
        return
    
    # Debug: Print chain_df info
    print(f"DEBUG: chain_df shape: {chain_df.shape}")
    print(f"DEBUG: chain_df columns: {list(chain_df.columns)}")
    print(f"DEBUG: centers_list length: {len(centers_list) if centers_list else 'None'}")
    
    # Calculate grid layout
    cols = min(3, n_junctions)  # Max 3 columns
    rows = (n_junctions + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 6), dpi=DEFAULT_PLOT_CONFIG.dpi)
    if n_junctions == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if isinstance(axes, list) else [axes]
    else:
        axes = axes.flatten()
    
    for i, junction in enumerate(junctions):
        ax = axes[i] if i < len(axes) else None
        if ax is None:
            continue
            
        # Set window bounds
        x_min, x_max = junction.cx - window_radius, junction.cx + window_radius
        z_min, z_max = junction.cz - window_radius, junction.cz + window_radius
        
        # Get branch assignments for this junction
        branch_col = f"branch_j{i}"
        print(f"DEBUG: Looking for column '{branch_col}' in chain_df")
        
        if branch_col in chain_df.columns:
            print(f"DEBUG: Found column '{branch_col}', unique values: {chain_df[branch_col].unique()}")
            
            # Create trajectory to branch mapping
            traj_branches = {}
            for _, row in chain_df.iterrows():
                traj_id = row.get('trajectory', 0)
                branch = row.get(branch_col)
                if pd.notna(branch) and branch >= 0:  # Only valid branches
                    traj_branches[traj_id] = int(branch)
            
            print(f"DEBUG: Junction {i} - traj_branches: {len(traj_branches)} trajectories assigned")
            
            # Plot trajectories colored by branch
            branch_colors = {}
            for traj_idx, traj in enumerate(trajectories):
                branch = traj_branches.get(traj_idx, -1)
                if branch >= 0:  # Valid branch
                    if branch not in branch_colors:
                        branch_colors[branch] = DEFAULT_PLOT_CONFIG.get_branch_color(branch)
                    
                    # Filter trajectory points within window
                    mask = ((traj.x >= x_min) & (traj.x <= x_max) & 
                           (traj.z >= z_min) & (traj.z <= z_max))
                    if np.any(mask):
                        ax.plot(traj.x[mask], traj.z[mask], color=branch_colors[branch], 
                               alpha=0.6, linewidth=1)
            
            print(f"DEBUG: Junction {i} - branch_colors: {branch_colors}")
            
            # Plot decision intercepts using the same logic as plot_decision_intercepts
            if centers_list and i < len(centers_list) and centers_list[i] is not None:
                centers = centers_list[i]
                r_outer = r_outer_list[i] if r_outer_list and i < len(r_outer_list) else junction.r * 2
                
                print(f"DEBUG: Junction {i} - centers shape: {centers.shape if hasattr(centers, 'shape') else 'No shape'}")
                print(f"DEBUG: Junction {i} - r_outer: {r_outer}")
                
                # Create assignments DataFrame for this junction
                junction_assignments = chain_df[['trajectory', branch_col]].copy()
                junction_assignments = junction_assignments.rename(columns={branch_col: 'branch'})
                junction_assignments = junction_assignments[junction_assignments['branch'] >= 0]
                
                print(f"DEBUG: Junction {i} - junction_assignments shape: {junction_assignments.shape}")
                
                if len(junction_assignments) > 0:
                    # Plot decision intercepts for each branch
                    for branch_idx, center in enumerate(centers):
                        if branch_idx in branch_colors:
                            color = branch_colors[branch_idx]
                            
                            # Get trajectories for this branch
                            branch_trajectories = junction_assignments[junction_assignments['branch'] == branch_idx]
                            
                            print(f"DEBUG: Junction {i}, Branch {branch_idx} - {len(branch_trajectories)} trajectories")
                            
                            # For each trajectory in this branch, find where it intersects the outer radius
                            for _, row in branch_trajectories.iterrows():
                                traj_id = row['trajectory']
                                if traj_id < len(trajectories):
                                    traj = trajectories[traj_id]
                                    
                                    # Find intersection with outer radius circle
                                    # Calculate distance from junction center for each point
                                    distances = np.sqrt((traj.x - junction.cx)**2 + (traj.z - junction.cz)**2)
                                    
                                    # Find points near the outer radius
                                    radius_tolerance = 5.0  # Tolerance for intersection detection
                                    near_radius = np.abs(distances - r_outer) < radius_tolerance
                                    
                                    if np.any(near_radius):
                                        # Get the first point that's near the outer radius
                                        intersect_idx = np.where(near_radius)[0][0]
                                        intersect_x = traj.x[intersect_idx]
                                        intersect_z = traj.z[intersect_idx]
                                        
                                        # Only plot if within window
                                        if (x_min <= intersect_x <= x_max and z_min <= intersect_z <= z_max):
                                            ax.plot(intersect_x, intersect_z, 'o', color=color, 
                                                   markersize=4, markeredgecolor='black', markeredgewidth=0.5)
            else:
                print(f"DEBUG: Junction {i} - No centers_list or centers is None")
            
            # Create legend for branches
            if branch_colors:
                legend_elements = []
                for branch in sorted(branch_colors.keys()):
                    legend_elements.append(plt.Line2D([0], [0], color=branch_colors[branch], 
                                                    label=f'Branch {branch}'))
                ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
        else:
            print(f"DEBUG: Column '{branch_col}' not found in chain_df")
            # Fallback: plot all trajectories in blue if no branch data
            for traj in trajectories:
                mask = ((traj.x >= x_min) & (traj.x <= x_max) & 
                       (traj.z >= z_min) & (traj.z <= z_max))
                if np.any(mask):
                    ax.plot(traj.x[mask], traj.z[mask], 'b-', alpha=0.6, linewidth=1)
        
        # Plot junction
        circle = plt.Circle((junction.cx, junction.cz), junction.r, 
                          color='black', alpha=0.8, edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        
        # Plot outer radius if available
        if r_outer_list and i < len(r_outer_list) and r_outer_list[i] is not None:
            outer_circle = plt.Circle((junction.cx, junction.cz), r_outer_list[i], 
                                    color='orange', alpha=0.3, edgecolor='orange', linewidth=2)
            ax.add_patch(outer_circle)
        
        # Add junction center dot
        ax.plot(junction.cx, junction.cz, 'ko', markersize=3)
        
        # Count intercepts for title
        intercept_count = 0
        if branch_col in chain_df.columns:
            intercept_count = len(chain_df[chain_df[branch_col] >= 0])
        
        # Set axis properties
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(z_min, z_max)
        ax.set_xlabel('X Position', fontsize=DEFAULT_PLOT_CONFIG.label_fontsize)
        ax.set_ylabel('Z Position', fontsize=DEFAULT_PLOT_CONFIG.label_fontsize)
        ax.set_title(f'J{i} ({intercept_count} intercepts)', fontsize=DEFAULT_PLOT_CONFIG.title_fontsize)
        ax.grid(True, alpha=DEFAULT_PLOT_CONFIG.grid_alpha)
        ax.set_aspect('equal')
    
    # Hide unused subplots
    for i in range(n_junctions, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Decision Chain Small Multiples', fontsize=DEFAULT_PLOT_CONFIG.title_fontsize + 2)
    DEFAULT_PLOT_CONFIG.apply_to_figure(fig)
    plt.savefig(out_path, dpi=DEFAULT_PLOT_CONFIG.dpi, bbox_inches='tight')
    plt.close()
