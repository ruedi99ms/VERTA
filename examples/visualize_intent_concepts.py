"""
Conceptual Visualizations for Intent Recognition

This script creates example visualizations showing what intent recognition
analysis outputs would look like (does not require actual data).
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle as MPLCircle, FancyArrowPatch
import matplotlib.patches as mpatches

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'


def visualize_intent_prediction_timeline():
    """Show how predictions evolve as user approaches junction"""
    
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(14, 10), 
                                             height_ratios=[2, 1])
    
    # === TOP PANEL: Spatial visualization ===
    
    # Junction
    junction_center = (100, 100)
    junction_radius = 15
    
    circle = MPLCircle(junction_center, junction_radius, 
                       fill=False, edgecolor='red', linewidth=3, 
                       label='Junction')
    ax_top.add_patch(circle)
    
    # Branch directions
    branches = [
        (100 + 60*np.cos(np.radians(30)), 100 + 60*np.sin(np.radians(30)), 'Branch 1'),
        (100 + 60*np.cos(np.radians(150)), 100 + 60*np.sin(np.radians(150)), 'Branch 2'),
        (100 + 60*np.cos(np.radians(270)), 100 + 60*np.sin(np.radians(270)), 'Branch 3'),
    ]
    
    for bx, bz, label in branches:
        ax_top.plot([100, bx], [100, bz], 'k--', alpha=0.3, linewidth=2)
        ax_top.text(bx, bz, label, fontsize=11, fontweight='bold',
                   ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # User trajectory approaching junction
    t = np.linspace(0, 1, 50)
    trajectory_x = 20 + 80*t + 10*np.sin(3*t)
    trajectory_z = 40 + 60*t + 5*np.cos(2*t)
    
    ax_top.plot(trajectory_x, trajectory_z, 'b-', linewidth=2, alpha=0.5, 
               label='Trajectory')
    
    # Prediction points
    prediction_distances = [100, 75, 50, 25]
    prediction_indices = [5, 15, 28, 40]
    confidences = [0.65, 0.73, 0.87, 0.93]
    colors = ['orange', 'gold', 'yellowgreen', 'green']
    
    for i, (idx, dist, conf, color) in enumerate(zip(prediction_indices, 
                                                       prediction_distances, 
                                                       confidences, colors)):
        x, z = trajectory_x[idx], trajectory_z[idx]
        
        # Prediction point
        ax_top.scatter([x], [z], s=300, c=[color], marker='o', 
                      edgecolors='black', linewidths=2, zorder=10)
        
        # Distance indicator
        ax_top.annotate(f'{dist}u', xy=(x, z), xytext=(x-15, z+8),
                       fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
        
        # Confidence indicator
        ax_top.text(x, z-12, f'{conf:.0%}', fontsize=8, ha='center',
                   fontweight='bold')
    
    # Arrow showing movement direction
    for i in range(0, len(trajectory_x)-1, 5):
        dx = trajectory_x[i+1] - trajectory_x[i]
        dz = trajectory_z[i+1] - trajectory_z[i]
        ax_top.arrow(trajectory_x[i], trajectory_z[i], dx, dz,
                    head_width=2, head_length=2, fc='blue', ec='blue',
                    alpha=0.3, linewidth=1)
    
    ax_top.set_xlim(0, 180)
    ax_top.set_ylim(20, 180)
    ax_top.set_aspect('equal')
    ax_top.set_xlabel('X Coordinate', fontsize=12)
    ax_top.set_ylabel('Z Coordinate', fontsize=12)
    ax_top.set_title('Intent Recognition: Predictions Along Trajectory', 
                     fontsize=14, fontweight='bold')
    ax_top.legend(loc='upper left', fontsize=10)
    ax_top.grid(True, alpha=0.3)
    
    # === BOTTOM PANEL: Prediction confidence over time ===
    
    time_points = np.array([10, 7.5, 5.0, 2.5, 0])
    distances = np.array([100, 75, 50, 25, 0])
    
    # Predicted probabilities for each branch
    branch1_probs = [0.30, 0.22, 0.10, 0.05, 0.02]
    branch2_probs = [0.65, 0.73, 0.87, 0.93, 0.97]  # Correct prediction
    branch3_probs = [0.05, 0.05, 0.03, 0.02, 0.01]
    
    ax_bottom.plot(time_points, branch1_probs, 'o-', linewidth=3, 
                   markersize=8, label='Branch 1', color='steelblue')
    ax_bottom.plot(time_points, branch2_probs, 'o-', linewidth=3, 
                   markersize=8, label='Branch 2 (Actual)', color='green')
    ax_bottom.plot(time_points, branch3_probs, 'o-', linewidth=3, 
                   markersize=8, label='Branch 3', color='coral')
    
    # Confidence threshold line
    ax_bottom.axhline(y=0.75, color='red', linestyle='--', linewidth=2, 
                     alpha=0.5, label='Confidence Threshold (75%)')
    
    # Shaded region for high confidence
    ax_bottom.fill_between(time_points, 0.75, 1.0, alpha=0.1, color='green')
    ax_bottom.text(8, 0.82, 'High Confidence\nZone', fontsize=10, 
                  fontweight='bold', alpha=0.5)
    
    # Add second x-axis for distance
    ax_bottom2 = ax_bottom.twiny()
    ax_bottom2.set_xlim(ax_bottom.get_xlim())
    ax_bottom2.set_xticks(time_points)
    ax_bottom2.set_xticklabels([f'{d}u' for d in distances])
    ax_bottom2.set_xlabel('Distance to Junction', fontsize=12)
    
    ax_bottom.set_xlabel('Time to Junction (seconds)', fontsize=12)
    ax_bottom.set_ylabel('Prediction Probability', fontsize=12)
    ax_bottom.set_title('Branch Prediction Confidence Over Time', 
                       fontsize=14, fontweight='bold')
    ax_bottom.legend(loc='center left', fontsize=10)
    ax_bottom.set_ylim(0, 1.0)
    ax_bottom.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('intent_prediction_timeline.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: intent_prediction_timeline.png")
    plt.close()


def visualize_feature_importance_comparison():
    """Compare feature importance at different prediction distances"""
    
    features = [
        'Lateral Offset',
        'Approach Angle',
        'Gaze Angle',
        'Current Speed',
        'Gaze Alignment',
        'Curvature',
        'Heart Rate Trend',
        'Sinuosity'
    ]
    
    # Importance at different distances (synthetic data)
    importance_100u = [0.05, 0.18, 0.15, 0.10, 0.08, 0.12, 0.06, 0.04]
    importance_50u = [0.15, 0.17, 0.14, 0.09, 0.10, 0.10, 0.08, 0.05]
    importance_25u = [0.22, 0.18, 0.14, 0.08, 0.09, 0.08, 0.07, 0.04]
    
    x = np.arange(len(features))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    bars1 = ax.barh([i - width for i in x], importance_100u, width, 
                    label='100 units before', color='lightcoral')
    bars2 = ax.barh(x, importance_50u, width,
                    label='50 units before', color='gold')
    bars3 = ax.barh([i + width for i in x], importance_25u, width,
                    label='25 units before', color='lightgreen')
    
    ax.set_yticks(x)
    ax.set_yticklabels(features, fontsize=11)
    ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
    ax.set_title('Feature Importance by Prediction Distance\n' + 
                 'How Different Features Matter at Different Stages',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            width_val = bar.get_width()
            ax.text(width_val + 0.005, bar.get_y() + bar.get_height()/2,
                   f'{width_val:.3f}', ha='left', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('intent_feature_importance_comparison.png', dpi=300, 
                bbox_inches='tight')
    print("✓ Saved: intent_feature_importance_comparison.png")
    plt.close()


def visualize_accuracy_vs_distance():
    """Show how prediction accuracy changes with distance"""
    
    distances = np.array([100, 75, 50, 25])
    
    # Synthetic data for different scenarios
    scenario1_acc = [0.68, 0.75, 0.84, 0.92]
    scenario1_std = [0.05, 0.05, 0.04, 0.03]
    
    scenario2_acc = [0.55, 0.62, 0.71, 0.83]
    scenario2_std = [0.08, 0.07, 0.06, 0.04]
    
    scenario3_acc = [0.78, 0.84, 0.89, 0.94]
    scenario3_std = [0.04, 0.03, 0.03, 0.02]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot scenarios
    ax.errorbar(distances, scenario1_acc, yerr=scenario1_std,
               marker='o', linewidth=3, markersize=10, capsize=5,
               label='Moderate Predictability', color='steelblue')
    
    ax.errorbar(distances, scenario2_acc, yerr=scenario2_std,
               marker='s', linewidth=3, markersize=10, capsize=5,
               label='Low Predictability (Exploratory)', color='coral')
    
    ax.errorbar(distances, scenario3_acc, yerr=scenario3_std,
               marker='^', linewidth=3, markersize=10, capsize=5,
               label='High Predictability (Clear Patterns)', color='green')
    
    # Threshold lines
    ax.axhline(y=0.80, color='orange', linestyle='--', linewidth=2, 
              alpha=0.5, label='Good Threshold (80%)')
    ax.axhline(y=0.90, color='darkgreen', linestyle='--', linewidth=2,
              alpha=0.5, label='Excellent Threshold (90%)')
    
    # Shaded confidence zones
    ax.fill_between([0, 110], 0.9, 1.0, alpha=0.1, color='green',
                    label='Excellent Zone')
    ax.fill_between([0, 110], 0.8, 0.9, alpha=0.1, color='yellow',
                    label='Good Zone')
    ax.fill_between([0, 110], 0.0, 0.8, alpha=0.1, color='red',
                    label='Challenging Zone')
    
    ax.set_xlabel('Distance Before Junction (units)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Prediction Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Intent Recognition Accuracy vs Prediction Distance\n' +
                'How Early Can We Reliably Predict User Choices?',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.set_xlim(15, 110)
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, alpha=0.3)
    
    # Add annotations
    ax.annotate('Closer = Higher Accuracy', 
               xy=(30, 0.88), xytext=(60, 0.65),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'),
               fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('intent_accuracy_vs_distance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: intent_accuracy_vs_distance.png")
    plt.close()


def visualize_real_time_prediction_dashboard():
    """Mock dashboard showing real-time intent prediction"""
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # === MAIN MAP ===
    ax_map = fig.add_subplot(gs[0:2, 0:2])
    
    # Junction and branches
    junction_x, junction_z = 100, 100
    circle = MPLCircle((junction_x, junction_z), 15, 
                      fill=False, edgecolor='red', linewidth=3)
    ax_map.add_patch(circle)
    
    # Current user position
    user_x, user_z = 60, 70
    ax_map.scatter([user_x], [user_z], s=500, c='blue', marker='o',
                  edgecolors='black', linewidths=3, zorder=10,
                  label='Current Position')
    
    # Predicted path (semi-transparent)
    predicted_path_x = np.linspace(user_x, 120, 30)
    predicted_path_z = np.linspace(user_z, 130, 30)
    ax_map.plot(predicted_path_x, predicted_path_z, 'g--', linewidth=3,
               alpha=0.5, label='Predicted Path')
    
    # Distance indicator
    dist = np.sqrt((user_x - junction_x)**2 + (user_z - junction_z)**2)
    ax_map.plot([user_x, junction_x], [user_z, junction_z], 'k--', 
               linewidth=1, alpha=0.5)
    ax_map.text((user_x + junction_x)/2, (user_z + junction_z)/2 + 5,
               f'{dist:.0f} units', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax_map.set_xlim(40, 140)
    ax_map.set_ylim(50, 150)
    ax_map.set_aspect('equal')
    ax_map.set_title('Live Position & Predicted Path', fontsize=12, fontweight='bold')
    ax_map.legend(loc='upper left')
    ax_map.grid(True, alpha=0.3)
    
    # === PREDICTION PROBABILITIES ===
    ax_probs = fig.add_subplot(gs[0, 2])
    
    branches = ['Branch 1', 'Branch 2', 'Branch 3']
    probabilities = [0.15, 0.78, 0.07]
    colors_prob = ['lightcoral', 'lightgreen', 'lightblue']
    
    bars = ax_probs.barh(branches, probabilities, color=colors_prob, 
                         edgecolor='black', linewidth=2)
    
    # Add percentage labels
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        ax_probs.text(prob + 0.02, i, f'{prob:.0%}', 
                     va='center', fontweight='bold', fontsize=11)
    
    ax_probs.set_xlim(0, 1.0)
    ax_probs.set_xlabel('Probability', fontsize=10)
    ax_probs.set_title('Branch Predictions\n(78% confidence)', 
                      fontsize=11, fontweight='bold')
    ax_probs.axvline(x=0.75, color='red', linestyle='--', alpha=0.5)
    ax_probs.grid(True, alpha=0.3, axis='x')
    
    # === KEY FEATURES ===
    ax_features = fig.add_subplot(gs[1, 2])
    
    feature_names = ['Approach\nAngle', 'Lateral\nOffset', 'Gaze\nAngle', 
                     'Speed', 'Curvature']
    feature_values = [0.85, 0.72, 0.91, 0.65, 0.58]
    colors_feat = ['green' if v > 0.7 else 'orange' if v > 0.5 else 'red' 
                   for v in feature_values]
    
    bars_feat = ax_features.barh(feature_names, feature_values, 
                                color=colors_feat, edgecolor='black', linewidth=1)
    
    ax_features.set_xlim(0, 1.0)
    ax_features.set_xlabel('Feature Score', fontsize=10)
    ax_features.set_title('Key Features\n(Normalized)', fontsize=11, fontweight='bold')
    ax_features.grid(True, alpha=0.3, axis='x')
    
    # === STATUS PANEL ===
    ax_status = fig.add_subplot(gs[2, :])
    ax_status.axis('off')
    
    status_text = """
    📊 INTENT RECOGNITION STATUS
    
    ⏱️  Time to Junction: 3.2 seconds
    📏 Distance: 52 units
    🎯 Predicted Branch: Branch 2 (78% confidence)
    
    🧠 Decision Factors:
       • Strong alignment between gaze and Branch 2 direction
       • Approach angle matches Branch 2 trajectory
       • Lateral offset indicates rightward bias
       • Speed consistent with confident navigation
    
    ✅ PREDICTION READY - Confidence threshold met (>75%)
    💡 Suggested Action: Highlight Branch 2 option in UI
    """
    
    ax_status.text(0.05, 0.5, status_text, fontsize=10, family='monospace',
                  verticalalignment='center',
                  bbox=dict(boxstyle='round', facecolor='lightgray', 
                           alpha=0.3, pad=1))
    
    fig.suptitle('Real-Time Intent Recognition Dashboard', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig('intent_realtime_dashboard.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: intent_realtime_dashboard.png")
    plt.close()


def visualize_use_cases():
    """Show various use cases for intent recognition"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    use_cases = [
        {
            'title': '1. Proactive Wayfinding',
            'description': 'Show navigation hints\nbefore junctions',
            'icon': '🗺️',
            'example': 'Predict user going to\nExit B → Show "Exit B\nAhead" sign early'
        },
        {
            'title': '2. Adaptive UI',
            'description': 'Highlight likely choices\nin real-time',
            'icon': '🎨',
            'example': 'User approaching menu\n→ Pre-highlight predicted\noption (saves time)'
        },
        {
            'title': '3. Congestion Management',
            'description': 'Predict traffic flows\n5-10 seconds ahead',
            'icon': '🚦',
            'example': '70% of users heading\nto Path A → Suggest\nalternative Path B'
        },
        {
            'title': '4. Anomaly Detection',
            'description': 'Detect unusual behavior\npatterns',
            'icon': '⚠️',
            'example': 'Predicted: Straight\nActual: Random turns\n→ User may be lost'
        },
        {
            'title': '5. Performance Optimization',
            'description': 'Preload assets based\non predictions',
            'icon': '⚡',
            'example': '85% confidence going\nleft → Preload left\narea assets now'
        },
        {
            'title': '6. Personalization',
            'description': 'Learn individual\npreferences',
            'icon': '👤',
            'example': 'User A always avoids\nstairs → Suggest\nelevator routes'
        }
    ]
    
    for ax, use_case in zip(axes, use_cases):
        ax.axis('off')
        
        # Background box
        from matplotlib.patches import FancyBboxPatch
        box = FancyBboxPatch((0.05, 0.05), 0.9, 0.9, 
                            boxstyle="round,pad=0.05",
                            edgecolor='black', facecolor='lightblue',
                            alpha=0.2, linewidth=2,
                            transform=ax.transAxes)
        ax.add_patch(box)
        
        # Icon
        ax.text(0.5, 0.75, use_case['icon'], fontsize=60, ha='center',
               transform=ax.transAxes)
        
        # Title
        ax.text(0.5, 0.60, use_case['title'], fontsize=13, 
               fontweight='bold', ha='center', transform=ax.transAxes)
        
        # Description
        ax.text(0.5, 0.45, use_case['description'], fontsize=10,
               ha='center', style='italic', transform=ax.transAxes)
        
        # Example
        ax.text(0.5, 0.20, use_case['example'], fontsize=9,
               ha='center', transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='white', 
                        alpha=0.8, pad=0.5))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    fig.suptitle('Intent Recognition: Real-World Applications', 
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('intent_use_cases.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: intent_use_cases.png")
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("  Generating Intent Recognition Concept Visualizations")
    print("=" * 60)
    print()
    
    print("Creating visualizations...")
    print()
    
    visualize_intent_prediction_timeline()
    visualize_feature_importance_comparison()
    visualize_accuracy_vs_distance()
    visualize_real_time_prediction_dashboard()
    visualize_use_cases()
    
    print()
    print("=" * 60)
    print("  ✅ All visualizations created successfully!")
    print("=" * 60)
    print()
    print("Generated files:")
    print("  • intent_prediction_timeline.png")
    print("  • intent_feature_importance_comparison.png")
    print("  • intent_accuracy_vs_distance.png")
    print("  • intent_realtime_dashboard.png")
    print("  • intent_use_cases.png")
    print()
    print("These visualizations show conceptual examples of what")
    print("intent recognition analysis would produce with real data.")

