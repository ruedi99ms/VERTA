"""
Example: Intent Recognition Analysis

This script demonstrates how to use the intent recognition module
to predict route choices before users reach decision points.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ra_data_loader import load_folder
from ra_geometry import Circle
from ra_decisions import discover_branches
from ra_intent_recognition import (
    analyze_intent_recognition,
    IntentRecognitionAnalyzer
)

# Configure visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def main():
    """Run complete intent recognition analysis"""
    
    print("=" * 70)
    print("  INTENT RECOGNITION ANALYSIS")
    print("  Predicting Route Choices Before Decision Points")
    print("=" * 70)
    print()
    
    # ========================================
    # 1. LOAD DATA
    # ========================================
    print("📂 Step 1: Loading trajectory data...")
    
    data_folder = "./data"  # Adjust to your data path
    output_folder = "./output_intent_recognition"
    os.makedirs(output_folder, exist_ok=True)
    
    # Load trajectories
    trajectories = load_folder(
        folder=data_folder,
        pattern="*.csv",
        require_time=True,
        scale=0.2  # Adjust scale as needed
    )
    
    print(f"   ✓ Loaded {len(trajectories)} trajectories")
    
    # Count trajectories with different data types
    n_with_gaze = sum(1 for tr in trajectories if tr.gaze_x is not None)
    n_with_physio = sum(1 for tr in trajectories if tr.heart_rate is not None)
    
    print(f"   • {n_with_gaze} trajectories with gaze data")
    print(f"   • {n_with_physio} trajectories with physiological data")
    print()
    
    # ========================================
    # 2. DEFINE JUNCTION
    # ========================================
    print("🎯 Step 2: Defining junction...")
    
    junction = Circle(cx=520.0, cz=330.0, r=20.0)
    
    print(f"   Junction: center=({junction.cx}, {junction.cz}), radius={junction.r}")
    print()
    
    # ========================================
    # 3. DISCOVER ACTUAL BRANCHES
    # ========================================
    print("🔍 Step 3: Discovering actual branch choices...")
    
    branch_output = os.path.join(output_folder, "branches")
    os.makedirs(branch_output, exist_ok=True)
    
    assignments, summary, centers = discover_branches(
        trajectories=trajectories,
        junction=junction,
        k=3,
        decision_mode="hybrid",
        r_outer=30.0,
        path_length=100.0,
        out_dir=branch_output,
        cluster_method="auto"
    )
    
    print(f"   ✓ Discovered {len(centers)} branches")
    print()
    print("   Branch distribution:")
    for _, row in summary.iterrows():
        print(f"      Branch {int(row['branch'])}: {int(row['count'])} trajectories ({row['percent']:.1f}%)")
    print()
    
    # ========================================
    # 4. INTENT RECOGNITION ANALYSIS
    # ========================================
    print("🧠 Step 4: Training intent recognition models...")
    print()
    
    # Define prediction distances (how far before junction to predict)
    prediction_distances = [100.0, 75.0, 50.0, 25.0]
    
    print(f"   Prediction distances: {prediction_distances}")
    print("   (These are distances BEFORE the junction where we'll try to predict choices)")
    print()
    
    # Run analysis
    intent_results = analyze_intent_recognition(
        trajectories=trajectories,
        junction=junction,
        actual_branches=assignments,
        output_dir=output_folder,
        prediction_distances=prediction_distances
    )
    
    if 'error' in intent_results:
        print(f"   ❌ Error: {intent_results['error']}")
        return
    
    # ========================================
    # 5. DISPLAY RESULTS
    # ========================================
    print()
    print("=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print()
    
    training_results = intent_results['training_results']
    
    # Show model performance at each distance
    print("📊 Model Performance by Prediction Distance:")
    print()
    
    models = training_results.get('models_trained', {})
    for dist in sorted(models.keys()):
        model_info = models[dist]
        acc = model_info['cv_mean_accuracy']
        std = model_info['cv_std_accuracy']
        n = model_info['n_samples']
        
        print(f"   Distance: {dist} units before junction")
        print(f"      • Accuracy: {acc:.1%} ± {std:.1%}")
        print(f"      • Training samples: {n}")
        print(f"      • Interpretation: {'🟢 Excellent' if acc > 0.8 else '🟡 Good' if acc > 0.6 else '🔴 Poor'}")
        print()
    
    # Show top features
    print("🎯 Most Important Features for Prediction:")
    print()
    
    feature_importance = training_results.get('feature_importance', {})
    if feature_importance:
        # Use the earliest (furthest) prediction distance
        earliest_dist = max(feature_importance.keys())
        importance = feature_importance[earliest_dist]
        
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        for i, (feature, score) in enumerate(sorted_features[:5], 1):
            bar = "█" * int(score * 50)
            print(f"   {i}. {feature:25s} {bar} {score:.3f}")
        print()
    
    # Show sample predictions
    print("🔮 Sample Predictions:")
    print()
    
    test_predictions = intent_results.get('test_predictions', {})
    
    for tid, pred_info in list(test_predictions.items())[:3]:
        actual = pred_info['actual_branch']
        print(f"   Trajectory: {tid}")
        print(f"   Actual choice: Branch {actual}")
        print()
        
        predictions = pred_info['predictions_by_distance']
        for dist in sorted(predictions.keys(), reverse=True):
            p = predictions[dist]
            status = "✓" if p['correct'] else "✗"
            print(f"      {dist} units before: Branch {p['predicted_branch']} "
                  f"(confidence: {p['confidence']:.1%}) {status}")
        print()
    
    # ========================================
    # 6. OUTPUT FILES
    # ========================================
    print("=" * 70)
    print("  OUTPUT FILES")
    print("=" * 70)
    print()
    print(f"   Output directory: {output_folder}")
    print()
    print("   Generated files:")
    print("      • intent_training_results.json    - Model training metrics")
    print("      • intent_test_predictions.json    - Sample prediction results")
    print("      • intent_feature_importance.png   - Feature importance plots")
    print("      • intent_accuracy_analysis.png    - Accuracy vs distance plot")
    print()
    
    # ========================================
    # 7. INTERPRETATION & INSIGHTS
    # ========================================
    print("=" * 70)
    print("  INTERPRETATION & INSIGHTS")
    print("=" * 70)
    print()
    
    print("📈 What These Results Mean:")
    print()
    
    # Calculate average accuracy
    if models:
        avg_accuracy = np.mean([m['cv_mean_accuracy'] for m in models.values()])
        
        print(f"   • Average prediction accuracy: {avg_accuracy:.1%}")
        print()
        
        if avg_accuracy > 0.8:
            print("   🟢 EXCELLENT: User intent is highly predictable")
            print("      - Route choices follow clear patterns")
            print("      - Early intervention/assistance is feasible")
            print("      - Predictive UI elements could be very effective")
        elif avg_accuracy > 0.6:
            print("   🟡 GOOD: User intent is moderately predictable")
            print("      - Some patterns exist but with variability")
            print("      - Contextual information may improve predictions")
            print("      - Adaptive systems could benefit users")
        else:
            print("   🔴 CHALLENGING: User behavior is highly variable")
            print("      - Route choices may be exploratory or random")
            print("      - More features or context may be needed")
            print("      - Consider analyzing subgroups separately")
        print()
    
    print("💡 Potential Applications:")
    print()
    print("   1. Proactive Wayfinding")
    print("      → Show navigation hints before users reach junctions")
    print()
    print("   2. Adaptive UI")
    print("      → Highlight relevant options based on predicted intent")
    print()
    print("   3. Congestion Prediction")
    print("      → Forecast traffic flow before bottlenecks occur")
    print()
    print("   4. User Experience Optimization")
    print("      → Identify confusing junctions where predictions fail")
    print()
    print("   5. Safety Systems")
    print("      → Detect unexpected behaviors early in emergencies")
    print()
    
    print("=" * 70)
    print("  ANALYSIS COMPLETE")
    print("=" * 70)
    print()
    print(f"   Review the output files in: {output_folder}")
    print()


def demo_realtime_prediction():
    """
    Demonstrate real-time intent prediction as a user approaches a junction
    """
    print("\n" + "=" * 70)
    print("  REAL-TIME PREDICTION DEMO")
    print("=" * 70)
    print()
    print("This demo shows how intent predictions change as a user")
    print("approaches a junction (from 100 units away to 25 units away)")
    print()
    
    # Note: This would require a trained analyzer and trajectory data
    # Shown here as a conceptual example
    
    print("Example output:")
    print()
    print("  Time: T-10.0s | Distance: 100 units")
    print("    Predicted: Branch 2 (confidence: 65%)")
    print("    Top features: approach_angle, current_speed")
    print()
    print("  Time: T-7.5s  | Distance: 75 units")
    print("    Predicted: Branch 2 (confidence: 72%)")
    print("    Top features: gaze_alignment, curvature")
    print()
    print("  Time: T-5.0s  | Distance: 50 units")
    print("    Predicted: Branch 2 (confidence: 85%)")
    print("    Top features: lateral_offset, gaze_angle")
    print()
    print("  Time: T-2.5s  | Distance: 25 units")
    print("    Predicted: Branch 2 (confidence: 93%)")
    print("    Top features: approach_angle, lateral_offset")
    print()
    print("  ACTUAL CHOICE: Branch 2 ✓")
    print()
    print("Confidence increases as user gets closer to junction!")
    print()


if __name__ == "__main__":
    try:
        main()
        demo_realtime_prediction()
        
    except FileNotFoundError as e:
        print()
        print("❌ Error: Data folder not found")
        print()
        print("Please update the 'data_folder' variable in this script")
        print("to point to your trajectory data directory.")
        print()
        print(f"Error details: {e}")
        
    except Exception as e:
        print()
        print(f"❌ Error: {e}")
        print()
        import traceback
        traceback.print_exc()

