"""
Intent Recognition Module for VERTA

Predicts user route choices before they reach decision points using machine learning on trajectory, gaze, and physiological features.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

# Suppress sklearn warnings about invalid values in division
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in divide')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='All-NaN slice encountered')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn')

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("scikit-learn not available. Intent recognition features disabled.")
    print("Install with: pip install scikit-learn")

try:
    from .verta_data_loader import Trajectory, has_gaze_data, has_physio_data
    from .verta_geometry import Circle
    from .verta_logging import get_logger
except ImportError:
    from verta.verta_data_loader import Trajectory, has_gaze_data, has_physio_data
    from verta.verta_geometry import Circle
    from verta.verta_logging import get_logger

logger = get_logger()


@dataclass
class IntentFeatures:
    """Features extracted for intent prediction"""
    # Spatial features
    distance_to_junction: float
    approach_angle: float  # Angle of approach trajectory
    lateral_offset: float  # How far off center-line
    
    # Kinematic features
    current_speed: float
    speed_change_rate: float  # Acceleration/deceleration
    curvature: float  # Path curvature
    sinuosity: float  # Path complexity
    
    # Temporal features
    time_to_junction: float  # Estimated based on current speed
    
    # Gaze features (if available)
    gaze_angle: Optional[float]  # Where they're looking
    gaze_alignment: Optional[float]  # Gaze-movement alignment
    head_rotation_rate: Optional[float]  # How fast head is turning
    
    # Physiological features (if available)
    heart_rate: Optional[float]
    heart_rate_trend: Optional[float]  # Increasing/decreasing
    pupil_dilation: Optional[float]
    pupil_change_rate: Optional[float]
    
    # Contextual features
    previous_junction_choice: Optional[int]  # Previous branch choice
    trajectory_id: str


class IntentRecognitionAnalyzer:
    """Analyze and predict user intent before decision points"""
    
    def __init__(self, 
                 prediction_distances: List[float] = [100.0, 75.0, 50.0, 25.0],
                 model_type: str = "random_forest",
                 output_dir: Optional[str] = None):
        """
        Initialize analyzer
        
        Args:
            prediction_distances: Distances before junction to make predictions
            model_type: "random_forest" or "gradient_boosting"
            output_dir: Directory to save models (optional)
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for intent recognition")
            
        self.prediction_distances = sorted(prediction_distances, reverse=True)
        self.model_type = model_type
        self.output_dir = output_dir
        self.models = {}  # One model per prediction distance
        self.scalers = {}  # Feature scalers
        self.feature_importance = {}
        
    def extract_features_at_distance(self,
                                     trajectory: Trajectory,
                                     junction: Circle,
                                     distance_before: float,
                                     previous_choice: Optional[int] = None) -> Optional[IntentFeatures]:
        """
        Extract features at a specific distance before junction
        
        Args:
            trajectory: Trajectory object
            junction: Junction circle
            distance_before: How far before junction to extract features
            previous_choice: Branch chosen at previous junction (if any)
        """
        # Find point approximately distance_before from junction
        distances = np.sqrt((trajectory.x - junction.cx)**2 + 
                           (trajectory.z - junction.cz)**2)
        
        # Find entry point into junction
        inside = distances <= junction.r
        if not inside.any():
            return None
            
        entry_idx = int(np.argmax(inside))
        
        # Find point at target distance before entry
        target_distance = junction.r + distance_before
        before_entry = distances[:entry_idx]
        
        if len(before_entry) < 10:  # Need sufficient history
            return None
            
        # Find closest point to target distance
        dist_diff = np.abs(before_entry - target_distance)
        feature_idx = int(np.argmin(dist_diff))
        
        if feature_idx < 5:  # Need some history for derivatives
            return None
            
        # Extract features
        features = self._compute_features(
            trajectory, junction, feature_idx, entry_idx, previous_choice
        )
        
        return features
    
    def _compute_features(self,
                         tr: Trajectory,
                         junction: Circle,
                         idx: int,
                         entry_idx: int,
                         previous_choice: Optional[int]) -> IntentFeatures:
        """Compute all features at a specific point in trajectory"""
        
        # Window for computing derivatives/trends
        window = min(10, idx)
        start_idx = max(0, idx - window)
        
        # === SPATIAL FEATURES ===
        dist_to_junction = float(np.sqrt(
            (tr.x[idx] - junction.cx)**2 + (tr.z[idx] - junction.cz)**2
        ))
        
        # Approach angle (direction toward junction vs current heading)
        dx = tr.x[idx] - tr.x[start_idx]
        dz = tr.z[idx] - tr.z[start_idx]
        heading_angle = np.arctan2(dz, dx)
        
        to_junction_x = junction.cx - tr.x[idx]
        to_junction_z = junction.cz - tr.z[idx]
        junction_angle = np.arctan2(to_junction_z, to_junction_x)
        
        approach_angle = float(np.abs(self._angle_diff(heading_angle, junction_angle)))
        
        # Lateral offset (perpendicular distance from straight line to junction)
        lateral_offset = float(np.abs(
            (to_junction_z * dx - to_junction_x * dz) / 
            (np.sqrt(dx**2 + dz**2) + 1e-6)
        ))
        
        # Kinematic Features
        dx_seg = np.diff(tr.x[start_idx:idx+1])
        dz_seg = np.diff(tr.z[start_idx:idx+1])
        speeds = np.sqrt(dx_seg**2 + dz_seg**2)
        
        if tr.t is not None and len(tr.t) > idx:
            dt = np.diff(tr.t[start_idx:idx+1])
            dt = np.where(dt > 0, dt, 1.0)  # Avoid division by zero
            speeds = speeds / dt
        
        current_speed = float(np.mean(speeds[-3:]) if len(speeds) >= 3 else speeds[-1] if len(speeds) > 0 else 0.0)
        
        # Acceleration (speed change rate)
        if len(speeds) >= 5:
            speed_change_rate = float(np.mean(np.diff(speeds[-5:])))
        else:
            speed_change_rate = 0.0
        
        # Curvature (angle changes along path)
        if len(dx_seg) >= 3:
            angles = np.arctan2(dz_seg, dx_seg)
            angle_diffs = np.diff(angles)
            # Normalize to [-pi, pi]
            angle_diffs = np.arctan2(np.sin(angle_diffs), np.cos(angle_diffs))
            curvature = float(np.mean(np.abs(angle_diffs)))
        else:
            curvature = 0.0
        
        # Sinuosity (path_length / straight_line_distance)
        path_length = float(np.sum(np.sqrt(dx_seg**2 + dz_seg**2)))
        straight_line = float(np.sqrt(
            (tr.x[idx] - tr.x[start_idx])**2 + 
            (tr.z[idx] - tr.z[start_idx])**2
        ))
        sinuosity = path_length / (straight_line + 1e-6)
        
        # Temporal Features
        time_to_junction = (dist_to_junction - junction.r) / (current_speed + 1e-6)
        
        # Gaze Features
        gaze_angle = None
        gaze_alignment = None
        head_rotation_rate = None
        
        if has_gaze_data(tr):
            # Gaze direction
            if tr.gaze_x is not None and tr.gaze_y is not None:
                gaze_angle = float(np.arctan2(tr.gaze_y[idx], tr.gaze_x[idx]))
                
                # Gaze-movement alignment (are they looking where they're going?)
                gaze_alignment = float(np.cos(gaze_angle - heading_angle))
            
            # Head rotation rate
            if tr.head_forward_x is not None and tr.head_forward_z is not None:
                head_angles = np.arctan2(
                    tr.head_forward_z[start_idx:idx+1],
                    tr.head_forward_x[start_idx:idx+1]
                )
                if len(head_angles) >= 2:
                    head_rotation_rate = float(np.std(np.diff(head_angles)))
        
        # Physiological Features
        heart_rate_val = None
        heart_rate_trend = None
        pupil_dilation = None
        pupil_change_rate = None
        
        if has_physio_data(tr):
            if tr.heart_rate is not None and not np.isnan(tr.heart_rate[idx]):
                heart_rate_val = float(tr.heart_rate[idx])
                
                # Heart rate trend
                hr_window = tr.heart_rate[start_idx:idx+1]
                hr_window = hr_window[~np.isnan(hr_window)]
                if len(hr_window) >= 2:
                    heart_rate_trend = float(hr_window[-1] - hr_window[0])
            
            # Pupil dilation (average of left and right)
            if tr.pupil_l is not None and tr.pupil_r is not None:
                pupil_l = tr.pupil_l[idx]
                pupil_r = tr.pupil_r[idx]
                if not (np.isnan(pupil_l) or np.isnan(pupil_r)):
                    pupil_dilation = float((pupil_l + pupil_r) / 2.0)
                    
                    # Pupil change rate
                    pupil_avg = (tr.pupil_l[start_idx:idx+1] + tr.pupil_r[start_idx:idx+1]) / 2.0
                    pupil_avg = pupil_avg[~np.isnan(pupil_avg)]
                    if len(pupil_avg) >= 2:
                        pupil_change_rate = float(pupil_avg[-1] - pupil_avg[0])
        
        return IntentFeatures(
            distance_to_junction=dist_to_junction,
            approach_angle=approach_angle,
            lateral_offset=lateral_offset,
            current_speed=current_speed,
            speed_change_rate=speed_change_rate,
            curvature=curvature,
            sinuosity=sinuosity,
            time_to_junction=time_to_junction,
            gaze_angle=gaze_angle,
            gaze_alignment=gaze_alignment,
            head_rotation_rate=head_rotation_rate,
            heart_rate=heart_rate_val,
            heart_rate_trend=heart_rate_trend,
            pupil_dilation=pupil_dilation,
            pupil_change_rate=pupil_change_rate,
            previous_junction_choice=previous_choice,
            trajectory_id=tr.tid
        )
    
    @staticmethod
    def _angle_diff(a1: float, a2: float) -> float:
        """Compute smallest angle difference between two angles"""
        diff = a1 - a2
        return np.arctan2(np.sin(diff), np.cos(diff))
    
    def features_to_array(self, features: IntentFeatures) -> np.ndarray:
        """Convert IntentFeatures to numpy array for ML"""
        # Build feature vector, replacing None with NaN
        # Also replace inf values with NaN to avoid sklearn warnings
        feature_vector = [
            features.distance_to_junction,
            features.approach_angle,
            features.lateral_offset,
            features.current_speed,
            features.speed_change_rate,
            features.curvature,
            features.sinuosity,
            features.time_to_junction,
            features.gaze_angle if features.gaze_angle is not None else np.nan,
            features.gaze_alignment if features.gaze_alignment is not None else np.nan,
            features.head_rotation_rate if features.head_rotation_rate is not None else np.nan,
            features.heart_rate if features.heart_rate is not None else np.nan,
            features.heart_rate_trend if features.heart_rate_trend is not None else np.nan,
            features.pupil_dilation if features.pupil_dilation is not None else np.nan,
            features.pupil_change_rate if features.pupil_change_rate is not None else np.nan,
            features.previous_junction_choice if features.previous_junction_choice is not None else -1,
        ]
        
        arr = np.array(feature_vector, dtype=float)
        # Replace inf with nan to avoid sklearn warnings
        arr[~np.isfinite(arr)] = np.nan
        return arr
    
    def get_feature_names(self) -> List[str]:
        """Get ordered list of feature names"""
        return [
            'distance_to_junction',
            'approach_angle',
            'lateral_offset',
            'current_speed',
            'speed_change_rate',
            'curvature',
            'sinuosity',
            'time_to_junction',
            'gaze_angle',
            'gaze_alignment',
            'head_rotation_rate',
            'heart_rate',
            'heart_rate_trend',
            'pupil_dilation',
            'pupil_change_rate',
            'previous_junction_choice',
        ]
    
    def train_models(self,
                    trajectories: List[Trajectory],
                    junction: Circle,
                    actual_branches: Dict[str, int],
                    previous_choices: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        """
        Train intent prediction models at multiple prediction distances
        
        Args:
            trajectories: List of trajectory objects
            junction: Junction to analyze
            actual_branches: Dict mapping trajectory_id -> actual branch chosen
            previous_choices: Dict mapping trajectory_id -> previous junction choice
            
        Returns:
            Training results and metrics
        """
        logger.info("Training intent recognition models...")
        
        if previous_choices is None:
            previous_choices = {}
        
        results = {
            'models_trained': {},
            'feature_importance': {},
            'cross_validation_scores': {},
            'sample_sizes': {}
        }
        
        # Train one model per prediction distance
        for dist in self.prediction_distances:
            logger.info(f"Training model for {dist} units before junction...")
            
            # Extract features for all trajectories
            X_list = []
            y_list = []
            valid_trajectories = []
            
            for tr in trajectories:
                if tr.tid not in actual_branches:
                    continue
                
                prev_choice = previous_choices.get(tr.tid, None)
                features = self.extract_features_at_distance(
                    tr, junction, dist, prev_choice
                )
                
                if features is None:
                    continue
                
                X_list.append(self.features_to_array(features))
                y_list.append(actual_branches[tr.tid])
                valid_trajectories.append(tr.tid)
            
            if len(X_list) < 10:
                logger.warning(f"Insufficient data for distance {dist} (only {len(X_list)} samples)")
                continue
            
            X = np.vstack(X_list)
            y = np.array(y_list)
            
            # Handle NaN values (impute with median)
            nan_mask = np.isnan(X)
            if nan_mask.any():
                # Suppress warning when computing median of all-NaN columns
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    col_medians = np.nanmedian(X, axis=0)
                
                for col in range(X.shape[1]):
                    # If entire column is NaN, use 0
                    if np.isnan(col_medians[col]):
                        X[nan_mask[:, col], col] = 0.0
                    else:
                        X[nan_mask[:, col], col] = col_medians[col]
            
            # Replace any remaining inf values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Standardize features
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
            
            # Train model
            if self.model_type == "random_forest":
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=10,
                    random_state=42
                )
            else:  # gradient_boosting
                model = GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                )
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_scaled, y, cv=5)
            
            # Train on full dataset
            model.fit(X_scaled, y)
            
            # Store model and scaler
            self.models[dist] = model
            self.scalers[dist] = scaler
            
            # Save model and scaler to disk
            if self.output_dir:
                models_dir = os.path.join(self.output_dir, "models")
                os.makedirs(models_dir, exist_ok=True)
                
                # Save model
                model_path = os.path.join(models_dir, f"model_{dist}.pkl")
                import pickle
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                
                # Save scaler
                scaler_path = os.path.join(models_dir, f"scaler_{dist}.pkl")
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scaler, f)
                
                logger.info(f"  ✓ Saved model to {model_path}")
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                self.feature_importance[dist] = dict(zip(
                    self.get_feature_names(), importance
                ))
            
            # Store results
            results['models_trained'][dist] = {
                'n_samples': len(X_list),
                'n_features': X.shape[1],
                'cv_mean_accuracy': float(np.mean(cv_scores)),
                'cv_std_accuracy': float(np.std(cv_scores)),
                'valid_trajectories': valid_trajectories
            }
            
            logger.info(f"  ✓ Distance {dist}: Accuracy = {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f} (n={len(X_list)})")
        
        results['feature_importance'] = self.feature_importance
        
        return results
    
    def predict_intent(self,
                      trajectory: Trajectory,
                      junction: Circle,
                      previous_choice: Optional[int] = None) -> Dict[float, Dict]:
        """
        Predict intent at multiple distances before junction
        
        Returns:
            Dict mapping distance -> {predicted_branch, confidence, probabilities}
        """
        predictions = {}
        
        for dist in self.prediction_distances:
            if dist not in self.models:
                continue
            
            features = self.extract_features_at_distance(
                trajectory, junction, dist, previous_choice
            )
            
            if features is None:
                continue
            
            X = self.features_to_array(features).reshape(1, -1)
            
            # Handle NaN and inf values
            nan_mask = np.isnan(X)
            if nan_mask.any():
                for col in range(X.shape[1]):
                    if nan_mask[0, col]:
                        X[0, col] = 0.0
            
            # Replace any remaining inf values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Scale features
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                X_scaled = self.scalers[dist].transform(X)
            
            # Predict
            predicted_branch = int(self.models[dist].predict(X_scaled)[0])
            probabilities = self.models[dist].predict_proba(X_scaled)[0]
            confidence = float(np.max(probabilities))
            
            predictions[dist] = {
                'predicted_branch': predicted_branch,
                'confidence': confidence,
                'probabilities': dict(enumerate(probabilities)),
                'features': features
            }
        
        return predictions


def analyze_intent_recognition(
    trajectories: List[Trajectory],
    junction: Circle,
    actual_branches: pd.DataFrame,
    output_dir: str,
    prediction_distances: List[float] = [100.0, 75.0, 50.0, 25.0],
    previous_choices: Optional[Dict[str, int]] = None
) -> Dict[str, Any]:
    """
    Complete intent recognition analysis
    
    Args:
        trajectories: List of trajectory objects
        junction: Junction to analyze
        actual_branches: DataFrame with columns ['trajectory', 'branch']
        output_dir: Directory for outputs
        prediction_distances: Distances before junction to make predictions
        previous_choices: Dict mapping trajectory_id -> previous branch choice
        
    Returns:
        Analysis results dictionary
    """
    logger.info("Starting intent recognition analysis...")
    
    # Convert branches to dict
    branch_dict = dict(zip(actual_branches['trajectory'], actual_branches['branch']))
    
    # Filter out invalid branches (-1, -2)
    branch_dict = {tid: b for tid, b in branch_dict.items() if b >= 0}
    
    if len(branch_dict) < 10:
        logger.error("Insufficient valid trajectories for intent recognition")
        return {'error': 'insufficient_data'}
    
    # Initialize analyzer with output_dir for model saving
    analyzer = IntentRecognitionAnalyzer(
        prediction_distances=prediction_distances,
        model_type="random_forest",
        output_dir=output_dir
    )
    
    # Train models
    training_results = analyzer.train_models(
        trajectories, junction, branch_dict, previous_choices
    )
    
    # Save training results
    with open(os.path.join(output_dir, 'intent_training_results.json'), 'w') as f:
        json.dump(training_results, f, indent=2)
    
    # Create visualizations
    _visualize_feature_importance(analyzer, output_dir)
    _visualize_prediction_accuracy(training_results, output_dir)
    
    # Test predictions on sample trajectories
    test_predictions = _test_sample_predictions(
        analyzer, trajectories, junction, branch_dict, previous_choices
    )
    
    # Save test predictions
    with open(os.path.join(output_dir, 'intent_test_predictions.json'), 'w') as f:
        json.dump(test_predictions, f, indent=2, default=str)
    
    logger.info("Intent recognition analysis complete!")
    
    return {
        'training_results': training_results,
        'test_predictions': test_predictions,
        'analyzer': analyzer
    }


def _visualize_feature_importance(analyzer: IntentRecognitionAnalyzer, output_dir: str):
    """Create feature importance visualization"""
    n_models = len(analyzer.feature_importance)
    
    if n_models == 0:
        return
    
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for ax, (dist, importance) in zip(axes, analyzer.feature_importance.items()):
        # Sort by importance
        sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        features, values = zip(*sorted_items[:10])  # Top 10
        
        # Plot
        ax.barh(range(len(features)), values)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel('Importance')
        ax.set_title(f'Feature Importance\n{dist} units before junction')
        ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'intent_feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Feature importance plot saved")


def _visualize_prediction_accuracy(training_results: Dict, output_dir: str):
    """Visualize prediction accuracy vs distance"""
    models = training_results.get('models_trained', {})
    
    if not models:
        return
    
    distances = sorted(models.keys())
    accuracies = [models[d]['cv_mean_accuracy'] for d in distances]
    stds = [models[d]['cv_std_accuracy'] for d in distances]
    samples = [models[d]['n_samples'] for d in distances]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy plot
    ax1.errorbar(distances, accuracies, yerr=stds, marker='o', linewidth=2, markersize=8)
    ax1.set_xlabel('Distance Before Junction (units)', fontsize=12)
    ax1.set_ylabel('Prediction Accuracy', fontsize=12)
    ax1.set_title('Intent Prediction Accuracy\nvs Distance to Junction', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.0)
    
    # Sample size plot
    ax2.bar(distances, samples, color='steelblue', alpha=0.7)
    ax2.set_xlabel('Distance Before Junction (units)', fontsize=12)
    ax2.set_ylabel('Number of Samples', fontsize=12)
    ax2.set_title('Training Sample Sizes', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'intent_accuracy_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Accuracy analysis plot saved")


def _test_sample_predictions(analyzer: IntentRecognitionAnalyzer,
                             trajectories: List[Trajectory],
                             junction: Circle,
                             actual_branches: Dict[str, int],
                             previous_choices: Optional[Dict[str, int]]) -> Dict:
    """Test predictions on sample trajectories"""
    
    # Select up to 10 random trajectories
    test_trajectories = [tr for tr in trajectories if tr.tid in actual_branches]
    test_trajectories = test_trajectories[:10]
    
    results = {}
    
    for tr in test_trajectories:
        prev_choice = previous_choices.get(tr.tid, None) if previous_choices else None
        predictions = analyzer.predict_intent(tr, junction, prev_choice)
        
        actual = actual_branches[tr.tid]
        
        results[tr.tid] = {
            'actual_branch': actual,
            'predictions_by_distance': {}
        }
        
        for dist, pred in predictions.items():
            correct = (pred['predicted_branch'] == actual)
            results[tr.tid]['predictions_by_distance'][float(dist)] = {
                'predicted_branch': pred['predicted_branch'],
                'confidence': pred['confidence'],
                'correct': correct
            }
    
    return results


if __name__ == "__main__":
    print("Intent Recognition Module")
    print("=" * 50)
    print("This module provides ML-based intent prediction")
    print("for trajectory route choices.")
    print()
    print("Usage:")
    print("  from verta.verta_intent_recognition import analyze_intent_recognition")
    print("  results = analyze_intent_recognition(trajectories, junction, branches, output_dir)")

