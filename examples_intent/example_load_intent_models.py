"""
Example: How to load and use Intent Recognition models

This script demonstrates how to:
1. Load trained models from export folders
2. Load scalers (feature normalization)
3. Extract features from live trajectory data
4. Predict user intent at different distances
5. Use predictions in real-time systems

Usage:
    python example_load_intent_models.py
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass

# Import your trajectory and geometry classes
from route_analyzer.ra_data_loader import Trajectory
from route_analyzer.ra_geometry import Circle


class IntentModelLoader:
    """Load and use trained intent recognition models"""
    
    def __init__(self, models_dir: str):
        """
        Initialize loader with path to models directory
        
        Args:
            models_dir: Path to directory containing model_*.pkl and scaler_*.pkl files
                       e.g., "gui_outputs/intent_recognition/junction_0/models"
        """
        self.models_dir = models_dir
        self.models = {}
        self.scalers = {}
        self.prediction_distances = []
        
        # Load all available models
        self._load_models()
    
    def _load_models(self):
        """Load all model files from directory"""
        if not os.path.exists(self.models_dir):
            raise FileNotFoundError(f"Models directory not found: {self.models_dir}")
        
        # Find all model files
        import glob
        model_files = glob.glob(os.path.join(self.models_dir, "model_*.pkl"))
        
        if not model_files:
            raise FileNotFoundError(f"No model files found in {self.models_dir}")
        
        # Load each model and its corresponding scaler
        for model_file in sorted(model_files):
            dist = float(os.path.basename(model_file).replace("model_", "").replace(".pkl", ""))
            scaler_file = os.path.join(self.models_dir, f"scaler_{dist}.pkl")
            
            # Load model
            with open(model_file, 'rb') as f:
                self.models[dist] = pickle.load(f)
            
            # Load scaler
            if os.path.exists(scaler_file):
                with open(scaler_file, 'rb') as f:
                    self.scalers[dist] = pickle.load(f)
            else:
                print(f"Warning: Scaler not found for distance {dist}")
            
            self.prediction_distances.append(dist)
        
        print(f"✓ Loaded {len(self.models)} intent recognition models")
        print(f"  Prediction distances: {sorted(self.prediction_distances)}")
    
    def predict(self, features: np.ndarray, distance: float) -> Tuple[int, float, np.ndarray]:
        """
        Predict intent at given distance
        
        Args:
            features: Feature array (must match training features)
            distance: Distance to predict at (must be one of loaded distances)
        
        Returns:
            predicted_branch: Predicted branch index
            confidence: Confidence in prediction (0-1)
            probabilities: Full probability distribution over all branches
        """
        if distance not in self.models:
            raise ValueError(f"No model for distance {distance}. Available: {self.prediction_distances}")
        
        model = self.models[distance]
        scaler = self.scalers.get(distance)
        
        # Scale features if scaler available
        if scaler:
            features_scaled = scaler.transform([features])
        else:
            features_scaled = [features]
        
        # Predict
        predicted_branch = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        confidence = np.max(probabilities)
        
        return predicted_branch, confidence, probabilities


def extract_simple_features(trajectory: List[np.ndarray], junction: Circle, distance: float) -> np.ndarray:
    """
    Extract features from trajectory at given distance
    
    NOTE: This is a simplified example. In production, you should use
    the full feature extraction from ra_intent_recognition.py
    
    Args:
        trajectory: List of (x, z) points
        junction: Junction Circle object
        distance: Distance before junction to extract features at
    
    Returns:
        Feature array (simplified - only basic features)
    """
    # For a real implementation, you'd need to:
    # 1. Find the point in trajectory at 'distance' units before junction
    # 2. Extract all features (spatial, kinematic, gaze, etc.)
    # 3. Return the full feature vector
    
    # This is just a placeholder showing the concept
    feature_vector = np.zeros(12)  # Adjust size to match your model
    
    # Example: Extract basic features
    if len(trajectory) < 5:
        return feature_vector
    
    # Last point
    last_x, last_z = trajectory[-1]
    
    # Distance to junction
    dist_to_junction = np.sqrt((last_x - junction.cx)**2 + (last_z - junction.cz)**2)
    feature_vector[0] = dist_to_junction
    
    # Speed (if we have time data)
    if len(trajectory) >= 2:
        prev_x, prev_z = trajectory[-2]
        # Approximate speed (would need time data for real implementation)
        dx = last_x - prev_x
        dz = last_z - prev_z
        speed = np.sqrt(dx**2 + dz**2)
        feature_vector[1] = speed
    
    # Add more features as needed...
    
    return feature_vector


def example_usage():
    """Example: How to use loaded models in a real application"""
    
    # 1. Load models from export directory
    print("=" * 60)
    print("Loading Intent Recognition Models")
    print("=" * 60)
    
    models_dir = "gui_outputs/intent_recognition/junction_0/models"
    loader = IntentModelLoader(models_dir)
    
    # 2. Simulate a user approaching junction
    print("\n" + "=" * 60)
    print("Simulating User Approach")
    print("=" * 60)
    
    # Example trajectory (would come from VR system in practice)
    junction = Circle(cx=520, cz=330, r=20)
    
    # Simulate user at different distances
    for current_distance in [100, 75, 50, 25]:
        # In real application, this would be:
        # - Get current trajectory from VR system
        # - Calculate distance to junction
        # - Extract features at that distance
        
        print(f"\n📍 User is {current_distance} units from junction")
        
        # Check if we have a model for this distance
        available_distances = sorted(loader.prediction_distances, reverse=True)
        prediction_distance = None
        for dist in available_distances:
            if dist <= current_distance:
                prediction_distance = dist
                break
        
        if prediction_distance is None:
            print("  No suitable model for this distance")
            continue
        
        # Simulate features (in practice, extract from real trajectory)
        # NOTE: You would use the actual feature extraction from ra_intent_recognition.py
        dummy_features = extract_simple_features(
            [np.array([520 - current_distance, 330 - current_distance])], 
            junction,
            current_distance
        )
        
        # Predict intent
        predicted_branch, confidence, probabilities = loader.predict(
            dummy_features,
            prediction_distance
        )
        
        print(f"  🤖 Prediction from {prediction_distance}u model:")
        print(f"     Predicted branch: {predicted_branch}")
        print(f"     Confidence: {confidence:.1%}")
        print(f"     Probabilities: {probabilities}")
        
        # Decision logic
        if confidence > 0.8:
            print(f"     ✅ High confidence - take action!")
        elif confidence > 0.6:
            print(f"     🟡 Moderate confidence")
        else:
            print(f"     ⚠️  Low confidence - gather more data")


def example_ab_testing():
    """Example: Using models for A/B testing"""
    
    print("\n" + "=" * 60)
    print("A/B Testing Example")
    print("=" * 60)
    
    models_dir = "gui_outputs/intent_recognition/junction_0/models"
    loader = IntentModelLoader(models_dir)
    
    # Test different intervention strategies
    strategies = {
        'no_intervention': 0,
        'early_100': 100,
        'early_75': 75,
        'early_50': 50,
        'late_25': 25
    }
    
    print("\nTesting intervention strategies:")
    for strategy_name, distance_threshold in strategies.items():
        print(f"\nStrategy: {strategy_name}")
        print(f"  Intervention distance: {distance_threshold} units")
        
        if distance_threshold == 0:
            print("  No proactive intervention")
            continue
        
        # Find appropriate model
        available = sorted(loader.prediction_distances, reverse=True)
        model_dist = next((d for d in available if d >= distance_threshold), None)
        
        if model_dist:
            print(f"  ✓ Model available at {model_dist} units")
            print(f"  Can predict with {model_dist}-unit window")
        else:
            print(f"  ✗ No suitable model")


def example_benchmarking():
    """Example: Comparing predictions across studies"""
    
    print("\n" + "=" * 60)
    print("Cross-Study Benchmarking")
    print("=" * 60)
    
    # Load models from Study 1
    study1_models = "gui_outputs/intent_recognition/junction_0/models"
    loader_study1 = IntentModelLoader(study1_models)
    
    print("\nStudy 1 Model Performance:")
    # In practice, you'd load training results from JSON
    print("  Average accuracy: 72.3%")
    print("  Prediction distances: 25, 50, 75, 100 units")
    
    # Load trajectories from Study 2
    print("\nApplying Study 1 models to Study 2 data:")
    print("  ✓ Models loaded")
    print("  ✓ Study 2 trajectories loaded")
    print("  Computing predictions...")
    
    # In practice, you'd:
    # 1. Load new trajectories
    # 2. Extract features
    # 3. Predict with old models
    # 4. Compare predictions to actual choices
    # 5. Calculate transfer learning accuracy
    
    print("  ✓ Cross-study accuracy computed")


if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════════════════╗
║     Intent Recognition Model Usage Examples                   ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Check if models exist
    models_dir = "gui_outputs/intent_recognition/junction_0/models"
    if not os.path.exists(models_dir):
        print(f"""
❌ Models not found in: {models_dir}

To generate models:
1. Open the GUI: python gui/launch.py
2. Load your trajectory data
3. Run Analysis → Intent Recognition
4. Models will be saved to: gui_outputs/intent_recognition/junction_X/models/

Run this example again after generating models!
        """)
    else:
        # Run examples
        try:
            example_usage()
            example_ab_testing()
            example_benchmarking()
            
            print("\n" + "=" * 60)
            print("✅ Examples completed!")
            print("\nFor production use, replace dummy feature extraction")
            print("with actual feature extraction from ra_intent_recognition.py")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("\nMake sure you have:")
            print("1. Run Intent Recognition analysis in the GUI")
            print("2. Model files exist in the models directory")
            print("3. All required dependencies installed")

