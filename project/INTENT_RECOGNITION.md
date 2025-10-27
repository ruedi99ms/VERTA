# Intent Recognition Guide

## 🧠 Overview

Intent Recognition uses machine learning to predict user route choices **before** they reach decision points. This enables proactive systems, predictive content loading, and adaptive environments.

## 🎯 Key Concepts

### What is Intent Recognition?

Instead of just analyzing which branch users **chose** at a junction (which we know after the fact), Intent Recognition predicts **which branch they will choose** based on their behavior **before** reaching the decision point.

### How It Works

1. **Extract features** from trajectories at multiple distances before junction (100, 75, 50, 25 units)
2. **Train ML models** (Random Forest or Gradient Boosting) to predict branch choice
3. **Evaluate accuracy** using cross-validation at each prediction distance
4. **Analyze features** to understand which behavioral cues matter most

### Why It Matters

- **Proactive wayfinding**: Show navigation hints before users reach junction
- **Predictive preloading**: Load next area based on predicted route  
- **Dynamic optimization**: Adjust environment before user arrives
- **Early intervention**: A/B test interventions at different prediction distances

## 📊 Features Extracted

### Spatial Features
- **Distance to junction**: How far from decision point
- **Approach angle**: Angle of approach vector
- **Lateral offset**: Perpendicular distance from straight path

### Kinematic Features  
- **Current speed**: Velocity magnitude
- **Speed change rate**: Acceleration/deceleration pattern
- **Curvature**: Path curvature (measures steering)
- **Sinuosity**: How much trajectory deviates from straight line
- **Time to junction**: Estimated time remaining

### Gaze Features (if available)
- **Gaze angle**: Head/gaze orientation
- **Gaze alignment**: How well gaze matches movement direction
- **Head rotation rate**: How quickly head is turning

### Physiological Features (if available)
- **Heart rate**: Physiological arousal level
- **Pupil dilation**: Cognitive load indicator

### Contextual Features
- **Previous choices**: Branch selections at earlier junctions
- **Trajectory ID**: User-specific patterns

## 🤖 Models

### Random Forest (Recommended)
- **Pros**: Fast, robust to overfitting, handles missing data well
- **Best for**: Initial exploration, smaller datasets
- **Training time**: ~1-5 seconds per model

### Gradient Boosting  
- **Pros**: Higher accuracy potential, learns complex patterns
- **Best for**: Maximum accuracy, larger datasets (>100 samples per branch)
- **Training time**: ~5-15 seconds per model

## 📈 Understanding Results

### Accuracy Levels

- **🟢 Excellent (85-100%)**: Highly predictable behavior
  - Use for: Proactive systems, critical navigation decisions
  - Examples: Clear paths, learned patterns, obvious destinations
  
- **🟡 Good (70-85%)**: Moderately predictable
  - Use for: Assisted navigation, likely predictions
  - Examples: Multi-option paths with preferences
  
- **🔴 Moderate (50-70%)**: Variable behavior
  - Use for: Exploratory systems, probabilistic predictions
  - Consider: Per-user models, additional features

### Accuracy vs. Distance

- **Farther away (100u)**: Lower accuracy - less information available
- **Closer (25u)**: Higher accuracy - more behavioral cues available
- **Analysis**: Shows how early you can reliably predict choices

### Feature Importance

Shows which behavioral cues the model relies on:

- **High importance** = Strong predictor of choice
- **Low importance** = Not informative for this junction
- **Spatial dominant** = Location-based decisions
- **Kinematic dominant** = Movement pattern-based decisions
- **Gaze dominant** = Looking ahead matters

## 🚀 Usage in GUI

### Basic Workflow

1. **Data Upload**: Load trajectory data
2. **Junction Editor**: Define decision points
3. **Analysis → Discover Branches**: Set clustering parameters (affects ground truth)
4. **Analysis → Intent Recognition**: 
   - Select prediction distances (100, 75, 50, 25 units)
   - Choose model type (Random Forest or Gradient Boosting)
   - Set cross-validation folds (default: 5)
   - Adjust test split (default: 20%)
5. **Run Analysis**: Train models and evaluate
6. **Visualize**: View results in Visualization tab
7. **Export**: Download models and predictions

### Configuration Tips

**Prediction Distances:**
- Select multiple distances to see accuracy progression
- For real-time systems: Focus on distances matching your prediction needs
- For research: Use all distances to understand behavior evolution

**Model Selection:**
- Start with **Random Forest** for speed
- Try **Gradient Boosting** if you need higher accuracy
- Compare both to see if extra accuracy is worth training time

**Cross-Validation:**
- **5 folds** (default): Balanced between speed and reliability
- **10 folds**: More reliable estimates, slower
- Use higher folds for smaller datasets (<50 samples)

**Test Split:**
- **20%** (default): Good balance for model evaluation
- **30-40%**: Use when you have lots of data (>200 samples)
- **10%**: Use when data is limited (<50 samples)

## 📊 Interpreting Results

### Accuracy Table

```
Junction  Distance  Accuracy  Std Dev  Samples
0        100       45.3%     ±8.9%    139
0        75        40.3%     ±6.3%    139
0        50        49.0%     ±4.8%    139
0        25        51.7%     ±5.4%    139
```

**Reading the table:**
- **Junction**: Which junction was analyzed
- **Distance**: Units before junction where prediction was made
- **Accuracy**: Cross-validated prediction accuracy
- **Std Dev**: Variation across CV folds (uncertainty)
- **Samples**: Number of trajectories used

**Example interpretation:**
- Junction 0: Poor predictability (40-52%)
  - Consider: Per-user models, contextual features
  - Likely: Exploratory behavior, multiple attractive options
  
### Feature Importance

Visualization showing which features matter:

```
Top Features for Junction 0 (100 units):
1. distance_to_junction:    0.234
2. approach_angle:          0.189
3. current_speed:          0.156
4. curvature:              0.098
...
```

**Spatial features dominant** → Users decide based on position
**Kinematic features dominant** → Users decide based on movement patterns
**Gaze features dominant** → Visual attention drives decisions

### Sample Predictions

Shows how well model predicted actual choices:

```
Distance  Predicted  Confidence  Correct
100u      Branch 2   65.3%       ✓
75u       Branch 2   72.1%       ✓
50u       Branch 2   88.4%       ✓
25u       Branch 2   95.2%       ✓
```

All correct + increasing confidence = Strong predictor!

## 🔧 Output Files

### Per-Junction Directory Structure

```
gui_outputs/intent_recognition/
  junction_0/
    intent_training_results.json    # Model metrics
    intent_feature_importance.png   # Feature importance plots
    intent_accuracy_analysis.png     # Accuracy vs. distance
    test_predictions.json           # Sample predictions
    models/                         # Trained model files
      model_100.pkl                 # Trained Random Forest/Gradient Boosting
      scaler_100.pkl                # Feature scaler
      model_75.pkl
      scaler_75.pkl
      model_50.pkl
      scaler_50.pkl
      model_25.pkl
      scaler_25.pkl
```

### Training Results JSON

```json
{
  "models_trained": {
    "100.0": {
      "model_type": "random_forest",
      "cv_mean_accuracy": 0.723,
      "cv_std_accuracy": 0.045,
      "n_samples": 139,
      "n_features": 12,
      "feature_importance": {
        "distance_to_junction": 0.234,
        "approach_angle": 0.189,
        ...
      }
    },
    ...
  },
  "overall_accuracy": 0.723
}
```

## 🎯 Best Practices

### Before Running

1. **Run Discover Branches first**: This sets ground truth labels
2. **Ensure sufficient data**: 
   - Minimum 30-50 trajectories per junction
   - At least 10-15 trajectories per branch
3. **Check data quality**: 
   - Trajectories should pass through junction
   - Time data available (for kinematic features)

### Model Selection

**Choose Random Forest if:**
- First time analyzing data
- Want fast results
- Have <100 samples per junction
- Need stable, reproducible results

**Choose Gradient Boosting if:**
- Already tried Random Forest
- Need maximum accuracy
- Have >100 samples per junction
- Willing to wait for longer training

### Improving Accuracy

**If accuracy is low (<70%):**

1. **Check branch distribution**: Is there actually a choice?
2. **Per-user models**: Train separate models per user
3. **Add features**: Contextual, environmental, task-specific
4. **Increase prediction distance**: More data available
5. **Combine with other junctions**: Multi-junction patterns

**If accuracy varies with distance:**

- **Decreasing accuracy**: Closer to junction, more variability
- **Increasing accuracy**: Users commit to choice earlier
- **Plateau**: Optimal prediction distance reached

## 💡 Use Cases

### 1. Proactive Wayfinding

**Scenario**: Help users navigate VR environment  
**When to predict**: 50-75 units before junction  
**Action**: Display subtle navigation hint  
**Required accuracy**: >80%

### 2. Predictive Content Loading

**Scenario**: VR world with large assets  
**When to predict**: 75-100 units before junction  
**Action**: Preload predicted next area  
**Required accuracy**: >70% (false preloads acceptable)

### 3. Dynamic Difficulty

**Scenario**: Adaptive game difficulty  
**When to predict**: 25-50 units before junction  
**Action**: Adjust environment based on predicted choice  
**Required accuracy**: >85%

### 4. A/B Testing

**Scenario**: Test navigation interventions  
**When to predict**: Multiple distances  
**Action**: Show intervention based on predicted path  
**Required accuracy**: Any level (learning experiment)

## 🔍 Troubleshooting

### Low Accuracy (<50%)

**Possible causes:**
- Insufficient data (<30 samples)
- No real choice (only one viable branch)
- Random behavior at this junction
- Feature extraction failing

**Solutions:**
- Collect more trajectories
- Verify junction is a true decision point
- Try different model settings
- Check feature extraction

### Missing Feature Importance

**Possible causes:**
- All trajectories go to same branch (trivial decision)
- Features not available (e.g., no gaze data)
- NaN/inf values in features

**Solutions:**
- Check if this is actually a decision point
- Verify data availability
- Feature importance not needed for trivial cases

### Slow Training

**Possible causes:**
- Large dataset (>500 trajectories)
- Gradient Boosting model
- Many prediction distances
- Many features extracted

**Solutions:**
- Use Random Forest for faster training
- Reduce prediction distances
- Reduce number of trajectories (sample)
- Train fewer models

## 📚 Advanced Topics

### Multi-Junction Context

Use choices at previous junctions to improve prediction:

```python
previous_choices = {
    'junction_0': 'branch_1',
    'junction_1': 'branch_0',
    ...
}
```

Models can learn sequential patterns.

### Transfer Learning

Train models on one dataset, apply to another:

1. Train models on "training" dataset
2. Extract models and feature extractors
3. Apply to "test" dataset
4. Evaluate performance

### Real-Time Prediction

For live systems:

1. Extract features from current trajectory
2. Load pre-trained models
3. Predict at current distance
4. Update as user approaches junction
5. Refine prediction at closer distances

## 🎓 References

- Random Forest: Breiman, L. (2001). "Random Forests"
- Gradient Boosting: Friedman, J. H. (2001). "Greedy Function Approximation"
- Feature Engineering: Correlation, covariance analysis
- Cross-Validation: Model evaluation best practices

## 💾 Loading and Using Saved Models

### After Training (Export from GUI)

After running Intent Recognition in the GUI, models are automatically saved to:
```
gui_outputs/intent_recognition/junction_0/models/
├── model_100.pkl    # Trained model for 100 units
├── scaler_100.pkl   # Feature scaler
├── model_75.pkl
├── scaler_75.pkl
├── model_50.pkl
├── scaler_50.pkl
├── model_25.pkl
└── scaler_25.pkl
```

### Loading Models in Python

```python
import pickle
import numpy as np

# Load a specific model and scaler
distance = 75.0

with open(f"gui_outputs/intent_recognition/junction_0/models/model_{distance}.pkl", 'rb') as f:
    model = pickle.load(f)

with open(f"gui_outputs/intent_recognition/junction_0/models/scaler_{distance}.pkl", 'rb') as f:
    scaler = pickle.load(f)

# Extract features from current trajectory (see ra_intent_recognition.py)
features = extract_features(trajectory, junction, distance)

# Scale features
features_scaled = scaler.transform([features])

# Predict
predicted_branch = model.predict(features_scaled)[0]
confidence = model.predict_proba(features_scaled)[0].max()
probabilities = model.predict_proba(features_scaled)[0]

print(f"Predicted branch: {predicted_branch}")
print(f"Confidence: {confidence:.1%}")
```

### Example Scripts

See `example_load_intent_models.py` for:
- ✅ Loading multiple models at once
- ✅ Real-time prediction in VR systems
- ✅ A/B testing different intervention strategies
- ✅ Cross-study benchmarking

### Integration into VR System

```python
# In your VR application loop
class NavigationSystem:
    def __init__(self):
        self.model_loader = IntentModelLoader("path/to/models")
    
    def update(self, user):
        """Called every frame"""
        junction = detect_upcoming_junction(user)
        distance = calculate_distance(user, junction)
        
        # Predict intent
        prediction, confidence, _ = self.model_loader.predict(
            extract_features(user.trajectory, junction),
            distance
        )
        
        # Take action if confident
        if confidence > 0.8 and distance < 75:
            show_navigation_hint(prediction)
            preload_content(prediction)
```

## 📝 Summary

Intent Recognition enables **predictive** rather than **reactive** systems by learning from historical user behavior to predict future route choices. Use it when you need to prepare the system before the user makes their decision!

**Key benefits:**
- ✅ **Models are saved** - Load and reuse in future studies
- ✅ **Production ready** - Use trained models in live VR systems
- ✅ **Transfer learning** - Apply models from one study to another
- ✅ **A/B testing** - Compare different intervention strategies
- ✅ **Research applications** - Understand when and why users decide

---

For more information, see [GUI_README.md](GUI_README.md) and the main [README.md](README.md).

