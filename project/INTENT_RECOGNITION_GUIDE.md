# Intent Recognition Analysis Guide

## 📖 Overview

**Intent Recognition** predicts which route users will choose **before** they reach decision points. Unlike traditional analysis that detects choices **at** junctions, intent recognition works **25-100 units ahead**, enabling proactive systems and early intervention.

---

## 🎯 Key Concept

### Traditional Decision Detection
```
User → Approaches Junction → Enters Junction → EXIT → Choice Detected ✓
                                                ↑
                                           Decision detected HERE
```

### Intent Recognition
```
User → 100u away → 75u away → 50u away → 25u away → Junction
         ↓            ↓           ↓          ↓
      Predict!    Predict!    Predict!   Predict!
      (65% conf)  (72% conf)  (85% conf) (93% conf)
```

**Benefits:**
- **Early prediction**: Know choices before they happen
- **Increasing confidence**: Predictions improve as users get closer
- **Actionable insights**: Enable proactive systems

---

## 🧬 Features Extracted

Intent recognition uses **16 feature categories**:

### 1. Spatial Features (4 features)
- `distance_to_junction` - How far from junction center
- `approach_angle` - Angle between heading and junction direction
- `lateral_offset` - How far off the center-line to junction
- *(Shows if user is already orienting toward a specific branch)*

### 2. Kinematic Features (4 features)
- `current_speed` - Instantaneous movement speed
- `speed_change_rate` - Acceleration/deceleration
- `curvature` - How much the path is curving
- `sinuosity` - Path complexity (meandering vs straight)
- *(Reveals movement patterns and decision-making behavior)*

### 3. Temporal Features (1 feature)
- `time_to_junction` - Estimated time until reaching junction
- *(Context for prediction urgency)*

### 4. Gaze Features (3 features, if available)
- `gaze_angle` - Direction user is looking
- `gaze_alignment` - Whether looking where they're going
- `head_rotation_rate` - How much head is moving
- *(Direct indicator of attention and intent)*

### 5. Physiological Features (4 features, if available)
- `heart_rate` - Current heart rate
- `heart_rate_trend` - Increasing/decreasing
- `pupil_dilation` - Average pupil size
- `pupil_change_rate` - Pupil dilation changes
- *(Indicates cognitive load, stress, uncertainty)*

### 6. Contextual Features (1 feature)
- `previous_junction_choice` - What they did at last junction
- *(Reveals sequential patterns and route preferences)*

---

## 🔬 How It Works

### Step 1: Feature Extraction
For each trajectory, extract features at multiple distances **before** the junction:
```python
# Example: Extract features 75 units before junction
features = analyzer.extract_features_at_distance(
    trajectory=trajectory,
    junction=junction,
    distance_before=75.0,
    previous_choice=None
)
```

**Extracted features** (example):
```
distance_to_junction: 95.3
approach_angle: 0.23 rad (13°)
lateral_offset: 2.4 units
current_speed: 1.5 units/sec
speed_change_rate: -0.05 (slowing down)
curvature: 0.12 (slight curve)
sinuosity: 1.08 (mostly straight)
gaze_angle: 0.35 rad (20°)
gaze_alignment: 0.92 (looking where going)
heart_rate: 78 bpm
```

### Step 2: Model Training
Train Random Forest or Gradient Boosting classifiers:
```python
results = analyzer.train_models(
    trajectories=trajectories,
    junction=junction,
    actual_branches=branch_labels,
    previous_choices=None
)
```

**Models trained at each distance:**
- **100 units before**: Early prediction (lower accuracy, ~60-70%)
- **75 units before**: Medium-range prediction (~70-80%)
- **50 units before**: Near-term prediction (~80-90%)
- **25 units before**: Immediate prediction (~85-95%)

### Step 3: Prediction
Predict intent at each distance:
```python
predictions = analyzer.predict_intent(
    trajectory=new_trajectory,
    junction=junction
)

# Results:
# {
#   100.0: {predicted_branch: 2, confidence: 0.67},
#   75.0:  {predicted_branch: 2, confidence: 0.74},
#   50.0:  {predicted_branch: 2, confidence: 0.87},
#   25.0:  {predicted_branch: 2, confidence: 0.93}
# }
```

---

## 📊 Example Analysis Results

### Model Performance
```
Distance: 100 units before junction
   • Accuracy: 68.5% ± 5.2%
   • Training samples: 147
   • Interpretation: 🟡 Good

Distance: 75 units before junction
   • Accuracy: 75.3% ± 4.8%
   • Training samples: 152
   • Interpretation: 🟡 Good

Distance: 50 units before junction
   • Accuracy: 84.2% ± 3.6%
   • Training samples: 158
   • Interpretation: 🟢 Excellent

Distance: 25 units before junction
   • Accuracy: 91.7% ± 2.9%
   • Training samples: 161
   • Interpretation: 🟢 Excellent
```

### Feature Importance Rankings

**At 100 units (early prediction):**
1. `approach_angle` ⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛ 0.184
2. `gaze_angle` ⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛ 0.152
3. `curvature` ⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛ 0.123
4. `current_speed` ⬛⬛⬛⬛⬛⬛⬛⬛⬛ 0.098
5. `gaze_alignment` ⬛⬛⬛⬛⬛⬛⬛ 0.081

**At 25 units (near junction):**
1. `lateral_offset` ⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛ 0.215
2. `approach_angle` ⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛ 0.178
3. `gaze_angle` ⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛ 0.143
4. `distance_to_junction` ⬛⬛⬛⬛⬛⬛⬛⬛ 0.091
5. `heart_rate_trend` ⬛⬛⬛⬛⬛⬛ 0.067

**Insights:**
- Early predictions rely on **approach trajectory** and **gaze**
- Near predictions rely on **body positioning** (lateral offset)
- Physiological signals matter more when close to decision
- Gaze is consistently important across all distances

---

## 🎨 Visualizations Generated

### 1. Feature Importance Plot
Shows which features matter most at each prediction distance.
- Horizontal bar charts per distance
- Helps understand what drives predictions
- Identifies most informative sensors

### 2. Accuracy vs Distance Plot
Shows how prediction accuracy changes with distance.
- Line plot with error bars
- Typically shows accuracy increasing as user approaches
- Identifies "sweet spot" for predictions

### 3. Sample Predictions Table
Shows example predictions for individual trajectories.
```
Trajectory: participant_001_trial_05
Actual choice: Branch 2

   100 units before: Branch 1 (confidence: 52%) ✗
    75 units before: Branch 2 (confidence: 68%) ✓
    50 units before: Branch 2 (confidence: 87%) ✓
    25 units before: Branch 2 (confidence: 94%) ✓
```

---

## 💡 Real-World Applications

### 1. **Proactive Wayfinding**
```python
# When user is 75 units from junction:
if prediction['confidence'] > 0.75:
    show_arrow_to_predicted_branch()
    preload_relevant_information()
```

**Use case**: VR navigation, museum guides, shopping malls

### 2. **Adaptive User Interfaces**
```python
# Highlight likely choices:
for branch in branches:
    if branch == predicted_branch:
        highlight_intensity = prediction['confidence']
        button.set_alpha(highlight_intensity)
```

**Use case**: VR menus, game interfaces, training simulations

### 3. **Congestion Management**
```python
# Predict traffic flows 5-10 seconds ahead:
predicted_flows = predict_all_users_intents()
if predicted_density[branch_2] > threshold:
    suggest_alternative_routes()
```

**Use case**: Crowd management, evacuation planning, theme parks

### 4. **Anomaly Detection**
```python
# Detect unusual behavior:
if actual_choice != predicted_choice and confidence > 0.9:
    flag_as_anomaly()  # Lost? Confused? Emergency?
```

**Use case**: Security systems, elderly care, emergency response

### 5. **Personalization**
```python
# Learn individual patterns:
user_model = train_per_user_intent_model()
if user_model.accuracy > generic_model.accuracy:
    use_personalized_predictions()
```

**Use case**: Personalized navigation, adaptive training, gaming

### 6. **Performance Optimization**
```python
# Preload assets based on predicted paths:
if distance < 50 and confidence > 0.8:
    preload_assets_for_branch(predicted_branch)
```

**Use case**: VR optimization, game engines, streaming content

---

## 🚀 Getting Started

### Basic Usage

```python
from ra_data_loader import load_folder
from ra_geometry import Circle
from ra_decisions import discover_branches
from ra_intent_recognition import analyze_intent_recognition

# Load data
trajectories = load_folder("./data", pattern="*.csv", require_time=True)

# Define junction
junction = Circle(cx=520.0, cz=330.0, r=20.0)

# Discover actual choices
assignments, summary, centers = discover_branches(
    trajectories, junction, k=3, decision_mode="hybrid"
)

# Run intent recognition analysis
results = analyze_intent_recognition(
    trajectories=trajectories,
    junction=junction,
    actual_branches=assignments,
    output_dir="./output",
    prediction_distances=[100.0, 75.0, 50.0, 25.0]
)

# Check accuracy
print(f"Accuracy at 75 units: {results['training_results']['models_trained'][75.0]['cv_mean_accuracy']:.1%}")
```

### Real-Time Prediction

```python
from ra_intent_recognition import IntentRecognitionAnalyzer

# Initialize and train
analyzer = IntentRecognitionAnalyzer(prediction_distances=[100, 75, 50, 25])
analyzer.train_models(trajectories, junction, branch_labels)

# Predict for new trajectory
predictions = analyzer.predict_intent(
    trajectory=new_trajectory,
    junction=junction
)

# Use prediction
if predictions[75.0]['confidence'] > 0.75:
    predicted_branch = predictions[75.0]['predicted_branch']
    show_guidance_to_branch(predicted_branch)
```

---

## 📈 Interpreting Results

### Accuracy Levels
- **> 85%**: Excellent - User behavior is highly predictable
- **70-85%**: Good - Clear patterns with some variability  
- **60-70%**: Moderate - Some predictability but significant variation
- **< 60%**: Poor - Behavior is exploratory or context-dependent

### When Accuracy is High
✅ **Implications:**
- Route choices follow clear patterns
- Users have strong preferences or habits
- Environment layout clearly guides behavior
- Early intervention systems will be effective

✅ **Applications:**
- Proactive navigation assistance
- Predictive UI highlighting
- Confident traffic flow predictions

### When Accuracy is Low
⚠️ **Possible reasons:**
- Users are exploring (first-time visitors)
- Multiple equally attractive options
- Environmental factors not captured in features
- Need more contextual information

⚠️ **Actions:**
- Analyze confused/uncertain users separately
- Add contextual features (time of day, user goals)
- Consider individual user models
- Focus on improving accuracy at shorter distances

---

## 🔧 Advanced Topics

### Multi-Junction Intent Prediction

For sequential junctions, use previous choices as features:

```python
# Build decision chain
previous_choices = {}
for trajectory in trajectories:
    junction_sequence = find_junction_sequence(trajectory)
    for i in range(1, len(junction_sequence)):
        prev_junction = junction_sequence[i-1]
        current_junction = junction_sequence[i]
        previous_choices[trajectory.tid] = prev_junction

# Train with history
results = analyze_intent_recognition(
    trajectories=trajectories,
    junction=current_junction,
    actual_branches=assignments,
    previous_choices=previous_choices,  # Include history!
    output_dir="./output"
)
```

### Individual User Models

Train per-user models for personalization:

```python
user_models = {}
for user_id in unique_users:
    user_trajectories = [t for t in trajectories if t.user == user_id]
    
    if len(user_trajectories) >= 10:  # Minimum samples
        user_models[user_id] = IntentRecognitionAnalyzer()
        user_models[user_id].train_models(
            user_trajectories, junction, branch_labels
        )
```

### Confidence Calibration

Adjust confidence thresholds based on application:

```python
# Safety-critical: High confidence required
if confidence > 0.90:
    take_action()

# User assistance: Medium confidence OK
if confidence > 0.70:
    show_suggestion()

# Analytics only: Any prediction
log_prediction(predicted_branch, confidence)
```

---

## 📚 Technical Details

### Feature Engineering Notes

1. **NaN Handling**: Gaze/physio features may be NaN if unavailable. Models impute with median values from training data.

2. **Scaling**: All features standardized (z-score normalization) before training.

3. **Angle Features**: Angles normalized to [-π, π] using `arctan2` for proper circular statistics.

4. **Temporal Alignment**: Features extracted at spatial distances, not time intervals, ensuring consistency across varying speeds.

### Model Selection

**Random Forest** (default):
- Robust to overfitting
- Handles missing values well
- Provides feature importance
- Good general-purpose choice

**Gradient Boosting** (alternative):
- Often higher accuracy
- Better for complex patterns
- Requires more tuning
- Risk of overfitting with small datasets

### Cross-Validation

Uses 5-fold cross-validation for robust accuracy estimates. Reports mean ± std to show prediction stability.

---

## 🎓 Research Applications

Intent recognition enables novel research questions:

1. **Cognitive Load**: Do complex environments reduce prediction accuracy?
2. **Learning Effects**: Does predictability increase with experience?
3. **Individual Differences**: Can we identify navigation "styles"?
4. **Design Optimization**: Which junction designs have highest predictability?
5. **Attention Patterns**: How does gaze predict choices across contexts?

---

## 📞 Support

For questions or issues:
1. Check the example script: `example_intent_recognition.py`
2. Review visualization outputs in `output_dir`
3. Examine feature importance to understand predictions
4. Try different prediction distances or model types

---

## 🔮 Future Enhancements

Potential additions:
- **LSTM/Transformer models** for full sequence modeling
- **Ensemble methods** combining multiple model types
- **Online learning** for real-time model updates
- **Explainable AI** showing why each prediction was made
- **Multi-modal fusion** explicitly modeling sensor interactions
- **Uncertainty quantification** beyond simple confidence scores

---

**Remember**: Intent recognition is most powerful when combined with your existing route analysis tools. Use it to enhance, not replace, traditional decision detection!

