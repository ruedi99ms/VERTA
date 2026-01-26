# VERTA - Virtual Environment Route and Trajectory Analyzer

![VERTA Logo](verta_logo.png)

`VERTA` is a comprehensive toolkit to analyze route choices from x–z movement trajectories: discover branch directions (common route choices that occur at an intersection), assign movement trajectories to existing branches (if new data was recorded), compute timing metrics, visualize results, and predict future route choices based on behavioral patterns. `VERTA` includes both command-line interface (CLI) and interactive web GUI.

## Installation

**Requirements:** Python ≥3.8

```bash
# Core installation
pip install verta

# With optional extras
pip install verta[yaml]      # YAML configuration support
pip install verta[parquet]    # Parquet file format support  
pip install verta[gui]        # GUI dependencies (streamlit, plotly)
```

This installs a console script `verta` and enables the web GUI.

## 🖥️ Web GUI

For an interactive, user-friendly interface, try the web-based GUI:

```bash
# Install GUI dependencies (same as: pip install verta[gui])
pip install verta[gui]

# Launch the web interface (recommended)
verta gui

# Alternative methods
python gui/launch.py
# or
streamlit run src/verta/verta_gui.py
```

You can also customize the port and host:
```bash
verta gui --port 8502              # Use a different port
verta gui --host 0.0.0.0           # Allow external connections
```

**Output Location:** The GUI saves all analysis results to a `gui_outputs/` directory in your current working directory. Different analysis types create subdirectories (e.g., `gui_outputs/junction_0/`, `gui_outputs/metrics/`, `gui_outputs/gaze_analysis/`).

See [`gui/README.md`](gui/README.md) for detailed GUI documentation.

The GUI provides:
- **Interactive junction editor** with drag-and-drop functionality
- **Visual zone definition** for start/end points
- **Real-time analysis** with live parameter adjustment
- **Interactive visualizations** with Plotly charts
- **Gaze and physiological analysis** (head yaw, pupil dilation)
- **Flow graph generation** and conditional probability analysis
- **Pattern recognition** and behavioral insights
- **Intent Recognition** - ML-based early route prediction (see below)
- **Export capabilities** in multiple formats (JSON, CSV, ZIP)
- **Multi-junction analysis** with evacuation planning features

### 🧠 Intent Recognition

Predict user route choices **before** they reach decision points using machine learning:

- **Features extracted**: Spatial (distance, angle, offset), kinematic (speed, acceleration, curvature), gaze (if available), physiological (if available), contextual
- **Models**: Random Forest (fast, robust) or Gradient Boosting (higher accuracy)
- **Prediction distances**: Configure multiple prediction points (100, 75, 50, 25 units before junction)
- **Cross-validation**: Built-in model evaluation with customizable folds
- **Feature importance**: Understand which cues users rely on for decisions
- **Accuracy analysis**: See how prediction improves as users approach the junction

**Use cases:**
- Proactive wayfinding systems
- Predictive content loading
- Dynamic environment optimization
- A/B testing of early interventions

📚 **Documentation**: See [`examples`](examples) for a comprehensive guide to Intent Recognition, including usage examples, model loading, and integration into VR systems.

## CLI Commands

VERTA provides 7 main commands for different types of analysis:

### Decision Modes

VERTA uses **decision modes** to determine where along a trajectory a route choice (branch decision) is made after passing through a junction. The decision point is used to extract the direction vector that represents the chosen route. Three modes are available:

- **`pathlen`**: Measures the path length traveled after entering the junction. The decision point is where the trajectory has traveled a specified distance (set by `--distance`, default: 100 units) from the junction entry point. This mode works well for consistent movement speeds and is computationally efficient.

- **`radial`**: Uses radial distance from the junction center. The decision point is where the trajectory crosses an outer radius (`--r_outer`) with an outward trend (moving away from the junction center). This mode is useful when trajectories move at varying speeds, as it's based on spatial position rather than path length.

- **`hybrid`** (recommended): Tries radial mode first, and if that doesn't find a decision point, falls back to pathlen mode. This provides the best of both approaches and handles a wider variety of trajectory patterns.

**When to use each mode:**
- Use `pathlen` for fast processing when trajectories have consistent speeds
- Use `radial` when trajectories vary significantly in speed or when you want spatial-based detection
- Use `hybrid` (default) for the most robust detection across different scenarios

### Parameters Reference

This section provides a comprehensive overview of all parameters used across VERTA commands. Individual command sections highlight only the most relevant parameters for that command.

#### Common Parameters

These parameters are used across most commands:

- **`--input`** (required): Path to input directory or file containing trajectory data
- **`--out`** / **`--output`**: Path to output directory for results (default: current directory)
- **`--glob`**: File pattern to match input files (default: `"*.csv"`)
- **`--columns`**: Column mapping in format `x=ColumnX,z=ColumnZ,t=TimeColumn`. Maps your CSV columns to VERTA's expected coordinate names
- **`--scale`**: Coordinate scaling factor (default: 1.0). Use if your coordinates need scaling (e.g., `0.2` to convert from millimeters to meters)
- **`--config`**: Path to YAML configuration file for batch parameter settings

#### Junction Parameters

Define the location and size of junctions (decision points):

- **`--junction`**: Junction center coordinates and radius as `x y radius` (e.g., `520 330 20`)
- **`--junctions`**: Multiple junctions as space-separated triplets: `x1 y1 r1 x2 y2 r2 ...`
- **`--radius`**: Junction radius (used with `--junction` when radius is specified separately)

#### Decision Mode Parameters

Control how decision points are detected (see [Decision Modes](#decision-modes) above):

- **`--decision_mode`**: Choose `pathlen`, `radial`, or `hybrid` (default varies by command)
- **`--distance`**: Path length after junction for decision detection in `pathlen` mode (default: 100.0 units)
- **`--r_outer`**: Outer radius for `radial` decision mode. Trajectory must cross this radius with outward movement
- **`--r_outer_list`**: List of outer radii for multiple junctions (one per junction)
- **`--linger_delta`**: Additional distance beyond junction required for decision detection (default: 5.0)
- **`--epsilon`**: Minimum step size for trajectory processing (default: 0.015). Smaller values detect finer movements
- **`--trend_window`**: Window size for trend analysis in radial mode (default: 5)
- **`--min_outward`**: Minimum outward movement threshold for radial detection (default: 0.0)

#### Clustering Parameters

Used by the `discover` command to identify branch directions:

- **`--cluster_method`**: Clustering algorithm - `kmeans` (fast, requires known k), `auto` (finds optimal k), or `dbscan` (density-based, finds variable number of clusters)
- **`--k`**: Number of clusters for kmeans (default: 3)
- **`--k_min`**, **`--k_max`**: Range for auto clustering (default: 2-6)
- **`--angle_eps`**: Angle epsilon for DBSCAN clustering in degrees (default: 15.0)
- **`--min_samples`**: Minimum samples per cluster for DBSCAN (default: 5)
- **`--min_sep_deg`**: Minimum angular separation in degrees between branch centers (default: 12.0). Branches closer than this are merged
- **`--seed`**: Random seed for reproducibility (default: 0)

#### Assignment Parameters

Used when assigning trajectories to existing branches:

- **`--centers`**: Path to previously discovered branch centers file (.npy format) - **required** for `assign` command
- **`--assign_angle_eps`**: Angle tolerance in degrees for branch assignment (default: 15.0). Trajectory direction must be within this angle of a branch center

#### Analysis Parameters

Control analysis behavior and output:

- **`--analyze_sequences`**: Enable route sequence analysis across multiple junctions
- **`--predict_examples`**: Number of concrete prediction examples to generate (default: 50)
- **`--physio_window`**: Time window in seconds for physiological data analysis (default: 3.0)

#### Machine Learning Parameters (Intent Recognition)

Parameters for ML-based intent recognition:

- **`--prediction_distances`**: Distances before junction to make predictions (default: `100 75 50 25` units)
- **`--model_type`**: ML model - `random_forest` (fast, robust) or `gradient_boosting` (higher accuracy, slower)
- **`--cv_folds`**: Number of cross-validation folds for model evaluation (default: 5)
- **`--test_split`**: Fraction of data reserved for testing (default: 0.2)
- **`--with_gaze`**: Include gaze and physiological data in feature extraction (if available)
- **`--assignments`**: Path to pre-computed branch assignments file (optional, speeds up analysis)

#### Visualization Parameters

Control plot generation:

- **`--plot_intercepts`**: Generate decision intercepts visualization (default: True)
- **`--show_paths`**: Show trajectory paths in plots (default: True)
- Use `--no-plot_intercepts` and `--no-show_paths` to disable specific visualizations

#### Chain Analysis Parameters

For multi-junction analysis:

- **`--evacuation_analysis`**: Enable evacuation efficiency analysis
- **`--generate_recommendations`**: Generate traffic flow recommendations
- **`--risk_assessment`**: Perform risk assessment metrics

### 1. Discover Branches

Discover branch directions from trajectory data using clustering algorithms:

```bash
verta discover \
  --input ./data \
  --glob "*.csv" \
  --columns x=Headset.Head.Position.X,z=Headset.Head.Position.Z,t=Time \
  --scale 0.2 \
  --junction 520 330 --radius 20 \
  --distance 100 \
  --decision_mode hybrid \
  --cluster_method auto \
  --out ./outputs
```

**Key Parameters:**
- `--cluster_method`: Choose from `kmeans`, `auto`, or `dbscan` (default: `kmeans`)
- `--decision_mode`: `pathlen`, `radial`, or `hybrid` (default: `hybrid`)
- `--k`: Number of clusters for kmeans (default: 3)
- `--k_min`, `--k_max`: Range for auto clustering (default: 2-6)

See [Parameters Reference](#parameters-reference) for all available parameters including decision mode, clustering, and visualization options.

### 2. Assign Branches

Assign new trajectories to previously discovered branch centers:

```bash
verta assign \
  --input ./new_data \
  --columns x=X,z=Z,t=time \
  --junction 520 330 --radius 20 \
  --distance 100 \
  --centers ./outputs/branch_centers.npy \
  --out ./outputs/new_assignments
```

**Key Parameters:**
- `--centers`: Path to previously discovered branch centers (.npy file) - **required**
- `--decision_mode`: `pathlen`, `radial`, or `hybrid` (default: `pathlen`)
- `--assign_angle_eps`: Angle tolerance for branch assignment (default: 15.0 degrees)

See [Parameters Reference](#parameters-reference) for decision mode and other common parameters.

**Use Cases:**
- Assign new test data to previously discovered branches
- Apply learned branch structure to new datasets
- Batch processing of multiple trajectory sets

### 3. Compute Metrics

Calculate timing and speed metrics for trajectories:

```bash
verta metrics \
  --input ./data \
  --columns x=X,z=Z,t=time \
  --junction 520 330 --radius 20 \
  --distance 100 \
  --decision_mode radial --r_outer 30 \
  --out ./outputs
```

**Key Parameters:**
- `--decision_mode`: `pathlen`, `radial`, or `hybrid` (default: `pathlen`)
- `--trend_window`: Window size for trend analysis (default: 5)
- `--min_outward`: Minimum outward movement threshold (default: 0.0)

See [Parameters Reference](#parameters-reference) for decision mode and junction parameters.

**Metrics Computed:**
- Time to travel specified path length after junction
- Speed through junction (entry, exit, average transit)
- Junction transit speed analysis
- Basic trajectory metrics (total distance, duration, average speed)

### 4. Gaze Analysis

Analyze head movement and physiological data at decision points:

```bash
verta gaze \
  --input ./gaze_data \
  --columns x=X,z=Z,t=time,yaw=HeadYaw,pupil=PupilDilation \
  --junction 520 330 --radius 20 \
  --distance 100 \
  --physio_window 3.0 \
  --out ./gaze_outputs
```

**Key Parameters:**
- `--physio_window`: Time window in seconds for physiological data analysis (default: 3.0)

See [Parameters Reference](#parameters-reference) for decision mode, junction, and other common parameters.

**Analysis Features:**
- Head yaw direction at decision points
- Pupil dilation trajectory analysis
- Physiological metrics (heart rate, etc.) at junctions
- Gaze-movement consistency analysis

### 5. Predict Choices

Analyze behavioral patterns and predict future route choices:

```bash
verta predict \
  --input ./data \
  --columns x=Headset.Head.Position.X,z=Headset.Head.Position.Z,t=Time \
  --scale 0.2 \
  --junctions 520 330 20  600 400 20  700 450 20 \
  --r_outer_list 30 32 30 \
  --distance 100 \
  --decision_mode hybrid \
  --cluster_method auto \
  --analyze_sequences \
  --predict_examples 50 \
  --out ./outputs/prediction
```

**Key Parameters:**
- `--analyze_sequences`: Enable route sequence analysis across multiple junctions
- `--predict_examples`: Number of concrete prediction examples to generate (default: 50)

See [Parameters Reference](#parameters-reference) for clustering, decision mode, and junction parameters.

**Prediction Features:**
- Behavioral pattern recognition (preferred, learned, direct)
- Conditional probability analysis between junctions
- Route sequence analysis
- Confidence scoring for predictions
- Concrete prediction examples

### 6. Intent Recognition

ML-based early route prediction - predict user route choices **before** they reach decision points:

```bash
verta intent \
  --input ./data \
  --columns x=Headset.Head.Position.X,z=Headset.Head.Position.Z,t=Time \
  --scale 0.2 \
  --junction 520 330 20 \
  --distance 100 \
  --prediction_distances 100 75 50 25 \
  --model_type random_forest \
  --cv_folds 5 \
  --test_split 0.2 \
  --out ./outputs/intent_recognition
```

**Key Parameters:**
- `--prediction_distances`: Distances before junction to make predictions (default: 100, 75, 50, 25 units)
- `--model_type`: Choose `random_forest` (fast) or `gradient_boosting` (higher accuracy)
- `--with_gaze`: Include gaze and physiological data if available
- `--centers`: Use pre-computed branch centers (optional)
- `--assignments`: Use pre-computed branch assignments (optional)

See [Parameters Reference](#parameters-reference) for all ML and decision mode parameters.

**Intent Recognition Features:**
- Multi-distance prediction models (train at 100, 75, 50, 25 units before junction)
- Feature importance analysis (spatial, kinematic, gaze, physiological)
- Accuracy analysis showing prediction improvement with proximity
- Cross-validated model evaluation
- Saved models for production deployment
- Sample predictions with confidence scores

**Use Cases:**
- Proactive wayfinding systems
- Predictive content loading
- Dynamic environment optimization
- A/B testing of early interventions

### 7. Enhanced Chain Analysis

Multi-junction analysis with evacuation planning features:

```bash
verta chain-enhanced \
  --input ./data \
  --columns x=X,z=Z,t=time \
  --junctions 520 330 20  600 400 20  700 450 20 \
  --r_outer_list 30 32 30 \
  --distance 100 \
  --evacuation_analysis \
  --generate_recommendations \
  --risk_assessment \
  --out ./outputs/chain_analysis
```

**Key Parameters:**
- `--evacuation_analysis`: Enable evacuation efficiency analysis
- `--generate_recommendations`: Generate traffic flow recommendations
- `--risk_assessment`: Perform risk assessment metrics

See [Parameters Reference](#parameters-reference) for decision mode, junction, and other common parameters.

**Enhanced Features:**
- Flow graph generation
- Evacuation efficiency analysis
- Risk assessment metrics
- Traffic flow recommendations

## Output Files Reference

Each command generates specific output files:

### Discover Command
- `branch_assignments.csv` - Main branch assignments
- `branch_assignments_all.csv` - All assignments including outliers
- `branch_centers.npy` / `branch_centers.json` - Branch center coordinates
- `branch_summary.csv` - Branch count statistics with entropy
- `Branch_Directions.png` - Visual plot of branch directions
- `Branch_Counts.png` - Bar chart of branch frequencies
- `Decision_Intercepts.png` - Trajectory decision points visualization
- `Decision_Map.png` - Overview map of decisions

### Assign Command
- `branch_assignments.csv` - New trajectory assignments
- `run_args.json` - Command parameters used

### Metrics Command
- `timing_and_speed_metrics.csv` - Comprehensive timing and speed data
- `run_args.json` - Command parameters used

### Gaze Command
- `gaze_analysis.csv` - Head yaw analysis at decision points
- `physiological_analysis.csv` - Physiological metrics at junctions
- `pupil_trajectory_analysis.csv` - Pupil dilation trajectory data
- `gaze_consistency_report.json` - Gaze-movement alignment statistics
- `Gaze_Directions.png` - Head movement visualization
- `Physiological_Analysis.png` - Physiological metrics by branch
- `Pupil_Trajectory_Analysis.png` - Pupil dilation plots

### Predict Command
- `choice_pattern_analysis.json` - Complete pattern analysis results
- `choice_patterns.png` - Behavioral pattern visualization
- `transition_heatmap.png` - Junction transition probabilities
- `prediction_examples.json` - Concrete prediction examples
- `prediction_confidence.png` - Confidence analysis plots
- `sequence_analysis.json` - Route sequence analysis (if `--analyze_sequences`)
- `analysis_summary.json` - High-level summary with recommendations

### Intent Recognition Command
- `intent_recognition_summary.csv` - Summary of model accuracy per junction and distance
- `intent_recognition_junction_summary.csv` - Average accuracy per junction
- `intent_recognition_results.json` - Complete analysis results
- `junction_*/models/` - Trained model files (model_*.pkl, scaler_*.pkl)
- `junction_*/intent_training_results.json` - Model metrics and feature importance
- `junction_*/intent_feature_importance.png` - Feature importance visualization
- `junction_*/intent_accuracy_analysis.png` - Accuracy vs. distance chart
- `junction_*/test_predictions.json` - Sample predictions with confidence scores

### Chain-Enhanced Command
- `Chain_Overview.png` - Multi-junction trajectory overview
- `Chain_SmallMultiples.png` - Detailed junction-by-junction view
- `Flow_Graph_Map.png` - Flow diagram between junctions
- `Per_Junction_Flow_Graph.png` - Individual junction flow analysis
- `branch_decisions_chain.csv` - Complete decision chain data

## Behavioral Pattern Analysis

VERTA identifies three types of behavioral patterns:

### Pattern Types

- **Preferred Patterns** (probability ≥ 0.7): Strong behavioral preferences that are highly predictable
- **Learned Patterns** (probability 0.5-0.7): Patterns that develop over time as participants learn the environment  
- **Direct Patterns** (probability 0.3-0.5): Basic route choices without strong preferences

### Example Analysis Results

```json
{
  "summary": {
    "total_sequences": 150,
    "total_transitions": 300,
    "unique_patterns": 12,
    "junctions_analyzed": 3
  },
  "pattern_types": {
    "preferred": 3,
    "learned": 5,
    "direct": 4
  },
  "top_patterns": [
    {
      "from_junction": 0,
      "to_junction": 1,
      "from_branch": 1,
      "to_branch": 2,
      "probability": 0.85,
      "confidence": 0.92,
      "sample_size": 23,
      "pattern_type": "preferred"
    }
  ]
}
```

### Applications

The analysis can help identify:
- Which junctions are most predictable vs. variable
- How participants adapt to the VR environment over time
- Optimal junction designs based on user behavior
- Potential traffic bottlenecks or flow issues
- Evacuation route efficiency and safety

## Configuration

Pass `--config path/to/config.yaml` to any subcommand. Keys under `defaults:` apply to all subcommands; subcommand-specific blocks (`discover:`, `assign:`, `metrics:`) override defaults.

Example `config.yaml`:
```yaml
defaults:
  glob: "*.csv"
  columns: { x: "Headset.Head.Position.X", z: "Headset.Head.Position.Z", t: "Time" }
  scale: 0.2
  motion_threshold: 0.001
  radius: 20
  distance: 100
  epsilon: 0.015
  junction: [520, 330]

discover:
  decision_mode: hybrid
  r_outer: 30
  linger_delta: 2.0
  cluster_method: dbscan
  angle_eps: 15
  show_paths: true
  plot_intercepts: true
```

## Dependencies

### Python Version

VERTA requires **Python ≥3.8**. Supported versions include Python 3.8, 3.9, 3.10, 3.11, and 3.12.

### Core Dependencies
- **numpy** (≥1.20.0) - Numerical computations
- **pandas** (≥1.3.0) - Data manipulation and analysis
- **matplotlib** (≥3.3.0) - Static plotting and visualization
- **tqdm** (≥4.60.0) - Progress bars
- **seaborn** (≥0.12.0) - Statistical data visualization

### ML Capabilities (Intent Recognition)
- **scikit-learn** (≥1.0.0) - Machine learning algorithms (Random Forest, Gradient Boosting)
- **plotly** (≥5.15.0) - Interactive visualization for feature importance and accuracy analysis

### Optional Dependencies

Install with `pip install verta[yaml]`, `pip install verta[parquet]`, `pip install verta[gui]`, or `pip install verta[test]`:

- **config** - YAML configuration file support (`pyyaml`)
- **parquet** - Parquet file format support (`pyarrow`)
- **gui** - Web GUI dependencies (`streamlit`, `plotly`)
- **test** - Testing dependencies (`pytest>=7.0.0`)

### GUI-Specific Dependencies

For the web interface, install:
```bash
pip install verta[gui]
```

This installs:
- **streamlit** (≥1.28.0) - Web framework
- **plotly** (≥5.15.0) - Interactive plotting

Note: Core dependencies (numpy, pandas, matplotlib, seaborn) are included in the main package installation.

## Troubleshooting

### Common Issues

**Import errors:**
```bash
# Ensure you're in the project directory
cd /path/to/verta

# Check Python path
python -c "import verta; print('Package OK')"
```

**GUI won't start:**
```bash
# Check GUI dependencies
pip install verta[gui]

# Verify Streamlit installation
python -c "import streamlit; print('Streamlit OK')"

# Launch the GUI (recommended)
verta gui

# Alternative: Launch with explicit path
streamlit run src/verta/verta_gui.py
```

**No trajectories loaded:**
- Check file paths and glob patterns
- Verify column mappings match your CSV headers
- Ensure files contain X, Z coordinates (Time optional)
- Check scale factor - try `--scale 1.0` for raw coordinates

**Clustering issues:**
- Increase `--k` if too few clusters found
- Try `--cluster_method auto` for automatic cluster detection
- Adjust `--min_samples` for DBSCAN clustering
- Check `--angle_eps` for angle-based clustering

**Performance issues:**
- Reduce number of trajectories for testing
- Use sample data for initial setup
- Close other browser tabs when using GUI
- Consider using `--decision_mode pathlen` for faster analysis

**Memory errors:**
- Process data in smaller batches
- Reduce `--distance` parameter
- Use `--scale` to reduce coordinate precision

### Getting Help

1. **Check console output** for detailed error messages
2. **Verify data format** - CSV files should have X, Z columns
3. **Try sample data first** to test installation
4. **Check file permissions** for output directories
5. **Review configuration** - use `--config` for complex setups

### Data Format Requirements

**Minimum CSV columns:**
- X coordinate (any column name)
- Z coordinate (any column name)
- Time (optional, enables timing metrics)

**Example CSV:**
```csv
Time,X,Z
0.0,100.0,200.0
0.1,101.0,201.0
...
```

**Column mapping:**
```bash
--columns x=Headset.Head.Position.X,z=Headset.Head.Position.Z,t=Time
```

## Tips

- Use `--no-plot_intercepts` and `--no-show_paths` to disable plotting
- The tool prints a suggested `--epsilon` based on step statistics
- For Parquet inputs, install the `[parquet]` extra
- Use `--config` for complex multi-command setups
- Try `--cluster_method auto` for automatic cluster detection

## AI Usage Disclosure

This project utilized AI-assisted development tools for various aspects of the codebase:

- **Cursor** and **ChatGPT** were used for:
  - Code refactoring
  - GUI design
  - Test scaffolding
  - Documentation refactoring and elaboration
  - Paper refinement

## License

MIT