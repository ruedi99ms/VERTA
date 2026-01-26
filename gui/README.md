# VERTA Web GUI

A modern, interactive web interface for VERTA (Virtual Environment Route and Trajectory Analyzer), built with Streamlit.

## 🚀 Quick Start

### 1. Install GUI Dependencies
```bash
pip install verta[gui]
```

### 2. Launch the GUI
```bash
# Option 1: Using the verta command (recommended)
verta gui

# Option 2: Using the launcher script
python gui/launch.py

# Option 3: Direct Streamlit command to GUI module
streamlit run src/verta/verta_gui.py
```

**Customize the server:**
```bash
verta gui --port 8502              # Use a different port (default: 8501)
verta gui --host 0.0.0.0           # Allow external connections (default: localhost)
```

### 3. Open in Browser
The GUI will automatically open at `http://localhost:8501`

**Output Location:** All analysis results are saved to a `gui_outputs/` directory in your current working directory (where you run `verta gui`). The structure includes:
- `gui_outputs/junction_{N}/` - Results for each junction (discover, assign operations)
- `gui_outputs/metrics/` - Movement metrics results
- `gui_outputs/gaze_analysis/` - Gaze and physiological analysis results
- `gui_outputs/gaze_plots/` - Global gaze visualizations
- `gui_outputs/intent_recognition/` - Intent recognition model outputs

## 🎯 Features

### 📁 Data Upload
- **File Upload**: Drag & drop CSV files or specify folder paths
- **Column Mapping**: Configure X, Z, and Time column names
- **Parameter Tuning**: Adjust scale factor and motion threshold
- **Sample Data**: Load demo data for testing

### 🎯 Junction Editor
- **Interactive Plot**: Visual junction management with Plotly
- **Add/Remove**: Click to add junctions, delete with one click
- **Real-time Preview**: See trajectories and junctions together
- **Sample Junctions**: Load predefined junction configurations

### 📍 Zone Definition
- **Start Zones**: Define circular or rectangular start areas
- **End Zones**: Configure destination zones
- **Visual Feedback**: See zones overlaid on trajectory data
- **Multiple Zones**: Support for multiple start/end zones

### 📊 Analysis
VERTA GUI provides 7 comprehensive analysis types:

1. **Discover Branches** - Automatic junction branch detection with clustering algorithms (kmeans, DBSCAN, auto)
2. **Assign Trajectories** - Assign new trajectories to previously discovered branch centers
3. **Movement Metrics** - Calculate trajectory movement patterns, timing, and speed metrics
4. **Gaze & Physiology** - Analyze eye tracking data, head yaw, pupil dilation, and physiological metrics
5. **Predict Choices** - Behavioral pattern recognition and junction choice prediction with confidence scoring
6. **Intent Recognition** - ML-based early route prediction (predict choices BEFORE reaching decision points)
7. **Enhanced Analysis** - Multi-junction evacuation planning, risk assessment, and efficiency metrics

**Additional Features:**
- **Parameter Control**: Adjust all analysis settings in real-time
- **Real-time Status**: Live progress and completion indicators
- **Multi-junction Support**: Analyze complex route networks
- **Batch Processing**: Process multiple datasets efficiently

### 📈 Visualization
- **Flow Graphs**: Interactive flow diagrams
- **Conditional Probabilities**: Heatmaps and probability matrices
- **Pattern Analysis**: Behavioral pattern identification
- **Intent Recognition**: Feature importance, accuracy analysis, sample predictions
- **Start/End Analysis**: Completion rate and trajectory classification

### 💾 Export Results
- **Multiple Formats**: JSON, CSV, ZIP archives
- **Selective Export**: Choose which data to export
- **Download Ready**: Direct download from browser
- **Clipboard Support**: Copy results for sharing

## 🎨 Interface Overview

### Navigation Sidebar
- **Step-by-step workflow**: Data → Junctions → Zones → Analysis → Visualization → Export
- **Status indicators**: Visual feedback on completion status
- **Quick access**: Jump between any step

### Main Workspace
- **Responsive layout**: Adapts to different screen sizes
- **Interactive plots**: Zoom, pan, hover for details
- **Real-time updates**: Changes reflect immediately
- **Error handling**: Clear error messages and recovery options

## 🔧 Technical Details

### Architecture
- **Streamlit**: Modern web framework for Python
- **Plotly**: Interactive plotting and visualization
- **Modular Design**: Clean separation of concerns
- **Session State**: Persistent data across interactions

### Integration
- **Package Compatible**: Full Python package functionality preserved
- **Standalone Mode**: Can be used independently
- **Extensible**: Easy to add new features and visualizations

### Performance
- **Efficient Rendering**: Optimized for large datasets
- **Progressive Loading**: Load data incrementally
- **Caching**: Smart caching of analysis results

## 📱 Usage Examples

### Basic Workflow
1. **Upload Data**: Load your (VR) trajectory CSV files
2. **Define Junctions**: Add decision points interactively
3. **Set Zones**: Configure start and end areas
4. **Run Analysis**: Execute junction-based choice prediction
5. **Visualize**: Explore results with interactive plots
6. **Export**: Download results in your preferred format

### Advanced Usage
- **Custom Parameters**: Fine-tune analysis settings
- **Multiple Datasets**: Compare different XR spatial studies
- **Batch Processing**: Analyze multiple trajectory sets
- **Custom Visualizations**: Create specialized plots

## 🛠️ Development

### Adding New Features
1. **Extend VERTAGUI class**: Add new methods
2. **Create new render methods**: Follow naming convention `render_*`
3. **Update navigation**: Add new steps to sidebar
4. **Test thoroughly**: Ensure compatibility with package mode

### Customization
- **Styling**: Modify CSS in the `render_header()` method
- **Layout**: Adjust column layouts and spacing
- **Colors**: Update Plotly color schemes
- **Icons**: Change emoji icons throughout

## 🔍 Troubleshooting

### Common Issues

**GUI won't start:**
```bash
# Check dependencies
pip install verta[gui]

# Verify Python path
python -c "import streamlit; print('Streamlit OK')"
```

**Import errors:**
```bash
# Ensure you're in the right directory
cd /path/to/verta

# Check project structure
ls src/verta/verta_gui.py
```

**Performance issues:**
- Reduce number of trajectories for testing
- Use sample data for initial setup
- Close other browser tabs

### Getting Help
- Check the console output for error messages
- Verify all dependencies are installed
- Ensure data files are in the correct format
- Try the sample data first

## 🎉 Benefits

### For Users
- **No coding required**: Visual interface for all operations
- **Interactive exploration**: Real-time parameter adjustment
- **Professional results**: Publication-ready visualizations
- **Easy sharing**: Export and share results easily

### For Developers
- **Package integration**: Use as Python package or GUI
- **Extensible**: Add new analysis methods easily
- **Modern stack**: Built with current web technologies
- **Cross-platform**: Works on Windows, Mac, Linux

## 🧠 Intent Recognition Feature

### Overview
The Intent Recognition feature uses machine learning to predict user route choices **before** they reach decision points. This enables proactive systems and predictive content loading.

### Key Features
- **Multi-distance prediction**: Train models at 100, 75, 50, 25 units before junction
- **Flexible model selection**: Random Forest (fast) or Gradient Boosting (accurate)
- **Cross-validation**: Built-in model evaluation with custom folds
- **Feature importance**: Understand which behavioral cues matter most
- **Accuracy analysis**: See how prediction improves with proximity

### Workflow
1. **Run Discover Branches** first (to set clustering parameters)
2. **Select Intent Recognition** in Analysis tab
3. **Configure** prediction distances, model type, test split
4. **Run Analysis** to train ML models on historical data
5. **Visualize** results - feature importance, accuracy over distance, sample predictions
6. **Export** models and results for production deployment

### Use Cases
- **Proactive wayfinding**: Show navigation hints before user reaches junction
- **Predictive content loading**: Preload next area based on predicted route
- **Dynamic optimization**: Adjust environment before user arrives
- **A/B testing**: Test early interventions at different prediction distances

### Model Details
- **Features**: Spatial (distance, angle, offset), kinematic (speed, acceleration, curvature), gaze (if available), physiological (if available), contextual (previous choices)
- **Evaluation**: Cross-validation with stratified splits
- **Output**: Feature importance plots, accuracy vs. distance charts, per-junction models
