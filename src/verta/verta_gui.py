"""
VERTA Web GUI
=============

A Streamlit-based web interface for VERTA (Virtual Environment Route and Trajectory Analyzer).
Provides interactive junction management and real-time analysis.

Usage:
    streamlit run verta_gui.py

Or as a Python package:
    from verta.verta_gui import launch_gui
    launch_gui()
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from typing import List, Dict, Optional, Tuple
import tempfile
import zipfile
from pathlib import Path

import sys
from pathlib import Path

# Package imports - no path manipulation needed

from verta.verta_data_loader import load_folder, load_folder_with_gaze, Trajectory, ColumnMapping
from verta.verta_decisions import discover_decision_chain, discover_branches, assign_branches
from verta.verta_geometry import Circle, entered_junction_idx
from verta.verta_prediction import analyze_junction_choice_patterns, JunctionChoiceAnalyzer
from verta.verta_plotting import plot_flow_graph_map, plot_per_junction_flow_graph, plot_chain_overview
from verta.verta_metrics import _timing_for_traj, time_between_regions, compute_basic_trajectory_metrics, speed_through_junction, junction_transit_speed
from verta.verta_gaze import (
    compute_head_yaw_at_decisions,
    analyze_physiological_at_junctions,
    plot_gaze_directions_at_junctions,
    plot_physiological_by_branch,
    gaze_movement_consistency_report,
    analyze_pupil_dilation_trajectory,
    plot_pupil_trajectory_analysis,
    plot_pupil_dilation_heatmap,
    create_per_junction_pupil_heatmap
)
from verta.verta_intent_recognition import analyze_intent_recognition, IntentRecognitionAnalyzer
from verta.verta_logging import get_logger

# Configure Streamlit page
st.set_page_config(
    page_title="VERTA",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .junction-list-container {
        max-height: 60vh;
        overflow-y: auto;
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 10px;
        margin-top: 10px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffeaa7;
    }
</style>
""", unsafe_allow_html=True)

class VERTAGUI:
    """Main GUI class for VERTA"""

    def __init__(self):
        self.initialize_session_state()
        self.logger = get_logger()

    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'junctions' not in st.session_state:
            st.session_state.junctions = []
        if 'junction_r_outer' not in st.session_state:
            st.session_state.junction_r_outer = {}  # Store r_outer for each junction
        if 'trajectories' not in st.session_state:
            st.session_state.trajectories = []
        # Unified model: trajectories may include optional gaze/physio fields
        if 'gaze_column_mappings' not in st.session_state:
            st.session_state.gaze_column_mappings = {}
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        if 'current_step' not in st.session_state:
            st.session_state.current_step = "data_upload"
        if 'scale_factor' not in st.session_state:
            st.session_state.scale_factor = 0.2  # Default scale factor
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        # Flash message shown after reruns (tuple: (level, text))
        if 'flash_message' not in st.session_state:
            st.session_state.flash_message = None

        # Track junction state for UI refresh
        if 'junction_state_hash' not in st.session_state:
            st.session_state.junction_state_hash = 0

        # Debug: Track session state changes
        if 'debug_session_state' not in st.session_state:
            st.session_state.debug_session_state = {
                'trajectories_count': len(st.session_state.trajectories) if st.session_state.trajectories else 0,
                'gaze_trajectories_count': 0,
                'last_modified': 'initialize'
            }

    def render_header(self):
        """Render the main header"""
        st.markdown('<h1 class="main-header">🗺️ VERTA</h1>', unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: center; color: #666; margin-bottom: 2rem;">
            A Python tool for route and trajectory analysis in virtual environments
        </div>
        """, unsafe_allow_html=True)
        # Show any pending flash message at the very top
        self._show_flash()

    def _show_flash(self) -> None:
        """Display and clear one-time flash message stored in session state."""
        msg = st.session_state.get('flash_message')
        if not msg:
            return
        # msg can be a tuple (level, text) or just a string
        if isinstance(msg, tuple) and len(msg) >= 2:
            level, text = msg[0], msg[1]
        else:
            level, text = 'success', str(msg)
        if level == 'warning':
            st.warning(text)
        elif level == 'error':
            st.error(text)
        elif level == 'info':
            st.info(text)
        else:
            st.success(text)
        # Clear after showing once
        st.session_state.flash_message = None

    def render_navigation(self):
        """Render the navigation sidebar"""
        st.sidebar.title("Navigation")

        steps = {
            "data_upload": "📁 Data Upload",
            "junction_editor": "🎯 Junction Editor",
            "analysis": "📊 Analysis",
            "visualization": "📈 Visualization",
            "export": "💾 Export Results"
        }

        for step_key, step_name in steps.items():
            if st.sidebar.button(step_name, key=f"nav_{step_key}"):
                st.session_state.current_step = step_key
                st.rerun()

        st.sidebar.markdown("---")
        st.sidebar.markdown("### Current Status")

        # Status indicators
        data_status = "✅" if (st.session_state.trajectories and getattr(st.session_state, 'data_loaded', False)) else "❌"
        junction_status = "✅" if st.session_state.junctions else "❌"

        st.sidebar.markdown(f"{data_status} Data Loaded")
        st.sidebar.markdown(f"{junction_status} Junctions Defined")

        if st.session_state.trajectories and st.session_state.junctions:
            st.sidebar.markdown("✅ Ready for Analysis")
        else:
            st.sidebar.markdown("⚠️ Complete setup steps")

    def render_data_upload(self):
        """Render the data upload interface"""
        st.markdown('<h2 class="section-header">📁 Data Upload</h2>', unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### Upload Trajectory Data")

            # File upload
            uploaded_files = st.file_uploader(
                "Choose CSV files",
                type=['csv'],
                accept_multiple_files=True,
                help="Upload one or more CSV files containing trajectory data"
            )

            # Folder path input
            st.markdown("### Or specify folder path")
            folder_path = st.text_input(
                "Folder path:",
                value="",
                help="Path to folder containing CSV files"
            )

            # Column mapping
            st.markdown("### Column Mapping")
            col_x, col_z, col_t = st.columns(3)

            with col_x:
                x_col = st.text_input("X Column:", value="Headset.Head.Position.X")
            with col_z:
                z_col = st.text_input("Z Column:", value="Headset.Head.Position.Z")
            with col_t:
                t_col = st.text_input("Time Column:", value="Time")

            # New: Gaze/Physiology column mapping now lives here
            with st.expander("🔧 Gaze/Physiology Column Mapping", expanded=False):
                col_g1, col_g2 = st.columns(2)
                with col_g1:
                    head_forward_x_col = st.text_input("Head Forward X", value=st.session_state.gaze_column_mappings.get('head_forward_x', 'Headset.Head.Forward.X'))
                    head_forward_z_col = st.text_input("Head Forward Z", value=st.session_state.gaze_column_mappings.get('head_forward_z', 'Headset.Head.Forward.Z'))
                    gaze_x_map = st.text_input("Gaze X", value=st.session_state.gaze_column_mappings.get('gaze_x', 'Headset.Gaze.X'))
                    gaze_y_map = st.text_input("Gaze Y", value=st.session_state.gaze_column_mappings.get('gaze_y', 'Headset.Gaze.Y'))
                with col_g2:
                    pupil_l_map = st.text_input("Pupil Left", value=st.session_state.gaze_column_mappings.get('pupil_l', 'Headset.PupilDilation.L'))
                    pupil_r_map = st.text_input("Pupil Right", value=st.session_state.gaze_column_mappings.get('pupil_r', 'Headset.PupilDilation.R'))
                    heart_rate_map = st.text_input("Heart Rate", value=st.session_state.gaze_column_mappings.get('heart_rate', 'Headset.HeartRate'))

                st.session_state.gaze_column_mappings = {
                    'head_forward_x': head_forward_x_col.strip(),
                    'head_forward_z': head_forward_z_col.strip(),
                    'gaze_x': gaze_x_map.strip(),
                    'gaze_y': gaze_y_map.strip(),
                    'pupil_l': pupil_l_map.strip(),
                    'pupil_r': pupil_r_map.strip(),
                    'heart_rate': heart_rate_map.strip(),
                }

            # Analysis parameters
            st.markdown("### Analysis Parameters")
            col_scale, col_threshold = st.columns(2)

            with col_scale:
                scale = st.number_input("Scale Factor:", value=st.session_state.get("scale_factor", 0.2), min_value=0.01, max_value=1.0, step=0.01)
                st.session_state.scale_factor = scale  # Store scale factor in session state
            with col_threshold:
                motion_threshold = st.number_input("Motion Threshold:", value=0.1, min_value=0.01, max_value=1.0, step=0.01)

        with col2:
            st.markdown("### Quick Actions")

            if st.button("🔄 Load Data", type="primary"):
                if uploaded_files and folder_path.strip():
                    # Both provided - show warning and ask user to choose
                    st.warning("⚠️ **Both file uploads and folder path are specified.**")
                    st.info("**Current behavior:** File uploads will be processed (folder path will be ignored).")
                    st.info("**To use folder path instead:** Clear the file uploads and click 'Load Data' again.")

                    # Process uploaded files (current behavior)
                    self.load_uploaded_files(uploaded_files, x_col, z_col, t_col, scale, motion_threshold)
                elif uploaded_files:
                    # Process uploaded files
                    self.load_uploaded_files(uploaded_files, x_col, z_col, t_col, scale, motion_threshold)
                elif folder_path.strip():
                    # Process folder path
                    self.load_trajectory_data(folder_path, x_col, z_col, t_col, scale, motion_threshold)
                else:
                    st.warning("⚠️ Please upload files or specify a folder path")

            if st.button("📋 Load Sample Data"):
                self.load_sample_data()

            if st.session_state.trajectories:
                st.markdown("### Data Summary")
                st.write(f"**Trajectories loaded:** {len(st.session_state.trajectories)}")

                if len(st.session_state.trajectories) > 0:
                    sample_traj = st.session_state.trajectories[0]
                    st.write(f"**Sample trajectory points:** {len(sample_traj.x)}")
                    st.write(f"**X range:** {min(sample_traj.x):.1f} to {max(sample_traj.x):.1f}")
                    st.write(f"**Z range:** {min(sample_traj.z):.1f} to {max(sample_traj.z):.1f}")

    def load_trajectory_data(self, folder_path: str, x_col: str, z_col: str, t_col: str, scale: float, motion_threshold: float):
        """Load trajectory data from folder using unified model"""
        try:
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("🔄 Initializing data loading...")
            progress_bar.progress(10)

            # Build comprehensive column mapping for VR headset data
            column_mapping = {
                'x': x_col,
                'z': z_col,
                't': t_col,
                # VR headset gaze/physio columns
                'head_forward_x': st.session_state.gaze_column_mappings.get('head_forward_x', 'Headset.Head.Forward.X'),
                'head_forward_y': st.session_state.gaze_column_mappings.get('head_forward_y', 'Headset.Head.Forward.Y'),
                'head_forward_z': st.session_state.gaze_column_mappings.get('head_forward_z', 'Headset.Head.Forward.Z'),
                'head_up_x': st.session_state.gaze_column_mappings.get('head_up_x', 'Headset.Head.Up.X'),
                'head_up_y': st.session_state.gaze_column_mappings.get('head_up_y', 'Headset.Head.Up.Y'),
                'head_up_z': st.session_state.gaze_column_mappings.get('head_up_z', 'Headset.Head.Up.Z'),
                'gaze_x': st.session_state.gaze_column_mappings.get('gaze_x', 'Headset.Gaze.X'),
                'gaze_y': st.session_state.gaze_column_mappings.get('gaze_y', 'Headset.Gaze.Y'),
                'pupil_l': st.session_state.gaze_column_mappings.get('pupil_l', 'Headset.PupilDilation.L'),
                'pupil_r': st.session_state.gaze_column_mappings.get('pupil_r', 'Headset.PupilDilation.R'),
                'heart_rate': st.session_state.gaze_column_mappings.get('heart_rate', 'Headset.HeartRate'),
            }

            status_text.text("🔍 Scanning folder for CSV files...")
            progress_bar.progress(30)

            # Add a small delay to ensure progress bar is visible
            import time
            time.sleep(0.1)

            # Check what columns are available in the first CSV file and auto-detect gaze columns
            import glob
            import pandas as pd
            csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
            if csv_files:
                sample_df = pd.read_csv(csv_files[0])

                # Auto-detect gaze columns if mappings are empty
                if not st.session_state.gaze_column_mappings:
                    st.info("🔍 **Auto-detecting gaze columns from CSV file...**")

                    # Try to find gaze columns by common patterns
                    detected_mappings = {}

                    # Look for pupil dilation columns
                    pupil_cols = [col for col in sample_df.columns if 'pupil' in col.lower() and ('l' in col.lower() or 'left' in col.lower())]
                    if pupil_cols:
                        detected_mappings['pupil_l'] = pupil_cols[0]

                    pupil_cols = [col for col in sample_df.columns if 'pupil' in col.lower() and ('r' in col.lower() or 'right' in col.lower())]
                    if pupil_cols:
                        detected_mappings['pupil_r'] = pupil_cols[0]

                    # Look for heart rate columns
                    hr_cols = [col for col in sample_df.columns if 'heart' in col.lower() or 'hr' in col.lower()]
                    if hr_cols:
                        detected_mappings['heart_rate'] = hr_cols[0]

                    # Look for gaze columns
                    gaze_x_cols = [col for col in sample_df.columns if 'gaze' in col.lower() and ('x' in col.lower() or 'horizontal' in col.lower())]
                    if gaze_x_cols:
                        detected_mappings['gaze_x'] = gaze_x_cols[0]

                    gaze_y_cols = [col for col in sample_df.columns if 'gaze' in col.lower() and ('y' in col.lower() or 'vertical' in col.lower())]
                    if gaze_y_cols:
                        detected_mappings['gaze_y'] = gaze_y_cols[0]

                    # Look for head forward columns
                    head_fwd_x_cols = [col for col in sample_df.columns if 'head' in col.lower() and 'forward' in col.lower() and 'x' in col.lower()]
                    if head_fwd_x_cols:
                        detected_mappings['head_forward_x'] = head_fwd_x_cols[0]

                    head_fwd_z_cols = [col for col in sample_df.columns if 'head' in col.lower() and 'forward' in col.lower() and ('z' in col.lower() or 'depth' in col.lower())]
                    if head_fwd_z_cols:
                        detected_mappings['head_forward_z'] = head_fwd_z_cols[0]

                    # Update session state with detected mappings
                    if detected_mappings:
                        st.session_state.gaze_column_mappings.update(detected_mappings)
                        st.success(f"✅ **Auto-detected gaze columns:** {detected_mappings}")

                        # Update column mapping with detected values
                        column_mapping.update(detected_mappings)
                    else:
                        st.warning("⚠️ **No gaze columns auto-detected**")

                # Check if gaze columns exist (using current mappings)
                gaze_columns = ['Headset.Head.Forward.X', 'Headset.Head.Forward.Z', 'Headset.Gaze.X',
                              'Headset.Gaze.Y', 'Headset.PupilDilation.L', 'Headset.PupilDilation.R', 'Headset.HeartRate']

                # Use detected mappings if available
                actual_gaze_columns = []
                for gaze_type in ['head_forward_x', 'head_forward_z', 'gaze_x', 'gaze_y', 'pupil_l', 'pupil_r', 'heart_rate']:
                    col_name = st.session_state.gaze_column_mappings.get(gaze_type, gaze_columns[['head_forward_x', 'head_forward_z', 'gaze_x', 'gaze_y', 'pupil_l', 'pupil_r', 'heart_rate'].index(gaze_type)])
                    actual_gaze_columns.append(col_name)

                missing_gaze_cols = [col for col in actual_gaze_columns if col not in sample_df.columns]
                if missing_gaze_cols:
                    st.warning(f"⚠️ **Missing gaze columns:** {missing_gaze_cols}")
                else:
                    st.success("✅ **All gaze columns found in CSV!**")

            status_text.text("📊 Loading trajectory data...")
            progress_bar.progress(60)

            # Add a small delay to ensure progress bar is visible
            import time
            time.sleep(0.1)

            # Create progress callback function
            def update_progress(current, total, message):
                progress_percent = int(60 + (current / total) * 30)  # 60-90% range
                progress_bar.progress(progress_percent)
                status_text.text(message)

            # Use unified loader - always returns Trajectory objects with optional gaze fields
            trajectories = load_folder(
                folder=folder_path,
                pattern="*.csv",
                columns=column_mapping,
                require_time=False,
                scale=scale,
                motion_threshold=motion_threshold,
                progress_callback=update_progress
            )

            status_text.text("💾 Storing trajectories in session...")
            progress_bar.progress(90)

            # Add another small delay to show progress
            time.sleep(0.1)

            # Store unified trajectories
            st.session_state.trajectories = trajectories

            # Update status display
            st.session_state.data_loaded = True

            progress_bar.progress(100)
            status_text.text("✅ Data loading completed!")

            # Clear progress elements first
            progress_bar.empty()
            status_text.empty()

            # Queue success flash for top-of-page after rerun
            st.session_state.flash_message = ('success', f"🎉 Successfully loaded {len(trajectories)} trajectories!")

            # Force rerun to update status display and show flash at top
            st.rerun()

        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            if 'progress_bar' in locals():
                progress_bar.empty()
            if 'status_text' in locals():
                status_text.empty()

    def load_uploaded_files(self, uploaded_files, x_col: str, z_col: str, t_col: str, scale: float, motion_threshold: float):
        """Load trajectory data from uploaded files using unified model"""
        try:
            with st.spinner("Loading uploaded files..."):
                import pandas as pd
                import io
                import numpy as np
                import tempfile
                import os
                from verta.verta_data_loader import Trajectory, TrajectoryLoader, ColumnMapping

                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.text("🔄 Initializing file processing...")
                progress_bar.progress(5)

                # Build comprehensive column mapping for VR headset data (same as folder loading)
                column_mapping = {
                    'x': x_col,
                    'z': z_col,
                    't': t_col,
                    # VR headset gaze/physio columns
                    'head_forward_x': st.session_state.gaze_column_mappings.get('head_forward_x', 'Headset.Head.Forward.X'),
                    'head_forward_y': st.session_state.gaze_column_mappings.get('head_forward_y', 'Headset.Head.Forward.Y'),
                    'head_forward_z': st.session_state.gaze_column_mappings.get('head_forward_z', 'Headset.Head.Forward.Z'),
                    'head_up_x': st.session_state.gaze_column_mappings.get('head_up_x', 'Headset.Head.Up.X'),
                    'head_up_y': st.session_state.gaze_column_mappings.get('head_up_y', 'Headset.Head.Up.Y'),
                    'head_up_z': st.session_state.gaze_column_mappings.get('head_up_z', 'Headset.Head.Up.Z'),
                    'gaze_x': st.session_state.gaze_column_mappings.get('gaze_x', 'Headset.Gaze.X'),
                    'gaze_y': st.session_state.gaze_column_mappings.get('gaze_y', 'Headset.Gaze.Y'),
                    'pupil_l': st.session_state.gaze_column_mappings.get('pupil_l', 'Headset.PupilDilation.L'),
                    'pupil_r': st.session_state.gaze_column_mappings.get('pupil_r', 'Headset.PupilDilation.R'),
                    'heart_rate': st.session_state.gaze_column_mappings.get('heart_rate', 'Headset.HeartRate'),
                }

                # Auto-detect gaze columns from first uploaded file (same as folder loading)
                if uploaded_files and not st.session_state.gaze_column_mappings:
                    st.info("🔍 **Auto-detecting gaze columns from uploaded file...**")

                    # Read first file to detect columns
                    first_file = uploaded_files[0]
                    df_sample = pd.read_csv(io.StringIO(first_file.read().decode('utf-8')))
                    first_file.seek(0)  # Reset file pointer

                    # Try to find gaze columns by common patterns (same logic as folder loading)
                    detected_mappings = {}

                    # Look for pupil dilation columns
                    pupil_cols = [col for col in df_sample.columns if 'pupil' in col.lower() and ('l' in col.lower() or 'left' in col.lower())]
                    if pupil_cols:
                        detected_mappings['pupil_l'] = pupil_cols[0]

                    pupil_cols = [col for col in df_sample.columns if 'pupil' in col.lower() and ('r' in col.lower() or 'right' in col.lower())]
                    if pupil_cols:
                        detected_mappings['pupil_r'] = pupil_cols[0]

                    # Look for heart rate columns
                    hr_cols = [col for col in df_sample.columns if 'heart' in col.lower() or 'hr' in col.lower()]
                    if hr_cols:
                        detected_mappings['heart_rate'] = hr_cols[0]

                    # Look for gaze columns
                    gaze_x_cols = [col for col in df_sample.columns if 'gaze' in col.lower() and ('x' in col.lower() or 'horizontal' in col.lower())]
                    if gaze_x_cols:
                        detected_mappings['gaze_x'] = gaze_x_cols[0]

                    gaze_y_cols = [col for col in df_sample.columns if 'gaze' in col.lower() and ('y' in col.lower() or 'vertical' in col.lower())]
                    if gaze_y_cols:
                        detected_mappings['gaze_y'] = gaze_y_cols[0]

                    # Look for head forward columns
                    head_fwd_x_cols = [col for col in df_sample.columns if 'head' in col.lower() and 'forward' in col.lower() and 'x' in col.lower()]
                    if head_fwd_x_cols:
                        detected_mappings['head_forward_x'] = head_fwd_x_cols[0]

                    head_fwd_z_cols = [col for col in df_sample.columns if 'head' in col.lower() and 'forward' in col.lower() and ('z' in col.lower() or 'depth' in col.lower())]
                    if head_fwd_z_cols:
                        detected_mappings['head_forward_z'] = head_fwd_z_cols[0]

                    # Update session state with detected mappings
                    if detected_mappings:
                        st.session_state.gaze_column_mappings.update(detected_mappings)
                        st.success(f"✅ **Auto-detected gaze columns:** {detected_mappings}")

                        # Update column mapping with detected values
                        column_mapping.update(detected_mappings)
                    else:
                        st.warning("⚠️ **No gaze columns auto-detected**")

                # Create temporary directory to store uploaded files
                with tempfile.TemporaryDirectory() as temp_dir:
                    status_text.text("📁 Saving uploaded files to temporary directory...")
                    progress_bar.progress(10)

                    # Save uploaded files to temporary directory
                    temp_files = []
                    for i, uploaded_file in enumerate(uploaded_files):
                        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(temp_file_path, 'wb') as f:
                            f.write(uploaded_file.getvalue())
                        temp_files.append(temp_file_path)

                    status_text.text("📊 Loading trajectories using unified loader...")
                    progress_bar.progress(20)

                    # Use the same unified loader as folder loading
                    loader = TrajectoryLoader(ColumnMapping.from_dict(column_mapping))

                    # Create progress callback function
                    def update_progress(current, total, message):
                        progress_percent = int(20 + (current / total) * 70)  # 20-90% range
                        progress_bar.progress(progress_percent)
                        status_text.text(message)

                    # Load trajectories using unified loader
                    trajectories = loader.load_folder(
                        folder=temp_dir,
                        pattern="*.csv",
                        trajectory_class=Trajectory,
                        require_time=False,
                        scale=scale,
                        motion_threshold=motion_threshold,
                        progress_callback=update_progress
                    )

                if trajectories:
                    status_text.text("💾 Storing trajectories in session...")
                    progress_bar.progress(90)

                    st.session_state.trajectories = trajectories

                    # Update status display
                    st.session_state.data_loaded = True

                    progress_bar.progress(100)
                    status_text.text("✅ File processing completed!")

                    # Clear progress elements first
                    progress_bar.empty()
                    status_text.empty()

                    # Queue success flash for top-of-page after rerun
                    st.session_state.flash_message = (
                        'success',
                        f"🎉 Successfully loaded {len(trajectories)} trajectories from {len(uploaded_files)} files!"
                    )

                    # Force rerun to update status display and show flash at top
                    st.rerun()
                else:
                    st.error("❌ No valid trajectories could be loaded from uploaded files")

        except Exception as e:
            st.error(f"❌ Error loading uploaded files: {str(e)}")

    def load_sample_data(self):
        """Load sample data for demonstration"""
        try:
            with st.spinner("Loading sample data..."):
                # Create sample trajectories
                np.random.seed(42)
                trajectories = []

                for i in range(10):
                    # Create a sample trajectory
                    n_points = np.random.randint(1000, 5000)
                    t = np.linspace(0, 100, n_points)
                    x = np.cumsum(np.random.normal(0, 0.5, n_points)) + 500
                    z = np.cumsum(np.random.normal(0, 0.5, n_points)) + 200

                    trajectory = Trajectory(
                        tid=str(i),
                        x=x,
                        z=z,
                        t=t
                    )
                    trajectories.append(trajectory)

                st.session_state.trajectories = trajectories
                st.success(f"✅ Loaded {len(trajectories)} sample trajectories!")

        except Exception as e:
            st.error(f"❌ Error loading sample data: {str(e)}")

    def load_assign_trajectories(self, assign_params: dict = None):
        """Load trajectories for assign function based on trajectory option"""
        try:
            # Get scale factor from assign parameters
            assign_scale = assign_params.get("assign_scale", 0.2) if assign_params else 0.2
            trajectory_option = assign_params.get("trajectory_option", "Upload files") if assign_params else "Upload files"

            if trajectory_option == "Upload files":
                trajectory_files = assign_params.get("trajectory_files") if assign_params else None
                if not trajectory_files:
                    st.error("❌ No trajectory files uploaded")
                    return None

                # Process uploaded files
                trajectories = []
                for uploaded_file in trajectory_files:
                    # Save uploaded file to temporary location
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name

                    # Load trajectory data
                    df = pd.read_csv(tmp_path)

                    # Use default column names if not specified
                    x_col = "Headset.Head.Position.X" if "Headset.Head.Position.X" in df.columns else df.columns[0]
                    z_col = "Headset.Head.Position.Z" if "Headset.Head.Position.Z" in df.columns else df.columns[1]
                    t_col = "Time" if "Time" in df.columns else df.columns[2] if len(df.columns) > 2 else None

                    # Filter out NaN values in coordinate columns
                    coord_mask = df[[x_col, z_col]].notnull().all(axis=1)
                    df_clean = df[coord_mask].copy()

                    if len(df_clean) == 0:
                        st.warning(f"⚠️ Skipping {uploaded_file.name}: All coordinates are NaN")
                        continue

                    if len(df_clean) < len(df):
                        st.info(f"ℹ️ Cleaned {uploaded_file.name}: Removed {len(df) - len(df_clean)} rows with NaN coordinates")

                    # Create trajectory with scale factor applied
                    trajectory = Trajectory(
                        tid=uploaded_file.name,
                        x=df_clean[x_col].values * assign_scale,  # Apply scale factor
                        z=df_clean[z_col].values * assign_scale,  # Apply scale factor
                        t=df_clean[t_col].values if t_col else np.arange(len(df_clean))
                    )
                    trajectories.append(trajectory)

                    # Clean up temporary file
                    os.unlink(tmp_path)

                return trajectories

            else:  # Select folder
                trajectory_folder = assign_params.get("trajectory_folder") if assign_params else None
                if not trajectory_folder or not trajectory_folder.strip():
                    st.error("❌ No trajectory folder path specified")
                    return None

                # Load from folder
                column_mapping = {
                    "x": "Headset.Head.Position.X",
                    "z": "Headset.Head.Position.Z",
                    "t": "Time"
                }

                trajectories = load_folder(
                    folder=trajectory_folder,
                    pattern="*.csv",
                    columns=column_mapping,
                    scale=assign_scale,  # Use assign-specific scale factor
                    motion_threshold=0.1
                )

                if not trajectories:
                    st.error(f"❌ No trajectories found in folder: {trajectory_folder}")
                    return None

                # Show data cleaning summary
                total_trajectories = len(trajectories)
                st.info(f"ℹ️ Loaded {total_trajectories} trajectories from folder")
                st.info(f"ℹ️ NaN filtering applied during loading (built into load_folder function)")

                return trajectories

        except Exception as e:
            st.error(f"❌ Failed to load assign trajectories: {str(e)}")
            return None

    def load_assign_centers(self, assign_params: dict = None):
        """Load junction centers for assign function based on centers option"""
        try:
            centers_option = assign_params.get("centers_option", "Use session centers") if assign_params else "Use session centers"

            if centers_option == "Use session centers":
                # Get centers from previous discover analysis
                if "branches" not in st.session_state.analysis_results:
                    st.error("❌ No centers found from previous discover analysis")
                    return None

                centers_dict = {}
                for junction_key, branch_data in st.session_state.analysis_results["branches"].items():
                    if "centers" in branch_data:
                        centers_dict[junction_key] = branch_data["centers"]

                return centers_dict

            elif centers_option == "Upload files":
                centers_files = assign_params.get("centers_files") if assign_params else None
                if not centers_files:
                    st.error("❌ No centers files uploaded")
                    return None

                centers_dict = {}
                for i, centers_file in enumerate(centers_files):
                    # Save uploaded file to temporary location
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.npy') as tmp_file:
                        tmp_file.write(centers_file.getvalue())
                        tmp_path = tmp_file.name

                    # Load centers
                    centers = np.load(tmp_path)
                    centers_dict[f"junction_{i}"] = centers

                    # Clean up temporary file
                    os.unlink(tmp_path)

                return centers_dict

            else:  # Select folder
                centers_folder = assign_params.get("centers_folder") if assign_params else None
                if not centers_folder or not centers_folder.strip():
                    st.error("❌ No centers folder path specified")
                    return None

                # Load centers from folder (search subfolders)
                centers_dict = {}
                for root, dirs, files in os.walk(centers_folder):
                    for file in files:
                        if file.startswith("branch_centers_j") and file.endswith(".npy"):
                            # Extract junction number from filename
                            junction_num = file.split("_")[-1].split(".")[0]
                            centers_path = os.path.join(root, file)
                            centers = np.load(centers_path)
                            centers_dict[f"junction_{junction_num}"] = centers

                if not centers_dict:
                    st.error(f"❌ No center files found in folder: {centers_folder}")
                    st.info("💡 Looking for files named: branch_centers_j*.npy")
                    return None

                return centers_dict

        except Exception as e:
            st.error(f"❌ Failed to load assign centers: {str(e)}")
            return None

    def render_junction_editor(self):
        """Render the interactive junction editor"""
        st.markdown('<h2 class="section-header">🎯 Junction Editor</h2>', unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### Interactive Junction Management")

            # Instructions
            st.info("""
            **How to use:**
            1. **Add junctions** using the controls on the right
            2. **Edit existing junctions** by changing position, radius, or r_outer values
            3. **Hover over junctions** on the map to see their properties
            4. **Delete junctions** using the 🗑️ button
            """)

            # Create interactive plot for junction editing
            if st.session_state.trajectories:
                self.render_junction_plot()
            else:
                st.warning("⚠️ Please load trajectory data first")

        with col2:
            st.markdown("### Junction Controls")

            # Add new junction section (always visible)
            with st.container():
                st.markdown("#### Add New Junction")
                col_x, col_z = st.columns(2)

                with col_x:
                    new_x = st.number_input("X Position:", value=500.0, step=10.0)
                with col_z:
                    new_z = st.number_input("Z Position:", value=300.0, step=10.0)

                col_radius, col_r_outer = st.columns(2)
                with col_radius:
                    new_radius = st.number_input("Radius:", value=30.0, min_value=5.0, max_value=100.0, step=5.0)
                with col_r_outer:
                    new_r_outer = st.number_input("R Outer:", value=50.0, min_value=10.0, max_value=200.0, step=5.0)

                if st.button("➕ Add Junction"):
                    new_junction = Circle(cx=new_x, cz=new_z, r=new_radius)
                    st.session_state.junctions.append(new_junction)
                    st.session_state.junction_r_outer[len(st.session_state.junctions)-1] = new_r_outer

                    # Update junction state hash to force UI refresh
                    st.session_state.junction_state_hash += 1

                    st.success(f"Added junction at ({new_x}, {new_z}) with r_outer={new_r_outer}")
                    st.rerun()

            st.markdown("---")

            # Bulk operations (always visible)
            st.markdown("#### Bulk Operations")
            col_clear, col_sample = st.columns(2)

            with col_clear:
                if st.button("🗑️ Clear All"):
                    st.session_state.junctions = []
                    st.session_state.junction_r_outer = {}

                    # Update junction state hash to force UI refresh
                    st.session_state.junction_state_hash += 1

                    st.rerun()

            with col_sample:
                if st.button("📋 Load Sample"):
                    self.load_sample_junctions()

            st.markdown("---")

            # Scrollable junction list
            st.markdown("#### Current Junctions")
            if st.session_state.junctions:
                # Create a scrollable container for the junction list
                st.markdown(f"**Total Junctions: {len(st.session_state.junctions)}**")
                with st.container():
                    st.markdown('<div class="junction-list-container">', unsafe_allow_html=True)

                    # Use a scrollable area for the junction list
                    for i, junction in enumerate(st.session_state.junctions):
                        with st.expander(f"Junction {i} - ({junction.cx:.1f}, {junction.cz:.1f})", expanded=False):
                            # Junction info and delete button
                            col_del, col_info = st.columns([1, 4])

                            with col_del:
                                if st.button("🗑️", key=f"del_{i}", help="Delete this junction"):
                                    # Store the deleted junction info for debugging
                                    deleted_junction = st.session_state.junctions[i]

                                    # Remove the junction from the list
                                    st.session_state.junctions.pop(i)

                                    # Remove the corresponding r_outer entry
                                    if i in st.session_state.junction_r_outer:
                                        del st.session_state.junction_r_outer[i]

                                    # Reindex remaining junctions and r_outer values
                                    new_r_outer = {}
                                    for j, junction in enumerate(st.session_state.junctions):
                                        old_idx = j + (1 if j >= i else 0)
                                        if old_idx in st.session_state.junction_r_outer:
                                            new_r_outer[j] = st.session_state.junction_r_outer[old_idx]
                                    st.session_state.junction_r_outer = new_r_outer

                                    # Show success message
                                    st.success(f"Deleted Junction {i} at ({deleted_junction.cx:.1f}, {deleted_junction.cz:.1f})")

                                    # Update junction state hash to force UI refresh
                                    st.session_state.junction_state_hash += 1

                                    # Force a complete rerun to refresh all UI elements
                                    st.rerun()

                            with col_info:
                                st.write(f"Position: ({junction.cx:.1f}, {junction.cz:.1f})")
                                st.write(f"Radius: {junction.r}")
                                st.write(f"R_outer: {st.session_state.junction_r_outer.get(i, 50.0)}")

                            # Position editing
                            st.markdown("**Edit Position:**")
                            col_x_edit, col_z_edit = st.columns(2)

                            with col_x_edit:
                                new_x = st.number_input(
                                    f"X:",
                                    value=float(junction.cx),
                                    step=1.0,
                                    key=f"x_edit_{i}"
                                )

                            with col_z_edit:
                                new_z = st.number_input(
                                    f"Z:",
                                    value=float(junction.cz),
                                    step=1.0,
                                    key=f"z_edit_{i}"
                                )

                            # Radius editing
                            new_radius = st.number_input(
                                f"Radius:",
                                value=float(junction.r),
                                min_value=5.0,
                                max_value=100.0,
                                step=1.0,
                                key=f"radius_edit_{i}"
                            )

                            # R_outer control
                            current_r_outer = st.session_state.junction_r_outer.get(i, 50.0)
                            new_r_outer = st.number_input(
                                f"R Outer:",
                                value=current_r_outer,
                                min_value=10.0,
                                max_value=200.0,
                                step=5.0,
                                key=f"r_outer_{i}"
                            )

                            # Update junction if any values changed
                            if (new_x != junction.cx or new_z != junction.cz or
                                new_radius != junction.r or new_r_outer != current_r_outer):

                                # Update junction
                                st.session_state.junctions[i] = Circle(cx=new_x, cz=new_z, r=new_radius)
                                st.session_state.junction_r_outer[i] = new_r_outer

                                # Update junction state hash to force UI refresh
                                st.session_state.junction_state_hash += 1

                                st.rerun()

                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No junctions defined yet")

    def render_junction_plot(self):
        """Render interactive plot for junction editing"""
        if not st.session_state.trajectories:
            return

        # Create plot
        fig = go.Figure()

        # Add ALL trajectories (with sampling for performance)
        all_trajectories = st.session_state.trajectories
        for i, traj in enumerate(all_trajectories):
            # Sample every 20th point for performance with many trajectories
            sample_rate = max(1, len(traj.x) // 1000)  # Adaptive sampling
            fig.add_trace(go.Scatter(
                x=traj.x[::sample_rate],
                y=traj.z[::sample_rate],
                mode='lines',
                line=dict(color='lightgray', width=0.5),
                name=f'Trajectory {i}',
                showlegend=False,
                opacity=0.6
            ))

        # Add junctions with r_outer circles
        for i, junction in enumerate(st.session_state.junctions):
            # Get r_outer for this junction
            r_outer = st.session_state.junction_r_outer.get(i, 50.0)

            # Junction center with hover info
            fig.add_trace(go.Scatter(
                x=[junction.cx],
                y=[junction.cz],
                mode='markers',
                marker=dict(size=20, color='red', symbol='circle'),
                name=f'J{i}',
                text=f'J{i}<br>Pos: ({junction.cx:.1f}, {junction.cz:.1f})<br>Radius: {junction.r}<br>R_outer: {r_outer}',
                textposition='middle center',
                textfont=dict(color='white', size=14, family="Arial Black"),
                hovertemplate=f'<b>Junction {i}</b><br>' +
                             f'Position: ({junction.cx:.1f}, {junction.cz:.1f})<br>' +
                             f'Radius: {junction.r}<br>' +
                             f'R_outer: {r_outer}<br>' +
                             '<extra></extra>'
            ))

            # Junction radius circle (decision radius)
            theta = np.linspace(0, 2*np.pi, 100)
            circle_x = junction.cx + junction.r * np.cos(theta)
            circle_z = junction.cz + junction.r * np.sin(theta)

            fig.add_trace(go.Scatter(
                x=circle_x,
                y=circle_z,
                mode='lines',
                line=dict(color='orange', width=3),
                showlegend=False,
                hovertemplate=f'<b>Junction {i} - Decision Radius</b><br>' +
                             f'Position: ({junction.cx:.1f}, {junction.cz:.1f})<br>' +
                             f'Decision Radius: {junction.r}<br>' +
                             f'R_outer: {r_outer}<br>' +
                             '<extra></extra>',
                name=f'J{i} Decision Radius'
            ))

            # R_outer circle (analysis radius)
            r_outer_x = junction.cx + r_outer * np.cos(theta)
            r_outer_z = junction.cz + r_outer * np.sin(theta)

            fig.add_trace(go.Scatter(
                x=r_outer_x,
                y=r_outer_z,
                mode='lines',
                line=dict(color='blue', width=2, dash='dash'),
                showlegend=False,
                hovertemplate=f'<b>Junction {i} - Analysis Radius</b><br>' +
                             f'Position: ({junction.cx:.1f}, {junction.cz:.1f})<br>' +
                             f'Decision Radius: {junction.r}<br>' +
                             f'R_outer: {r_outer}<br>' +
                             '<extra></extra>',
                name=f'J{i} R_outer'
            ))

            # Add invisible junction area for better hover capture
            # Create a filled circle area that will capture hover events
            theta_dense = np.linspace(0, 2*np.pi, 200)
            area_x = junction.cx + r_outer * np.cos(theta_dense)
            area_z = junction.cz + r_outer * np.sin(theta_dense)

            fig.add_trace(go.Scatter(
                x=area_x,
                y=area_z,
                mode='lines',
                fill='toself',
                fillcolor='rgba(255, 0, 0, 0.05)',  # Very light red fill
                line=dict(width=0),  # No visible line
                showlegend=False,
                hovertemplate=f'<b>Junction {i} Area</b><br>' +
                             f'Position: ({junction.cx:.1f}, {junction.cz:.1f})<br>' +
                             f'Decision Radius: {junction.r}<br>' +
                             f'R_outer: {r_outer}<br>' +
                             '<extra></extra>',
                name=f'J{i} Area',
                hoveron='fills'  # Only show hover on filled area
            ))

        # Update layout
        fig.update_layout(
            title="Interactive Junction Editor - All Trajectories",
            xaxis_title="X Position",
            yaxis_title="Z Position",
            hovermode='closest',
            showlegend=False
        )

        # Set equal aspect ratio for both axes
        fig.update_xaxes(scaleanchor="y", scaleratio=1)
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

        # Add legend manually
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color='red'),
            name='Junction Center',
            showlegend=True
        ))
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color='orange', width=3),
            name='Decision Radius',
            showlegend=True
        ))
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color='blue', width=2, dash='dash'),
            name='R_outer (Analysis Radius)',
            showlegend=True
        ))
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color='lightgray', width=1),
            name='Trajectories',
            showlegend=True
        ))

        st.plotly_chart(fig, config={'displayModeBar': True}, width='stretch')

    def load_sample_junctions(self):
        """Load sample junctions"""
        sample_junctions = [
            Circle(cx=685, cz=170, r=30),
            Circle(cx=550, cz=-90, r=30),
            Circle(cx=730, cz=440, r=20),
            Circle(cx=520, cz=340, r=40),
            Circle(cx=500, cz=515, r=20),
            Circle(cx=575, cz=430, r=15),
            Circle(cx=500, cz=205, r=20)
        ]

        # Sample r_outer values
        sample_r_outer = [100.0, 50.0, 45.0, 45.0, 45.0, 30.0, 45.0]

        st.session_state.junctions = sample_junctions
        st.session_state.junction_r_outer = {i: r_outer for i, r_outer in enumerate(sample_r_outer)}
        st.success("✅ Loaded sample junctions with r_outer values!")
        st.rerun()

    def render_analysis(self):
        """Render the analysis interface"""
        st.markdown('<h2 class="section-header">📊 Analysis</h2>', unsafe_allow_html=True)

        if not st.session_state.trajectories:
            st.warning("⚠️ Please load trajectory data first")
            return

        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown("### Analysis Configuration")

            # Analysis type selection
            analysis_type_options = {
                "discover": "🔍 Discover Branches - Find decision branches at junctions",
                "assign": "📊 Assign Trajectories - Assign trajectories to discovered branches",
                "metrics": "📈 Movement Metrics - Calculate trajectory movement patterns and timing",
                "gaze": "👁️ Gaze & Physiology - Analyze eye tracking and physiological data",
                "predict": "🔮 Predict Choices - Predict junction choice patterns",
                "intent": "🧠 Intent Recognition - Predict route choices BEFORE decision points (ML)",
                "enhanced": "🚨 Enhanced Analysis - Evacuation planning, risk assessment, and efficiency metrics"
            }

            # Reset analysis type if we're coming from another tab
            if st.session_state.current_step == "analysis":
                # Clear any cached analysis type to ensure fresh selection
                if "cached_analysis_type" in st.session_state:
                    del st.session_state.cached_analysis_type

            analysis_type = st.selectbox(
                "Analysis Type:",
                list(analysis_type_options.keys()),
                format_func=lambda x: analysis_type_options[x],
                help="Select the type of analysis to perform",
                key=f"analysis_type_select_{st.session_state.current_step}"
            )

            # Store the selected analysis type
            st.session_state.cached_analysis_type = analysis_type

            # Initialize default values
            decision_mode = "pathlen"
            cluster_method = "dbscan"
            seed = 42

            # Initialize cluster parameters with defaults
            dbscan_eps = 0.5
            dbscan_min_samples = 5
            dbscan_angle_eps = 15.0
            kmeans_k = 3
            kmeans_k_min = 2
            kmeans_k_max = 6
            auto_k_min = 2
            auto_k_max = 6
            auto_min_sep_deg = 12.0
            auto_angle_eps = 15.0

            # Initialize decision mode parameters with defaults
            radial_r_outer = 50.0
            radial_epsilon = 0.05
            pathlen_path_length = 100.0
            pathlen_linger_delta = 0.0
            hybrid_r_outer = 50.0
            hybrid_path_length = 100.0
            hybrid_linger_delta = 0.0

            # Initialize metrics parameters with defaults
            metrics_decision_mode = "pathlen"
            metrics_distance = 100.0
            metrics_r_outer = 50.0
            metrics_trend_window = 5
            metrics_min_outward = 0.0

            # Initialize discover parameters with defaults
            discover_decision_mode = "hybrid"
            discover_r_outer = 50.0
            discover_epsilon = 0.05
            discover_path_length = 100.0
            discover_linger_delta = 0.0
            discover_hybrid_r_outer = 50.0
            discover_hybrid_path_length = 100.0

            # Show parameters based on analysis type
            if analysis_type == "predict":
                # Predict analysis uses only spatial tracking - no parameters needed
                st.info("ℹ️ Predict analysis uses spatial tracking only. No additional parameters required.")

                # Set default values for compatibility (not used in analysis)
                cluster_method = "kmeans"
                seed = 42
                decision_mode = "hybrid"

                # REMOVED: All cluster method parameters - not needed for spatial tracking only

            elif analysis_type == "intent":
                # Intent Recognition - ML-based early prediction
                st.markdown("#### 🧠 Intent Recognition Configuration")
                st.info("🤖 Machine Learning-based prediction of route choices **before** users reach decision points.")

                # Check if Discover has been run
                has_discover_results = (st.session_state.analysis_results and
                                       'branches' in st.session_state.analysis_results)

                if has_discover_results:
                    st.success("✅ Will use branch assignments from your previous 'Discover Branches' analysis")
                else:
                    st.warning("⚠️ No 'Discover Branches' results found. Will use default clustering parameters.")
                    st.info("💡 **Recommended:** Run 'Discover Branches' analysis first to control clustering settings!")

                # Prediction distances
                st.markdown("##### Prediction Distances")
                dist_col1, dist_col2, dist_col3, dist_col4 = st.columns(4)

                with dist_col1:
                    intent_dist_100 = st.checkbox("100 units", value=True, help="Predict 100 units before junction")
                with dist_col2:
                    intent_dist_75 = st.checkbox("75 units", value=True, help="Predict 75 units before junction")
                with dist_col3:
                    intent_dist_50 = st.checkbox("50 units", value=True, help="Predict 50 units before junction")
                with dist_col4:
                    intent_dist_25 = st.checkbox("25 units", value=True, help="Predict 25 units before junction")

                # Build prediction distances list
                intent_prediction_distances = []
                if intent_dist_100:
                    intent_prediction_distances.append(100.0)
                if intent_dist_75:
                    intent_prediction_distances.append(75.0)
                if intent_dist_50:
                    intent_prediction_distances.append(50.0)
                if intent_dist_25:
                    intent_prediction_distances.append(25.0)

                if not intent_prediction_distances:
                    st.warning("⚠️ Select at least one prediction distance!")
                    intent_prediction_distances = [50.0]  # Default

                # Model configuration
                st.markdown("##### Model Configuration")
                model_col1, model_col2 = st.columns(2)

                with model_col1:
                    intent_model_type = st.selectbox(
                        "ML Model:",
                        ["random_forest", "gradient_boosting"],
                        index=0,
                        help="Random Forest: Fast, robust | Gradient Boosting: More accurate but slower"
                    )

                with model_col2:
                    intent_cv_folds = st.number_input(
                        "Cross-validation Folds:",
                        value=5,
                        min_value=2,
                        max_value=10,
                        help="Number of folds for cross-validation"
                    )

                # Feature configuration
                with st.expander("🔧 Advanced: Feature Configuration"):
                    st.markdown("**Features Used:**")
                    st.markdown("""
                    - ✅ **Spatial**: Distance, approach angle, lateral offset
                    - ✅ **Kinematic**: Speed, acceleration, curvature, sinuosity
                    - ✅ **Gaze** (if available): Gaze angle, alignment, head rotation
                    - ✅ **Physiological** (if available): Heart rate, pupil dilation
                    - ✅ **Contextual**: Previous junction choices
                    """)

                    intent_test_split = st.slider(
                        "Test Set Size (%):",
                        min_value=10,
                        max_value=40,
                        value=20,
                        help="Percentage of data reserved for testing"
                    )

                # Store intent parameters in session state
                if 'intent_params' not in st.session_state:
                    st.session_state.intent_params = {}

                st.session_state.intent_params = {
                    'prediction_distances': intent_prediction_distances,
                    'model_type': intent_model_type,
                    'cv_folds': intent_cv_folds,
                    'test_split': intent_test_split / 100.0
                }

                # Set default values for compatibility
                cluster_method = "kmeans"
                seed = 42
                decision_mode = "hybrid"

            elif analysis_type == "discover":
                # Clustering parameters (used by discover analysis)
                st.markdown("#### Clustering Parameters")
                col_method, col_seed = st.columns(2)

                with col_method:
                    cluster_method = st.selectbox(
                        "Cluster Method:",
                        ["dbscan", "kmeans", "auto"],
                        index=0,  # Default to DBSCAN
                        help="Clustering method for discover analysis"
                    )

                with col_seed:
                    seed = st.number_input(
                        "Random Seed:",
                        value=42,
                        min_value=0,
                        max_value=10000,
                        step=1,
                        help="Random seed for reproducibility"
                    )

                # Decision mode parameters (needed for discover analysis)
                st.markdown("#### Decision Mode Parameters")
                col_decision_mode, col_decision_param = st.columns(2)

                with col_decision_mode:
                    discover_decision_mode = st.selectbox(
                        "Decision Mode:",
                        ["radial", "pathlen", "hybrid"],
                        index=2,  # Default to hybrid
                        help="Decision mode for discover analysis"
                    )

                with col_decision_param:
                    if discover_decision_mode == "radial":
                        st.info("ℹ️ Using junction-specific r_outer values from the Junctions tab")
                        # r_outer will be overridden by junction-specific values
                        discover_r_outer = 50.0
                        discover_epsilon = st.number_input(
                            "Epsilon:",
                            value=0.05,
                            min_value=0.01,
                            max_value=1.0,
                            step=0.01,
                            help="Epsilon parameter"
                        )
                    elif discover_decision_mode == "pathlen":
                        discover_path_length = st.number_input(
                            "Path Length:",
                            value=100.0,
                            min_value=10.0,
                            max_value=500.0,
                            step=10.0,
                            help="Path length for pathlen mode"
                        )
                        discover_linger_delta = st.number_input(
                            "Linger Delta:",
                            value=0.0,
                            min_value=0.0,
                            max_value=50.0,
                            step=1.0,
                            help="Linger distance beyond junction"
                        )
                    elif discover_decision_mode == "hybrid":
                        st.info("ℹ️ Using junction-specific r_outer values from the Junctions tab")
                        # r_outer will be overridden by junction-specific values
                        discover_hybrid_r_outer = 50.0
                        discover_hybrid_path_length = st.number_input(
                            "Hybrid Path Length:",
                            value=100.0,
                            min_value=10.0,
                            max_value=500.0,
                            step=10.0,
                            help="Path length for hybrid mode"
                        )
                        discover_hybrid_linger_delta = st.number_input(
                            "Hybrid Linger Delta:",
                            value=0.0,
                            min_value=0.0,
                            max_value=50.0,
                            step=1.0,
                            help="Linger distance beyond junction for hybrid mode"
                        )

                # Dynamic parameters based on cluster method (for discover analysis)
                st.markdown("#### Cluster Method Parameters")
                if cluster_method == "dbscan":
                    col_eps, col_min_samples = st.columns(2)
                    with col_eps:
                        dbscan_eps = st.number_input(
                            "DBSCAN Epsilon (eps):",
                            value=0.5,
                            min_value=0.1,
                            max_value=10.0,
                            step=0.1,
                            help="Maximum distance between samples in the same neighborhood"
                        )
                    with col_min_samples:
                        dbscan_min_samples = st.number_input(
                            "DBSCAN Min Samples:",
                            value=5,
                            min_value=1,
                            max_value=50,
                            step=1,
                            help="Minimum number of samples in a neighborhood"
                        )

                    # Add angle_eps parameter for DBSCAN
                    dbscan_angle_eps = st.number_input(
                        "DBSCAN Angle Epsilon (degrees):",
                        value=11.0,
                        min_value=1.0,
                        max_value=90.0,
                        step=1.0,
                        help="Angle epsilon for DBSCAN clustering (angular separation between clusters)"
                    )
                elif cluster_method == "kmeans":
                    col_k, col_k_range = st.columns(2)
                    with col_k:
                        kmeans_k = st.number_input(
                            "K-Means K (number of clusters):",
                            value=3,
                            min_value=2,
                            max_value=20,
                            step=1,
                            help="Number of clusters to form"
                        )
                    with col_k_range:
                        kmeans_k_min = st.number_input(
                            "K-Means K Min:",
                            value=2,
                            min_value=2,
                            max_value=10,
                            step=1,
                            help="Minimum number of clusters for auto selection"
                        )
                        kmeans_k_max = st.number_input(
                            "K-Means K Max:",
                            value=6,
                            min_value=3,
                            max_value=20,
                            step=1,
                            help="Maximum number of clusters for auto selection"
                        )
                elif cluster_method == "auto":
                    col_k_range, col_separation = st.columns(2)
                    with col_k_range:
                        auto_k_min = st.number_input(
                            "Auto K Min:",
                            value=2,
                            min_value=2,
                            max_value=10,
                            step=1,
                            help="Minimum number of clusters for auto selection"
                        )
                        auto_k_max = st.number_input(
                            "Auto K Max:",
                            value=6,
                            min_value=3,
                            max_value=20,
                            step=1,
                            help="Maximum number of clusters for auto selection"
                        )
                    with col_separation:
                        auto_min_sep_deg = st.number_input(
                            "Auto Min Separation (degrees):",
                            value=12.0,
                            min_value=1.0,
                            max_value=90.0,
                            step=1.0,
                            help="Minimum separation in degrees between clusters"
                        )
                        auto_angle_eps = st.number_input(
                            "Auto Angle Epsilon (degrees):",
                            value=11.0,
                            min_value=1.0,
                            max_value=90.0,
                            step=1.0,
                            help="Angle epsilon for auto clustering"
                        )

            elif analysis_type == "enhanced":
                # Enhanced analysis parameters (same as discover since it uses discover_decision_chain)
                st.markdown("#### Enhanced Analysis Parameters")
                st.info("🚨 Enhanced analysis uses the same clustering and decision parameters as discover analysis, then performs evacuation planning, risk assessment, and efficiency analysis.")

                # Clustering parameters (same as discover)
                st.markdown("##### Clustering Parameters")
                col_method, col_seed = st.columns(2)

                with col_method:
                    cluster_method = st.selectbox(
                        "Cluster Method:",
                        ["dbscan", "kmeans", "auto"],
                        index=0,  # Default to DBSCAN
                        help="Clustering method for enhanced analysis"
                    )

                with col_seed:
                    seed = st.number_input(
                        "Random Seed:",
                        value=42,
                        min_value=0,
                        max_value=10000,
                        step=1,
                        help="Random seed for reproducibility"
                    )

                # Decision mode parameters (same as discover)
                st.markdown("##### Decision Mode Parameters")
                col_decision_mode, col_decision_param = st.columns(2)

                with col_decision_mode:
                    discover_decision_mode = st.selectbox(
                        "Decision Mode:",
                        ["radial", "pathlen", "hybrid"],
                        index=2,  # Default to hybrid
                        help="Decision mode for enhanced analysis"
                    )

                with col_decision_param:
                    if discover_decision_mode == "radial":
                        st.info("ℹ️ Using junction-specific r_outer values from the Junctions tab")
                        # r_outer will be overridden by junction-specific values
                        discover_r_outer = 50.0
                        discover_epsilon = st.number_input(
                            "Epsilon:",
                            value=0.05,
                            min_value=0.01,
                            max_value=1.0,
                            step=0.01,
                            help="Epsilon parameter"
                        )
                    elif discover_decision_mode == "pathlen":
                        discover_path_length = st.number_input(
                            "Path Length:",
                            value=100.0,
                            min_value=10.0,
                            max_value=500.0,
                            step=10.0,
                            help="Path length for pathlen mode"
                        )
                        discover_linger_delta = st.number_input(
                            "Linger Delta:",
                            value=0.0,
                            min_value=0.0,
                            max_value=50.0,
                            step=1.0,
                            help="Linger distance beyond junction"
                        )
                    elif discover_decision_mode == "hybrid":
                        st.info("ℹ️ Using junction-specific r_outer values from the Junctions tab")
                        # r_outer will be overridden by junction-specific values
                        discover_hybrid_r_outer = 50.0
                        discover_hybrid_path_length = st.number_input(
                            "Hybrid Path Length:",
                            value=100.0,
                            min_value=10.0,
                            max_value=500.0,
                            step=10.0,
                            help="Path length for hybrid mode"
                        )
                        discover_hybrid_linger_delta = st.number_input(
                            "Hybrid Linger Delta:",
                            value=0.0,
                            min_value=0.0,
                            max_value=50.0,
                            step=1.0,
                            help="Linger distance beyond junction for hybrid mode"
                        )

                # Dynamic parameters based on cluster method (same as discover)
                st.markdown("##### Cluster Method Parameters")
                if cluster_method == "dbscan":
                    col_eps, col_min_samples = st.columns(2)
                    with col_eps:
                        dbscan_eps = st.number_input(
                            "DBSCAN Epsilon (eps):",
                            value=0.5,
                            min_value=0.1,
                            max_value=10.0,
                            step=0.1,
                            help="Maximum distance between samples in the same neighborhood"
                        )
                    with col_min_samples:
                        dbscan_min_samples = st.number_input(
                            "DBSCAN Min Samples:",
                            value=5,
                            min_value=1,
                            max_value=50,
                            step=1,
                            help="Minimum number of samples in a neighborhood"
                        )

                    # Add angle_eps parameter for DBSCAN
                    dbscan_angle_eps = st.number_input(
                        "DBSCAN Angle Epsilon (degrees):",
                        value=11.0,
                        min_value=1.0,
                        max_value=90.0,
                        step=1.0,
                        help="Angle epsilon for DBSCAN clustering (angular separation between clusters)"
                    )
                elif cluster_method == "kmeans":
                    col_k, col_k_range = st.columns(2)
                    with col_k:
                        kmeans_k = st.number_input(
                            "K-Means K (number of clusters):",
                            value=3,
                            min_value=2,
                            max_value=20,
                            step=1,
                            help="Number of clusters to form"
                        )
                    with col_k_range:
                        kmeans_k_min = st.number_input(
                            "K-Means K Min:",
                            value=2,
                            min_value=2,
                            max_value=10,
                            step=1,
                            help="Minimum number of clusters for auto selection"
                        )
                        kmeans_k_max = st.number_input(
                            "K-Means K Max:",
                            value=6,
                            min_value=3,
                            max_value=20,
                            step=1,
                            help="Maximum number of clusters for auto selection"
                        )
                elif cluster_method == "auto":
                    col_k_range, col_separation = st.columns(2)
                    with col_k_range:
                        auto_k_min = st.number_input(
                            "Auto K Min:",
                            value=2,
                            min_value=2,
                            max_value=10,
                            step=1,
                            help="Minimum number of clusters for auto selection"
                        )
                        auto_k_max = st.number_input(
                            "Auto K Max:",
                            value=6,
                            min_value=3,
                            max_value=20,
                            step=1,
                            help="Maximum number of clusters for auto selection"
                        )
                    with col_separation:
                        auto_min_sep_deg = st.number_input(
                            "Auto Min Separation (degrees):",
                            value=12.0,
                            min_value=1.0,
                            max_value=90.0,
                            step=1.0,
                            help="Minimum separation in degrees between clusters"
                        )
                        auto_angle_eps = st.number_input(
                            "Auto Angle Epsilon (degrees):",
                            value=11.0,
                            min_value=1.0,
                            max_value=90.0,
                            step=1.0,
                            help="Angle epsilon for auto clustering"
                        )

            elif analysis_type == "metrics":
                # Metrics-specific parameters
                st.markdown("#### Metrics Parameters")
                st.info("Metrics analysis computes timing and distance metrics for trajectories. The decision mode determines how junction timing is calculated - 'pathlen' measures time to reach a distance threshold, 'radial' measures time to exit a radius, and 'hybrid' tries radial first then falls back to pathlen.")

                col_metrics_mode, col_metrics_distance = st.columns(2)

                with col_metrics_mode:
                    st.session_state.metrics_decision_mode = st.selectbox(
                        "Decision Mode:",
                        ["pathlen", "radial", "hybrid"],
                        index=2,  # Default to hybrid
                        help="Decision mode for junction timing analysis"
                    )

                with col_metrics_distance:
                    st.session_state.metrics_distance = st.number_input(
                        "Distance Threshold:",
                        value=100.0,
                        min_value=10.0,
                        max_value=500.0,
                        step=10.0,
                        help="Path length for decision timing (pathlen mode) or outer radius (radial mode)"
                    )

                # Additional metrics parameters
                col_trend_window, col_min_outward = st.columns(2)

                with col_trend_window:
                    st.session_state.metrics_trend_window = st.number_input(
                        "Metrics Trend Window:",
                        value=5,
                        min_value=1,
                        max_value=20,
                        step=1,
                        help="Trend window for radial mode"
                    )

                st.session_state.metrics_min_outward = st.number_input(
                    "Metrics Min Outward:",
                    value=0.0,
                    min_value=0.0,
                    max_value=10.0,
                    step=0.1,
                    help="Minimum outward movement for radial mode"
                )

                # Show info about using junction-specific r_outer values
                if st.session_state.metrics_decision_mode in ["radial", "hybrid"]:
                    st.info("ℹ️ Using junction-specific r_outer values from the Junctions tab")

            elif analysis_type == "gaze":
                # Gaze analysis parameters
                st.markdown("#### Gaze & Physiological Analysis Parameters")

                # Prefer gaze trajectories if available
                active_trajs = st.session_state.trajectories

                # Check if we have proper gaze trajectory data
                has_gaze_data = self._check_for_gaze_data(active_trajs)

                # Show simple status message
                if active_trajs:
                    from verta.verta_data_loader import has_vr_headset_data
                    has_vr = any(has_vr_headset_data(traj) for traj in active_trajs)

                    if has_vr:
                        st.success("✅ This dataset contains VR headset data!")

                # Show analysis options
                if has_gaze_data:

                    # Check if we have existing branch assignments
                    has_existing_assignments = (st.session_state.analysis_results is not None and
                                             "branches" in st.session_state.analysis_results)

                    if has_existing_assignments:
                        st.success("✅ **Existing branch assignments found!**")
                        st.info("🔍 Gaze analysis will use existing branch assignments from previous discover analysis.")
                        st.write("**💡 To create new assignments:** Run '🔍 Discover Branches' analysis first, then return here for gaze analysis.")

                        # Always use existing assignments
                        st.session_state.use_existing_assignments = True
                        st.session_state.run_custom_discover = False

                    else:
                        st.warning("⚠️ **No existing branch assignments found!**")
                        st.error("**Prerequisite:** You must run '🔍 Discover Branches' analysis first to create branch assignments.")
                        st.write("**Steps:**")
                        st.write("1. Go to the '🔍 Discover Branches' analysis")
                        st.write("2. Run the discover analysis to create branch assignments")
                        st.write("3. Return here to run gaze analysis")

                        # Disable gaze analysis if no assignments
                        st.session_state.use_existing_assignments = False
                        st.session_state.run_custom_discover = False

                        # Show a disabled button
                        st.button("Run Analysis", disabled=True, help="Run '🔍 Discover Branches' analysis first to create branch assignments")
                        return

                    # Pupil Dilation Heatmap Settings
                    st.markdown("---")
                    st.markdown("#### 🗺️ Pupil Dilation Heatmap Settings")
                    st.info("Configure spatial heatmap visualization of pupil dilation patterns")

                    col_grid, col_norm = st.columns(2)

                    with col_grid:
                        # Initialize session state with default value if not set
                        if 'pupil_heatmap_cell_size' not in st.session_state:
                            st.session_state.pupil_heatmap_cell_size = 10.0

                        st.session_state.pupil_heatmap_cell_size = st.slider(
                            "Cell Size (coordinate units):",
                            min_value=1.0,
                            max_value=200.0,
                            value=float(st.session_state.pupil_heatmap_cell_size),
                            step=1.0,
                            help="Size of each grid cell in coordinate units (smaller = finer resolution)"
                        )

                    with col_norm:
                        # Initialize session state with default value if not set
                        if 'pupil_heatmap_normalization' not in st.session_state:
                            st.session_state.pupil_heatmap_normalization = 'relative'

                        st.session_state.pupil_heatmap_normalization = st.selectbox(
                            "Normalization:",
                            ["relative", "zscore"],
                            index=0 if st.session_state.pupil_heatmap_normalization == 'relative' else 1,
                            help="Relative: % change from baseline. Z-score: standard deviations from mean"
                        )

                    # Show expected grid dimensions
                    if st.session_state.trajectories:
                        # Calculate approximate grid dimensions
                        all_x = np.concatenate([t.x for t in st.session_state.trajectories[:10] if hasattr(t, 'x')])
                        all_z = np.concatenate([t.z for t in st.session_state.trajectories[:10] if hasattr(t, 'z')])
                        x_range = np.max(all_x) - np.min(all_x)
                        z_range = np.max(all_z) - np.min(all_z)
                        cell_size = st.session_state.pupil_heatmap_cell_size
                        grid_x = int(np.ceil(x_range / cell_size))
                        grid_z = int(np.ceil(z_range / cell_size))

                        st.info(f"📏 **Expected grid:** {grid_x} × {grid_z} cells ({cell_size}×{cell_size} units each)")

                    # Run Analysis button
                    if st.button("Run Analysis", type="primary"):
                        # Run gaze analysis
                        self.run_analysis("gaze", "hybrid", "dbscan", 42)

                else:
                    st.info("ℹ️ Gaze/Physiological analysis requires VR headset data with eye tracking.")

            elif analysis_type == "assign":
                st.markdown("#### Assign Parameters")

                # Add scaling warning
                st.info("💡 **Important**: Ensure the scale factor used for trajectories matches the scale factor used during discover analysis. Mismatched scaling will cause assignment failures.")

                # Scale factor input for assign analysis
                st.markdown("**📏 Scale Factor for Assignment:**")
                col_scale_assign, col_scale_info = st.columns([1, 2])

                with col_scale_assign:
                    assign_scale = st.number_input(
                        "Scale Factor:",
                        value=st.session_state.get("scale_factor", 0.2),
                        min_value=0.01,
                        max_value=1.0,
                        step=0.01,
                        key="assign_scale_factor",
                        help="Scale factor to apply to trajectory coordinates"
                    )

                with col_scale_info:
                    # Show scale factor from discover analysis if available
                    analysis_results = st.session_state.get("analysis_results")
                    if analysis_results and "branches" in analysis_results:
                        discover_scale = None
                        for junction_key, branch_data in analysis_results["branches"].items():
                            if "scale" in branch_data:
                                discover_scale = branch_data["scale"]
                                break

                        if discover_scale is not None:
                            st.info(f"🔍 **Discover used scale**: {discover_scale:.2f}")
                            if abs(assign_scale - discover_scale) > 0.01:
                                st.warning(f"⚠️ **Scale mismatch detected!** Consider using {discover_scale:.2f} for consistency.")
                        else:
                            st.info("🔍 No discover scale factor found")
                    else:
                        st.info("🔍 No discover analysis found")

                # Simplified data input options
                st.markdown("**📁 Data Input Options:**")

                # Trajectories input
                st.markdown("**Trajectories:**")
                trajectory_option = st.radio(
                    "Trajectory Source:",
                    ["Upload files", "Select folder"],
                    key="assign_trajectory_option",
                    help="Upload new trajectories to be assigned to existing branches"
                )

                # Centers input
                st.markdown("**Junction Centers:**")
                centers_option = st.radio(
                    "Centers Source:",
                    ["Use session centers", "Upload files", "Select folder"],
                    key="assign_centers_option",
                    help="Choose how to provide junction center data"
                )

                # File upload and folder selection based on options
                if trajectory_option == "Upload files":
                    st.markdown("**📤 Upload Trajectory Files:**")
                    trajectory_files = st.file_uploader(
                        "Choose trajectory CSV files:",
                        type=['csv'],
                        accept_multiple_files=True,
                        key="assign_trajectory_files",
                        help="Upload CSV files containing trajectory data to be assigned to existing branches"
                    )
                else:  # Select folder
                    st.markdown("**📁 Select Trajectory Folder:**")
                    trajectory_folder = st.text_input(
                        "Trajectory folder path:",
                        key="assign_trajectory_folder",
                        help="Enter the path to the folder containing trajectory CSV files to be assigned to existing branches"
                    )

                if centers_option == "Upload files":
                    st.markdown("**📤 Upload Center Files:**")
                    centers_files = st.file_uploader(
                        "Choose center files (.npy or .zip):",
                        type=['npy', 'zip'],
                        accept_multiple_files=True,
                        key="assign_centers_files",
                        help="Upload .npy files containing junction centers or .zip files with multiple centers"
                    )
                elif centers_option == "Select folder":
                    st.markdown("**📁 Select Centers Folder:**")
                    centers_folder = st.text_input(
                        "Centers folder path:",
                        key="assign_centers_folder",
                        help="Enter the path to the folder containing junction centers (will search subfolders for branch_centers_j*.npy files)"
                    )

                # Assignment parameters
                st.markdown("**⚙️ Assignment Parameters:**")
                # Decision mode selector (mirror discover logic with selectbox)
                # Default from discover if available
                default_mode = "hybrid"
                if centers_option == "Use session centers" and st.session_state.get("analysis_results") and "branches" in st.session_state.analysis_results:
                    # Try to fetch from first matching junction block
                    for _jk, _bd in st.session_state.analysis_results["branches"].items():
                        if isinstance(_bd, dict) and "decision_mode" in _bd:
                            default_mode = _bd.get("decision_mode", default_mode)
                            break

                col_decision_mode, col_decision_param = st.columns(2)
                with col_decision_mode:
                    assign_decision_mode = st.selectbox(
                        "Decision Mode:",
                        ["pathlen", "radial", "hybrid"],
                        index=["pathlen","radial","hybrid"].index(default_mode) if default_mode in ["pathlen","radial","hybrid"] else 2,
                        key="assign_decision_mode",
                        help="How to compute initial direction vectors for assignment"
                    )

                # Auto-rediscovery (always available when uploading new trajectories)
                st.markdown("**🧭 Auto-Discover New Branches (optional):**")
                auto_col1, auto_col2, auto_col3 = st.columns(3)
                with auto_col1:
                    st.checkbox(
                        "Enable auto-rediscover",
                        value=False,
                        key="assign_auto_rediscover",
                        help="If outlier assignments form a dense region of size ≥ min samples, rerun discovery for this junction using all trajectories (existing + newly uploaded)."
                    )
                with auto_col2:
                    st.number_input(
                        "Min samples for new branch",
                        value=5,
                        min_value=2,
                        max_value=100,
                        step=1,
                        key="assign_auto_min_samples",
                        help="Minimum outlier vectors required to trigger rediscovery."
                    )
                with auto_col3:
                    st.number_input(
                        "Angle eps (deg)",
                        value=11.0,
                        min_value=1.0,
                        max_value=90.0,
                        step=1.0,
                        key="assign_auto_angle_eps",
                        help="Angular neighborhood size for detecting a dense outlier region."
                    )

                # Decision mode parameters in second column (mirror discover UI)
                with col_decision_param:
                    # Fetch defaults from discover if using session centers
                    pref_path_length = 100.0
                    pref_linger = 0.0
                    pref_epsilon = 0.05
                    if centers_option == "Use session centers" and st.session_state.get("analysis_results") and "branches" in st.session_state.analysis_results:
                        for _jk, _bd in st.session_state.analysis_results["branches"].items():
                            if isinstance(_bd, dict):
                                if "path_length" in _bd:
                                    pref_path_length = float(_bd.get("path_length", pref_path_length))
                                if "linger_delta" in _bd:
                                    pref_linger = float(_bd.get("linger_delta", pref_linger))
                                if "epsilon" in _bd:
                                    pref_epsilon = float(_bd.get("epsilon", pref_epsilon))
                                break

                    if assign_decision_mode == "radial":
                        st.info("ℹ️ Using r_outer from junctions (or stored discover results)")
                        assign_r_outer = None  # Will be fetched per junction
                        assign_epsilon = st.number_input(
                            "Epsilon:",
                            value=pref_epsilon,
                            min_value=0.001,
                            max_value=1.0,
                            step=0.001,
                            format="%.3f",
                            key="assign_epsilon",
                            help="Minimum movement threshold"
                        )
                        assign_path_length = 100.0
                        assign_linger_delta = 0.0
                    elif assign_decision_mode == "pathlen":
                        assign_path_length = st.number_input(
                            "Path Length:",
                            value=pref_path_length,
                            min_value=10.0,
                            max_value=500.0,
                            step=10.0,
                            key="assign_path_length",
                            help="Path length for decision point"
                        )
                        assign_linger_delta = st.number_input(
                            "Linger Delta:",
                            value=pref_linger,
                            min_value=0.0,
                            max_value=200.0,
                            step=1.0,
                            key="assign_linger_delta",
                            help="Linger distance beyond junction"
                        )
                        assign_epsilon = st.number_input(
                            "Epsilon:",
                            value=pref_epsilon,
                            min_value=0.001,
                            max_value=1.0,
                            step=0.001,
                            format="%.3f",
                            key="assign_epsilon",
                            help="Minimum movement threshold"
                        )
                        assign_r_outer = None
                    elif assign_decision_mode == "hybrid":
                        st.info("ℹ️ Using r_outer from junctions (or stored discover results)")
                        assign_r_outer = None  # Will be fetched per junction
                        assign_path_length = st.number_input(
                            "Hybrid Path Length:",
                            value=pref_path_length,
                            min_value=10.0,
                            max_value=500.0,
                            step=10.0,
                            key="assign_path_length",
                            help="Path length for hybrid mode"
                        )
                        assign_linger_delta = st.number_input(
                            "Hybrid Linger Delta:",
                            value=pref_linger,
                            min_value=0.0,
                            max_value=200.0,
                            step=1.0,
                            key="assign_linger_delta",
                            help="Linger distance for hybrid mode"
                        )
                        assign_epsilon = st.number_input(
                            "Epsilon:",
                            value=pref_epsilon,
                            min_value=0.001,
                            max_value=1.0,
                            step=0.001,
                            format="%.3f",
                            key="assign_epsilon",
                            help="Minimum movement threshold"
                        )

                # Junction parameters for external data (only show if not using session centers)
                if centers_option != "Use session centers":
                    st.markdown("**🎯 Junction Parameters (for external data):**")
                    st.info("If using external trajectory data, you may need to specify junction parameters manually.")

                    col_junction_cx, col_junction_cz, col_junction_r = st.columns(3)

                    with col_junction_cx:
                        assign_junction_cx = st.number_input(
                            "Junction Center X:",
                            value=0.0,
                            key="assign_junction_cx",
                            help="X coordinate of junction center"
                        )

                    with col_junction_cz:
                        assign_junction_cz = st.number_input(
                            "Junction Center Z:",
                            value=0.0,
                            key="assign_junction_cz",
                            help="Z coordinate of junction center"
                        )

                    with col_junction_r:
                        assign_junction_r = st.number_input(
                            "Junction Radius:",
                            value=50.0,
                            min_value=1.0,
                            max_value=200.0,
                            step=1.0,
                            key="assign_junction_r",
                            help="Radius of the junction area"
                        )

                # Legacy code removed - using simplified interface above

        with col2:
            st.markdown("### Run Analysis")

            # Check if junctions are defined for analysis
            has_junctions = bool(st.session_state.junctions)

            if not has_junctions:
                st.warning("⚠️ **No junctions defined!** Please define junctions in the Junction Editor before running analysis.")
                st.info("💡 **Tip:** Go to the Junction Editor tab to define junctions for your analysis.")

            if st.button("🚀 Run Analysis", type="primary", disabled=not has_junctions):
                # Collect cluster method parameters
                cluster_params = {}
                if analysis_type == "discover" or analysis_type == "enhanced":
                    if cluster_method == "dbscan":
                        cluster_params = {"eps": dbscan_eps, "min_samples": dbscan_min_samples, "angle_eps": dbscan_angle_eps}
                    elif cluster_method == "kmeans":
                        cluster_params = {"k": kmeans_k, "k_min": kmeans_k_min, "k_max": kmeans_k_max}
                    elif cluster_method == "auto":
                        cluster_params = {"k_min": auto_k_min, "k_max": auto_k_max, "min_sep_deg": auto_min_sep_deg, "angle_eps": auto_angle_eps}

                # Collect decision mode parameters
                decision_params = {}
                if analysis_type == "predict":
                    if decision_mode == "radial":
                        decision_params = {"r_outer": radial_r_outer, "epsilon": radial_epsilon}
                    elif decision_mode == "pathlen":
                        decision_params = {"path_length": pathlen_path_length, "linger_delta": pathlen_linger_delta}
                    elif decision_mode == "hybrid":
                        decision_params = {"r_outer": hybrid_r_outer, "path_length": hybrid_path_length, "linger_delta": hybrid_linger_delta}
                elif analysis_type == "discover" or analysis_type == "enhanced":
                    if discover_decision_mode == "radial":
                        decision_params = {"r_outer": discover_r_outer, "epsilon": discover_epsilon}
                    elif discover_decision_mode == "pathlen":
                        decision_params = {"path_length": discover_path_length, "linger_delta": discover_linger_delta}
                    elif discover_decision_mode == "hybrid":
                        decision_params = {"r_outer": discover_hybrid_r_outer, "path_length": discover_hybrid_path_length, "linger_delta": discover_hybrid_linger_delta}

                # Add metrics-specific parameters if analysis type is metrics
                if analysis_type == "metrics":
                    decision_params.update({
                        "decision_mode": st.session_state.get("metrics_decision_mode", "pathlen"),
                        "distance": st.session_state.get("metrics_distance", 100.0),
                        "r_outer": st.session_state.get("metrics_r_outer", 50.0),
                        "trend_window": st.session_state.get("metrics_trend_window", 5),
                        "min_outward": st.session_state.get("metrics_min_outward", 0.0)
                    })

                # Collect assign parameters if needed
                assign_params = {}
                if analysis_type == "assign":
                    # Get assignment parameters
                    assign_path_length = st.session_state.get("assign_path_length", 100.0)
                    assign_epsilon = st.session_state.get("assign_epsilon", 0.05)
                    # Auto-rediscover controls
                    assign_auto_rediscover = st.session_state.get("assign_auto_rediscover", False)
                    assign_auto_min_samples = st.session_state.get("assign_auto_min_samples", 5)
                    assign_auto_angle_eps = st.session_state.get("assign_auto_angle_eps", 15.0)

                    # Get trajectory and centers options
                    trajectory_option = st.session_state.get("assign_trajectory_option", "Upload files")
                    centers_option = st.session_state.get("assign_centers_option", "Use session centers")

                    # Collect trajectory data
                    trajectory_files = None
                    trajectory_folder = None
                    if trajectory_option == "Upload files":
                        trajectory_files = st.session_state.get("assign_trajectory_files")
                    else:  # Select folder
                        trajectory_folder = st.session_state.get("assign_trajectory_folder")

                    # Collect centers data
                    centers_files = None
                    centers_folder = None
                    if centers_option == "Upload files":
                        centers_files = st.session_state.get("assign_centers_files")
                    elif centers_option == "Select folder":
                        centers_folder = st.session_state.get("assign_centers_folder")

                    assign_params = {
                        "path_length": assign_path_length,
                        "epsilon": assign_epsilon,
                        "assign_scale": assign_scale,  # Add assign-specific scale factor
                        "decision_mode": assign_decision_mode,
                        "r_outer": assign_r_outer,
                        "linger_delta": assign_linger_delta,
                        "auto_rediscover": assign_auto_rediscover,
                        "auto_min_samples": assign_auto_min_samples,
                        "auto_angle_eps": assign_auto_angle_eps,
                        "trajectory_option": trajectory_option,
                        "centers_option": centers_option,
                        "trajectory_files": trajectory_files,
                        "trajectory_folder": trajectory_folder,
                        "centers_files": centers_files,
                        "centers_folder": centers_folder,
                        "junction_cx": st.session_state.get("assign_junction_cx", 0.0),
                        "junction_cz": st.session_state.get("assign_junction_cz", 0.0),
                        "junction_r": st.session_state.get("assign_junction_r", 50.0)
                    }

                self.run_analysis(analysis_type, decision_mode, cluster_method, seed, cluster_params, decision_params, assign_params, discover_decision_mode)

            if st.session_state.analysis_results:
                st.markdown("### Analysis Results")
                st.success("✅ Analysis completed successfully!")


                # Show summary based on analysis type
                if analysis_type == "discover":
                    if "branches" in st.session_state.analysis_results:
                        st.write(f"**Branches discovered:** {len(st.session_state.analysis_results['branches'])}")

                elif analysis_type == "assign":
                    if "assignments" in st.session_state.analysis_results:
                        st.write(f"**Trajectories assigned:** {len(st.session_state.analysis_results['assignments'])}")

                        # Show debug information if available
                        if "assign_debug_info" in st.session_state and st.session_state.assign_debug_info:
                            st.markdown("### 🔍 Debug Information")
                            for junction_key, debug_info in st.session_state.assign_debug_info.items():
                                with st.expander(f"Debug Info for {junction_key}", expanded=False):
                                    st.write("**Junction Parameters:**")
                                    st.write(f"- Center: {debug_info['junction_params']['center']}")
                                    st.write(f"- Radius: {debug_info['junction_params']['radius']}")
                                    st.write(f"- R_outer: {debug_info['junction_params']['r_outer']}")

                                    st.write("**Assignment Parameters:**")
                                    st.write(f"- Path length: {debug_info['assignment_params']['path_length']}")
                                    st.write(f"- Epsilon: {debug_info['assignment_params']['epsilon']}")

                                    st.write("**Data Info:**")
                                    st.write(f"- Centers shape: {debug_info['data_info']['centers_shape']}")
                                    st.write(f"- Trajectories: {debug_info['data_info']['trajectories']}")

                                    st.write("**Assignment Distribution:**")
                                    total_trajectories = sum(debug_info['assignment_distribution'].values())
                                    for branch, count in debug_info['assignment_distribution'].items():
                                        percentage = (count / total_trajectories) * 100
                                        st.write(f"- Branch {branch}: {count} trajectories ({percentage:.1f}%)")

                                    # Add troubleshooting info if most trajectories are -2/-1
                                    neg2_count = debug_info['assignment_distribution'].get(-2, 0)
                                    neg1_count = debug_info['assignment_distribution'].get(-1, 0)

                                    if (neg2_count + neg1_count) / total_trajectories > 0.8:  # More than 80% are -2/-1
                                        st.warning("⚠️ **Troubleshooting:** Most trajectories are getting -2/-1 assignments!")
                                        st.write("**Possible solutions:**")
                                        st.write("1. **Increase junction radius** - Current radius might be too small")
                                        st.write("2. **Adjust junction center** - Center might not match trajectory paths")
                                        st.write("3. **Check trajectory data** - Ensure trajectories actually pass through junction area")
                                        st.write("4. **Use manual junction parameters** - Try different center coordinates and radius")

                                    st.write("**First 10 Assignments:**")
                                    if debug_info['assignments_sample']:
                                        st.dataframe(pd.DataFrame(debug_info['assignments_sample']), width='stretch')

                elif analysis_type == "metrics":
                    if "metrics" in st.session_state.analysis_results:
                        st.write(f"**Metrics computed:** {len(st.session_state.analysis_results['metrics'])}")

                        # Show debug information for metrics
                        st.markdown("---")
                        st.markdown("### 🔍 Debug Information")

                        # Get debug information from the first few trajectories
                        if st.session_state.trajectories:
                            st.write("**Debug Status:**")
                            st.write(f"- Total trajectories: {len(st.session_state.trajectories)}")

                            # Sample first 5 trajectories for debug
                            time_data_debug = []
                            for i, traj in enumerate(st.session_state.trajectories[:5]):
                                time_debug = {
                                    "trajectory_id": i,
                                    "time_data_type": str(type(traj.t)),
                                    "time_data_shape": traj.t.shape if traj.t is not None else None,
                                    "time_data_sample": traj.t[:3].tolist() if traj.t is not None and len(traj.t) > 0 else None,
                                    "time_data_dtype": str(traj.t.dtype) if traj.t is not None else None,
                                    "time_is_none": traj.t is None,
                                    "time_length": len(traj.t) if traj.t is not None else 0,
                                    # Add position data diagnostics
                                    "x_data_type": str(type(traj.x)),
                                    "x_data_shape": traj.x.shape if traj.x is not None else None,
                                    "x_data_sample": traj.x[:3].tolist() if traj.x is not None and len(traj.x) > 0 else None,
                                    "x_data_dtype": str(traj.x.dtype) if traj.x is not None else None,
                                    "x_is_none": traj.x is None,
                                    "x_length": len(traj.x) if traj.x is not None else 0,
                                    "z_data_type": str(type(traj.z)),
                                    "z_data_shape": traj.z.shape if traj.z is not None else None,
                                    "z_data_sample": traj.z[:3].tolist() if traj.z is not None and len(traj.z) > 0 else None,
                                    "z_data_dtype": str(traj.z.dtype) if traj.z is not None else None,
                                    "z_is_none": traj.z is None,
                                    "z_length": len(traj.z) if traj.z is not None else 0
                                }
                                time_data_debug.append(time_debug)

                            st.write(f"- time_data_debug length: {len(time_data_debug)}")

                            if time_data_debug:
                                with st.expander("🔍 Trajectory Data Debug Information", expanded=True):
                                    st.write("**First 5 trajectories data analysis:**")
                                    for debug_info in time_data_debug:
                                        st.write(f"**Trajectory {debug_info['trajectory_id']}:**")

                                        # Time data
                                        st.write("**Time Data:**")
                                        st.write(f"- Is None: {debug_info['time_is_none']}")
                                        st.write(f"- Length: {debug_info['time_length']}")
                                        st.write(f"- Type: {debug_info['time_data_type']}")
                                        st.write(f"- Shape: {debug_info['time_data_shape']}")
                                        st.write(f"- Dtype: {debug_info['time_data_dtype']}")
                                        st.write(f"- Sample: {debug_info['time_data_sample']}")

                                        # Position data
                                        st.write("**Position Data (X):**")
                                        st.write(f"- Is None: {debug_info['x_is_none']}")
                                        st.write(f"- Length: {debug_info['x_length']}")
                                        st.write(f"- Type: {debug_info['x_data_type']}")
                                        st.write(f"- Shape: {debug_info['x_data_shape']}")
                                        st.write(f"- Dtype: {debug_info['x_data_dtype']}")
                                        st.write(f"- Sample: {debug_info['x_data_sample']}")

                                        st.write("**Position Data (Z):**")
                                        st.write(f"- Is None: {debug_info['z_is_none']}")
                                        st.write(f"- Length: {debug_info['z_length']}")
                                        st.write(f"- Type: {debug_info['z_data_type']}")
                                        st.write(f"- Shape: {debug_info['z_data_shape']}")
                                        st.write(f"- Dtype: {debug_info['z_data_dtype']}")
                                        st.write(f"- Sample: {debug_info['z_data_sample']}")

                                        st.write("---")
                            else:
                                st.info("No time data debug information available")

                elif analysis_type == "gaze":
                    if "gaze_results" in st.session_state.analysis_results:
                        st.write(f"**Gaze analysis completed**")

                elif analysis_type == "predict":
                    if "choice_patterns" in st.session_state.analysis_results:
                        st.write(f"**Choice patterns analyzed**")

    def run_analysis(self, analysis_type: str, decision_mode: str, cluster_method: str, seed: int, cluster_params: dict = None, decision_params: dict = None, assign_params: dict = None, discover_decision_mode: str = "hybrid"):
        """Run the selected analysis"""
        try:
            with st.spinner(f"Running {analysis_type} analysis..."):

                if analysis_type == "discover":
                    # Unified multi-junction discovery for consistent decisions and assignments
                    import os
                    output_dir = "gui_outputs"
                    os.makedirs(output_dir, exist_ok=True)

                    # Cluster/decision parameters
                    k_value = cluster_params.get("k", 3) if cluster_params else 3
                    min_samples = cluster_params.get("min_samples", 5) if cluster_params else 5
                    k_min = cluster_params.get("k_min", 2) if cluster_params else 2
                    k_max = cluster_params.get("k_max", 6) if cluster_params else 6
                    min_sep_deg = cluster_params.get("min_sep_deg", 12.0) if cluster_params else 12.0
                    angle_eps = cluster_params.get("angle_eps", 15.0) if cluster_params else 15.0

                    path_length = decision_params.get("path_length", 100.0) if decision_params else 100.0
                    epsilon = decision_params.get("epsilon", 0.05) if decision_params else 0.05
                    linger_delta = decision_params.get("linger_delta", 5.0) if decision_params else 5.0
                    r_outer_list = [st.session_state.junction_r_outer.get(i, 50.0) for i in range(len(st.session_state.junctions))]

                    # Run consolidated discovery
                    chain_df, centers_list, decisions_chain_df = discover_decision_chain(
                        trajectories=st.session_state.trajectories,
                        junctions=st.session_state.junctions,
                        path_length=path_length,
                        epsilon=epsilon,
                        seed=seed,
                        decision_mode=discover_decision_mode,
                        r_outer_list=r_outer_list,
                        linger_delta=linger_delta,
                        out_dir=output_dir,
                        cluster_method=cluster_method,
                        k=k_value,
                        k_min=k_min,
                        k_max=k_max,
                        min_sep_deg=min_sep_deg,
                        angle_eps=angle_eps,
                        min_samples=min_samples,
                    )

                    # Build per-junction results view from chain_df/centers_list
                    results = {}
                    for i, junction in enumerate(st.session_state.junctions):
                        junction_key = f"junction_{i}"
                        col = f"branch_j{i}"
                        if col in chain_df.columns:
                            df_i = chain_df[["trajectory", col]].copy()
                            df_i = df_i.rename(columns={col: "branch"})
                            # summary counts for main branches (>=0)
                            vc = df_i[df_i["branch"] >= 0]["branch"].value_counts().sort_index()
                            summary_i = vc.rename_axis("branch").to_frame("count").reset_index()
                            total_i = int(summary_i["count"].sum()) if len(summary_i) else 0
                            summary_i["percent"] = summary_i["count"] / max(1, total_i) * 100.0
                        else:
                            import pandas as _pd
                            df_i = _pd.DataFrame(columns=["trajectory","branch"])  # empty
                            summary_i = _pd.DataFrame(columns=["branch","count","percent"])  # empty

                        results[junction_key] = {
                            "assignments": df_i,
                            "summary": summary_i,
                            "centers": centers_list[i] if i < len(centers_list) else None,
                            "junction": junction,
                            "r_outer": r_outer_list[i] if i < len(r_outer_list) else None,
                            "path_length": path_length,
                            "epsilon": epsilon,
                            "linger_delta": linger_delta,  # Store linger_delta for gaze analysis
                            "decision_mode": discover_decision_mode,
                            "scale": st.session_state.get("scale_factor", 1.0),
                        }

                    # Flow graph generation removed - discover should only do discovery, not flow analysis

                    # Persist results and decisions for gaze reuse
                    if st.session_state.analysis_results is None:
                        st.session_state.analysis_results = {}
                    st.session_state.analysis_results["branches"] = results

                    # Debug: Check chain_decisions DataFrame
                    st.write(f"🔍 **Chain Decisions Debug:**")
                    st.write(f"- decisions_chain_df is not None: {decisions_chain_df is not None}")
                    if decisions_chain_df is not None:
                        st.write(f"- decisions_chain_df length: {len(decisions_chain_df)}")
                        st.write(f"- decisions_chain_df columns: {list(decisions_chain_df.columns)}")
                        if not decisions_chain_df.empty:
                            st.write(f"- Junction indices in decisions_chain_df: {sorted(decisions_chain_df['junction_index'].unique())}")
                        else:
                            st.write("- decisions_chain_df is empty!")
                    else:
                        st.write("- decisions_chain_df is None!")

                    # Store branch assignments (chain_df) as chain_decisions for gaze analysis
                    if chain_df is not None and len(chain_df) > 0:
                        st.session_state.analysis_results.setdefault("branches", {})
                        st.session_state.analysis_results["branches"]["chain_decisions"] = chain_df
                        st.write(f"✅ **Stored branch assignments (chain_df) with {len(chain_df)} rows in session state**")
                        st.write(f"🔍 **Branch assignment columns:** {list(chain_df.columns)}")

                        # Debug: Check for branch_jX columns specifically
                        branch_cols = [col for col in chain_df.columns if col.startswith('branch_j')]
                        st.write(f"🔍 **Branch columns found:** {branch_cols}")
                        if len(branch_cols) > 0:
                            st.write(f"🔍 **Sample branch data:** {chain_df[branch_cols].head()}")
                        else:
                            st.error("❌ **No branch_jX columns found in chain_df!**")
                    else:
                        st.write(f"❌ **Not storing branch assignments - chain_df is None or empty**")

                    # Also store decision points separately for reference
                    if decisions_chain_df is not None and len(decisions_chain_df) > 0:
                        st.session_state.analysis_results.setdefault("branches", {})
                        st.session_state.analysis_results["branches"]["decision_points"] = decisions_chain_df
                        st.write(f"✅ **Stored decision points with {len(decisions_chain_df)} rows in session state**")
                    else:
                        st.write(f"❌ **Not storing decision points - DataFrame is None or empty**")

                    # Add debugging information for flow analysis
                    try:
                        st.markdown("#### 🔍 Flow Analysis Debug")

                        # Count trajectories that visit multiple junctions
                        multi_junction_trajectories = 0
                        junction_visit_counts = {}

                        for i, junction in enumerate(st.session_state.junctions):
                            junction_key = f"junction_{i}"
                            if junction_key in results and "assignments" in results[junction_key]:
                                assignments = results[junction_key]["assignments"]
                                if not assignments.empty:
                                    visited_trajectories = set(assignments["trajectory"].unique())
                                    junction_visit_counts[i] = visited_trajectories

                        # Find trajectories that visit multiple junctions
                        all_trajectories = set()
                        for trajectories in junction_visit_counts.values():
                            all_trajectories.update(trajectories)

                        for traj_id in all_trajectories:
                            visited_junctions = [i for i, trajs in junction_visit_counts.items() if traj_id in trajs]
                            if len(visited_junctions) > 1:
                                multi_junction_trajectories += 1

                        st.info(f"📊 **Flow Analysis Summary:**")
                        st.write(f"- Total trajectories: {len(st.session_state.trajectories)}")
                        st.write(f"- Trajectories visiting multiple junctions: {multi_junction_trajectories}")
                        st.write(f"- Junction visit counts: {[len(trajs) for trajs in junction_visit_counts.values()]}")

                        if multi_junction_trajectories == 0:
                            st.warning("⚠️ **No trajectories visit multiple junctions!** This explains the zero flow matrix.")
                            st.write("**Possible causes:**")
                            st.write("1. Trajectories are too short to reach multiple junctions")
                            st.write("2. Junction r_outer values are too small")
                            st.write("3. Junctions are too far apart")
                            st.write("4. Trajectory data needs different scaling")

                    except Exception as e:
                        st.warning(f"Debug analysis failed: {str(e)}")

                    #st.success(f"✅ Discover analysis completed successfully for all {len(st.session_state.junctions)} junctions!")
                    self.generate_cli_command("discover", results, cluster_method, cluster_params, decision_mode, decision_params)

                elif analysis_type == "assign":
                    # Assign trajectories to branches using simplified interface
                    import numpy as np

                    # Load trajectories based on trajectory option
                    trajectories = self.load_assign_trajectories(assign_params)
                    if trajectories is None:
                        return

                    # Load centers based on centers option
                    centers_dict = self.load_assign_centers(assign_params)
                    if centers_dict is None:
                        return

                    # Validate that we have compatible data
                    if not trajectories:
                        st.error("❌ No trajectories loaded for assignment")
                        return

                    if not centers_dict:
                        st.error("❌ No junction centers loaded for assignment")
                        return

                    # Get assignment parameters
                    path_length = assign_params.get("path_length", 100.0) if assign_params else 100.0
                    epsilon = assign_params.get("epsilon", 0.05) if assign_params else 0.05

                    results = {}
                    successful_assignments = 0

                    # Process each junction
                    for junction_key, centers in centers_dict.items():
                        try:
                            # Extract junction number from key (e.g., "junction_0" -> 0)
                            junction_num = int(junction_key.split('_')[1])

                            # Get assignment parameters - use stored values from discover analysis if available
                            centers_option = assign_params.get("centers_option", "Use session centers") if assign_params else "Use session centers"
                            if centers_option == "Use session centers":
                                if junction_key in st.session_state.analysis_results["branches"]:
                                    branch_data = st.session_state.analysis_results["branches"][junction_key]
                                    # Use stored parameters from discover analysis
                                    stored_path_length = branch_data.get("path_length", path_length)
                                    stored_epsilon = branch_data.get("epsilon", epsilon)
                                    stored_scale = branch_data.get("scale", 1.0)
                                    st.info(f"📊 Using assignment parameters from discover analysis: path_length={stored_path_length:.1f}, epsilon={stored_epsilon:.3f}, scale={stored_scale:.1f}")
                                else:
                                    stored_path_length = path_length
                                    stored_epsilon = epsilon
                                    stored_scale = 1.0
                            else:
                                stored_path_length = path_length
                                stored_epsilon = epsilon
                                stored_scale = 1.0

                            # Get junction and r_outer - prioritize stored parameters from discover analysis
                            if centers_option == "Use session centers":
                                # Try to get junction parameters from discover analysis first
                                if junction_key in st.session_state.analysis_results["branches"]:
                                    branch_data = st.session_state.analysis_results["branches"][junction_key]
                                    if "junction" in branch_data and "r_outer" in branch_data:
                                        junction = branch_data["junction"]
                                        r_outer = branch_data["r_outer"]
                                        stored_scale = branch_data.get("scale", 1.0)
                                        st.info(f"📊 Using junction parameters from discover analysis: center=({junction.cx:.1f}, {junction.cz:.1f}), radius={junction.r:.1f}, r_outer={r_outer:.1f}")
                                        st.info(f"📊 Scale factor from discover analysis: {stored_scale:.1f}")

                                        # Check if current trajectories use different scale factor
                                        if hasattr(st.session_state, 'trajectories') and st.session_state.trajectories:
                                            # Estimate scale factor from trajectory coordinates
                                            sample_traj = st.session_state.trajectories[0]
                                            if hasattr(sample_traj, 'x') and len(sample_traj.x) > 0:
                                                # Simple heuristic: if coordinates are much larger than expected, scale might be different
                                                max_coord = max(abs(sample_traj.x.max()), abs(sample_traj.z.max()))
                                                if max_coord > 1000 and stored_scale < 0.5:
                                                    st.warning(f"⚠️ Scale factor mismatch detected! Discover used {stored_scale:.1f}, but current trajectories appear to use a different scale.")
                                                    st.warning(f"⚠️ This may cause assignment failures. Consider using the same scale factor as discover analysis.")
                                    else:
                                        # Fallback to session state junctions
                                        if junction_num < len(st.session_state.junctions):
                                            junction = st.session_state.junctions[junction_num]
                                            r_outer = st.session_state.junction_r_outer.get(junction_num, 50.0)
                                        else:
                                            st.error(f"❌ No junction parameters found for {junction_key}")
                                            continue
                                else:
                                    st.error(f"❌ No discover analysis data found for {junction_key}")
                                    continue
                            elif junction_num < len(st.session_state.junctions):
                                junction = st.session_state.junctions[junction_num]
                                r_outer = st.session_state.junction_r_outer.get(junction_num, 50.0)
                            else:
                                # If using external data, use manual parameters or estimate from trajectory data
                                manual_cx = assign_params.get("junction_cx", 0.0)
                                manual_cz = assign_params.get("junction_cz", 0.0)
                                manual_r = assign_params.get("junction_r", 50.0)

                                if manual_cx != 0.0 or manual_cz != 0.0 or manual_r != 50.0:
                                    # Use manual parameters
                                    junction = Circle(cx=manual_cx, cz=manual_cz, r=manual_r)
                                    r_outer = manual_r * 2.0
                                    st.info(f"📊 Using manual junction: center=({manual_cx:.1f}, {manual_cz:.1f}), radius={manual_r:.1f}")
                                else:
                                    # Estimate junction from trajectory data
                                    st.warning(f"⚠️ No junction defined for {junction_key}. Attempting to estimate from trajectory data...")

                                    # Estimate junction center from trajectory data
                                    all_x = np.concatenate([tr.x for tr in trajectories])
                                    all_z = np.concatenate([tr.z for tr in trajectories])

                                    # Show trajectory data range for debugging
                                    st.info(f"📊 Trajectory data range:")
                                    st.write(f"- X range: {np.min(all_x):.1f} to {np.max(all_x):.1f}")
                                    st.write(f"- Z range: {np.min(all_z):.1f} to {np.max(all_z):.1f}")

                                    # Use median as center (more robust than mean)
                                    estimated_cx = float(np.median(all_x))
                                    estimated_cz = float(np.median(all_z))

                                    # Estimate radius based on data spread - use a more conservative approach
                                    distances = np.sqrt((all_x - estimated_cx)**2 + (all_z - estimated_cz)**2)
                                    estimated_r = float(np.percentile(distances, 75))  # Use 75th percentile for radius

                                    junction = Circle(cx=estimated_cx, cz=estimated_cz, r=max(estimated_r, 20.0))
                                    r_outer = estimated_r * 3.0  # Make r_outer much larger than junction radius

                                    st.info(f"📊 Estimated junction: center=({estimated_cx:.1f}, {estimated_cz:.1f}), radius={estimated_r:.1f}")
                                    st.warning(f"⚠️ Using estimated junction for {junction_key}. Consider defining junctions manually for better results.")
                                    st.info(f"💡 Tip: If trajectories still get -2/-1, try increasing the junction radius or adjusting the center coordinates.")

                            # Create output directory for this junction
                            import os
                            out_dir = os.path.join("gui_outputs", f"junction_{junction_num}")
                            os.makedirs(out_dir, exist_ok=True)

                            # Run assignment
                            # Determine decision parameters to use
                            dm = assign_params.get("decision_mode", "pathlen")
                            ld = assign_params.get("linger_delta", 0.0)
                            dm_r_outer = assign_params.get("r_outer", r_outer)

                            assignments = assign_branches(
                                trajectories=trajectories,
                                centers=centers,
                                junction=junction,
                                path_length=stored_path_length,
                                epsilon=stored_epsilon,
                                decision_mode=dm,
                                r_outer=dm_r_outer,
                                linger_delta=ld,
                                out_dir=out_dir
                            )

                            # Optional: auto-rediscover if outlier cluster among new assignments is large enough
                            try:
                                auto_flag = st.session_state.get("assign_auto_rediscover", False)
                                min_samples_new = int(st.session_state.get("assign_auto_min_samples", 5))
                                angle_eps_new = float(st.session_state.get("assign_auto_angle_eps", 15.0))
                                # Auto-rediscover: detect dense outlier regions among newly uploaded trajectories
                                if auto_flag and len(assignments) > 0:
                                    # Identify outliers (-1) that entered junction and have usable vectors
                                    from verta.verta_decisions import compute_assignment_vectors
                                    from verta.verta_clustering import cluster_angles_dbscan
                                    # Compute vectors for these trajectories with same decision params
                                    vec_df = compute_assignment_vectors(
                                        trajectories=trajectories,
                                        junction=junction,
                                        path_length=stored_path_length,
                                        decision_mode=dm,
                                        r_outer=dm_r_outer,
                                        epsilon=stored_epsilon,
                                    )
                                    # Merge to filter to current outliers only
                                    outlier_ids = set(assignments[assignments["branch"] == -1]["trajectory"].tolist())
                                    if outlier_ids:
                                        use = vec_df[(vec_df["trajectory"].isin(outlier_ids)) & (vec_df["entered"]) & (vec_df["usable"])].copy()
                                        if len(use) >= min_samples_new:
                                            V = np.vstack([use[["vx","vz"]].to_numpy()]) if len(use) else np.zeros((0,2))
                                            if V.size:
                                                labels_o, centers_o = cluster_angles_dbscan(V, eps_deg=angle_eps_new, min_samples=min_samples_new)
                                                # If any valid cluster exists (label >=0), trigger rediscovery using all available trajectories
                                                if (labels_o >= 0).any():
                                                    st.info(f"🔄 Auto-rediscover triggered for {junction_key}: detected dense outlier region (min_samples={min_samples_new}, angle_eps={angle_eps_new}°)")
                                                    # Build combined trajectory set: existing session trajectories + newly provided
                                                    all_trajs = []
                                                    if hasattr(st.session_state, 'trajectories') and st.session_state.trajectories:
                                                        all_trajs.extend(st.session_state.trajectories)
                                                    all_trajs.extend([t for t in trajectories if t not in all_trajs])
                                                    # Rerun discovery for this junction
                                                    from verta.verta_decisions import discover_branches
                                                    new_assign, _sum, new_centers = discover_branches(
                                                        trajectories=all_trajs,
                                                        junction=junction,
                                                        k=centers.shape[0] if centers is not None and centers.size else 3,
                                                        path_length=stored_path_length,
                                                        epsilon=stored_epsilon,
                                                        seed=seed,
                                                        decision_mode=dm,
                                                        r_outer=dm_r_outer,
                                                        out_dir=out_dir,
                                                        cluster_method=cluster_method,
                                                        k_min=st.session_state.get("discover_k_min", 2),
                                                        k_max=st.session_state.get("discover_k_max", 6),
                                                        min_sep_deg=st.session_state.get("discover_min_sep_deg", 12.0),
                                                        angle_eps=st.session_state.get("discover_angle_eps", 15.0),
                                                        min_samples=min_samples_new,
                                                        junction_number=junction_num,
                                                        all_junctions=[junction]
                                                    )
                                                    # Update centers to the rediscovered ones and reassign current trajectories for display
                                                    centers = new_centers
                                                    assignments = assign_branches(
                                                        trajectories=trajectories,
                                                        centers=centers,
                                                        junction=junction,
                                                        path_length=stored_path_length,
                                                        epsilon=stored_epsilon,
                                                        decision_mode=dm,
                                                        r_outer=dm_r_outer,
                                                        linger_delta=ld,
                                                        out_dir=out_dir
                                                    )
                                                    st.warning("⚠️ Branch IDs may have been renumbered due to rediscovery.")
                            except Exception as _e:
                                # Keep assignment results even if auto-rediscover path fails
                                pass

                            # Enhanced debugging for assignment issues
                            if assignments is not None and len(assignments) > 0:
                                # Check if assignments is a string (error message) or pandas DataFrame
                                if isinstance(assignments, str):
                                    st.error(f"🚨 **Assignment Error:** {assignments}")
                                    st.error("This indicates the assign_branches function returned an error message instead of assignment results.")
                                    st.error("Check the assign_branches function implementation or input parameters.")
                                elif hasattr(assignments, 'iterrows'):  # pandas DataFrame
                                    # Count assignment types from DataFrame
                                    assignment_counts = assignments['branch'].value_counts().to_dict()

                                    total_trajectories = len(assignments)
                                    neg2_count = assignment_counts.get(-2, 0)
                                    neg1_count = assignment_counts.get(-1, 0)

                                    # If most trajectories are -2/-1, provide enhanced debugging
                                    if (neg2_count + neg1_count) / total_trajectories > 0.8:
                                        st.error(f"🚨 **Assignment Issue Detected for {junction_key}:**")
                                        st.error(f"   -2 (never entered): {neg2_count} trajectories ({neg2_count/total_trajectories*100:.1f}%)")
                                        st.error(f"   -1 (no usable vector): {neg1_count} trajectories ({neg1_count/total_trajectories*100:.1f}%)")

                                        # Enhanced debugging analysis
                                        st.info(f"🔍 **Enhanced Debug Analysis:**")

                                        # Show trajectory data ranges
                                        all_x = []
                                        all_z = []
                                        for traj in trajectories:
                                            all_x.extend(traj.x)
                                            all_z.extend(traj.z)

                                        if all_x and all_z:
                                            st.info(f"📊 **Trajectory Data Ranges:**")
                                            st.info(f"   X: {min(all_x):.1f} to {max(all_x):.1f} (range: {max(all_x)-min(all_x):.1f})")
                                            st.info(f"   Z: {min(all_z):.1f} to {max(all_z):.1f} (range: {max(all_z)-min(all_z):.1f})")
                                            st.info(f"   Total points: {len(all_x)}")

                                            # Show how many trajectories actually pass through the junction area
                                            trajectories_in_junction = 0
                                            trajectories_with_usable_vectors = 0

                                            for traj in trajectories:
                                                # Check if trajectory passes through junction area
                                                distances = np.sqrt((traj.x - junction.cx)**2 + (traj.z - junction.cz)**2)
                                                if np.any(distances <= junction.r):
                                                    trajectories_in_junction += 1

                                                    # Check if trajectory has usable vectors (length > epsilon)
                                                    if len(traj.x) > 1:
                                                        dx = np.diff(traj.x)
                                                        dz = np.diff(traj.z)
                                                        movement = np.sqrt(dx**2 + dz**2)
                                                        if np.any(movement > stored_epsilon):
                                                            trajectories_with_usable_vectors += 1

                                            st.info(f"📊 **Junction Analysis:**")
                                            st.info(f"   Trajectories passing through junction: {trajectories_in_junction}/{len(trajectories)}")
                                            st.info(f"   Trajectories with usable vectors: {trajectories_with_usable_vectors}/{len(trajectories)}")

                                            # Analyze movement patterns for -1 assignments
                                            if neg1_count > neg2_count:  # More -1 than -2 assignments
                                                st.error(f"🚨 **Critical Issue: Most trajectories are -1 (entered junction but no usable vectors)!**")

                                                # Analyze movement patterns
                                                st.info(f"🔍 **Movement Analysis:**")
                                                all_movements = []
                                                nan_trajectories = 0
                                                for traj in trajectories:
                                                    if len(traj.x) > 1:
                                                        # Check for NaN values
                                                        if np.any(np.isnan(traj.x)) or np.any(np.isnan(traj.z)):
                                                            nan_trajectories += 1
                                                            continue

                                                        dx = np.diff(traj.x)
                                                        dz = np.diff(traj.z)
                                                        movement = np.sqrt(dx**2 + dz**2)
                                                        # Filter out NaN movements
                                                        valid_movements = movement[~np.isnan(movement)]
                                                        all_movements.extend(valid_movements)

                                                if nan_trajectories > 0:
                                                    st.error(f"🚨 **CRITICAL: {nan_trajectories} trajectories contain NaN coordinates!**")
                                                    st.error("This will cause assignment failures. Check your trajectory data for missing/invalid coordinates.")

                                                if all_movements:
                                                    percentile_5 = np.percentile(all_movements, 5)
                                                    percentile_10 = np.percentile(all_movements, 10)
                                                    percentile_25 = np.percentile(all_movements, 25)
                                                    mean_movement = np.mean(all_movements)

                                                    st.info(f"📊 **Movement Statistics:**")
                                                    st.info(f"   Mean movement: {mean_movement:.4f}")
                                                    st.info(f"   5th percentile: {percentile_5:.4f}")
                                                    st.info(f"   10th percentile: {percentile_10:.4f}")
                                                    st.info(f"   25th percentile: {percentile_25:.4f}")
                                                    st.info(f"   Current epsilon: {stored_epsilon:.3f}")

                                                    # Suggest epsilon adjustment
                                                    if stored_epsilon > percentile_25:
                                                        suggested_epsilon = percentile_10
                                                        st.warning(f"⚠️ **Epsilon too high!** Try: {suggested_epsilon:.4f} (current: {stored_epsilon:.3f})")
                                                    elif stored_epsilon < percentile_5:
                                                        suggested_epsilon = percentile_10
                                                        st.warning(f"⚠️ **Epsilon too low!** Try: {suggested_epsilon:.4f} (current: {stored_epsilon:.3f})")
                                                    else:
                                                        suggested_epsilon = percentile_5
                                                        st.warning(f"⚠️ **Try lower epsilon:** {suggested_epsilon:.4f} (current: {stored_epsilon:.3f})")
                                                else:
                                                    st.error(f"🚨 **NO VALID MOVEMENTS FOUND!**")
                                                    st.error("All trajectories have NaN coordinates or invalid movement data.")
                                                    st.error("This explains why all trajectories get -1 assignments.")
                                                    st.error("**SOLUTION**: Check your trajectory data for missing/invalid coordinates.")

                                                    # Show sample trajectory analysis
                                                    st.info(f"🔍 **Sample Trajectory Analysis (first 3):**")
                                                    for i, traj in enumerate(trajectories[:3]):
                                                        if len(traj.x) > 1:
                                                            # Check for NaN values
                                                            has_nan = np.any(np.isnan(traj.x)) or np.any(np.isnan(traj.z))
                                                            if has_nan:
                                                                st.error(f"   Trajectory {i}: ⚠️ CONTAINS NaN COORDINATES!")
                                                                continue

                                                            dx = np.diff(traj.x)
                                                            dz = np.diff(traj.z)
                                                            movement = np.sqrt(dx**2 + dz**2)
                                                            max_movement = np.max(movement) if len(movement) > 0 else 0
                                                            mean_movement = np.mean(movement) if len(movement) > 0 else 0

                                                            # Check if trajectory passes through junction
                                                            distances = np.sqrt((traj.x - junction.cx)**2 + (traj.z - junction.cz)**2)
                                                            in_junction = np.any(distances <= junction.r)
                                                            min_distance = np.min(distances)

                                                            st.info(f"   Trajectory {i}: max_movement={max_movement:.3f}, mean_movement={mean_movement:.3f}, in_junction={in_junction}, min_distance={min_distance:.1f}")

                                            # Suggest junction radius adjustment
                                            if trajectories_in_junction < len(trajectories) * 0.5:
                                                st.warning(f"⚠️ **Low junction coverage!** Only {trajectories_in_junction}/{len(trajectories)} trajectories pass through the junction area.")

                                                # Calculate suggested radius to cover more trajectories
                                                all_distances = []
                                                for traj in trajectories:
                                                    distances = np.sqrt((traj.x - junction.cx)**2 + (traj.z - junction.cz)**2)
                                                    all_distances.extend(distances)

                                                suggested_radius = np.percentile(all_distances, 80)  # Cover 80% of trajectory points
                                                st.warning(f"⚠️ **Suggested radius:** {suggested_radius:.1f} (current: {junction.r:.1f})")
                                                st.warning(f"⚠️ **Suggested center:** ({junction.cx:.1f}, {junction.cz:.1f}) - verify this matches your junction location")
                                else:
                                    st.warning(f"⚠️ Unexpected assignment format: {type(assignments)} - {assignments}")
                                    st.warning("Expected pandas DataFrame or string, but got something else.")

                            results[junction_key] = {
                                "assignments": assignments,
                                "centers": centers,
                                "junction": junction,
                                "path_length": path_length,
                                "epsilon": epsilon
                            }

                            successful_assignments += 1
                            st.success(f"✅ Completed assignment for {junction_key} ({len(assignments)} trajectories)")

                            # Store debug information in session state
                            branch_counts = assignments['branch'].value_counts().sort_index()
                            debug_info = {
                                "junction_params": {
                                    "center": f"({junction.cx:.1f}, {junction.cz:.1f})",
                                    "radius": f"{junction.r:.1f}",
                                    "r_outer": f"{r_outer:.1f}"
                                },
                                "assignment_params": {
                                    "path_length": f"{path_length:.1f}",
                                    "epsilon": f"{epsilon:.3f}"
                                },
                                "data_info": {
                                    "centers_shape": str(centers.shape),
                                    "trajectories": len(trajectories)
                                },
                                "assignment_distribution": dict(branch_counts),
                                "assignments_sample": assignments.head(10).to_dict('records')
                            }

                            # Store in session state
                            if "assign_debug_info" not in st.session_state:
                                st.session_state.assign_debug_info = {}
                            st.session_state.assign_debug_info[junction_key] = debug_info

                        except Exception as e:
                            st.error(f"❌ Assignment failed for {junction_key}: {str(e)}")
                            continue

                    # Store results - preserve existing analysis results
                    if st.session_state.analysis_results is None:
                        st.session_state.analysis_results = {}
                    st.session_state.analysis_results["assignments"] = results

                    # Show summary
                    total_junctions = len(centers_dict)
                    if successful_assignments == total_junctions:
                        st.success(f"✅ Assign analysis completed successfully for all {total_junctions} junctions!")
                    else:
                        st.warning(f"⚠️ Assign analysis completed for {successful_assignments}/{total_junctions} junctions")

                    # Show assignment statistics
                    if results:
                        st.markdown("### Assignment Summary")
                        total_trajectories = len(trajectories)
                        st.write(f"**Total trajectories processed:** {total_trajectories}")
                        st.write(f"**Junctions processed:** {successful_assignments}")

                        # Show branch distribution for first junction as example
                        first_junction = list(results.keys())[0]
                        assignments_df = results[first_junction]["assignments"]
                        branch_counts = assignments_df["branch"].value_counts().sort_index()

                        st.markdown(f"**Branch distribution for {first_junction}:**")
                        for branch, count in branch_counts.items():
                            percentage = (count / len(assignments_df)) * 100
                            st.write(f"- Branch {branch}: {count} trajectories ({percentage:.1f}%)")

                    # Generate CLI command for easy copying
                    self.generate_cli_command("assign", results, cluster_method, cluster_params, decision_mode, decision_params)

                elif analysis_type == "metrics":
                    # Compute timing metrics
                    metrics = []
                    trajectories_with_time_data = 0
                    trajectories_without_time_data = 0
                    time_data_debug = []

                    # Get junctions from session state or estimate from data (ONCE, before the loop)
                    junctions_to_use = st.session_state.junctions if st.session_state.junctions else []

                    # If no junctions defined, estimate from trajectory data
                    if not junctions_to_use:
                        st.info("📊 No junctions defined. Estimating junctions from trajectory data...")

                        # Estimate junction from trajectory data (similar to assign analysis)
                        all_x = np.concatenate([tr.x for tr in st.session_state.trajectories])
                        all_z = np.concatenate([tr.z for tr in st.session_state.trajectories])

                        # Use median as center (more robust than mean)
                        estimated_cx = float(np.median(all_x))
                        estimated_cz = float(np.median(all_z))

                        # Estimate radius based on data spread
                        distances = np.sqrt((all_x - estimated_cx)**2 + (all_z - estimated_cz)**2)
                        estimated_r = float(np.percentile(distances, 75))  # Use 75th percentile for radius

                        # Create estimated junction
                        estimated_junction = Circle(cx=estimated_cx, cz=estimated_cz, r=max(estimated_r, 20.0))
                        junctions_to_use = [estimated_junction]

                        st.info(f"📊 Estimated junction: center=({estimated_cx:.1f}, {estimated_cz:.1f}), radius={estimated_r:.1f}")

                        # Debug: Show trajectory data range
                        st.info(f"📊 Trajectory data range:")
                        st.write(f"- X range: {np.min(all_x):.1f} to {np.max(all_x):.1f}")
                        st.write(f"- Z range: {np.min(all_z):.1f} to {np.max(all_z):.1f}")
                        st.write(f"- Total trajectories: {len(st.session_state.trajectories)}")
                        st.write(f"- Total coordinate points: {len(all_x)}")

                    # Debug: Count how many trajectories pass through the estimated junction
                    if junctions_to_use:
                        junction = junctions_to_use[0]
                        trajectories_through_junction = 0
                        for traj in st.session_state.trajectories:
                            dist = np.hypot(traj.x - junction.cx, traj.z - junction.cz)
                            if np.any(dist <= junction.r):
                                trajectories_through_junction += 1

                        st.info(f"📊 Junction Analysis:")
                        st.write(f"- Trajectories passing through estimated junction: {trajectories_through_junction}/{len(st.session_state.trajectories)} ({trajectories_through_junction/len(st.session_state.trajectories)*100:.1f}%)")

                    for i, traj in enumerate(st.session_state.trajectories):
                        try:
                            # Debug time data for first few trajectories
                            if i < 5:  # Debug first 5 trajectories
                                time_debug = {
                            "trajectory_id": i,
                                    "time_data_type": str(type(traj.t)),
                                    "time_data_shape": traj.t.shape if traj.t is not None else None,
                                    "time_data_sample": traj.t[:3].tolist() if traj.t is not None and len(traj.t) > 0 else None,
                                    "time_data_dtype": str(traj.t.dtype) if traj.t is not None else None
                                }
                                time_data_debug.append(time_debug)

                            # Compute basic trajectory metrics
                            basic_metrics = compute_basic_trajectory_metrics(traj)

                            # Track time data availability
                            if basic_metrics["total_time"] > 0:
                                trajectories_with_time_data += 1
                            else:
                                trajectories_without_time_data += 1

                            # Compute junction-specific timing metrics
                            junction_metrics = {}

                            # Compute timing for each junction
                            for j, junction in enumerate(junctions_to_use):
                                try:
                                    # First check if trajectory actually passes through this junction
                                    entered, _ = entered_junction_idx(traj.x, traj.z, junction)

                                    if not entered:
                                        # Trajectory doesn't pass through this junction, set NaN
                                        junction_metrics[f"junction_{j}_time"] = float('nan')
                                        junction_metrics[f"junction_{j}_mode"] = "no_entry"
                                        # Set speed metrics to NaN as well
                                        junction_metrics[f"junction_{j}_speed"] = float('nan')
                                        junction_metrics[f"junction_{j}_speed_mode"] = "no_entry"
                                        junction_metrics[f"junction_{j}_entry_speed"] = float('nan')
                                        junction_metrics[f"junction_{j}_exit_speed"] = float('nan')
                                        junction_metrics[f"junction_{j}_avg_transit_speed"] = float('nan')
                                        continue

                                    # Use the selected decision mode from GUI (with defaults)
                                    decision_mode = getattr(st.session_state, 'metrics_decision_mode', 'pathlen')
                                    distance = getattr(st.session_state, 'metrics_distance', 100.0)
                                    # Use r_outer from junction definitions like other functions do
                                    r_outer_list = [st.session_state.junction_r_outer.get(i, 50.0) for i in range(len(st.session_state.junctions))]
                                    r_outer = r_outer_list[j] if j < len(r_outer_list) else 50.0
                                    trend_window = getattr(st.session_state, 'metrics_trend_window', 5)
                                    min_outward = getattr(st.session_state, 'metrics_min_outward', 0.0)

                                    # Dynamic path length adjustment for all modes
                                    if len(traj.x) > 1:
                                        dx = np.diff(traj.x)
                                        dz = np.diff(traj.z)
                                        segments = np.hypot(dx, dz)
                                        total_distance = float(np.sum(segments))

                                        # Apply dynamic adjustment based on mode
                                        if decision_mode == "pathlen":
                                            # Use 5% of total distance, but at least 0.1 and at most 10.0
                                            dynamic_distance = max(0.1, min(10.0, total_distance * 0.05))
                                            distance = dynamic_distance
                                        elif decision_mode == "radial":
                                            # For radial mode, use a smaller r_outer based on trajectory data
                                            dynamic_r_outer = max(5.0, min(50.0, total_distance * 0.1))
                                            r_outer = dynamic_r_outer

                                        # Debug: Show dynamic parameters for first few trajectories
                                        if i < 5:
                                            if decision_mode == "pathlen":
                                                st.write(f"🔍 Trajectory {i}: total_distance={total_distance:.2f}, dynamic_distance={dynamic_distance:.2f}")
                                            elif decision_mode == "radial":
                                                st.write(f"🔍 Trajectory {i}: total_distance={total_distance:.2f}, dynamic_r_outer={dynamic_r_outer:.2f}")

                                    # Try the timing calculation with the dynamic parameters
                                    t_val, mode_used = _timing_for_traj(
                                        tr=traj,
                                        junction=junction,
                                        decision_mode=decision_mode,
                                        distance=distance,
                                        r_outer=r_outer,
                                        trend_window=trend_window,
                                        min_outward=min_outward,
                                    )

                                    # Compute speed analysis for this junction
                                    speed_val, speed_mode = speed_through_junction(
                                        tr=traj,
                                        junction=junction,
                                        decision_mode=decision_mode,
                                        path_length=distance,
                                        r_outer=r_outer,
                                        window=trend_window,
                                        min_outward=min_outward,
                                    )

                                    # Compute junction transit speeds
                                    entry_speed, exit_speed, avg_transit_speed = junction_transit_speed(traj, junction)

                                    # If still NaN and we have some movement, try with even smaller parameters
                                    if np.isnan(t_val) and len(traj.x) > 1:
                                        if decision_mode == "pathlen":
                                            # Try with just 1% of total distance, minimum 0.01
                                            fallback_distance = max(0.01, total_distance * 0.01)
                                            t_val, mode_used = _timing_for_traj(
                                                tr=traj,
                                                junction=junction,
                                                decision_mode=decision_mode,
                                                distance=fallback_distance,
                                                r_outer=r_outer,
                                                trend_window=trend_window,
                                                min_outward=min_outward,
                                            )
                                        elif decision_mode == "radial":
                                            # Try with even smaller r_outer
                                            fallback_r_outer = max(2.0, total_distance * 0.05)
                                            t_val, mode_used = _timing_for_traj(
                                                tr=traj,
                                                junction=junction,
                                                decision_mode=decision_mode,
                                                distance=distance,
                                                r_outer=fallback_r_outer,
                                                trend_window=trend_window,
                                                min_outward=min_outward,
                                            )

                                        if i < 5 and not np.isnan(t_val):
                                            if decision_mode == "pathlen":
                                                st.write(f"🔍 Trajectory {i}: Fallback worked! fallback_distance={fallback_distance:.3f}")
                                            elif decision_mode == "radial":
                                                st.write(f"🔍 Trajectory {i}: Fallback worked! fallback_r_outer={fallback_r_outer:.3f}")

                                    junction_metrics[f"junction_{j}_time"] = t_val
                                    junction_metrics[f"junction_{j}_mode"] = mode_used
                                    # Add speed analysis metrics
                                    junction_metrics[f"junction_{j}_speed"] = speed_val
                                    junction_metrics[f"junction_{j}_speed_mode"] = speed_mode
                                    junction_metrics[f"junction_{j}_entry_speed"] = entry_speed
                                    junction_metrics[f"junction_{j}_exit_speed"] = exit_speed
                                    junction_metrics[f"junction_{j}_avg_transit_speed"] = avg_transit_speed
                                except Exception as e:
                                    st.warning(f"⚠️ Junction {j} timing failed for trajectory {i}: {e}")
                                    junction_metrics[f"junction_{j}_time"] = float('nan')
                                    junction_metrics[f"junction_{j}_mode"] = "error"

                            # Combine basic and junction metrics
                            combined_metrics = {
                                "trajectory_id": i,
                                "trajectory_tid": traj.tid,
                                **basic_metrics,
                                **junction_metrics
                            }
                            metrics.append(combined_metrics)

                        except Exception as e:
                            st.error(f"❌ Failed to compute metrics for trajectory {i} ({traj.tid}): {e}")
                            # Add error entry to maintain consistency
                            error_metrics = {
                                "trajectory_id": i,
                                "trajectory_tid": traj.tid,
                                "total_time": 0.0,
                                "total_distance": 0.0,
                                "average_speed": 0.0,
                                "error": str(e)
                            }
                            metrics.append(error_metrics)

                    # Show time data debug information (always show for metrics analysis)
                    st.markdown("---")
                    st.markdown("### 🔍 Debug Information")

                    # Debug: Show what we have
                    st.write(f"**Debug Status:**")
                    st.write(f"- time_data_debug length: {len(time_data_debug)}")
                    st.write(f"- trajectories_without_time_data: {trajectories_without_time_data}")
                    st.write(f"- trajectories_with_time_data: {trajectories_with_time_data}")

                    if time_data_debug:
                        with st.expander("🔍 Time Data Debug Information", expanded=True):
                            st.write("**First 5 trajectories time data analysis:**")
                            for debug_info in time_data_debug:
                                st.write(f"**Trajectory {debug_info['trajectory_id']}:**")
                                st.write(f"- Type: {debug_info['time_data_type']}")
                                st.write(f"- Shape: {debug_info['time_data_shape']}")
                                st.write(f"- Dtype: {debug_info['time_data_dtype']}")
                                st.write(f"- Sample: {debug_info['time_data_sample']}")
                                st.write("---")
                    else:
                        st.info("No time data debug information available")

                    # Show detailed analysis of failing trajectories
                    if trajectories_without_time_data > 0:
                        with st.expander("🔍 Detailed Time Data Analysis", expanded=True):
                            st.write(f"**Analysis of {trajectories_without_time_data} trajectories with invalid time data:**")

                            # Sample a few failing trajectories for detailed analysis
                            failing_samples = []
                            for i, traj in enumerate(st.session_state.trajectories):
                                if i >= 10:  # Limit to first 10 for performance
                                    break
                                try:
                                    basic_metrics = compute_basic_trajectory_metrics(traj)
                                    if basic_metrics["total_time"] == 0:
                                        failing_samples.append({
                                            "id": i,
                                            "tid": traj.tid,
                                            "time_data": traj.t[:5].tolist() if traj.t is not None and len(traj.t) > 0 else None,
                                            "time_dtype": str(traj.t.dtype) if traj.t is not None else None,
                                            "time_shape": traj.t.shape if traj.t is not None else None,
                                            "time_is_none": traj.t is None,
                                            "time_length": len(traj.t) if traj.t is not None else 0
                                        })
                                except Exception as e:
                                    failing_samples.append({
                                        "id": i,
                                        "tid": traj.tid,
                                        "error": str(e)
                                    })

                            if failing_samples:
                                st.write("**Sample failing trajectories:**")
                                for sample in failing_samples:
                                    st.write(f"**Trajectory {sample['id']} ({sample['tid']}):**")
                                    if 'error' in sample:
                                        st.write(f"- Error: {sample['error']}")
                                    else:
                                        st.write(f"- Time data is None: {sample['time_is_none']}")
                                        st.write(f"- Time data length: {sample['time_length']}")
                                        st.write(f"- Time data sample: {sample['time_data']}")
                                        st.write(f"- Time dtype: {sample['time_dtype']}")
                                        st.write(f"- Time shape: {sample['time_shape']}")
                                    st.write("---")

                            # Check if time data is completely missing
                            none_count = sum(1 for traj in st.session_state.trajectories[:10] if traj.t is None)
                            if none_count > 0:
                                st.warning(f"⚠️ {none_count} out of 10 sample trajectories have NO time data (t=None)")
                                st.write("**This suggests:**")
                                st.write("- Time column mapping is incorrect")
                                st.write("- Time column doesn't exist in CSV files")
                                st.write("- Time column is completely empty")

                                # Try to diagnose column mapping issue
                                st.write("**Column Mapping Diagnosis:**")
                                st.write("The GUI is looking for a time column, but it might not exist or be named differently.")
                                st.write("**Common time column names:**")
                                st.write("- 'Time' (most common)")
                                st.write("- 'time'")
                                st.write("- 't'")
                                st.write("- 'timestamp'")
                                st.write("- 'Timestamp'")
                                st.write("- 'TIME'")
                                st.write("")
                                st.write("**To fix this:**")
                                st.write("1. Check the Data Upload tab")
                                st.write("2. Look at the 'Time Column' field")
                                st.write("3. Make sure it matches a column name in your CSV files")
                                st.write("4. If unsure, try common names like 'Time', 'time', or 't'")

                            st.write("**Common time data issues:**")
                            st.write("- Empty strings or null values")
                            st.write("- Non-numeric text (e.g., 'Time: 1.23', '1.23s')")
                            st.write("- Mixed data types in the same column")
                            st.write("- Missing or corrupted time data")
                            st.write("- Incorrect column mapping")

                    # Store results - preserve existing analysis results
                    if st.session_state.analysis_results is None:
                        st.session_state.analysis_results = {}
                    st.session_state.analysis_results["metrics"] = metrics

                    # Save metrics to CSV file
                    try:
                        import os
                        import pandas as pd
                        os.makedirs("gui_outputs", exist_ok=True)
                        df = pd.DataFrame(metrics)
                        csv_path = os.path.join("gui_outputs", "metrics_results.csv")
                        df.to_csv(csv_path, index=False)
                        st.info(f"📁 Metrics saved to: {csv_path}")
                    except Exception as e:
                        st.warning(f"⚠️ Could not save metrics to file: {e}")

                    # Generate and save metrics plots for export and visualization reuse
                    try:
                        import os
                        import glob
                        import pandas as pd
                        import matplotlib.pyplot as plt
                        import math

                        metrics_dir = os.path.join("gui_outputs", "metrics")
                        os.makedirs(metrics_dir, exist_ok=True)

                        metrics_df = pd.DataFrame(metrics)

                        # Helper to safely save and close figures
                        def _save_fig(path):
                            plt.tight_layout()
                            plt.savefig(path, dpi=150)
                            plt.close()

                        # ---- Utilities for KDE and distribution fitting ----
                        def _kde_curve(values):
                            arr = np.asarray(values, dtype=float)
                            arr = arr[np.isfinite(arr)]
                            if len(arr) < 2:
                                return None
                            std = np.std(arr)
                            if std == 0:
                                return None
                            n = len(arr)
                            # Silverman's rule of thumb
                            h = 1.06 * std * (n ** (-1.0 / 5.0))
                            xs = np.linspace(np.percentile(arr, 1), np.percentile(arr, 99), 200)
                            diffs = (xs[:, None] - arr[None, :]) / h
                            kernel = np.exp(-0.5 * diffs * diffs) / (math.sqrt(2.0 * math.pi))
                            density = np.sum(kernel, axis=1) / (n * h)
                            return xs, density

                        def _loglik_normal(arr, mu, sigma):
                            if sigma <= 0:
                                return -np.inf
                            return np.sum(-0.5 * np.log(2 * np.pi) - np.log(sigma) - 0.5 * ((arr - mu) / sigma) ** 2)

                        def _loglik_lognormal(arr, mu_log, sigma_log):
                            if sigma_log <= 0:
                                return -np.inf
                            if np.any(arr <= 0):
                                return -np.inf
                            z = (np.log(arr) - mu_log) / sigma_log
                            return np.sum(-0.5 * np.log(2 * np.pi) - np.log(sigma_log) - np.log(arr) - 0.5 * z * z)

                        def _loglik_gamma(arr, k, theta):
                            if k <= 0 or theta <= 0:
                                return -np.inf
                            if np.any(arr <= 0):
                                return -np.inf
                            # log pdf = (k-1)ln x - x/theta - k ln theta - lgamma(k)
                            return np.sum((k - 1) * np.log(arr) - arr / theta - k * np.log(theta) - math.lgamma(k))

                        def _fit_and_plot_overlays(ax, arr, xlabel, base_color, alt_color):
                            # KDE overlay
                            kde = _kde_curve(arr)
                            if kde is not None:
                                xs_kde, dens_kde = kde
                                ax.plot(xs_kde, dens_kde, color=alt_color, linewidth=2, alpha=0.9, label="KDE")

                            # Candidate distributions and AIC
                            candidates = []
                            n = len(arr)
                            if n >= 2:
                                # Normal
                                mu = float(np.mean(arr))
                                sigma = float(np.std(arr))
                                ll_n = _loglik_normal(arr, mu, sigma) if sigma > 0 else -np.inf
                                aic_n = 2 * 2 - 2 * ll_n  # 2 params
                                candidates.append({
                                    "name": "Normal",
                                    "params": (mu, sigma),
                                    "aic": aic_n,
                                    "pdf": lambda x: (1.0 / (sigma * math.sqrt(2 * math.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2),
                                    "label": f"Normal (μ={mu:.2f}, σ={sigma:.2f})"
                                })

                                # Log-normal (positive values)
                                pos = arr[arr > 0]
                                if len(pos) >= 2 and np.std(np.log(pos)) > 0:
                                    mu_l = float(np.mean(np.log(pos)))
                                    sigma_l = float(np.std(np.log(pos)))
                                    ll_l = _loglik_lognormal(pos, mu_l, sigma_l)
                                    aic_l = 2 * 2 - 2 * ll_l  # 2 params
                                    candidates.append({
                                        "name": "LogNormal",
                                        "params": (mu_l, sigma_l),
                                        "aic": aic_l,
                                        "pdf": lambda x: np.where(x > 0, (1.0 / (x * sigma_l * math.sqrt(2 * math.pi))) * np.exp(-0.5 * ((np.log(x) - mu_l) / sigma_l) ** 2), 0.0),
                                        "label": f"LogNormal (μlog={mu_l:.2f}, σlog={sigma_l:.2f})"
                                    })

                                # Gamma (method of moments)
                                if np.all(arr > 0) and np.var(arr) > 0:
                                    mean = float(np.mean(arr))
                                    var = float(np.var(arr))
                                    k = mean * mean / var
                                    theta = var / mean
                                    ll_g = _loglik_gamma(arr, k, theta)
                                    aic_g = 2 * 2 - 2 * ll_g  # 2 params (k, theta)
                                    candidates.append({
                                        "name": "Gamma",
                                        "params": (k, theta),
                                        "aic": aic_g,
                                        "pdf": lambda x: np.where(x > 0, (x ** (k - 1)) * np.exp(-x / theta) / (math.gamma(k) * (theta ** k)), 0.0),
                                        "label": f"Gamma (k={k:.2f}, θ={theta:.2f})"
                                    })

                            if candidates:
                                best = min(candidates, key=lambda c: c["aic"])
                                xs = np.linspace(np.percentile(arr, 1), np.percentile(arr, 99), 200)
                                ax.plot(xs, best["pdf"](xs), color=base_color, linewidth=2.5, label=f"Best: {best['label']} (AIC {best['aic']:.1f})")
                                ax.legend()

                        # 1) Total Time Distribution (+ KDE and best-fit distribution)
                        if "total_time" in metrics_df.columns and metrics_df["total_time"].notna().any():
                            plt.figure(figsize=(8, 4))
                            vals = metrics_df["total_time"].dropna().to_numpy()
                            ax = plt.gca()
                            ax.hist(vals, bins=30, color="#4C78A8", alpha=0.55, density=True, edgecolor="none")
                            _fit_and_plot_overlays(ax, vals, xlabel="Seconds", base_color="#1F77B4", alt_color="#4C78A8")
                            ax.set_title("Total Time Distribution (s)")
                            ax.set_xlabel("Seconds")
                            ax.set_ylabel("Density")
                            _save_fig(os.path.join(metrics_dir, "total_time_distribution.png"))

                        # 2) Average Speed Distribution (+ KDE and best-fit distribution)
                        if "average_speed" in metrics_df.columns and metrics_df["average_speed"].notna().any():
                            plt.figure(figsize=(8, 4))
                            vals = metrics_df["average_speed"].dropna().to_numpy()
                            ax = plt.gca()
                            ax.hist(vals, bins=30, color="#F58518", alpha=0.55, density=True, edgecolor="none")
                            _fit_and_plot_overlays(ax, vals, xlabel="Speed", base_color="#DD8452", alt_color="#F58518")
                            ax.set_title("Average Speed Distribution")
                            ax.set_xlabel("Speed")
                            ax.set_ylabel("Density")
                            _save_fig(os.path.join(metrics_dir, "average_speed_distribution.png"))

                        # 3) Total Distance Distribution (+ KDE and best-fit distribution)
                        if "total_distance" in metrics_df.columns and metrics_df["total_distance"].notna().any():
                            plt.figure(figsize=(8, 4))
                            vals = metrics_df["total_distance"].dropna().to_numpy()
                            ax = plt.gca()
                            ax.hist(vals, bins=30, color="#54A24B", alpha=0.55, density=True, edgecolor="none")
                            _fit_and_plot_overlays(ax, vals, xlabel="Distance", base_color="#2CA02C", alt_color="#54A24B")
                            ax.set_title("Total Distance Distribution")
                            ax.set_xlabel("Distance")
                            ax.set_ylabel("Density")
                            _save_fig(os.path.join(metrics_dir, "total_distance_distribution.png"))

                        # Discover junction-related columns
                        junction_time_cols = [c for c in metrics_df.columns if c.startswith("junction_") and c.endswith("_time")]
                        speed_cols = [c for c in metrics_df.columns if c.startswith("junction_") and c.endswith("_speed")]

                        # 4) Speed vs Time Correlation (means per junction for selected speed metrics)
                        # Use average transit speed if available; else fall back to generic _speed
                        suffix_candidates = ["_avg_transit_speed", "_speed"]
                        chosen_speed_cols = []
                        for sfx in suffix_candidates:
                            candidate = [c for c in speed_cols if c.endswith(sfx)]
                            if candidate:
                                chosen_speed_cols = candidate
                                break
                        if chosen_speed_cols and junction_time_cols:
                            means_speed = []
                            means_time = []
                            labels = []
                            for sc in chosen_speed_cols:
                                jn = sc.split("_")[1]
                                tc = f"junction_{jn}_time"
                                if tc in metrics_df.columns:
                                    df_pair = metrics_df[[sc, tc]].dropna()
                                    if len(df_pair) > 0:
                                        means_speed.append(df_pair[sc].mean())
                                        means_time.append(df_pair[tc].mean())
                                        labels.append(f"J{jn}")
                            if means_speed and means_time:
                                plt.figure(figsize=(6, 6))
                                plt.scatter(means_time, means_speed, c="#4C78A8")
                                for x, y, lab in zip(means_time, means_speed, labels):
                                    plt.annotate(lab, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
                                plt.title("Speed vs Time (means per junction)")
                                plt.xlabel("Time (s)")
                                plt.ylabel("Speed")
                                _save_fig(os.path.join(metrics_dir, "speed_vs_time_correlation.png"))

                        # 5) Entry vs Exit Speed by Junction (grouped bars)
                        entry_cols = [c for c in metrics_df.columns if c.endswith("_entry_speed")]
                        exit_cols = [c for c in metrics_df.columns if c.endswith("_exit_speed")]
                        if entry_cols and exit_cols:
                            data = []
                            labels_j = []
                            for ec in entry_cols:
                                jn = ec.split("_")[1]
                                xc = f"junction_{jn}_exit_speed"
                                if xc in metrics_df.columns:
                                    e_vals = metrics_df[ec].dropna()
                                    x_vals = metrics_df[xc].dropna()
                                    if len(e_vals) > 0 and len(x_vals) > 0:
                                        data.append((e_vals.mean(), x_vals.mean()))
                                        labels_j.append(f"J{jn}")
                            if data:
                                entry_means = [d[0] for d in data]
                                exit_means = [d[1] for d in data]
                                x = np.arange(len(labels_j))
                                width = 0.35
                                plt.figure(figsize=(max(6, len(labels_j) * 0.6), 4))
                                plt.bar(x - width/2, entry_means, width, label='Entry', color='#72B7B2')
                                plt.bar(x + width/2, exit_means, width, label='Exit', color='#E45756')
                                plt.xticks(x, labels_j)
                                plt.title("Entry vs Exit Speed by Junction")
                                plt.xlabel("Junction")
                                plt.ylabel("Speed")
                                plt.legend()
                                _save_fig(os.path.join(metrics_dir, "entry_exit_speed_by_junction.png"))

                        # 6) Junction Timing Comparison (average times)
                        if junction_time_cols:
                            jt_labels = []
                            jt_means = []
                            for tc in sorted(junction_time_cols, key=lambda c: int(c.split('_')[1])):
                                jn = tc.split("_")[1]
                                vals = metrics_df[tc].dropna()
                                if len(vals) > 0:
                                    jt_labels.append(f"J{jn}")
                                    jt_means.append(vals.mean())
                            if jt_labels:
                                x = np.arange(len(jt_labels))
                                plt.figure(figsize=(max(6, len(jt_labels) * 0.6), 4))
                                plt.bar(x, jt_means, color="#4C78A8")
                                plt.xticks(x, jt_labels)
                                plt.title("Average Junction Timing")
                                plt.xlabel("Junction")
                                plt.ylabel("Time (s)")
                                _save_fig(os.path.join(metrics_dir, "junction_timing_comparison.png"))

                        # 7) Individual Junction Timing Distributions (per junction) (+ KDE and best-fit distribution)
                        if junction_time_cols:
                            for tc in sorted(junction_time_cols, key=lambda c: int(c.split('_')[1])):
                                jn = tc.split("_")[1]
                                vals = metrics_df[tc].dropna().to_numpy()
                                if len(vals) > 0:
                                    plt.figure(figsize=(8, 4))
                                    ax = plt.gca()
                                    ax.hist(vals, bins=30, color="#B279A2", alpha=0.55, density=True, edgecolor="none")
                                    _fit_and_plot_overlays(ax, vals, xlabel="Seconds", base_color="#A05FA3", alt_color="#B279A2")
                                    ax.set_title(f"Junction {jn} Timing Distribution (s)")
                                    ax.set_xlabel("Seconds")
                                    ax.set_ylabel("Density")
                                    _save_fig(os.path.join(metrics_dir, f"timing_distribution_J{jn}.png"))

                        # Store paths in session for downstream tabs
                        try:
                            images = {}
                            for p in glob.glob(os.path.join(metrics_dir, "*.png")):
                                images[os.path.basename(p)] = p
                            st.session_state.analysis_results.setdefault("metrics_images", images)
                        except Exception:
                            pass

                    except Exception as e:
                        st.warning(f"⚠️ Could not generate metrics plots: {e}")

                    # Provide detailed success message
                    success_msg = f"✅ Computed metrics for {len(metrics)} trajectories"
                    if trajectories_with_time_data > 0:
                        success_msg += f" ({trajectories_with_time_data} with time data)"
                    if trajectories_without_time_data > 0:
                        success_msg += f" ({trajectories_without_time_data} without valid time data)"

                    st.success(success_msg)

                    # Show warning if many trajectories lack time data
                    if trajectories_without_time_data > len(metrics) * 0.5:
                        st.warning(f"⚠️ {trajectories_without_time_data} trajectories lack valid time data. This may indicate time data format issues. Check the debug information above.")

                        # Provide suggestions for fixing time data issues
                        with st.expander("💡 Suggestions for Fixing Time Data Issues", expanded=False):
                            st.write("**To fix time data issues, try these solutions:**")
                            st.write("")
                            st.write("**1. Check Column Mapping:**")
                            st.write("- Verify the time column is correctly mapped")
                            st.write("- Look for columns like 'Time', 'timestamp', 't', etc.")
                            st.write("")
                            st.write("**2. Check Data Format:**")
                            st.write("- Time data should be numeric (e.g., 0, 1.5, 2.3)")
                            st.write("- Avoid text formats like 'Time: 1.23' or '1.23s'")
                            st.write("- Remove quotes around time values")
                            st.write("")
                            st.write("**3. Data Cleaning:**")
                            st.write("- Remove empty rows or null values")
                            st.write("- Ensure consistent data types in time column")
                            st.write("- Check for mixed formats in the same column")
                            st.write("")
                            st.write("**4. File Format:**")
                            st.write("- Ensure CSV files are properly formatted")
                            st.write("- Check for encoding issues (UTF-8 recommended)")
                            st.write("- Verify file is not corrupted")
                            st.write("")
                            st.write("**5. Manual Inspection:**")
                            st.write("- Open the CSV file in a text editor")
                            st.write("- Look at the first few rows of the time column")
                            st.write("- Check for patterns in the data format")

                    # Generate CLI command for easy copying
                    # Build results dict for CLI command generation
                    metrics_results = {}
                    for i, junction in enumerate(junctions_to_use):
                        junction_key = f"junction_{i}"
                        metrics_results[junction_key] = {
                            "junction": junction,
                            "r_outer": st.session_state.junction_r_outer.get(i, 50.0) if i < len(st.session_state.junctions) else 50.0,
                            "decision_mode": getattr(st.session_state, 'metrics_decision_mode', 'pathlen'),
                            "distance": getattr(st.session_state, 'metrics_distance', 100.0),
                            "r_outer_value": st.session_state.junction_r_outer.get(i, 50.0) if i < len(st.session_state.junctions) else 50.0,
                            "trend_window": getattr(st.session_state, 'metrics_trend_window', 5),
                            "min_outward": getattr(st.session_state, 'metrics_min_outward', 0.0),
                        }
                    self.generate_cli_command("metrics", metrics_results, cluster_method, cluster_params, decision_mode, decision_params)

                elif analysis_type == "gaze":
                    # Analyze gaze and physiological data
                    gaze_results = {}

                    # Create dedicated debug container for gaze analysis
                    gaze_debug_container = st.empty()

                    def update_gaze_debug_display():
                        """Update the persistent gaze debug display"""

                    # Initialize gaze debug info
                    st.session_state['gaze_debug_info'] = {}

                    # Check if we have proper gaze trajectory data (prefer trajectories that actually carry data)
                    active_trajs = st.session_state.trajectories
                    # Build a filtered list that actually has gaze OR physio data
                    try:
                        from verta.verta_data_loader import has_gaze_data as _has_gaze, has_physio_data as _has_physio
                        trajs_with_signals = [t for t in active_trajs if (_has_gaze(t) or _has_physio(t))]

                        # Debug: Show filtering results
                        st.info(f"🔍 **Trajectory Filtering Debug:**")
                        st.write(f"- Total trajectories: {len(active_trajs)}")
                        st.write(f"- Trajectories with gaze/physio data: {len(trajs_with_signals)}")

                        if len(trajs_with_signals) < len(active_trajs):
                            skipped_count = len(active_trajs) - len(trajs_with_signals)
                            st.info(f"ℹ️ Skipped {skipped_count} trajectories without gaze/physiological data")

                    except Exception as e:
                        st.warning(f"⚠️ Error filtering trajectories: {e}")
                        trajs_with_signals = active_trajs

                    # Use the filtered list if it has any valid trajectories
                    if trajs_with_signals and len(trajs_with_signals) > 0:
                        active_trajs = trajs_with_signals
                    else:
                        st.warning("⚠️ No trajectories with gaze/physiological data found")
                        active_trajs = []
                    has_gaze_data = self._check_for_gaze_data(active_trajs)
                    # If global check still fails but we have some physio data, allow comprehensive path to proceed
                    try:
                        from verta.verta_data_loader import has_physio_data as _has_physio
                        has_any_physio = any(_has_physio(t) for t in active_trajs)
                    except Exception:
                        has_any_physio = False

                    # Get column mappings from session state (if they were specified)
                    column_mappings = self._get_gaze_column_mappings()

                    if not has_gaze_data and not has_any_physio and not column_mappings:
                        st.warning("⚠️ **No Gaze Data Available**")
                        st.info("""
                        **Gaze analysis requires trajectories with:**
                        - Head tracking data (`head_forward_x`, `head_forward_z`)
                        - Eye tracking data (`gaze_x`, `gaze_y`)
                        - Physiological data (`pupil_l`, `pupil_r`, `heart_rate`)

                        **Current trajectories only have position data (x, z, t).**

                        Proceeding with movement-only fallback so visualizations still render.
                        """)

                        # Compute movement-only fallback per junction
                        for i, junction in enumerate(st.session_state.junctions):
                            junction_key = f"junction_{i}"
                            r_outer = st.session_state.junction_r_outer.get(i, 50.0)
                            try:
                                movement_df = self._analyze_movement_patterns_optimized(
                                    active_trajs, junction, r_outer, decision_mode, path_length=100.0, epsilon=0.05
                                )
                                # Tag as movement analysis for downstream UI
                                if hasattr(movement_df, 'assign'):
                                    movement_df = movement_df.assign(analysis_type='movement')
                                gaze_results[junction_key] = movement_df
                            except Exception as e:
                                st.warning(f"⚠️ Movement fallback failed for {junction_key}: {e}")
                                gaze_results[junction_key] = None
                    else:
                        # Perform gaze analysis (use the active trajectories we already determined)

                        # Debug: Track session state before analysis
                        st.session_state.debug_session_state = {
                            'trajectories_count': len(st.session_state.trajectories) if st.session_state.trajectories else 0,
                            'gaze_trajectories_count': 0,
                            'last_modified': 'before_gaze_analysis'
                        }

                        # Create global heatmap ONCE (outside junction loop)
                        global_heatmap_data = None
                        cell_size = st.session_state.get('pupil_heatmap_cell_size', 50.0)
                        normalization = st.session_state.get('pupil_heatmap_normalization', 'relative')

                        # Define output directory for global plots
                        import os
                        global_out_dir = os.path.join("gui_outputs", "gaze_plots")
                        os.makedirs(global_out_dir, exist_ok=True)

                        st.info("🗺️ Creating global pupil dilation heatmap...")
                        try:
                            global_heatmap_data = create_pupil_dilation_heatmap(
                                trajectories=active_trajs,
                                junctions=st.session_state.junctions,
                                cell_size=cell_size,
                                normalization=normalization
                            )
                            st.success("✅ Global heatmap created")

                            # Store global heatmap data for consistent scaling calculation
                            st.session_state['global_heatmap_data'] = global_heatmap_data

                            # Generate global heatmap plot
                            try:
                                from verta.verta_gaze import plot_pupil_dilation_heatmap
                                import matplotlib.pyplot as plt
                                import os

                                # Create global heatmap plot
                                global_plot_path = os.path.join(global_out_dir, "global_pupil_heatmap.png")

                                fig = plot_pupil_dilation_heatmap(
                                    heatmap_data=global_heatmap_data,
                                    junctions=st.session_state.junctions,
                                    trajectories=active_trajs,
                                    all_trajectories=active_trajs,
                                    title="Global Pupil Dilation Heatmap",
                                    show_sample_counts=False,
                                    show_minimap=False,
                                    vmin=None,  # Let the function determine scaling
                                    vmax=None
                                )

                                # Save the plot
                                fig.savefig(global_plot_path, dpi=150, bbox_inches="tight")
                                plt.close(fig)

                                st.info(f"📁 Global heatmap plot saved to: {global_plot_path}")
                                st.write(f"🔍 **Debug:** Global plot saved to: `{global_plot_path}`")
                                st.write(f"🔍 **Debug:** File exists after save: {os.path.exists(global_plot_path)}")

                            except Exception as plot_e:
                                st.warning(f"⚠️ Could not generate global heatmap plot: {plot_e}")

                        except Exception as e:
                            st.warning(f"⚠️ Could not create global heatmap: {e}")

                        # Debug: Check junctions
                        st.info(f"🔍 **Junction Debug:**")
                        st.write(f"- Number of junctions: {len(st.session_state.junctions)}")
                        if st.session_state.junctions:
                            for i, junction in enumerate(st.session_state.junctions):
                                r_outer = st.session_state.junction_r_outer.get(i, 50.0)
                                st.write(f"- Junction {i}: Circle(cx={junction.cx}, cz={junction.cz}, r={junction.r}), r_outer={r_outer}")
                        else:
                            st.error("❌ **No junctions defined!** Gaze analysis requires junctions to be defined.")
                            st.write("**Solution:** Go to the Junction Editor tab and define at least one junction.")
                            return

                        # CRITICAL FIX: Perform comprehensive gaze analysis for ALL junctions at once
                        # This ensures all junctions have access to the complete assignments DataFrame
                        st.info("🔍 **Performing comprehensive gaze analysis for all junctions...**")

                        # Create output directory for all junctions
                        import os
                        import pandas as pd
                        out_dir = os.path.join("gui_outputs", "gaze_analysis")
                        os.makedirs(out_dir, exist_ok=True)

                        # Get r_outer values for all junctions
                        r_outer_list = [st.session_state.junction_r_outer.get(i, 50.0) for i in range(len(st.session_state.junctions))]

                        # Perform comprehensive gaze analysis for ALL junctions at once
                        try:
                            gaze_data_all = self._perform_comprehensive_gaze_analysis_all_junctions(
                                trajectories=active_trajs,
                                junctions=st.session_state.junctions,
                                r_outer_list=r_outer_list,
                                decision_mode=decision_mode,
                                path_length=100.0,
                                epsilon=0.05,
                                linger_delta=0.0,
                                out_dir=out_dir
                            )
                        except Exception as e:
                            st.error(f"❌ **Comprehensive gaze analysis failed:** {e}")
                            st.write(f"**Error type:** {type(e).__name__}")
                            st.write(f"**Error message:** {str(e)}")

                            # Fall back to individual junction analysis
                            st.info("🔄 **Falling back to individual junction analysis...**")
                            for i, junction in enumerate(st.session_state.junctions):
                                junction_key = f"junction_{i}"
                                r_outer = r_outer_list[i]

                                try:
                                    gaze_data = self._perform_gaze_analysis_with_mappings(
                                        trajectories=active_trajs,
                                        junction=junction,
                                        r_outer=r_outer,
                                        decision_mode=decision_mode,
                                        path_length=100.0,
                                        epsilon=0.05,
                                        linger_delta=0.0,
                                        out_dir=out_dir,
                                        column_mappings=column_mappings,
                                        scale_factor=1.0
                                    )

                                    # Normalize columns so downstream plots find expected names
                                    if isinstance(gaze_data, dict):
                                        gaze_data = self._normalize_gaze_result_frames(gaze_data)

                                    # Store the comprehensive gaze analysis results
                                    gaze_results[junction_key] = gaze_data

                                    st.success(f"✅ Completed fallback gaze analysis for {junction_key}")

                                    # Generate gaze plots immediately after analysis
                                    try:
                                        st.info(f"📊 Generating gaze plots for {junction_key}...")
                                        self._generate_gaze_plots_during_analysis(gaze_data, junction_key, out_dir)
                                        st.success(f"✅ Generated plots for {junction_key}")
                                    except Exception as e:
                                        st.warning(f"⚠️ Plot generation failed for {junction_key}: {e}")

                                    # Update debug display after each junction
                                    update_gaze_debug_display()

                                except Exception as e:
                                    st.warning(f"⚠️ Fallback gaze analysis failed for {junction_key}: {e}")

                                    # Store error information
                                    gaze_results[junction_key] = {
                                        'error': str(e),
                                        'error_type': type(e).__name__,
                                        'junction': junction,
                                        'r_outer': r_outer,
                                        'physiological': None,
                                        'pupil_dilation': None,
                                        'head_yaw': None
                                    }

                                    # Update debug display even on error
                                    update_gaze_debug_display()
                                    continue

                            # Skip the rest of the multi-junction processing
                            gaze_data_all = None

                        # The comprehensive analysis returns data for all junctions
                        # We need to split it by junction for storage
                        if gaze_data_all is not None and isinstance(gaze_data_all, dict) and 'head_yaw' in gaze_data_all:
                                # Split the results by junction
                                head_yaw_df = gaze_data_all['head_yaw']
                                physio_df = gaze_data_all.get('physiological')
                                pupil_df = gaze_data_all.get('pupil_dilation')
                                all_heatmaps = gaze_data_all.get('pupil_heatmap_junction', {})

                                # Process each junction's data
                                for i, junction in enumerate(st.session_state.junctions):
                                    junction_key = f"junction_{i}"

                                    # Filter data for this junction
                                    junction_head_yaw = head_yaw_df[head_yaw_df['junction'] == i] if not head_yaw_df.empty else pd.DataFrame()
                                    junction_physio = physio_df[physio_df['junction'] == i] if physio_df is not None and not physio_df.empty else None
                                    junction_pupil = pupil_df[pupil_df['junction'] == i] if pupil_df is not None and not pupil_df.empty else None

                                    # Get heatmap data for this junction
                                    junction_heatmap = all_heatmaps.get(i) if all_heatmaps else None

                                    # Create junction-specific gaze data
                                    gaze_data = {
                                        'head_yaw': junction_head_yaw,
                                        'physiological': junction_physio,
                                        'pupil_dilation': junction_pupil,
                                        'pupil_heatmap_junction': {i: junction_heatmap} if junction_heatmap else {},
                                        'junction': junction,
                                        'r_outer': r_outer_list[i]
                                    }

                                    st.info(f"🔍 **Processing junction {i}: {junction_key}**")
                                    st.write(f"- Junction: Circle(cx={junction.cx}, cz={junction.cz}, r={junction.r})")
                                    st.write(f"- R_outer: {r_outer_list[i]}")
                                    st.write(f"- Head yaw records: {len(junction_head_yaw)}")
                                    st.write(f"- Physiological records: {len(junction_physio) if junction_physio is not None else 0}")
                                    st.write(f"- Pupil records: {len(junction_pupil) if junction_pupil is not None else 0}")

                                    # Normalize columns so downstream plots find expected names
                                    if isinstance(gaze_data, dict):
                                        gaze_data = self._normalize_gaze_result_frames(gaze_data)

                                    # Store the comprehensive gaze analysis results
                                    gaze_results[junction_key] = gaze_data

                                    # Debug: Verify data was saved correctly
                                    st.info(f"🔍 **Data Storage Verification for {junction_key}:**")
                                    if isinstance(gaze_data, dict):
                                        for data_type, data in gaze_data.items():
                                            if data is not None:
                                                if hasattr(data, 'shape'):
                                                    st.write(f"- {data_type}: {data.shape} DataFrame")
                                                elif isinstance(data, list):
                                                    st.write(f"- {data_type}: {len(data)} records")
                                                else:
                                                    st.write(f"- {data_type}: {type(data).__name__}")
                                            else:
                                                st.write(f"- {data_type}: None")
                                    else:
                                        st.write(f"- Raw data type: {type(gaze_data).__name__}")

                                    st.success(f"✅ Completed gaze analysis for {junction_key}")

                                    # Generate gaze plots immediately after analysis
                                    try:
                                        st.info(f"📊 Generating gaze plots for {junction_key}...")
                                        self._generate_gaze_plots_during_analysis(gaze_data, junction_key, out_dir)
                                        st.success(f"✅ Generated plots for {junction_key}")
                                    except Exception as e:
                                        st.warning(f"⚠️ Plot generation failed for {junction_key}: {e}")

                                    # Update debug display after each junction
                                    update_gaze_debug_display()
                        else:
                            st.error("❌ **Comprehensive gaze analysis failed to return expected data structure**")
                            st.write(f"Returned data type: {type(gaze_data_all)}")
                            if isinstance(gaze_data_all, dict):
                                st.write(f"Keys: {list(gaze_data_all.keys())}")

                            # Fall back to individual junction analysis
                            st.info("🔄 **Falling back to individual junction analysis...**")
                            for i, junction in enumerate(st.session_state.junctions):
                                junction_key = f"junction_{i}"
                                r_outer = r_outer_list[i]

                                try:
                                    gaze_data = self._perform_gaze_analysis_with_mappings(
                                        trajectories=active_trajs,
                                        junction=junction,
                                        r_outer=r_outer,
                                        decision_mode=decision_mode,
                                        path_length=100.0,
                                        epsilon=0.05,
                                        linger_delta=0.0,
                                        out_dir=out_dir,
                                        column_mappings=column_mappings,
                                        scale_factor=1.0
                                    )

                                    # Normalize columns so downstream plots find expected names
                                    if isinstance(gaze_data, dict):
                                        gaze_data = self._normalize_gaze_result_frames(gaze_data)

                                    # Store the comprehensive gaze analysis results
                                    gaze_results[junction_key] = gaze_data

                                    st.success(f"✅ Completed fallback gaze analysis for {junction_key}")

                                    # Generate gaze plots immediately after analysis
                                    try:
                                        st.info(f"📊 Generating gaze plots for {junction_key}...")
                                        self._generate_gaze_plots_during_analysis(gaze_data, junction_key, out_dir)
                                        st.success(f"✅ Generated plots for {junction_key}")
                                    except Exception as e:
                                        st.warning(f"⚠️ Plot generation failed for {junction_key}: {e}")

                                    # Update debug display after each junction
                                    update_gaze_debug_display()

                                except Exception as e:
                                    st.warning(f"⚠️ Fallback gaze analysis failed for {junction_key}: {e}")

                                    # Store error information
                                    gaze_results[junction_key] = {
                                        'error': str(e),
                                        'error_type': type(e).__name__,
                                        'junction': junction,
                                        'r_outer': r_outer,
                                        'physiological': None,
                                        'pupil_dilation': None,
                                        'head_yaw': None
                                    }

                                    # Update debug display even on error
                                    update_gaze_debug_display()
                                    continue

                    # Store results - preserve existing analysis results
                    if st.session_state.analysis_results is None:
                        st.session_state.analysis_results = {}
                    st.session_state.analysis_results["gaze_results"] = gaze_results

                    # Store global heatmap separately (only once)
                    if global_heatmap_data is not None:
                        st.session_state.analysis_results["pupil_heatmap_global"] = global_heatmap_data

                    # Show summary
                    successful_junctions = len([k for k, v in gaze_results.items() if v is not None])
                    total_junctions = len(st.session_state.junctions)
                    if successful_junctions == total_junctions:
                        st.success(f"✅ Gaze analysis completed successfully for all {total_junctions} junctions!")
                    else:
                        st.warning(f"⚠️ Gaze analysis completed for {successful_junctions}/{total_junctions} junctions")

                    # Generate CLI command for easy copying
                    # Build results dict for CLI command generation
                    gaze_results_dict = {}
                    for i, junction in enumerate(st.session_state.junctions):
                        junction_key = f"junction_{i}"
                        gaze_results_dict[junction_key] = {
                            "junction": junction,
                            "r_outer": st.session_state.junction_r_outer.get(i, 50.0),
                            "decision_mode": decision_mode,
                            "path_length": decision_params.get("path_length", 100.0) if decision_params else 100.0,
                            "epsilon": decision_params.get("epsilon", 0.05) if decision_params else 0.05,
                            "linger_delta": decision_params.get("linger_delta", 5.0) if decision_params else 5.0,
                        }
                    self.generate_cli_command("gaze", gaze_results_dict, cluster_method, cluster_params, decision_mode, decision_params)

                elif analysis_type == "predict":
                    # Run prediction analysis using spatial tracking only
                    # Create output directory for prediction results
                    import os
                    output_dir = "gui_outputs"
                    os.makedirs(output_dir, exist_ok=True)

                    # Skip discover_decision_chain - use spatial tracking only
                    # Create empty chain_df for compatibility
                    import pandas as pd
                    chain_df = pd.DataFrame(columns=['trajectory'])
                    chain_df['trajectory'] = [i for i in range(len(st.session_state.trajectories))]

                    print(f"🔍 DEBUG: Using spatial tracking only - skipping discover_decision_chain")
                    print(f"🔍 DEBUG: Created empty chain_df with {len(chain_df)} trajectories")

                    # Define r_outer_list for predict analysis
                    r_outer_list = [st.session_state.junction_r_outer.get(i, 50.0) for i in range(len(st.session_state.junctions))]

                    print(f"🔍 DEBUG: Starting analyze_junction_choice_patterns with spatial tracking only...")

                    # Debug: Check what trajectories visit multiple junctions
                    print(f"\n🔍 DEBUG: Analyzing trajectory junction visits...")
                    multi_junction_trajectories = 0
                    consecutive_junction_trajectories = 0

                    # COMMENTED OUT: This code referenced norm_df which we removed
                    # for idx, row in norm_df.iterrows():
                    #     traj_id = row['trajectory']
                    #     visited_junctions = []
                    #     for i in range(7):  # 7 junctions
                    #         col = f'branch_j{i}'
                    #         if pd.notna(row[col]) and row[col] >= 0:  # Valid branch assignment
                    #             visited_junctions.append(i)
                    #
                    #     if len(visited_junctions) > 1:
                    #         multi_junction_trajectories += 1
                    #         # Check if junctions are consecutive (for flow analysis)
                    #         if len(visited_junctions) >= 2:
                    #             consecutive_junction_trajectories += 1
                    #             if consecutive_junction_trajectories <= 5:  # Show first 5 examples
                    #                 print(f"🔍 DEBUG: Trajectory {traj_id} visits junctions: {visited_junctions}")

                    print(f"🔍 DEBUG: Trajectories visiting multiple junctions: {multi_junction_trajectories}")
                    print(f"🔍 DEBUG: Trajectories with consecutive visits: {consecutive_junction_trajectories}")

                    # Debug: Check r_outer_list values
                    print(f"\n🔍 DEBUG: r_outer_list values: {r_outer_list}")
                    print(f"🔍 DEBUG: r_outer_list length: {len(r_outer_list)}")
                    print(f"🔍 DEBUG: r_outer_list types: {[type(r) for r in r_outer_list]}")

                    # Debug: Check junction radii
                    print(f"\n🔍 DEBUG: Junction radii:")
                    for i, junction in enumerate(st.session_state.junctions):
                        print(f"  Junction {i}: radius={junction.r}, r_outer={r_outer_list[i] if i < len(r_outer_list) else 'N/A'}")

                    # Debug: Test trajectory sequence tracking with multiple trajectories
                    print(f"\n🔍 DEBUG: Testing trajectory sequence tracking...")
                    from verta.verta_plotting import _track_trajectory_junction_sequence

                    # Test trajectories 1-3 instead of trajectory 0 (which has limited range)
                    test_trajectories = st.session_state.trajectories[1:4]  # Trajectories 1-3

                    for test_idx, test_traj in enumerate(test_trajectories):
                        traj_id = getattr(test_traj, 'tid', test_idx + 1)
                        print(f"\n🔍 DEBUG: === Testing Trajectory {traj_id} ===")

                        # Debug: Check trajectory data
                        print(f"🔍 DEBUG: Trajectory {traj_id} data:")
                        print(f"  - Length: {len(test_traj.x)} points")
                        print(f"  - X range: {min(test_traj.x):.2f} to {max(test_traj.x):.2f}")
                        print(f"  - Z range: {min(test_traj.z):.2f} to {max(test_traj.z):.2f}")

                        # Debug: Check junction data
                        print(f"🔍 DEBUG: Junction data:")
                        for i, junction in enumerate(st.session_state.junctions):
                            print(f"  Junction {i}: center=({junction.cx:.2f}, {junction.cz:.2f}), radius={junction.r}, r_outer={r_outer_list[i]}")

                        # Test spatial tracking
                        test_sequence = _track_trajectory_junction_sequence(test_traj, st.session_state.junctions, r_outer_list)
                        print(f"🔍 DEBUG: Trajectory {traj_id} sequence: {test_sequence}")

                        if len(test_sequence) > 0:
                            print(f"🔍 DEBUG: ✅ Trajectory {traj_id} has valid sequence!")
                            print(f"🔍 DEBUG: Stopping debug testing - spatial tracking is working!")
                            break
                        else:
                            print(f"🔍 DEBUG: ❌ Trajectory {traj_id} has no sequence")

                    # If no trajectories worked, test trajectory 0 as fallback
                    if all(len(_track_trajectory_junction_sequence(traj, st.session_state.junctions, r_outer_list)) == 0 for traj in test_trajectories):
                        print(f"\n🔍 DEBUG: === Testing Trajectory 0 as fallback ===")
                        test_traj = st.session_state.trajectories[0]
                        test_sequence = _track_trajectory_junction_sequence(test_traj, st.session_state.junctions, r_outer_list)
                        print(f"🔍 DEBUG: Trajectory 0 sequence: {test_sequence}")

                    # Debug: Show sample of norm_df data
                    # print(f"\n🔍 DEBUG: Sample norm_df data (first 10 rows):")
                    # print(norm_df.head(10))

                    # Then analyze junction choice patterns
                    print("\n" + "="*60)
                    print("🔍 PREDICT ANALYSIS - FLOW GRAPH DEBUG OUTPUT")
                    print("="*60)

                    # Run prediction analysis using spatial tracking only
                    results = analyze_junction_choice_patterns(
                        trajectories=st.session_state.trajectories,
                        chain_df=chain_df,  # Empty chain_df for compatibility
                        junctions=st.session_state.junctions,
                        output_dir=output_dir,
                        r_outer_list=r_outer_list,
                        gui_mode=False  # Enable console debug output
                    )

                    print("="*60)
                    print("✅ Predict analysis completed")
                    print("="*60 + "\n")

                    # Store results - preserve existing analysis results
                    if st.session_state.analysis_results is None:
                        st.session_state.analysis_results = {}
                    st.session_state.analysis_results["predictions"] = results

                    # Generate CLI command for easy copying
                    self.generate_cli_command("predict", results, cluster_method, cluster_params, decision_mode, decision_params)

                elif analysis_type == "intent":
                    # Run intent recognition analysis
                    import pandas as pd
                    import numpy as np
                    import os

                    st.info("🧠 Running Intent Recognition Analysis...")

                    # Get parameters
                    intent_params = st.session_state.get('intent_params', {
                        'prediction_distances': [100.0, 75.0, 50.0, 25.0],
                        'model_type': 'random_forest',
                        'cv_folds': 5,
                        'test_split': 0.2
                    })

                    # Check if we have time data
                    has_time = all(tr.t is not None and len(tr.t) > 0 for tr in st.session_state.trajectories[:5])
                    if not has_time:
                        st.warning("⚠️ Time data not detected. Intent recognition requires temporal information for velocity/acceleration features.")
                        st.info("💡 Tip: Ensure your CSV files have a time column specified in column mapping.")

                    # Check for sklearn
                    try:
                        import sklearn
                    except ImportError:
                        st.error("❌ scikit-learn not installed!")
                        st.markdown("""
                        Intent recognition requires scikit-learn. Install with:
                        ```bash
                        pip install scikit-learn
                        ```
                        """)
                        return

                    # Create output directory
                    output_dir = "gui_outputs/intent_recognition"
                    os.makedirs(output_dir, exist_ok=True)

                    # For each junction, run intent recognition
                    intent_results = {}

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for junction_idx, junction in enumerate(st.session_state.junctions):
                        status_text.text(f"Analyzing junction {junction_idx + 1}/{len(st.session_state.junctions)}...")

                        try:
                            # Create junction output directory
                            junction_output = os.path.join(output_dir, f"junction_{junction_idx}")
                            os.makedirs(junction_output, exist_ok=True)

                            # Try to load existing branch assignments from previous Discover analysis
                            assignments_df = None
                            centers = None

                            # Check if we have results from a previous analysis
                            if st.session_state.analysis_results and 'branches' in st.session_state.analysis_results:
                                branch_results = st.session_state.analysis_results['branches']
                                junction_key = f"junction_{junction_idx}"

                                if junction_key in branch_results:
                                    assignments_df = branch_results[junction_key].get('assignments')
                                    centers = branch_results[junction_key].get('centers')

                                    if assignments_df is not None:
                                        st.info(f"📋 Using existing branch assignments from previous Discover analysis for Junction {junction_idx}")

                            # If no existing assignments, run discovery
                            if assignments_df is None:
                                st.warning(f"⚠️ No existing branch assignments found for Junction {junction_idx}")
                                st.info(f"🔍 Running branch discovery with default parameters...")
                                st.info("💡 **Tip:** Run 'Discover Branches' analysis first to control clustering parameters!")

                                r_outer = st.session_state.junction_r_outer.get(junction_idx, 50.0)

                                # Run branch discovery with default parameters
                                assignments_df, summary_df, centers = discover_branches(
                                    trajectories=st.session_state.trajectories,
                                    junction=junction,
                                    k=3,
                                    decision_mode="hybrid",
                                    r_outer=r_outer,
                                    path_length=100.0,
                                    epsilon=0.05,
                                    cluster_method="auto",
                                    out_dir=junction_output
                                )

                            # Filter valid branches (>= 0)
                            valid_assignments = assignments_df[assignments_df['branch'] >= 0]

                            if len(valid_assignments) < 10:
                                st.warning(f"⚠️ Junction {junction_idx}: Insufficient valid trajectories ({len(valid_assignments)}). Skipping intent analysis.")
                                intent_results[f"junction_{junction_idx}"] = {
                                    'error': 'insufficient_data',
                                    'n_valid_trajectories': len(valid_assignments)
                                }
                                continue

                            # Count unique branches
                            n_branches = len(assignments_df[assignments_df['branch'] >= 0]['branch'].unique())
                            st.success(f"✅ Using {n_branches} branches with {len(valid_assignments)} valid trajectories")

                            # Run intent recognition
                            st.info(f"🤖 Training intent recognition models...")

                            results = analyze_intent_recognition(
                                trajectories=st.session_state.trajectories,
                                junction=junction,
                                actual_branches=assignments_df,
                                output_dir=junction_output,
                                prediction_distances=intent_params['prediction_distances'],
                                previous_choices=None  # TODO: Could add multi-junction support
                            )

                            if 'error' in results:
                                st.error(f"❌ Junction {junction_idx}: {results['error']}")
                                intent_results[f"junction_{junction_idx}"] = results
                            else:
                                st.success(f"✅ Junction {junction_idx}: Intent recognition complete!")

                                # Display quick summary
                                models_trained = results['training_results'].get('models_trained', {})
                                if models_trained:
                                    avg_acc = np.mean([m['cv_mean_accuracy'] for m in models_trained.values()])
                                    st.metric(f"Junction {junction_idx} Avg Accuracy", f"{avg_acc:.1%}")

                                intent_results[f"junction_{junction_idx}"] = results

                        except Exception as e:
                            st.error(f"❌ Junction {junction_idx} failed: {str(e)}")
                            intent_results[f"junction_{junction_idx}"] = {
                                'error': str(e),
                                'error_type': type(e).__name__
                            }

                        progress_bar.progress((junction_idx + 1) / len(st.session_state.junctions))

                    status_text.text("✅ Intent recognition analysis complete!")
                    progress_bar.empty()

                    # Store results
                    if st.session_state.analysis_results is None:
                        st.session_state.analysis_results = {}
                    st.session_state.analysis_results["intent_recognition"] = intent_results

                    # Display summary
                    st.markdown("### 📊 Intent Recognition Summary")

                    successful_junctions = [k for k, v in intent_results.items()
                                          if 'error' not in v]

                    if successful_junctions:
                        st.success(f"✅ Successfully analyzed {len(successful_junctions)}/{len(st.session_state.junctions)} junctions")

                        # Create summary table
                        summary_data = []
                        for junction_key in successful_junctions:
                            results = intent_results[junction_key]
                            models = results['training_results'].get('models_trained', {})

                            for dist, model_info in models.items():
                                summary_data.append({
                                    'Junction': junction_key.replace('junction_', 'J'),
                                    'Distance (units)': dist,
                                    'Accuracy': f"{model_info['cv_mean_accuracy']:.1%}",
                                    'Std Dev': f"±{model_info['cv_std_accuracy']:.1%}",
                                    'Samples': model_info['n_samples']
                                })

                        if summary_data:
                            summary_df = pd.DataFrame(summary_data)
                            st.dataframe(summary_df, width='stretch')

                        # Show interpretation
                        st.markdown("#### 💡 Interpretation")
                        avg_accuracy = np.mean([float(d['Accuracy'].strip('%')) / 100 for d in summary_data])

                        if avg_accuracy > 0.85:
                            st.success("🟢 **Excellent Predictability**: User intent is highly predictable. Early intervention systems will be very effective!")
                        elif avg_accuracy > 0.70:
                            st.info("🟡 **Good Predictability**: Clear patterns exist. Adaptive systems can benefit users.")
                        else:
                            st.warning("🔴 **Moderate Predictability**: Behavior is variable. Consider per-user models or additional features.")
                    else:
                        st.error("❌ Intent recognition failed for all junctions")

                    st.info(f"📁 Detailed results saved to: {output_dir}")

                    # Generate CLI command for easy copying
                    # Build results dict for CLI command generation
                    intent_results_dict = {}
                    for i, junction in enumerate(st.session_state.junctions):
                        junction_key = f"junction_{i}"
                        intent_results_dict[junction_key] = {
                            "junction": junction,
                            "r_outer": st.session_state.junction_r_outer.get(i, 50.0),
                            "decision_mode": decision_mode,
                            "path_length": decision_params.get("path_length", 100.0) if decision_params else 100.0,
                            "epsilon": decision_params.get("epsilon", 0.05) if decision_params else 0.05,
                            "linger_delta": decision_params.get("linger_delta", 5.0) if decision_params else 5.0,
                            "prediction_distances": intent_params.get('prediction_distances', [100.0, 75.0, 50.0, 25.0]),
                            "model_type": intent_params.get('model_type', 'random_forest'),
                            "cv_folds": intent_params.get('cv_folds', 5),
                            "test_split": intent_params.get('test_split', 0.2),
                        }
                    self.generate_cli_command("intent", intent_results_dict, cluster_method, cluster_params, decision_mode, decision_params)

                elif analysis_type == "enhanced":
                    # Run enhanced analysis for evacuation planning and risk assessment
                    import os
                    output_dir = "gui_outputs"
                    os.makedirs(output_dir, exist_ok=True)

                    # First run discovery to get chain data
                    k_value = cluster_params.get("k", 3) if cluster_params else 3
                    min_samples = cluster_params.get("min_samples", 5) if cluster_params else 5
                    k_min = cluster_params.get("k_min", 2) if cluster_params else 2
                    k_max = cluster_params.get("k_max", 6) if cluster_params else 6
                    min_sep_deg = cluster_params.get("min_sep_deg", 12.0) if cluster_params else 12.0
                    angle_eps = cluster_params.get("angle_eps", 15.0) if cluster_params else 15.0

                    path_length = decision_params.get("path_length", 100.0) if decision_params else 100.0
                    epsilon = decision_params.get("epsilon", 0.05) if decision_params else 0.05
                    linger_delta = decision_params.get("linger_delta", 5.0) if decision_params else 5.0
                    r_outer_list = [st.session_state.junction_r_outer.get(i, 50.0) for i in range(len(st.session_state.junctions))]

                    # Run discovery first
                    chain_df, centers_list, decisions_chain_df = discover_decision_chain(
                        trajectories=st.session_state.trajectories,
                        junctions=st.session_state.junctions,
                        path_length=path_length,
                        epsilon=epsilon,
                        seed=seed,
                        decision_mode=discover_decision_mode,
                        r_outer_list=r_outer_list,
                        linger_delta=linger_delta,
                        out_dir=output_dir,
                        cluster_method=cluster_method,
                        k=k_value,
                        k_min=k_min,
                        k_max=k_max,
                        min_sep_deg=min_sep_deg,
                        angle_eps=angle_eps,
                        min_samples=min_samples,
                    )

                    # Run enhanced analysis
                    enhanced_results = self._run_enhanced_analysis(
                        trajectories=st.session_state.trajectories,
                        chain_df=chain_df,
                        junctions=st.session_state.junctions,
                        r_outer_list=r_outer_list,
                        centers_list=centers_list,
                        decisions_df=decisions_chain_df
                    )

                    # Store results
                    if st.session_state.analysis_results is None:
                        st.session_state.analysis_results = {}
                    st.session_state.analysis_results["enhanced"] = enhanced_results

                #st.success(f"✅ {analysis_type.capitalize()} analysis completed!")
                #st.rerun()

        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            st.exception(e)

    def _run_enhanced_analysis(self, trajectories, chain_df, junctions, r_outer_list, centers_list, decisions_df):
        """Run enhanced analysis for evacuation planning and risk assessment"""
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import os

        results = {
            "evacuation_analysis": {},
            "recommendations": [],
            "risk_assessment": {},
            "efficiency_metrics": {}
        }

        # 1. Evacuation Analysis - Identify bottlenecks and optimal routes
        st.info("🚨 Running evacuation analysis...")
        evacuation_results = self._analyze_evacuation_patterns(
            trajectories, chain_df, junctions, r_outer_list, centers_list
        )
        results["evacuation_analysis"] = evacuation_results

        # 2. Generate Recommendations
        st.info("💡 Generating recommendations...")
        recommendations = self._generate_recommendations(evacuation_results, chain_df, junctions)
        results["recommendations"] = recommendations

        # 3. Risk Assessment
        st.info("⚠️ Assessing risks...")
        risk_results = self._assess_risks(trajectories, chain_df, junctions, r_outer_list)
        results["risk_assessment"] = risk_results

        # 4. Efficiency Metrics
        st.info("📊 Computing efficiency metrics...")
        efficiency_results = self._compute_efficiency_metrics(trajectories, chain_df, junctions, r_outer_list)
        results["efficiency_metrics"] = efficiency_results

        # Save enhanced analysis results to CSV files
        try:
            enhanced_data_dir = os.path.join("gui_outputs", "enhanced_analysis")
            os.makedirs(enhanced_data_dir, exist_ok=True)

            # Save evacuation analysis results
            if evacuation_results["bottlenecks"]:
                bottlenecks_df = pd.DataFrame(evacuation_results["bottlenecks"])
                bottlenecks_file = os.path.join(enhanced_data_dir, "evacuation_bottlenecks.csv")
                bottlenecks_df.to_csv(bottlenecks_file, index=False)
                st.info(f"📁 Evacuation bottlenecks saved to: {bottlenecks_file}")

            if evacuation_results["optimal_routes"]:
                optimal_routes_df = pd.DataFrame(evacuation_results["optimal_routes"])
                optimal_routes_file = os.path.join(enhanced_data_dir, "optimal_routes.csv")
                optimal_routes_df.to_csv(optimal_routes_file, index=False)
                st.info(f"📁 Optimal routes saved to: {optimal_routes_file}")

            # Save flow analysis results
            if evacuation_results["flow_analysis"]:
                flow_data = []
                for junction_key, flow_info in evacuation_results["flow_analysis"].items():
                    flow_data.append({
                        "junction": junction_key,
                        "total_trajectories": flow_info["total_trajectories"],
                        "entropy": flow_info["entropy"],
                        "branch_distribution": str(flow_info["branch_distribution"])
                    })
                flow_df = pd.DataFrame(flow_data)
                flow_file = os.path.join(enhanced_data_dir, "flow_analysis.csv")
                flow_df.to_csv(flow_file, index=False)
                st.info(f"📁 Flow analysis saved to: {flow_file}")

            # Save recommendations
            if recommendations:
                recommendations_df = pd.DataFrame(recommendations)
                recommendations_file = os.path.join(enhanced_data_dir, "recommendations.csv")
                recommendations_df.to_csv(recommendations_file, index=False)
                st.info(f"📁 Recommendations saved to: {recommendations_file}")

            # Save risk assessment results
            if risk_results["high_risk_junctions"]:
                risk_df = pd.DataFrame(risk_results["high_risk_junctions"])
                risk_file = os.path.join(enhanced_data_dir, "risk_assessment.csv")
                risk_df.to_csv(risk_file, index=False)
                st.info(f"📁 Risk assessment saved to: {risk_file}")

            # Save efficiency metrics
            if efficiency_results:
                efficiency_data = []
                for metric_name, metric_value in efficiency_results.items():
                    if isinstance(metric_value, dict):
                        for key, value in metric_value.items():
                            efficiency_data.append({
                                "metric_category": metric_name,
                                "metric_name": key,
                                "value": value
                            })
                    else:
                        efficiency_data.append({
                            "metric_category": "overall",
                            "metric_name": metric_name,
                            "value": metric_value
                        })

                if efficiency_data:
                    efficiency_df = pd.DataFrame(efficiency_data)
                    efficiency_file = os.path.join(enhanced_data_dir, "efficiency_metrics.csv")
                    efficiency_df.to_csv(efficiency_file, index=False)
                    st.info(f"📁 Efficiency metrics saved to: {efficiency_file}")

            # Save overall enhanced analysis summary
            summary_data = {
                "overall_risk_score": risk_results.get("overall_risk_score", 0),
                "total_bottlenecks": len(evacuation_results["bottlenecks"]),
                "total_optimal_routes": len(evacuation_results["optimal_routes"]),
                "total_recommendations": len(recommendations),
                "high_risk_junctions_count": len(risk_results.get("high_risk_junctions", [])),
                "analysis_timestamp": pd.Timestamp.now().isoformat()
            }

            summary_df = pd.DataFrame([summary_data])
            summary_file = os.path.join(enhanced_data_dir, "enhanced_analysis_summary.csv")
            summary_df.to_csv(summary_file, index=False)
            st.info(f"📁 Enhanced analysis summary saved to: {summary_file}")

        except Exception as e:
            st.warning(f"⚠️ Could not save enhanced analysis files: {e}")

        return results

    def _analyze_evacuation_patterns(self, trajectories, chain_df, junctions, r_outer_list, centers_list):
        """Analyze evacuation patterns and identify bottlenecks"""
        import numpy as np
        import pandas as pd

        evacuation_results = {
            "bottlenecks": [],
            "optimal_routes": [],
            "flow_analysis": {},
            "capacity_analysis": {}
        }

        # Analyze flow patterns for each junction
        for i, junction in enumerate(junctions):
            branch_col = f"branch_j{i}"
            if branch_col in chain_df.columns:
                junction_assignments = chain_df[['trajectory', branch_col]].copy()
                junction_assignments = junction_assignments.rename(columns={branch_col: 'branch'})
                junction_assignments = junction_assignments[junction_assignments['branch'] >= 0]

                # Calculate branch flow rates
                branch_counts = junction_assignments['branch'].value_counts()
                total_trajectories = len(junction_assignments)

                # Identify bottlenecks (branches with high concentration)
                for branch, count in branch_counts.items():
                    concentration = count / total_trajectories
                    if concentration > 0.6:  # More than 60% use same route
                        evacuation_results["bottlenecks"].append({
                            "junction": i,
                            "branch": branch,
                            "concentration": concentration,
                            "trajectory_count": count,
                            "risk_level": "HIGH" if concentration > 0.8 else "MEDIUM"
                        })

                # Identify optimal routes (balanced distribution)
                if len(branch_counts) > 1:
                    entropy = -sum((count/total_trajectories) * np.log2(count/total_trajectories)
                                for count in branch_counts.values)
                    max_entropy = np.log2(len(branch_counts))
                    balance_ratio = entropy / max_entropy

                    if balance_ratio > 0.7:  # Well-balanced distribution
                        evacuation_results["optimal_routes"].append({
                            "junction": i,
                            "balance_ratio": balance_ratio,
                            "entropy": entropy,
                            "branch_count": len(branch_counts)
                        })

                evacuation_results["flow_analysis"][f"junction_{i}"] = {
                    "total_trajectories": total_trajectories,
                    "branch_distribution": branch_counts.to_dict(),
                    "entropy": entropy if len(branch_counts) > 1 else 0
                }

        return evacuation_results

    def _generate_recommendations(self, evacuation_results, chain_df, junctions):
        """Generate actionable recommendations based on analysis"""
        recommendations = []

        # Recommendations based on bottlenecks
        for bottleneck in evacuation_results["bottlenecks"]:
            if bottleneck["risk_level"] == "HIGH":
                recommendations.append({
                    "type": "Signage",
                    "priority": "HIGH",
                    "junction": bottleneck["junction"],
                    "message": f"Add directional signage at Junction {bottleneck['junction']} to distribute traffic away from Branch {int(bottleneck['branch'])} (currently {bottleneck['concentration']:.1%} of traffic)"
                })
                recommendations.append({
                    "type": "Route Modification",
                    "priority": "HIGH",
                    "junction": bottleneck["junction"],
                    "message": f"Consider widening or adding alternative routes at Junction {bottleneck['junction']} to reduce bottleneck risk"
                })

        # Recommendations based on optimal routes
        for route in evacuation_results["optimal_routes"]:
            recommendations.append({
                "type": "Maintenance",
                "priority": "LOW",
                "junction": route["junction"],
                "message": f"Junction {route['junction']} shows good traffic distribution (balance ratio: {route['balance_ratio']:.2f}) - maintain current design"
            })

        # General recommendations
        if len(evacuation_results["bottlenecks"]) > len(junctions) * 0.5:
            recommendations.append({
                "type": "System-wide",
                "priority": "MEDIUM",
                "junction": "ALL",
                "message": "High number of bottlenecks detected - consider system-wide evacuation route optimization"
            })

        return recommendations

    def _assess_risks(self, trajectories, chain_df, junctions, r_outer_list):
        """Assess potential safety risks in flow patterns"""
        risk_results = {
            "high_risk_junctions": [],
            "overall_risk_score": 0
        }

        total_risk_score = 0

        for i, junction in enumerate(junctions):
            branch_col = f"branch_j{i}"
            if branch_col in chain_df.columns:
                junction_assignments = chain_df[['trajectory', branch_col]].copy()
                junction_assignments = junction_assignments.rename(columns={branch_col: 'branch'})
                junction_assignments = junction_assignments[junction_assignments['branch'] >= 0]

                branch_counts = junction_assignments['branch'].value_counts()
                total_trajectories = len(junction_assignments)

                # Calculate unified risk factors
                risk_factors = []
                risk_score = 0.0

                # 1. Concentration Risk (0.0-1.0)
                if len(branch_counts) > 0:
                    max_concentration = branch_counts.max() / total_trajectories
                    if max_concentration > 0.7:
                        concentration_risk = (max_concentration - 0.7) / 0.3  # Scale 0.7-1.0 to 0.0-1.0
                        risk_factors.append(("high_concentration", concentration_risk))
                        risk_score += concentration_risk

                # 2. Diversity Risk (0.0-1.0)
                if len(branch_counts) < 2:
                    diversity_risk = 1.0
                    risk_factors.append(("low_diversity", diversity_risk))
                    risk_score += diversity_risk
                elif len(branch_counts) == 2:
                    diversity_risk = 0.3  # Moderate risk for only 2 routes
                    risk_factors.append(("limited_diversity", diversity_risk))
                    risk_score += diversity_risk

                # 3. Crowding Risk (0.0-1.0)
                if total_trajectories > 50:
                    if total_trajectories > 100:
                        crowding_risk = 1.0  # High crowding
                        risk_factors.append(("high_crowding", crowding_risk))
                    else:
                        crowding_risk = (total_trajectories - 50) / 50  # Scale 50-100 to 0.0-1.0
                        risk_factors.append(("moderate_crowding", crowding_risk))
                    risk_score += crowding_risk

                # Normalize total risk score to 0-1 scale
                # Max possible: 1.0 (concentration) + 1.0 (diversity) + 1.0 (crowding) = 3.0
                risk_score = min(risk_score / 3.0, 1.0)
                total_risk_score += risk_score

                # Classify risk level
                if risk_score >= 0.7:
                    risk_level = "HIGH"
                elif risk_score >= 0.4:
                    risk_level = "MEDIUM"
                else:
                    risk_level = "LOW"

                # Include all junctions with risk score >= 0.4 for comprehensive assessment
                if risk_score >= 0.4:
                    risk_results["high_risk_junctions"].append({
                        "junction": i,
                        "risk_score": risk_score,
                        "risk_level": risk_level,
                        "risk_factors": risk_factors,
                        "trajectory_count": total_trajectories,
                        "concentration": max_concentration if len(branch_counts) > 0 else 0,
                        "route_count": len(branch_counts)
                    })

        # Calculate overall risk score
        risk_results["overall_risk_score"] = total_risk_score / len(junctions) if junctions else 0

        return risk_results

    def _compute_efficiency_metrics(self, trajectories, chain_df, junctions, r_outer_list):
        """Compute efficiency metrics for navigation"""
        efficiency_results = {
            "average_travel_times": {},
            "route_efficiency": {},
            "capacity_utilization": {},
            "overall_efficiency": 0
        }

        total_efficiency = 0

        for i, junction in enumerate(junctions):
            branch_col = f"branch_j{i}"
            if branch_col in chain_df.columns:
                junction_assignments = chain_df[['trajectory', branch_col]].copy()
                junction_assignments = junction_assignments.rename(columns={branch_col: 'branch'})
                junction_assignments = junction_assignments[junction_assignments['branch'] >= 0]

                # Calculate efficiency metrics for this junction
                branch_counts = junction_assignments['branch'].value_counts()
                total_trajectories = len(junction_assignments)

                # Route efficiency (entropy-based)
                if len(branch_counts) > 1:
                    entropy = -sum((count/total_trajectories) * np.log2(count/total_trajectories)
                                for count in branch_counts.values)
                    max_entropy = np.log2(len(branch_counts))
                    route_efficiency = entropy / max_entropy
                else:
                    route_efficiency = 0

                # Capacity utilization
                capacity_utilization = total_trajectories / 100.0  # Assuming capacity of 100
                capacity_utilization = min(capacity_utilization, 1.0)  # Cap at 100%

                efficiency_results["route_efficiency"][f"junction_{i}"] = route_efficiency
                efficiency_results["capacity_utilization"][f"junction_{i}"] = capacity_utilization

                total_efficiency += route_efficiency

        # Calculate overall efficiency
        efficiency_results["overall_efficiency"] = total_efficiency / len(junctions) if junctions else 0

        return efficiency_results

    def render_visualization(self):
        """Render the visualization interface"""
        st.markdown('<h2 class="section-header">📈 Visualization</h2>', unsafe_allow_html=True)

        if not st.session_state.analysis_results:
            st.warning("⚠️ Please run an analysis first")
            return

        # Debug: Show what analysis results are available
        with st.expander("🔍 Debug: Available Analysis Results", expanded=False):
            if st.session_state.analysis_results is not None:
                st.write("Analysis results keys:", list(st.session_state.analysis_results.keys()))
            else:
                st.write("No analysis results available")

        # Show different visualizations based on analysis type
        # If multiple analysis types are available, let user choose
        # Prioritize "branches" as the default selection
        available_analyses = []

        # Add "branches" first if available (for default selection)
        if "branches" in st.session_state.analysis_results:
            available_analyses.append("branches")

        # Add other analysis types
        if st.session_state.analysis_results is not None:
            if "metrics" in st.session_state.analysis_results:
                available_analyses.append("metrics")
            if "assignments" in st.session_state.analysis_results:
                available_analyses.append("assignments")
        if "predictions" in st.session_state.analysis_results:
            available_analyses.append("predictions")
        if "choice_patterns" in st.session_state.analysis_results:
            available_analyses.append("choice_patterns")
        if "intent_recognition" in st.session_state.analysis_results:
            available_analyses.append("intent_recognition")
        if "enhanced" in st.session_state.analysis_results:
            available_analyses.append("enhanced")
        if "gaze_results" in st.session_state.analysis_results:
            available_analyses.append("gaze_results")

        if len(available_analyses) > 1:
            # Multiple analysis types available - let user choose
            st.markdown("### Multiple Analysis Results Available")
            selected_analysis = st.selectbox(
                "Choose analysis to visualize:",
                available_analyses,
                help="Select which analysis results to display"
            )

            if selected_analysis == "metrics":
                self.render_metrics_visualizations()
            elif selected_analysis == "assignments":
                self.render_assign_visualizations()
            elif selected_analysis == "branches":
                self.render_discover_visualizations()
            elif selected_analysis == "predictions":
                self.render_predict_visualizations()
            elif selected_analysis == "choice_patterns":
                self.render_flow_graphs()
                self.render_conditional_probabilities()
                self.render_pattern_analysis()
            elif selected_analysis == "intent_recognition":
                self.render_intent_visualizations()
            elif selected_analysis == "enhanced":
                self.render_enhanced_visualizations()
            elif selected_analysis == "gaze_results":
                self.render_gaze_visualizations()
        else:
            # Single analysis type - show automatically
            if st.session_state.analysis_results is not None:
                if "metrics" in st.session_state.analysis_results:
                    self.render_metrics_visualizations()
                elif "assignments" in st.session_state.analysis_results:
                    self.render_assign_visualizations()
                elif "branches" in st.session_state.analysis_results:
                    self.render_discover_visualizations()
                    # Also show flow graphs if available
                    if "flow_graph_map" in st.session_state.analysis_results:
                        self.render_flow_graphs()
                elif "predictions" in st.session_state.analysis_results:
                    self.render_predict_visualizations()
                elif "intent_recognition" in st.session_state.analysis_results:
                    self.render_intent_visualizations()
                elif "enhanced" in st.session_state.analysis_results:
                    self.render_enhanced_visualizations()
                elif "gaze_results" in st.session_state.analysis_results:
                    self.render_gaze_visualizations()
                else:
                    st.info("No visualizations available for this analysis type")
                    st.write("Available analysis results:", list(st.session_state.analysis_results.keys()))
            else:
                st.info("No analysis results available. Please run an analysis first.")

    def render_predict_visualizations(self):
        """Render predict analysis visualizations"""
        st.markdown("### Predict Analysis Results")

        # Check if predict analysis results exist
        if (st.session_state.analysis_results is None or
            "predictions" not in st.session_state.analysis_results):
            st.info("No predict analysis results available. Run predict analysis first.")
            return

        predictions_data = st.session_state.analysis_results["predictions"]

        # Display flow graphs
        st.markdown("#### Flow Graphs")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### Overall Flow Graph")
            flow_map_path = os.path.join("gui_outputs", "Flow_Graph_Map.png")
            if os.path.exists(flow_map_path):
                st.image(flow_map_path, width='stretch')
            else:
                st.info("Flow graph map not available")

        with col2:
            st.markdown("##### Per-Junction Flow Graph")
            per_junction_path = os.path.join("gui_outputs", "Per_Junction_Flow_Graph.png")
            if os.path.exists(per_junction_path):
                st.image(per_junction_path, width='stretch')
            else:
                st.info("Per-junction flow graph not available")

        # Display conditional probability heatmap
        st.markdown("#### Conditional Probability Analysis")
        heatmap_path = os.path.join("gui_outputs", "conditional_probability_heatmap.png")
        if os.path.exists(heatmap_path):
            st.image(heatmap_path, width='stretch')
        else:
            st.info("Conditional probability heatmap not available")

        # Display behavioral pattern analysis
        st.markdown("#### Behavioral Pattern Distribution")
        pattern_path = os.path.join("gui_outputs", "behavioral_patterns.png")
        if os.path.exists(pattern_path):
            st.image(pattern_path, width='stretch')
        else:
            st.info("Behavioral pattern analysis not available")

        # Display summary statistics
        if "summary" in predictions_data:
            st.markdown("#### Analysis Summary")
            summary = predictions_data["summary"]

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Trajectories", summary.get("total_trajectories", 0))
            with col2:
                st.metric("Total Junctions", summary.get("total_junctions", 0))
            with col3:
                st.metric("Total Transitions", summary.get("total_transitions", 0))
            with col4:
                st.metric("Unique Patterns", summary.get("unique_patterns", 0))

        # Interactive Junction Prediction Tool
        st.markdown("#### Interactive Junction Prediction")
        self._render_interactive_prediction_tool(predictions_data)

    def _render_interactive_prediction_tool(self, predictions_data):
        """Render interactive junction prediction tool"""
        if "conditional_probabilities" not in predictions_data:
            st.info("No conditional probability data available for predictions.")
            return

        conditional_probs = predictions_data["conditional_probabilities"]

        # Get available junctions from the conditional probabilities
        available_junctions = []
        for origin_key in conditional_probs.keys():
            junction_num = int(origin_key.split('_')[1][1:])  # Extract from "from_J0"
            available_junctions.append(junction_num)

        available_junctions = sorted(set(available_junctions))

        if not available_junctions:
            st.info("No junction data available for predictions.")
            return

        st.markdown("Select a decision junction and analyze probabilities for connected junctions:")

        col1, col2 = st.columns(2)

        with col1:
            # Decision junction selection
            decision_junction = st.selectbox(
                "Decision Junction",
                options=available_junctions,
                format_func=lambda x: f"J{x}",
                key="prediction_decision_junction"
            )

        with col2:
            # Direction selection
            direction = st.selectbox(
                "Analysis Direction",
                options=["Predecessor Analysis", "Successor Analysis"],
                key="prediction_direction"
            )

        # Get connected junctions based on direction
        if direction == "Predecessor Analysis":
            # Find junctions that lead TO the decision junction
            connected_junctions = []
            for origin_key, destinations in conditional_probs.items():
                origin_num = int(origin_key.split('_')[1][1:])
                if f"J{decision_junction}" in destinations:
                    connected_junctions.append(origin_num)

            if not connected_junctions:
                st.info(f"No predecessors found for J{decision_junction}")
                return

            # Predecessor selection
            predecessor = st.selectbox(
                "Select Predecessor Junction",
                options=sorted(connected_junctions),
                format_func=lambda x: f"J{x}",
                key="prediction_predecessor"
            )

            # Calculate probabilities
            self._calculate_predecessor_probabilities(conditional_probs, decision_junction, predecessor)

        else:  # Successor Analysis
            # Find junctions that the decision junction leads TO
            origin_key = f"from_J{decision_junction}"
            if origin_key not in conditional_probs:
                st.info(f"No successors found for J{decision_junction}")
                return

            destinations = conditional_probs[origin_key]
            connected_junctions = [int(dest[1:]) for dest in destinations.keys()]

            if not connected_junctions:
                st.info(f"No successors found for J{decision_junction}")
                return

            # Successor selection
            successor = st.selectbox(
                "Select Successor Junction",
                options=sorted(connected_junctions),
                format_func=lambda x: f"J{x}",
                key="prediction_successor"
            )

            # Calculate probabilities
            self._calculate_successor_probabilities(conditional_probs, decision_junction, successor)

    def _calculate_predecessor_probabilities(self, conditional_probs, decision_junction, predecessor):
        """Calculate probabilities for predecessor analysis - what happens AFTER decision junction when coming FROM predecessor"""
        st.markdown(f"### Analysis: J{predecessor} → J{decision_junction} → ?")
        st.markdown(f"**Question**: What junctions do trajectories visit AFTER J{decision_junction} when they came FROM J{predecessor}? (excluding self-loops)**")

        # Get cached sequences from analysis results
        if "cached_sequences" not in st.session_state.analysis_results.get("predictions", {}):
            st.error("No trajectory sequence data available. Please rerun the predict analysis.")
            return

        cached_sequences = st.session_state.analysis_results["predictions"]["cached_sequences"]

        # Find trajectories that follow the J{predecessor} → J{decision_junction} sequence
        relevant_trajectories = []
        successor_counts = {}

        for traj_idx, sequence in cached_sequences.items():
            # Check if this trajectory follows the predecessor → decision sequence
            for i in range(len(sequence) - 1):
                if sequence[i] == predecessor and sequence[i + 1] == decision_junction:
                    relevant_trajectories.append(traj_idx)

                    # Find what happens after the decision junction (only count different junctions)
                    if i + 2 < len(sequence):  # There's a junction after the decision junction
                        successor = sequence[i + 2]
                        # Only count as successor if it's a different junction (no self-loops)
                        if successor != decision_junction:
                            successor_counts[successor] = successor_counts.get(successor, 0) + 1
                    break

        if not relevant_trajectories:
            st.info(f"No trajectories found that follow the J{predecessor} → J{decision_junction} sequence")
            return

        total_trajectories = len(relevant_trajectories)
        st.markdown(f"**Found {total_trajectories} trajectories that follow J{predecessor} → J{decision_junction}**")

        if not successor_counts:
            st.info(f"None of these trajectories continue to another junction after J{decision_junction}")
            return

        # Calculate probabilities
        successor_probs = {}
        for successor, count in successor_counts.items():
            prob = (count / total_trajectories) * 100
            successor_probs[f"J{successor}"] = prob

        # Create visualization
        import matplotlib.pyplot as plt
        import pandas as pd

        successor_names = list(successor_probs.keys())
        probabilities = list(successor_probs.values())

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(successor_names, probabilities, color='skyblue', alpha=0.7)

        ax.set_xlabel('Successor Junction')
        ax.set_ylabel('Probability (%)')
        ax.set_title(f'Direct Successor Probabilities: J{predecessor} → J{decision_junction} → ?')
        ax.set_ylim(0, max(probabilities) * 1.1)

        # Add value labels on bars
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{prob:.1f}%', ha='center', va='bottom')

        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        # Show detailed table
        st.markdown("#### Detailed Analysis:")
        prob_data = []
        for successor, prob in successor_probs.items():
            count = successor_counts[int(successor[1:])]
            prob_data.append({
                'Successor Junction': successor,
                'Trajectory Count': count,
                'Probability (%)': f"{prob:.1f}",
                'Sequence': f"J{predecessor} → J{decision_junction} → {successor}"
            })

        df = pd.DataFrame(prob_data)
        st.dataframe(df, width='stretch')

        # Show trajectory examples
        st.markdown("#### Example Trajectory Sequences:")
        example_count = 0
        for traj_idx in relevant_trajectories[:5]:  # Show first 5 examples
            sequence = cached_sequences[traj_idx]
            seq_str = " → ".join([f"J{j}" for j in sequence])
            st.write(f"**Trajectory {traj_idx}**: {seq_str}")
            example_count += 1

        if len(relevant_trajectories) > 5:
            st.write(f"... and {len(relevant_trajectories) - 5} more trajectories")

    def _calculate_successor_probabilities(self, conditional_probs, decision_junction, successor):
        """Calculate probabilities for successor analysis - what happened BEFORE decision junction when going TO successor"""
        st.markdown(f"### Analysis: ? → J{decision_junction} → J{successor}")
        st.markdown(f"**Question**: What junctions did trajectories visit BEFORE J{decision_junction} when they went TO J{successor}? (excluding self-loops)**")

        # Get cached sequences from analysis results
        if "cached_sequences" not in st.session_state.analysis_results.get("predictions", {}):
            st.error("No trajectory sequence data available. Please rerun the predict analysis.")
            return

        cached_sequences = st.session_state.analysis_results["predictions"]["cached_sequences"]

        # Find trajectories that follow the J{decision_junction} → J{successor} sequence
        relevant_trajectories = []
        predecessor_counts = {}

        for traj_idx, sequence in cached_sequences.items():
            # Check if this trajectory follows the decision → successor sequence
            for i in range(len(sequence) - 1):
                if sequence[i] == decision_junction and sequence[i + 1] == successor:
                    relevant_trajectories.append(traj_idx)

                    # Find what happened before the decision junction (only count different junctions)
                    if i > 0:  # There's a junction before the decision junction
                        predecessor = sequence[i - 1]
                        # Only count as predecessor if it's a different junction (no self-loops)
                        if predecessor != decision_junction:
                            predecessor_counts[predecessor] = predecessor_counts.get(predecessor, 0) + 1
                        else:
                            # Skip self-loops and look further back
                            j = i - 1
                            while j >= 0 and sequence[j] == decision_junction:
                                j -= 1
                            if j >= 0:  # Found a different junction
                                predecessor = sequence[j]
                                predecessor_counts[predecessor] = predecessor_counts.get(predecessor, 0) + 1
                    break

        if not relevant_trajectories:
            st.info(f"No trajectories found that follow the J{decision_junction} → J{successor} sequence")
            return

        total_trajectories = len(relevant_trajectories)
        st.markdown(f"**Found {total_trajectories} trajectories that follow J{decision_junction} → J{successor}**")

        if not predecessor_counts:
            st.info(f"None of these trajectories came from another junction before J{decision_junction}")
            return

        # Calculate probabilities
        predecessor_probs = {}
        for predecessor, count in predecessor_counts.items():
            prob = (count / total_trajectories) * 100
            predecessor_probs[f"J{predecessor}"] = prob

        # Create visualization
        import matplotlib.pyplot as plt
        import pandas as pd

        predecessor_names = list(predecessor_probs.keys())
        probabilities = list(predecessor_probs.values())

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(predecessor_names, probabilities, color='lightgreen', alpha=0.7)

        ax.set_xlabel('Predecessor Junction')
        ax.set_ylabel('Probability (%)')
        ax.set_title(f'Direct Predecessor Probabilities: ? → J{decision_junction} → J{successor}')
        ax.set_ylim(0, max(probabilities) * 1.1)

        # Add value labels on bars
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{prob:.1f}%', ha='center', va='bottom')

        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        # Show detailed table
        st.markdown("#### Detailed Analysis:")
        prob_data = []
        for predecessor, prob in predecessor_probs.items():
            count = predecessor_counts[int(predecessor[1:])]
            prob_data.append({
                'Predecessor Junction': predecessor,
                'Trajectory Count': count,
                'Probability (%)': f"{prob:.1f}",
                'Sequence': f"{predecessor} → J{decision_junction} → J{successor}"
            })

        df = pd.DataFrame(prob_data)
        st.dataframe(df, width='stretch')

        # Show trajectory examples
        st.markdown("#### Example Trajectory Sequences:")
        example_count = 0
        for traj_idx in relevant_trajectories[:5]:  # Show first 5 examples
            sequence = cached_sequences[traj_idx]
            seq_str = " → ".join([f"J{j}" for j in sequence])
            st.write(f"**Trajectory {traj_idx}**: {seq_str}")
            example_count += 1

        if len(relevant_trajectories) > 5:
            st.write(f"... and {len(relevant_trajectories) - 5} more trajectories")

    def render_flow_graphs(self):
        """Render flow graph visualizations"""
        st.markdown("### Flow Graphs")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Overall Flow Graph")
            if "flow_graph_map" in st.session_state.analysis_results:
                st.image(st.session_state.analysis_results["flow_graph_map"], width='stretch')

        with col2:
            st.markdown("#### Per-Junction Flow Graph")
            if "per_junction_flow_graph" in st.session_state.analysis_results:
                st.image(st.session_state.analysis_results["per_junction_flow_graph"], width='stretch')

    def render_discover_visualizations(self):
        """Render discover analysis visualizations"""
        st.markdown("### Discover Analysis Results")

        # Display decision intercepts for each junction
        for junction_key, branches_data in st.session_state.analysis_results["branches"].items():
            if junction_key == "chain_decisions":  # Skip the chain decisions data
                continue

            st.markdown(f"#### {junction_key.replace('_', ' ').title()}")

            # Show decision intercepts plot
            junction_num = junction_key.split('_')[1]
            junction_dir = os.path.join("gui_outputs", f"junction_{junction_num}")

            # Display available plots
            intercepts_path = os.path.join(junction_dir, "Decision_Intercepts.png")
            if os.path.exists(intercepts_path):
                st.image(intercepts_path, caption=f"Decision Intercepts - {junction_key}", width='stretch')
            else:
                st.warning(f"Decision intercepts plot not found for {junction_key}")

            # Check for other available plots that might be generated
            other_plots = [
                ("Decision_Map.png", "Decision Map"),
                ("Branch_Counts.png", "Branch Counts"),
                ("Branch_Directions.png", "Branch Directions")
            ]

            for plot_file, plot_name in other_plots:
                plot_path = os.path.join(junction_dir, plot_file)
                if os.path.exists(plot_path):
                    st.image(plot_path, caption=f"{plot_name} - {junction_key}", width='stretch')

            # Show branch summary
            if "summary" in branches_data and branches_data["summary"] is not None:
                st.markdown("**Branch Summary:**")
                st.dataframe(branches_data["summary"], width='stretch')

            # Show assignments preview
            if "assignments" in branches_data and branches_data["assignments"] is not None:
                st.markdown("**Branch Assignments (first 20):**")
                st.dataframe(branches_data["assignments"].head(20), width='stretch')

    def render_assign_visualizations(self):
        """Render assign analysis visualizations"""
        st.markdown("### Assign Analysis Results")

        # Display assignment results for each junction
        for junction_key, assignments_data in st.session_state.analysis_results["assignments"].items():
            st.markdown(f"#### {junction_key.replace('_', ' ').title()}")

            # Extract the actual assignments DataFrame from the nested structure
            if isinstance(assignments_data, dict) and "assignments" in assignments_data:
                assignments_df = assignments_data["assignments"]
            else:
                assignments_df = assignments_data

            # Show assignment data
            if assignments_df is not None and hasattr(assignments_df, 'head'):
                st.markdown("**Branch Assignments:**")
                st.dataframe(assignments_df.head(20), width='stretch')

                if len(assignments_df) > 20:
                    st.info(f"Showing first 20 of {len(assignments_df)} assignments")

                # Show assignment statistics
                if 'branch' in assignments_df.columns:
                    branch_counts = assignments_df['branch'].value_counts()
                    st.markdown("**Branch Distribution:**")
                    st.bar_chart(branch_counts)

                    # Show detailed statistics
                    st.markdown("**Assignment Statistics:**")
                    total_trajectories = len(assignments_df)
                    for branch, count in branch_counts.items():
                        percentage = (count / total_trajectories) * 100
                        st.write(f"- Branch {branch}: {count} trajectories ({percentage:.1f}%)")
            else:
                st.info(f"No assignment data available for {junction_key}")

    def render_metrics_visualizations(self):
        """Render metrics analysis visualizations"""
        st.markdown("### Metrics Analysis Results")

        metrics_data = st.session_state.analysis_results["metrics"]
        metrics_images = st.session_state.analysis_results.get("metrics_images", {})

        if metrics_data:
            # Convert to DataFrame for better display
            import pandas as pd
            df = pd.DataFrame(metrics_data)

            # Display metrics table
            st.markdown("**Trajectory Metrics:**")
            st.dataframe(df, width='stretch')

            # Create distribution visualizations (prefer pre-generated images)
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Total Time Distribution**")
                img = metrics_images.get("total_time_distribution.png")
                if img and os.path.exists(img):
                    st.image(img, width='stretch')
                else:
                    if 'total_time' in df.columns:
                        valid_times = df['total_time'].dropna()
                        if len(valid_times) > 0:
                            sorted_times = valid_times.sort_values().reset_index(drop=True)
                            st.bar_chart(sorted_times)
                            st.caption(f"Range: {sorted_times.min():.1f}s - {sorted_times.max():.1f}s")
                            st.caption(f"Mean: {sorted_times.mean():.1f}s, Median: {sorted_times.median():.1f}s")
                        else:
                            st.info("No valid time data available")

            with col2:
                st.markdown("**Average Speed Distribution**")
                img = metrics_images.get("average_speed_distribution.png")
                if img and os.path.exists(img):
                    st.image(img, width='stretch')
                else:
                    if 'average_speed' in df.columns:
                        valid_speeds = df['average_speed'].dropna()
                        if len(valid_speeds) > 0:
                            sorted_speeds = valid_speeds.sort_values().reset_index(drop=True)
                            st.bar_chart(sorted_speeds)
                            st.caption(f"Range: {sorted_speeds.min():.2f} - {sorted_speeds.max():.2f}")
                            st.caption(f"Mean: {sorted_speeds.mean():.2f}, Median: {sorted_speeds.median():.2f}")
                        else:
                            st.info("No valid speed data available")

            # Add distance visualization
            col3, col4 = st.columns(2)

            with col3:
                st.markdown("**Total Distance Distribution**")
                img = metrics_images.get("total_distance_distribution.png")
                if img and os.path.exists(img):
                    st.image(img, width='stretch')
                else:
                    if 'total_distance' in df.columns:
                        valid_distances = df['total_distance'].dropna()
                        if len(valid_distances) > 0:
                            sorted_distances = valid_distances.sort_values().reset_index(drop=True)
                            st.bar_chart(sorted_distances)
                            st.caption(f"Range: {sorted_distances.min():.1f} - {sorted_distances.max():.1f}")
                            st.caption(f"Mean: {sorted_distances.mean():.1f}, Median: {sorted_distances.median():.1f}")
                        else:
                            st.info("No valid distance data available")

            with col4:
                st.markdown("**Summary Statistics**")
                if len(df) > 0:
                    summary_stats = {
                        "Total Trajectories": len(df),
                        "Avg Total Time": f"{df['total_time'].mean():.2f}s" if 'total_time' in df.columns else "N/A",
                        "Avg Total Distance": f"{df['total_distance'].mean():.2f}" if 'total_distance' in df.columns else "N/A",
                        "Avg Speed": f"{df['average_speed'].mean():.2f}" if 'average_speed' in df.columns else "N/A"
                    }
                    for key, value in summary_stats.items():
                        st.metric(key, value)

            # Define junction columns early for use in speed analysis
            junction_cols = [col for col in df.columns if col.startswith('junction_') and col.endswith('_time')]

            # Speed analysis visualizations
            speed_cols = [col for col in df.columns if col.startswith('junction_') and col.endswith('_speed')]
            if speed_cols:
                st.markdown("### Junction Speed Analysis")

                # Create speed analysis summary
                speed_summary = []
                for col in speed_cols:
                    junction_num = col.split('_')[1]
                    speed_mode_col = f"junction_{junction_num}_speed_mode"
                    entry_speed_col = f"junction_{junction_num}_entry_speed"
                    exit_speed_col = f"junction_{junction_num}_exit_speed"
                    avg_transit_col = f"junction_{junction_num}_avg_transit_speed"

                    valid_speeds = df[col].dropna()
                    total_trajectories = len(df)
                    valid_count = len(valid_speeds)

                    if valid_count > 0:
                        speed_summary.append({
                            "Junction": f"Junction {junction_num}",
                            "Avg Speed Through": f"{valid_speeds.mean():.2f}",
                            "Std Speed Through": f"{valid_speeds.std():.2f}",
                            "Valid Count": valid_count,
                            "NaN Count": total_trajectories - valid_count
                        })

                if speed_summary:
                    speed_df = pd.DataFrame(speed_summary)
                    st.markdown("**Junction Speed Statistics:**")
                    st.dataframe(speed_df, width='stretch')

                    # One concise explanation above both diagrams
                    st.markdown("### Speed Analysis")
                    st.info("""
                    **Available Speed Metrics:** Entry (2–5 s before), Exit (2–5 s after), and Average Transit (inside junction).
                    Use the selector in the correlation plot to switch the speed metric.
                    """)

                    # Show correlation (left) and entry/exit bars (right) side-by-side
                    col_speed1, col_speed2 = st.columns(2)

                    with col_speed1:
                        st.markdown("**Speed vs Time Correlation**")
                        img = metrics_images.get("speed_vs_time_correlation.png")
                        if img and os.path.exists(img):
                            st.image(img, width='stretch')
                        else:
                            st.info("Correlation plot not available yet. Re-run metrics analysis to generate.")

                    with col_speed2:
                        st.markdown("**Entry vs Exit Speed Analysis**")
                        st.caption("**Entry Speed**: Average speed in 2-5 second window before entering junction")
                        st.caption("**Exit Speed**: Average speed in 2-5 second window after leaving junction")
                        img = metrics_images.get("entry_exit_speed_by_junction.png")
                        if img and os.path.exists(img):
                            st.image(img, width='stretch')
                        else:
                            st.info("Entry/Exit bar chart not available yet. Re-run metrics analysis to generate.")

                # Detailed speed metrics table
                st.markdown("### Detailed Speed Metrics")
                speed_detail_cols = [col for col in df.columns if 'speed' in col.lower()]
                if speed_detail_cols:
                    speed_detail_df = df[speed_detail_cols + ['trajectory_id', 'trajectory_tid']]
                    st.dataframe(speed_detail_df, width='stretch')

            # Junction-specific metrics if available
            if junction_cols:
                st.markdown("### Junction Timing Analysis")

                # Check for NaN values and provide explanation
                total_junction_measurements = len(df) * len(junction_cols)
                valid_junction_measurements = sum(len(df[col].dropna()) for col in junction_cols)
                nan_count = total_junction_measurements - valid_junction_measurements

                if nan_count > 0:
                    st.info(f"ℹ️ **Note**: {nan_count} out of {total_junction_measurements} junction timing measurements returned NaN. This typically means trajectories didn't pass through those junctions or timing couldn't be computed.")

                # Create junction timing summary
                junction_summary = []
                for col in junction_cols:
                    junction_num = col.split('_')[1]
                    mode_col = f"junction_{junction_num}_mode"
                    if mode_col in df.columns:
                        valid_times = df[col].dropna()
                        total_trajectories = len(df)
                        valid_count = len(valid_times)
                        nan_count_junction = total_trajectories - valid_count

                        if valid_count > 0:
                            junction_summary.append({
                                "Junction": f"Junction {junction_num}",
                                "Avg Time": f"{valid_times.mean():.2f}s",
                                "Std Time": f"{valid_times.std():.2f}s",
                                "Valid Count": valid_count,
                                "NaN Count": nan_count_junction
                            })
                        else:
                            junction_summary.append({
                                "Junction": f"Junction {junction_num}",
                                "Avg Time": "N/A",
                                "Std Time": "N/A",
                                "Valid Count": 0,
                                "NaN Count": total_trajectories
                            })

                if junction_summary:
                    junction_df = pd.DataFrame(junction_summary)
                    st.markdown("**Junction Statistics (Only trajectories that actually pass through each junction):**")
                    st.dataframe(junction_df, width='stretch')

                    # Junction timing visualization
                    st.markdown("**Junction Timing Comparison**")
                    img = metrics_images.get("junction_timing_comparison.png")
                    if img and os.path.exists(img):
                        st.image(img, width='stretch')
                    else:
                        st.info("Timing comparison chart not available yet. Re-run metrics analysis to generate.")

                    # Show individual junction timing distributions
                    st.markdown("**Individual Junction Timing Distributions**")
                    if metrics_images:
                        # display per-junction histograms if present
                        for name, path in sorted(metrics_images.items()):
                            if name.startswith("timing_distribution_J") and os.path.exists(path):
                                jlabel = name.replace("timing_distribution_", "").replace(".png", "")
                                st.markdown(f"**{jlabel.replace('_', ' ')}**")
                                st.image(path, width='stretch')
                    else:
                        for col in junction_cols:
                            junction_num = col.split('_')[1]
                            valid_times = df[col].dropna()
                            if len(valid_times) > 0:
                                st.markdown(f"**Junction {junction_num} Timing Distribution**")
                                sorted_times = valid_times.sort_values().reset_index(drop=True)
                                st.bar_chart(sorted_times)
                                st.caption(f"Range: {sorted_times.min():.2f}s - {sorted_times.max():.2f}s, Mean: {sorted_times.mean():.2f}s")


    def _analyze_movement_patterns_at_junction(self, trajectories, junction, r_outer, decision_mode, path_length, epsilon):
        """Analyze movement patterns at a junction for regular trajectories (without head tracking data)."""
        import numpy as np
        import pandas as pd

        results = []

        for traj_idx, trajectory in enumerate(trajectories):
            # Find decision point using the same logic as the gaze analysis
            if decision_mode == "radial" or (decision_mode == "hybrid" and r_outer > junction.r):
                # Use radial decision point
                decision_idx = self._find_radial_decision_point(trajectory, junction, r_outer)
            else:
                # Use path length decision point
                decision_idx = self._find_path_length_decision_point(trajectory, junction, path_length, epsilon)

            if decision_idx is None:
                # Fallback to nearest point to junction center
                decision_idx = self._find_nearest_to_center(trajectory, junction)

            if decision_idx is not None and decision_idx < len(trajectory.x):
                # Calculate movement direction at decision point with better edge case handling
                movement_yaw = np.nan
                if decision_idx > 0 and decision_idx < len(trajectory.x) - 1:
                    dx = trajectory.x[decision_idx + 1] - trajectory.x[decision_idx - 1]
                    dz = trajectory.z[decision_idx + 1] - trajectory.z[decision_idx - 1]
                    movement_magnitude = np.hypot(dx, dz)
                    if movement_magnitude > 1e-3:  # Increased threshold for numerical stability
                        movement_yaw = np.degrees(np.arctan2(dx, dz))

                # Calculate approach direction (direction from previous point to decision point)
                approach_yaw = np.nan
                if decision_idx > 0:
                    dx_approach = trajectory.x[decision_idx] - trajectory.x[decision_idx - 1]
                    dz_approach = trajectory.z[decision_idx] - trajectory.z[decision_idx - 1]
                    approach_magnitude = np.hypot(dx_approach, dz_approach)
                    if approach_magnitude > 1e-3:
                        approach_yaw = np.degrees(np.arctan2(dx_approach, dz_approach))

                # Calculate exit direction (direction from decision point to next point)
                exit_yaw = np.nan
                if decision_idx < len(trajectory.x) - 1:
                    dx_exit = trajectory.x[decision_idx + 1] - trajectory.x[decision_idx]
                    dz_exit = trajectory.z[decision_idx + 1] - trajectory.z[decision_idx]
                    exit_magnitude = np.hypot(dx_exit, dz_exit)
                    if exit_magnitude > 1e-3:
                        exit_yaw = np.degrees(np.arctan2(dx_exit, dz_exit))

                # Calculate distance from junction center
                distance_from_center = np.sqrt(
                    (trajectory.x[decision_idx] - junction.cx)**2 +
                    (trajectory.z[decision_idx] - junction.cz)**2
                )

                # Calculate trajectory length for context
                trajectory_length = len(trajectory.x)

                results.append({
                    "trajectory": traj_idx,
                    "junction": 0,  # Single junction analysis
                    "decision_idx": decision_idx,
                    "trajectory_length": trajectory_length,
                    "decision_ratio": decision_idx / trajectory_length if trajectory_length > 0 else 0,
                    "movement_yaw": movement_yaw,
                    "approach_yaw": approach_yaw,
                    "exit_yaw": exit_yaw,
                    "distance_from_center": distance_from_center,
                    "decision_x": trajectory.x[decision_idx],
                    "decision_z": trajectory.z[decision_idx],
                    "time_at_decision": self._safe_get_time_value(trajectory, decision_idx)
                })

        return pd.DataFrame(results)

    def _analyze_movement_patterns_optimized(self, trajectories, junction, r_outer, decision_mode, path_length, epsilon):
        """Optimized movement pattern analysis for a single junction."""
        import numpy as np
        import pandas as pd

        results = []

        # Pre-calculate junction center for efficiency
        jx, jz = junction.cx, junction.cz

        for traj_idx, trajectory in enumerate(trajectories):
            # Fast decision point detection
            decision_idx = self._find_decision_point_fast(trajectory, jx, jz, r_outer, decision_mode, path_length, epsilon)

            if decision_idx is not None and decision_idx < len(trajectory.x):
                # Calculate movement direction with optimized approach
                movement_yaw = np.nan
                if decision_idx > 0 and decision_idx < len(trajectory.x) - 1:
                    dx = trajectory.x[decision_idx + 1] - trajectory.x[decision_idx - 1]
                    dz = trajectory.z[decision_idx + 1] - trajectory.z[decision_idx - 1]
                    movement_magnitude = np.hypot(dx, dz)
                    if movement_magnitude > 1e-3:
                        movement_yaw = np.degrees(np.arctan2(dx, dz))

                # Calculate approach and exit directions
                approach_yaw = np.nan
                if decision_idx > 0:
                    dx_approach = trajectory.x[decision_idx] - trajectory.x[decision_idx - 1]
                    dz_approach = trajectory.z[decision_idx] - trajectory.z[decision_idx - 1]
                    approach_magnitude = np.hypot(dx_approach, dz_approach)
                    if approach_magnitude > 1e-3:
                        approach_yaw = np.degrees(np.arctan2(dx_approach, dz_approach))

                exit_yaw = np.nan
                if decision_idx < len(trajectory.x) - 1:
                    dx_exit = trajectory.x[decision_idx + 1] - trajectory.x[decision_idx]
                    dz_exit = trajectory.z[decision_idx + 1] - trajectory.z[decision_idx]
                    exit_magnitude = np.hypot(dx_exit, dz_exit)
                    if exit_magnitude > 1e-3:
                        exit_yaw = np.degrees(np.arctan2(dx_exit, dz_exit))

                # Calculate distance from junction center
                distance_from_center = np.sqrt(
                    (trajectory.x[decision_idx] - jx)**2 +
                    (trajectory.z[decision_idx] - jz)**2
                )

                # Calculate trajectory position metrics
                trajectory_length = len(trajectory.x)
                decision_ratio = decision_idx / trajectory_length if trajectory_length > 0 else 0

                results.append({
                    "trajectory": traj_idx,
                    "junction": 0,  # Single junction analysis
                    "decision_idx": decision_idx,
                    "trajectory_length": trajectory_length,
                    "decision_ratio": decision_ratio,
                    "movement_yaw": movement_yaw,
                    "approach_yaw": approach_yaw,
                    "exit_yaw": exit_yaw,
                    "distance_from_center": distance_from_center,
                    "decision_x": trajectory.x[decision_idx],
                    "decision_z": trajectory.z[decision_idx],
                    "time_at_decision": self._safe_get_time_value(trajectory, decision_idx)
                })

        return pd.DataFrame(results)

    def _find_decision_point_fast(self, trajectory, jx, jz, r_outer, decision_mode, path_length, epsilon):
        """Fast decision point detection optimized for performance."""
        import numpy as np

        # Vectorized distance calculation
        distances = np.sqrt((trajectory.x - jx)**2 + (trajectory.z - jz)**2)

        if decision_mode == "radial" or (decision_mode == "hybrid" and r_outer > 50.0):
            # Find first point within junction
            within_junction = distances <= r_outer
            if np.any(within_junction):
                return np.argmax(within_junction)  # First True value
        else:
            # Find closest point and search around it
            closest_idx = np.argmin(distances)
            search_window = min(int(path_length), len(trajectory.x) // 10, 50)
            start_idx = max(0, closest_idx - search_window)
            end_idx = min(len(trajectory.x), closest_idx + search_window)

            # Look for point within junction radius
            for i in range(start_idx, end_idx):
                if distances[i] <= 50.0 + epsilon:  # Use junction radius + epsilon
                    return i
            return closest_idx

        return None

    def _check_for_gaze_data(self, trajectories):
        """Check if trajectories have gaze/physiological data using unified model."""
        if not trajectories:
            return False

        # Use capability helpers from unified model
        from verta.verta_data_loader import has_gaze_data, has_physio_data, has_vr_headset_data

        # Check if ANY trajectory has gaze/physio capabilities
        has_gaze = any(has_gaze_data(traj) for traj in trajectories)
        has_physio = any(has_physio_data(traj) for traj in trajectories)
        has_vr = any(has_vr_headset_data(traj) for traj in trajectories)

        # Also check the first trajectory for debugging
        sample_traj = trajectories[0]
        sample_has_gaze = has_gaze_data(sample_traj)
        sample_has_physio = has_physio_data(sample_traj)
        sample_has_vr = has_vr_headset_data(sample_traj)

        st.write(f"- Debug _check_for_gaze_data: has_gaze={has_gaze}, has_physio={has_physio}, has_vr_headset={has_vr}")
        st.write(f"- Sample trajectory (first): has_gaze={sample_has_gaze}, has_physio={sample_has_physio}, has_vr_headset={sample_has_vr}")

        return has_gaze or has_physio or has_vr



    def _get_gaze_column_mappings(self):
        """Get gaze column mappings from session state."""
        return getattr(st.session_state, 'gaze_column_mappings', {})

    def _normalize_gaze_result_frames(self, results: dict) -> dict:
        """Normalize result column names so GUI plots find expected columns."""
        # Physiological: ensure heart_rate_change, pupil_change
        if 'physiological' in results and results['physiological'] is not None:
            phys = results['physiological']
            # Handle both DataFrames and lists (converted from DataFrames)
            if hasattr(phys, 'rename'):  # DataFrame
                phys = phys.rename(columns={
                    'hr_change': 'heart_rate_change',
                    'hr_delta': 'heart_rate_change',
                    'pupil_delta': 'pupil_change',
                    'pupil_dilation_change': 'pupil_change'
                })
                results['physiological'] = phys
            elif isinstance(phys, list):  # List of dictionaries (converted DataFrame)
                # Convert back to DataFrame, rename, then convert back to list
                import pandas as pd
                phys_df = pd.DataFrame(phys)
                phys_df = phys_df.rename(columns={
                    'hr_change': 'heart_rate_change',
                    'hr_delta': 'heart_rate_change',
                    'pupil_delta': 'pupil_change',
                    'pupil_dilation_change': 'pupil_change'
                })
                results['physiological'] = phys_df.to_dict('records')

        # Pupil dilation: ensure pupil_change
        if 'pupil_dilation' in results and results['pupil_dilation'] is not None:
            pup = results['pupil_dilation']
            if hasattr(pup, 'rename'):  # DataFrame
                pup = pup.rename(columns={
                    'pupil_delta': 'pupil_change',
                    'pupil_dilation_change': 'pupil_change'
                })
                results['pupil_dilation'] = pup
            elif isinstance(pup, list):  # List of dictionaries (converted DataFrame)
                import pandas as pd
                pup_df = pd.DataFrame(pup)
                pup_df = pup_df.rename(columns={
                    'pupil_delta': 'pupil_change',
                    'pupil_dilation_change': 'pupil_change'
                })
                results['pupil_dilation'] = pup_df.to_dict('records')

        # Head yaw: ensure head_yaw, yaw_difference, intercept_x, intercept_z
        if 'head_yaw' in results and results['head_yaw'] is not None:
            yaw = results['head_yaw']
            if hasattr(yaw, 'rename'):  # DataFrame
                yaw = yaw.rename(columns={
                    'yaw': 'head_yaw',
                    'delta_yaw': 'yaw_difference',
                    'gaze_movement_diff': 'yaw_difference',
                    'decision_x': 'intercept_x',
                    'decision_z': 'intercept_z'
                })
                results['head_yaw'] = yaw
            elif isinstance(yaw, list):  # List of dictionaries (converted DataFrame)
                import pandas as pd
                yaw_df = pd.DataFrame(yaw)
                yaw_df = yaw_df.rename(columns={
                    'yaw': 'head_yaw',
                    'delta_yaw': 'yaw_difference',
                    'gaze_movement_diff': 'yaw_difference',
                    'decision_x': 'intercept_x',
                    'decision_z': 'intercept_z'
                })
                results['head_yaw'] = yaw_df.to_dict('records')

        return results

    # (Removed) _plot_head_yaw_arrows_at_intercepts helper per request

    def _perform_gaze_analysis_with_mappings(self, trajectories, junction, r_outer, decision_mode, path_length, epsilon, linger_delta, out_dir, column_mappings, scale_factor=1.0):
        """Perform gaze analysis using column mappings with scaling support."""
        import pandas as pd
        import numpy as np
        from verta.verta_geometry import Circle

        # Filter trajectories to only include those with gaze or physiological data
        try:
            from verta.verta_data_loader import has_gaze_data as _has_gaze, has_physio_data as _has_physio
            filtered_trajectories = [t for t in trajectories if (_has_gaze(t) or _has_physio(t))]

            st.info(f"🔍 **Trajectory Filtering in Analysis:**")
            st.write(f"- Total trajectories: {len(trajectories)}")
            st.write(f"- Trajectories with gaze/physio data: {len(filtered_trajectories)}")

            if len(filtered_trajectories) < len(trajectories):
                skipped_count = len(trajectories) - len(filtered_trajectories)
                st.info(f"ℹ️ Skipped {skipped_count} trajectories without gaze/physiological data")

            if not filtered_trajectories:
                st.warning("⚠️ No trajectories with gaze/physiological data found")
                return None

            trajectories = filtered_trajectories

        except Exception as e:
            st.warning(f"⚠️ Error filtering trajectories: {e}")
            # Continue with original trajectories if filtering fails

        # Check if we have standard gaze data first
        has_gaze_data = self._check_for_gaze_data(trajectories)

        # Debug: Show which code path is being used
        st.info(f"🔍 **Gaze Analysis Code Path Debug:**")
        st.write(f"- Has gaze data: {has_gaze_data}")
        st.write(f"- Column mappings provided: {bool(column_mappings)}")
        if column_mappings:
            st.write(f"- Column mappings: {list(column_mappings.keys())}")

        # Debug: Check what data is actually available in trajectories
        if trajectories:
            sample_traj = trajectories[0]
            st.write(f"**Sample trajectory data availability:**")
            st.write(f"- head_forward_x: {hasattr(sample_traj, 'head_forward_x') and sample_traj.head_forward_x is not None}")
            st.write(f"- head_forward_z: {hasattr(sample_traj, 'head_forward_z') and sample_traj.head_forward_z is not None}")
            st.write(f"- gaze_x: {hasattr(sample_traj, 'gaze_x') and sample_traj.gaze_x is not None}")
            st.write(f"- gaze_y: {hasattr(sample_traj, 'gaze_y') and sample_traj.gaze_y is not None}")
            st.write(f"- pupil_l: {hasattr(sample_traj, 'pupil_l') and sample_traj.pupil_l is not None}")
            st.write(f"- pupil_r: {hasattr(sample_traj, 'pupil_r') and sample_traj.pupil_r is not None}")
            st.write(f"- heart_rate: {hasattr(sample_traj, 'heart_rate') and sample_traj.heart_rate is not None}")

            if hasattr(sample_traj, 'pupil_l') and sample_traj.pupil_l is not None:
                st.write(f"- pupil_l length: {len(sample_traj.pupil_l)}")
                st.write(f"- pupil_l sample: {sample_traj.pupil_l[:5] if len(sample_traj.pupil_l) > 5 else sample_traj.pupil_l}")
            if hasattr(sample_traj, 'heart_rate') and sample_traj.heart_rate is not None:
                st.write(f"- heart_rate length: {len(sample_traj.heart_rate)}")
                st.write(f"- heart_rate sample: {sample_traj.heart_rate[:5] if len(sample_traj.heart_rate) > 5 else sample_traj.heart_rate}")

        if has_gaze_data:
            st.write("**→ Using comprehensive gaze analysis (with enhanced debugging)**")
            # Apply scaling to trajectories AND junction if needed
            if scale_factor != 1.0:
                st.info(f"🔧 Applying scale factor {scale_factor} to trajectory coordinates and junction...")
                scaled_trajectories = []
                for traj in trajectories:
                    # Create a copy with scaled coordinates
                    scaled_traj = Trajectory(
                        tid=traj.tid,
                        x=traj.x * scale_factor,
                        z=traj.z * scale_factor,
                        t=traj.t,
                        head_forward_x=traj.head_forward_x,
                        head_forward_z=traj.head_forward_z,
                        gaze_x=traj.gaze_x,
                        gaze_y=traj.gaze_y,
                        pupil_l=traj.pupil_l,
                        pupil_r=traj.pupil_r,
                        heart_rate=traj.heart_rate
                    )
                    scaled_trajectories.append(scaled_traj)
                trajectories = scaled_trajectories

                # Scale the junction coordinates to match scaled trajectories
                scaled_junction = Circle(
                    cx=junction.cx * scale_factor,
                    cz=junction.cz * scale_factor,
                    r=junction.r * scale_factor
                )
                junction = scaled_junction
            else:
                # Junction coordinates are correct and should NOT be scaled
                # The issue is that trajectories might be getting double-scaled somewhere
                st.info(f"🔧 Using original junction coordinates (no scaling needed)")
                st.write(f"- Junction coordinates: ({junction.cx}, {junction.cz}), r={junction.r}")

            # Call the comprehensive gaze analysis function
            return self._perform_comprehensive_gaze_analysis(
                trajectories=trajectories,
                junction=junction,
                r_outer=r_outer,
                decision_mode=decision_mode,
                path_length=path_length,
                epsilon=epsilon,
                linger_delta=linger_delta,
                out_dir=out_dir,
                run_custom_discover=st.session_state.get('run_custom_discover', False)
            )

        elif column_mappings:
            st.write("**→ Using custom gaze analysis (with column mappings)**")
            # Use custom gaze analysis with column mappings
            gaze_data = self._perform_custom_gaze_analysis(
                trajectories, junction, r_outer, decision_mode, path_length, epsilon, out_dir, column_mappings, scale_factor=1.0
            )
        else:
            # Check if we have any physiological data (even if incomplete)
            has_any_physio = any(
                (hasattr(traj, 'pupil_l') and traj.pupil_l is not None) or
                (hasattr(traj, 'pupil_r') and traj.pupil_r is not None) or
                (hasattr(traj, 'heart_rate') and traj.heart_rate is not None)
                for traj in trajectories
            )

            if has_any_physio:
                st.write("**→ Using physiological-only analysis (incomplete gaze data)**")
                # Try to perform analysis with whatever physiological data we have
                return self._perform_comprehensive_gaze_analysis(
                    trajectories=trajectories,
                    junction=junction,
                    r_outer=r_outer,
                    decision_mode=decision_mode,
                    path_length=path_length,
                    epsilon=epsilon,
                    linger_delta=linger_delta,
                    out_dir=out_dir,
                    run_custom_discover=st.session_state.get('run_custom_discover', False)
                )
            else:
                st.error("❌ No gaze data, physiological data, or column mappings available")
            return None

        # Debug: Check trajectory types
        st.info(f"🔍 **Trajectory Type Debug:**")
        if trajectories:
            sample_traj = trajectories[0]
            st.write(f"- Sample trajectory type: {type(sample_traj).__name__}")
            st.write(f"- Sample trajectory ID: {sample_traj.tid}")
            st.write(f"- Has gaze_x: {hasattr(sample_traj, 'gaze_x') and sample_traj.gaze_x is not None}")
            st.write(f"- Has heart_rate: {hasattr(sample_traj, 'heart_rate') and sample_traj.heart_rate is not None}")
            st.write(f"- Has pupil_l: {hasattr(sample_traj, 'pupil_l') and sample_traj.pupil_l is not None}")

        # CRITICAL FIX: Use the same trajectory objects that were used for discover analysis
        # to ensure trajectory IDs match between assignments and gaze analysis
        if "branches" in st.session_state.analysis_results and st.session_state.analysis_results["branches"]:
            st.info("🔧 **Using trajectory objects from discover analysis to ensure ID consistency**")
            # Use the original trajectories that were used for discover analysis
            discover_trajectories = st.session_state.trajectories
            st.write(f"- Discover trajectories: {len(discover_trajectories)}")
            st.write(f"- Gaze trajectories: {len(trajectories)}")

            # DEBUG: Check coordinate ranges
            if discover_trajectories and trajectories:
                sample_discover = discover_trajectories[0]
                sample_gaze = trajectories[0]
                st.error("🔍 **COORDINATE SYSTEM DEBUG:**")
                st.write(f"- Discover trajectory {sample_discover.tid}: X range {np.min(sample_discover.x):.1f} to {np.max(sample_discover.x):.1f}")

                # Check if gaze trajectory has valid coordinates
                if np.all(np.isnan(sample_gaze.x)):
                    st.error(f"❌ **CRITICAL ISSUE:** Gaze trajectory {sample_gaze.tid} has ALL NaN coordinates!")
                    st.write("**This explains why arrows are not visible - trajectory coordinates are invalid!**")
                    st.write("**Root cause:** Gaze trajectories were loaded BEFORE NaN handling was added")

                    # Show sample data for debugging
                    st.write(f"**Sample gaze trajectory data:**")
                    st.write(f"- X values: {sample_gaze.x[:5]} (first 5)")
                    st.write(f"- Z values: {sample_gaze.z[:5]} (first 5)")
                    st.write(f"- X type: {type(sample_gaze.x)}")
                    st.write(f"- Z type: {type(sample_gaze.z)}")

                    # Provide solution
                    st.error("🔧 **SOLUTION:** Reload gaze data to apply NaN handling")
                    st.write("**Steps to fix:**")
                    st.write("1. Go to 'Data Loader' tab")
                    st.write("2. Re-upload or reload your gaze trajectory files")
                    st.write("3. The new NaN handling will clean the coordinate data")
                    st.write("4. Run gaze analysis again")
                else:
                    st.write(f"- Gaze trajectory {sample_gaze.tid}: X range {np.min(sample_gaze.x):.1f} to {np.max(sample_gaze.x):.1f}")

                    # Check if coordinates are in the same scale
                    discover_scale = np.max(sample_discover.x) / np.max(sample_gaze.x) if np.max(sample_gaze.x) > 0 else 1
                    st.write(f"- Coordinate scale ratio (discover/gaze): {discover_scale:.2f}")
                    if abs(discover_scale - 1.0) > 0.1:
                        st.error(f"❌ **COORDINATE MISMATCH DETECTED!** Scale ratio: {discover_scale:.2f}")
                        st.write("**This explains why arrows are not visible - they're in different coordinate systems!**")

                st.write(f"- Junction coordinates: ({junction.cx}, {junction.cz})")

            # Check if we can convert discover trajectories to Trajectory objects
            if True:
                st.write("🔄 Using unified Trajectory objects (IDs already consistent)...")
                # Create a mapping from trajectory ID to trajectories currently loaded
                gaze_traj_map = {gt.tid: gt for gt in trajectories}
                # Create a mapping for discover trajectories to allow coordinate repair when needed
                discover_traj_map = {dt.tid: dt for dt in discover_trajectories}

                # Convert discover trajectories to Trajectory objects where possible
                converted_trajectories = []
                for dt in discover_trajectories:
                    if dt.tid in gaze_traj_map:
                        gt = gaze_traj_map[dt.tid]
                        # Repair NaN coordinate issue by borrowing x/z/t from discover trajectory when needed
                        try:
                            needs_repair = (
                                gt.x is None or gt.z is None or
                                (hasattr(gt.x, 'shape') and hasattr(gt.z, 'shape') and
                                 (np.all(np.isnan(gt.x)) or np.all(np.isnan(gt.z))))
                            )
                        except Exception:
                            needs_repair = True
                        if needs_repair and dt.tid in discover_traj_map:
                            repaired = Trajectory(
                                tid=gt.tid,
                                x=np.asarray(discover_traj_map[dt.tid].x),
                                z=np.asarray(discover_traj_map[dt.tid].z),
                                t=getattr(discover_traj_map[dt.tid], 't', None),
                                head_forward_x=getattr(gt, 'head_forward_x', None),
                                head_forward_z=getattr(gt, 'head_forward_z', None),
                                gaze_x=getattr(gt, 'gaze_x', None),
                                gaze_y=getattr(gt, 'gaze_y', None),
                                pupil_l=getattr(gt, 'pupil_l', None),
                                pupil_r=getattr(gt, 'pupil_r', None),
                                heart_rate=getattr(gt, 'heart_rate', None),
                            )
                            converted_trajectories.append(repaired)
                        else:
                            converted_trajectories.append(gt)
                    else:
                        st.warning(f"⚠️ No Trajectory found for discover trajectory ID: {dt.tid}")

                st.write(f"- Converted trajectories: {len(converted_trajectories)}")
                trajectories = converted_trajectories
            else:
                st.warning("⚠️ No gaze trajectories available - using discover trajectories (may not have gaze data)")
                trajectories = discover_trajectories

        # Return the movement data as fallback (this should not be reached if has_gaze_data is True)
        return self._analyze_movement_patterns_optimized(trajectories, junction, r_outer, decision_mode, path_length, epsilon)


    def _perform_custom_gaze_analysis(self, trajectories, junction, r_outer, decision_mode, path_length, epsilon, column_mappings, scale_factor=1.0):
        """Perform gaze analysis using custom column mappings with scaling support."""
        import pandas as pd
        import numpy as np
        from verta.verta_geometry import Circle

        results = []

        # Apply scaling to junction coordinates if needed
        if scale_factor != 1.0:
            jx = junction.cx * scale_factor
            jz = junction.cz * scale_factor
            junction_radius = junction.r * scale_factor
        else:
            # Junction coordinates are correct and should NOT be scaled
            jx = junction.cx
            jz = junction.cz
            junction_radius = junction.r

        for traj_idx, trajectory in enumerate(trajectories):
            # Apply scaling to coordinates if needed
            if scale_factor != 1.0:
                scaled_x = trajectory.x * scale_factor
                scaled_z = trajectory.z * scale_factor
            else:
                scaled_x = trajectory.x
                scaled_z = trajectory.z

            # Find decision point using scaled coordinates
            decision_idx = self._find_decision_point_fast(trajectory, jx, jz, r_outer, decision_mode, path_length, epsilon)

            if decision_idx is not None and decision_idx < len(scaled_x):
                # Extract gaze data using column mappings
                gaze_data = self._extract_gaze_data_from_trajectory(trajectory, decision_idx, column_mappings)

                # Calculate trajectory position metrics
                trajectory_length = len(scaled_x)
                decision_ratio = decision_idx / trajectory_length if trajectory_length > 0 else 0

                # Calculate distance from junction center using scaled coordinates
                distance_from_center = np.sqrt(
                    (scaled_x[decision_idx] - jx)**2 +
                    (scaled_z[decision_idx] - jz)**2
                )

                results.append({
                    "trajectory": traj_idx,
                    "junction": 0,  # Single junction analysis
                    "decision_idx": decision_idx,
                    "trajectory_length": trajectory_length,
                    "decision_ratio": decision_ratio,
                    "distance_from_center": distance_from_center,
                    "decision_x": trajectory.x[decision_idx],
                    "decision_z": trajectory.z[decision_idx],
                    "time_at_decision": self._safe_get_time_value(trajectory, decision_idx),
                    # Gaze-specific data
                    "head_yaw": gaze_data.get('head_yaw', np.nan),
                    "gaze_x": gaze_data.get('gaze_x', np.nan),
                    "gaze_y": gaze_data.get('gaze_y', np.nan),
                    "pupil_l": gaze_data.get('pupil_l', np.nan),
                    "pupil_r": gaze_data.get('pupil_r', np.nan),
                    "heart_rate": gaze_data.get('heart_rate', np.nan),
                    "analysis_type": "gaze"
                })

        return pd.DataFrame(results)

    def _extract_gaze_data_from_trajectory(self, trajectory, decision_idx, column_mappings):
        """Extract gaze data from trajectory using column mappings."""
        import numpy as np

        gaze_data = {}

        # Map the standard gaze fields to the actual column names
        field_mappings = {
            'head_yaw': ['head_forward_x', 'head_forward_z'],
            'gaze_x': ['gaze_x'],
            'gaze_y': ['gaze_y'],
            'pupil_l': ['pupil_l'],
            'pupil_r': ['pupil_r'],
            'heart_rate': ['heart_rate']
        }

        for field, required_columns in field_mappings.items():
            try:
                if field == 'head_yaw':
                    # Calculate head yaw from forward direction
                    head_x_col = column_mappings.get('head_forward_x', '')
                    head_z_col = column_mappings.get('head_forward_z', '')

                    if head_x_col and head_z_col and hasattr(trajectory, head_x_col) and hasattr(trajectory, head_z_col):
                        head_x = getattr(trajectory, head_x_col)[decision_idx] if decision_idx < len(getattr(trajectory, head_x_col)) else np.nan
                        head_z = getattr(trajectory, head_z_col)[decision_idx] if decision_idx < len(getattr(trajectory, head_z_col)) else np.nan

                        if not (np.isnan(head_x) or np.isnan(head_z)):
                            gaze_data[field] = np.degrees(np.arctan2(head_x, head_z))
                        else:
                            gaze_data[field] = np.nan
                    else:
                        gaze_data[field] = np.nan

                else:
                    # For other fields, directly map the column
                    col_name = column_mappings.get(required_columns[0], '')
                    if col_name and hasattr(trajectory, col_name):
                        data_array = getattr(trajectory, col_name)
                        if decision_idx < len(data_array):
                            gaze_data[field] = data_array[decision_idx]
                        else:
                            gaze_data[field] = np.nan
                    else:
                        gaze_data[field] = np.nan

            except (IndexError, AttributeError, KeyError):
                gaze_data[field] = np.nan

        return gaze_data

    def _perform_comprehensive_gaze_analysis_all_junctions(self, trajectories, junctions, r_outer_list, decision_mode, path_length, epsilon, linger_delta, out_dir):
        """Perform comprehensive gaze analysis for all junctions at once."""
        import pandas as pd
        import numpy as np
        from verta.verta_gaze import compute_head_yaw_at_decisions, analyze_physiological_at_junctions, analyze_pupil_dilation_trajectory

        st.info("🔍 **Performing comprehensive gaze analysis for all junctions...**")

        # First check if we have existing branch assignments from previous discover analysis
        existing_assignments = None
        use_existing_assignments = False

        if "branches" in st.session_state.analysis_results:
            st.info("🔍 Found existing branch assignments from previous discover analysis!")

            # Try to get chain_decisions (contains all junctions' assignments)
            if "chain_decisions" in st.session_state.analysis_results["branches"]:
                existing_assignments = st.session_state.analysis_results["branches"]["chain_decisions"]
                st.write(f"🔍 **Chain decisions shape:** {existing_assignments.shape}")
                st.write(f"🔍 **Chain decisions columns:** {list(existing_assignments.columns)}")

                # Debug: Check what type of data we're getting
                if 'junction_index' in existing_assignments.columns:
                    st.error("❌ **ERROR: Found junction_index column in chain_decisions - this is decision points data, not branch assignments!**")
                    st.write("🔍 **This means the wrong data was stored as chain_decisions**")
                else:
                    st.success("✅ **No junction_index column found - this looks like proper branch assignments**")

                # Check if we have assignments for all junctions
                branch_cols = [col for col in existing_assignments.columns if col.startswith('branch_j')]
                st.write(f"🔍 **Found branch columns:** {branch_cols}")

                # If no branch_j columns found, check if we have junction_index column
                if len(branch_cols) == 0 and 'junction_index' in existing_assignments.columns:
                    st.info("🔍 **Found junction_index column - this appears to be decision points data, not branch assignments**")
                    st.write("**Solution:** Run '🔍 Discover Branches' analysis first to create proper branch assignments")
                    existing_assignments = None
                elif len(branch_cols) >= len(junctions):
                    use_existing_assignments = True
                    st.success(f"✅ **Found assignments for all {len(junctions)} junctions!**")
                else:
                    st.warning(f"⚠️ **Only found assignments for {len(branch_cols)} junctions, but have {len(junctions)} junctions**")
                    existing_assignments = None

        if existing_assignments is not None and use_existing_assignments:
            chain_df = existing_assignments

            # For multi-junction analysis, we'll pass decision points separately to each gaze function
            # This avoids the complex merging issues with junction-specific data
            decisions_chain_df = st.session_state.analysis_results.get("branches", {}).get("decision_points")
            if decisions_chain_df is not None and len(decisions_chain_df) > 0:
                st.info("🔗 **Found decision points - will use precomputed intercept coordinates**")
                st.write(f"🔍 **Decision points available:** {len(decisions_chain_df)} records")
            else:
                st.warning("⚠️ **No decision points found in session state - will calculate from scratch**")

            st.success(f"✅ Using existing assignments - found {len(chain_df)} assignments")
            st.info(f"🔍 **GAZE ANALYSIS PATH:** Using existing assignments from previous discover analysis")
        else:
            st.error("❌ **No existing assignments found!**")
            st.write("**Solution:** Run '🔍 Discover Branches' analysis first to create proper assignments")
            return {
                'head_yaw': pd.DataFrame(),
                'physiological': pd.DataFrame(),
                'pupil_dilation': pd.DataFrame(),
                'error': 'No assignments found'
            }

        # Preprocess trajectories to convert time values to numeric format
        processed_trajectories = []
        for traj in trajectories:
            if hasattr(traj, 't') and traj.t is not None:
                # Convert time values to numeric if they're strings
                if isinstance(traj.t[0], str):
                    try:
                        import pandas as pd
                        # Convert string time format to numeric seconds
                        numeric_times = []
                        for t_val in traj.t:
                            if isinstance(t_val, str):
                                # Parse time string like "00:00:17.425"
                                time_parts = t_val.split(':')
                                if len(time_parts) == 3:
                                    hours = float(time_parts[0])
                                    minutes = float(time_parts[1])
                                    seconds = float(time_parts[2])
                                    total_seconds = hours * 3600 + minutes * 60 + seconds
                                    numeric_times.append(total_seconds)
                                else:
                                    numeric_times.append(float(t_val))
                            else:
                                numeric_times.append(float(t_val))
                        traj.t = np.array(numeric_times)
                    except Exception as e:
                        st.warning(f"⚠️ Could not convert time values for trajectory {traj.tid}: {e}")
                        # Keep original time values
                processed_trajectories.append(traj)
            else:
                processed_trajectories.append(traj)

        # Merge decision points with assignments for each junction to avoid multi-junction merging issues
        chain_df_with_decisions = chain_df.copy()
        if decisions_chain_df is not None and len(decisions_chain_df) > 0:
            st.info("🔗 **Merging decision points with assignments per junction...**")
            try:
                from verta.verta_consistency import normalize_assignments

                # For each junction, merge its decision points
                # Ensure junction_index is numeric for comparison (do this once outside the loop)
                decisions_df_numeric = decisions_chain_df.copy()
                st.write(f"🔍 **Debug: Original junction_index values:** {decisions_df_numeric['junction_index'].unique()[:10]}")
                st.write(f"🔍 **Debug: Original junction_index types:** {[type(x) for x in decisions_df_numeric['junction_index'].unique()[:5]]}")

                decisions_df_numeric["junction_index"] = pd.to_numeric(decisions_df_numeric["junction_index"], errors='coerce')
                st.write(f"🔍 **Debug: After conversion junction_index values:** {decisions_df_numeric['junction_index'].unique()[:10]}")
                st.write(f"🔍 **Debug: NaN count after conversion:** {decisions_df_numeric['junction_index'].isna().sum()}")

                for junction_idx in range(len(junctions)):
                    # Filter decision points for this junction
                    junction_decisions = decisions_df_numeric[decisions_df_numeric["junction_index"] == junction_idx].copy()

                    if len(junction_decisions) > 0:
                        # Remove junction_index column to avoid filtering conflicts in normalize_assignments
                        junction_decisions_clean = junction_decisions.drop(columns=['junction_index']).copy()

                        # Debug: Check trajectory ID types
                        st.write(f"🔍 **Debug Junction {junction_idx}:**")
                        st.write(f"- Chain_df trajectory types: {[type(x) for x in chain_df['trajectory'].unique()[:5]]}")
                        st.write(f"- Junction_decisions trajectory types: {[type(x) for x in junction_decisions_clean['trajectory'].unique()[:5]]}")
                        st.write(f"- Chain_df trajectory values: {chain_df['trajectory'].unique()[:5]}")
                        st.write(f"- Junction_decisions trajectory values: {junction_decisions_clean['trajectory'].unique()[:5]}")

                        # Normalize assignments for this junction
                        chain_df_normalized, norm_report = normalize_assignments(
                            assignments_df=chain_df,
                            trajectories=processed_trajectories,
                            junctions=[junctions[junction_idx]],  # Single junction
                            current_junction_idx=None,  # Don't filter by junction since we already did
                            decisions_df=junction_decisions_clean,
                            prefer_decisions=True,
                            include_outliers=False,
                            strict=False,
                        )

                        # Merge the decision columns from this junction's normalized data
                        decision_cols = ['decision_idx', 'intercept_x', 'intercept_z']
                        for col in decision_cols:
                            if col in chain_df_normalized.columns:
                                # Create junction-specific column names
                                junction_col = f"{col}_j{junction_idx}"
                                chain_df_with_decisions[junction_col] = chain_df_normalized[col]

                # Check if any decision points were merged
                decision_cols_merged = [col for col in chain_df_with_decisions.columns if col.startswith('decision_idx_')]
                if decision_cols_merged:
                    st.success(f"✅ **Successfully merged decision points for {len(decision_cols_merged)} junctions**")

                    # Debug: Show coverage of decision points
                    for col in decision_cols_merged:
                        junction_num = col.split('_')[-1]
                        non_null_count = chain_df_with_decisions[col].notna().sum()
                        total_count = len(chain_df_with_decisions)
                        st.write(f"🔍 **Junction {junction_num}:** {non_null_count}/{total_count} trajectories have precomputed decision points")

                    chain_df = chain_df_with_decisions
                else:
                    st.warning("⚠️ **No decision points merged - will calculate from scratch**")

            except Exception as e:
                st.warning(f"⚠️ **Could not merge decision points:** {e}")
                st.write("**Will calculate decision points from scratch**")

        # Use the actual gaze analysis functions with proper assignments
        # Get decision mode and parameters from discover analysis if available
        discover_decision_mode = decision_mode  # Default fallback
        discover_path_length = path_length
        discover_epsilon = epsilon
        discover_linger_delta = linger_delta
        discover_r_outer = r_outer_list[0] if r_outer_list else None  # Default fallback

        if "branches" in st.session_state.analysis_results:
            # Try to get parameters from the first junction's stored data
            for junction_key, branch_data in st.session_state.analysis_results["branches"].items():
                if isinstance(branch_data, dict) and "decision_mode" in branch_data:
                    discover_decision_mode = branch_data["decision_mode"]
                    discover_path_length = branch_data.get("path_length", path_length)
                    discover_epsilon = branch_data.get("epsilon", epsilon)
                    discover_linger_delta = branch_data.get("linger_delta", linger_delta)
                    discover_r_outer = branch_data.get("r_outer", discover_r_outer)
                    st.info(f"🔧 **Using discover analysis parameters:** decision_mode={discover_decision_mode}, path_length={discover_path_length}, epsilon={discover_epsilon}, linger_delta={discover_linger_delta}, r_outer={discover_r_outer}")
                    break

        try:
            st.info("🔬 Analyzing head yaw data for all junctions...")
            with st.spinner("Processing head yaw data..."):
                head_yaw_df = compute_head_yaw_at_decisions(
                    trajectories=processed_trajectories,
                    junctions=junctions,
                    assignments_df=chain_df,
                    decision_mode=discover_decision_mode,  # Use discover decision mode
                    r_outer_list=r_outer_list,
                    path_length=discover_path_length,  # Use discover path length
                    epsilon=discover_epsilon,  # Use discover epsilon
                    linger_delta=discover_linger_delta,  # Use discover linger delta
                    base_index=0  # Start from 0 for all junctions
                )

            st.info("🔬 Analyzing physiological data for all junctions...")
            with st.spinner("Processing physiological data..."):
                physio_df = analyze_physiological_at_junctions(
                    trajectories=processed_trajectories,
                    junctions=junctions,
                    assignments_df=chain_df,
                    decision_mode=discover_decision_mode,  # Use discover decision mode
                    r_outer_list=r_outer_list,
                    path_length=discover_path_length,  # Use discover path length
                    epsilon=discover_epsilon,  # Use discover epsilon
                    linger_delta=discover_linger_delta,  # Use discover linger delta
                    physio_window=3.0,
                    base_index=0,
                )

            st.info("🔬 Analyzing pupil dilation trajectories for all junctions...")
            with st.spinner("Processing pupil dilation data..."):
                pupil_df = analyze_pupil_dilation_trajectory(
                    trajectories=processed_trajectories,
                    junctions=junctions,
                    assignments_df=chain_df,
                    decision_mode=discover_decision_mode,  # Use discover decision mode
                    r_outer_list=r_outer_list,
                    path_length=discover_path_length,  # Use discover path length
                    epsilon=discover_epsilon,  # Use discover epsilon
                    linger_delta=discover_linger_delta,  # Use discover linger delta
                    physio_window=3.0,
                    base_index=0,
                )

            # Generate pupil dilation heatmaps for all junctions
            st.info("🗺️ Generating pupil dilation heatmaps for all junctions...")
            with st.spinner("Creating spatial heatmaps..."):
                from verta.verta_gaze import create_per_junction_pupil_heatmap

                # Get heatmap parameters from session state
                cell_size = st.session_state.get('pupil_heatmap_cell_size', 3.0)
                normalization = st.session_state.get('pupil_heatmap_normalization', 'relative')

                # Generate heatmaps for all junctions
                all_heatmaps = create_per_junction_pupil_heatmap(
                    trajectories=processed_trajectories,
                    junctions=junctions,
                    r_outer_list=r_outer_list,
                    cell_size=cell_size,
                    normalization=normalization,
                    base_index=0  # Start from 0 for all junctions
                )

                st.write(f"🔍 **Generated heatmaps for {len(all_heatmaps)} junctions**")

            # Debug: Show results summary
            st.info(f"🔍 **Gaze Analysis Results Summary:**")
            st.write(f"- Head yaw records: {len(head_yaw_df)}")
            st.write(f"- Physiological records: {len(physio_df)}")
            st.write(f"- Pupil dilation records: {len(pupil_df)}")
            st.write(f"- Heatmaps generated: {len(all_heatmaps)}")

            if len(head_yaw_df) > 0:
                st.write(f"- Junctions with head yaw data: {sorted(head_yaw_df['junction'].unique())}")
            if len(physio_df) > 0:
                st.write(f"- Junctions with physiological data: {sorted(physio_df['junction'].unique())}")
            if len(pupil_df) > 0:
                st.write(f"- Junctions with pupil data: {sorted(pupil_df['junction'].unique())}")

            return {
                'head_yaw': head_yaw_df,
                'physiological': physio_df,
                'pupil_dilation': pupil_df,
                'pupil_heatmap_junction': all_heatmaps,  # Add heatmaps to results
                'junction': junctions[0],  # Reference junction
                'r_outer': r_outer_list[0]  # Reference r_outer
            }

        except Exception as e:
            st.error(f"❌ **Gaze analysis failed:** {e}")
            st.write(f"**Error type:** {type(e).__name__}")
            st.write(f"**Error message:** {str(e)}")

            # Show suggestions based on error type
            if "No assignments found" in str(e):
                st.info("💡 **Suggestion:** Run '🔍 Discover Branches' analysis first to create proper assignments")
            elif "trajectory" in str(e).lower():
                st.info("💡 **Suggestion:** Check if trajectories actually pass through the junctions")
            elif "column" in str(e).lower():
                st.info("💡 **Suggestion:** Check your gaze column mappings in the Data tab")

            return {
                'head_yaw': pd.DataFrame(),
                'physiological': pd.DataFrame(),
                'pupil_dilation': pd.DataFrame(),
                'error': str(e),
                'error_type': type(e).__name__
            }

    def _perform_comprehensive_gaze_analysis(self, trajectories, junction, r_outer, decision_mode, path_length, epsilon, linger_delta, out_dir, run_custom_discover=False):
        """Perform comprehensive gaze analysis using the actual gaze functions."""
        import pandas as pd
        import numpy as np
        from verta.verta_decisions import discover_decision_chain

        # Debug: Check r_outer parameter at function start
        st.write(f"🔍 **DEBUG: r_outer parameter received:** {r_outer} (type: {type(r_outer)})")

        # Get decision mode and parameters from discover analysis if available
        discover_decision_mode = decision_mode  # Default fallback
        discover_path_length = path_length
        discover_epsilon = epsilon
        discover_linger_delta = linger_delta

        if "branches" in st.session_state.analysis_results:
            # Try to get parameters from the first junction's stored data
            for junction_key, branch_data in st.session_state.analysis_results["branches"].items():
                if isinstance(branch_data, dict) and "decision_mode" in branch_data:
                    discover_decision_mode = branch_data["decision_mode"]
                    discover_path_length = branch_data.get("path_length", path_length)
                    discover_epsilon = branch_data.get("epsilon", epsilon)
                    discover_linger_delta = branch_data.get("linger_delta", linger_delta)
                    st.info(f"🔧 **Using discover analysis parameters:** decision_mode={discover_decision_mode}, path_length={discover_path_length}, epsilon={discover_epsilon}")
                    break

        # Always define r_outer_list upfront to avoid unbound local errors later
        r_outer_list = [r_outer] if r_outer is not None else [None]

        # First check if we have existing branch assignments from previous discover analysis
        existing_assignments = None
        use_existing_assignments = False

        if "branches" in st.session_state.analysis_results:
            st.info("🔍 Found existing branch assignments from previous discover analysis!")

            # Debug: Show available keys
            available_keys = list(st.session_state.analysis_results["branches"].keys())
            st.write(f"🔍 **Available branch keys:** {available_keys}")
            st.write(f"🔍 **Looking for junction:** ({junction.cx}, {junction.cz}, r={junction.r})")

            # First try to find junction-specific assignments (from recent discover analysis)
            junction_found = False
            for junction_key, junction_data in st.session_state.analysis_results["branches"].items():
                if junction_key.startswith("junction_") and isinstance(junction_data, dict):
                    junction_obj = junction_data.get("junction")
                    if junction_obj and junction_obj.cx == junction.cx and junction_obj.cz == junction.cz and junction_obj.r == junction.r:
                        st.success(f"✅ **Found junction-specific assignments for {junction_key}!**")
                        assignments_df = junction_data.get("assignments")
                        if assignments_df is not None and not assignments_df.empty:
                            st.write(f"🔍 **Assignments shape:** {assignments_df.shape}")
                            st.write(f"🔍 **Assignments columns:** {list(assignments_df.columns)}")
                            st.write(f"🔍 **Sample assignments:**")
                            st.write(assignments_df.head())

                            existing_assignments = assignments_df
                            use_existing_assignments = True
                            junction_found = True
                            break
                        else:
                            st.warning(f"⚠️ **No assignments found in {junction_key}**")

            # If no junction-specific assignments found, try chain_decisions as fallback
            if not junction_found and "chain_decisions" in st.session_state.analysis_results["branches"]:
                st.info("🔍 **No junction-specific assignments found, trying chain_decisions as fallback**")
                existing_assignments = st.session_state.analysis_results["branches"]["chain_decisions"]
                st.write(f"🔍 **Chain decisions shape:** {existing_assignments.shape}")
                st.write(f"🔍 **Chain decisions columns:** {list(existing_assignments.columns)}")

                # Check if this junction has assignments in the chain decisions
                try:
                    junction_index = next(i for i, j in enumerate(st.session_state.junctions)
                                          if j.cx == junction.cx and j.cz == junction.cz and j.r == junction.r)

                    # Look for junction-specific branch column
                    branch_col = f"branch_j{junction_index}"
                    if branch_col in existing_assignments.columns:
                        st.success(f"✅ **Found assignments for Junction {junction_index} in chain_decisions!**")
                        st.write(f"🔍 **Branch column:** {branch_col}")

                        # Filter to only trajectories with assignments for this junction
                        assigned_mask = existing_assignments[branch_col].notna() & (existing_assignments[branch_col] >= 0)
                        junction_assignments = existing_assignments[assigned_mask].copy()

                        st.write(f"🔍 **Junction {junction_index} assignments:** {len(junction_assignments)} trajectories")
                        if len(junction_assignments) > 0:
                            st.write(f"🔍 **Sample assignments:**")
                            st.write(junction_assignments[['trajectory', branch_col]].head())

                            # Set the found assignments
                            existing_assignments = junction_assignments
                            use_existing_assignments = True
                        else:
                            st.warning(f"⚠️ **No assignments found for Junction {junction_index} in chain_decisions**")
                            existing_assignments = None
                    else:
                        st.warning(f"⚠️ **Branch column {branch_col} not found in chain_decisions**")
                        st.write(f"Available branch columns: {[col for col in existing_assignments.columns if col.startswith('branch_j')]}")
                        existing_assignments = None
                        found_key = None

                except StopIteration:
                    st.warning("⚠️ **Could not find junction index in session state junctions**")
                    existing_assignments = None

            # If we found assignments, show summary
            if existing_assignments is not None and use_existing_assignments:
                try:
                    num_rows = len(existing_assignments)
                    st.success(f"✅ **Using existing assignments: {num_rows} trajectories**")
                except Exception:
                    st.success("✅ **Using existing assignments**")

                # Debug: Show sample of assignments data
                st.write(f"🔍 **Sample assignments data:**")
                if hasattr(existing_assignments, 'head'):
                    st.write(existing_assignments.head())
                st.write(f"🔍 **Assignments columns:** {list(existing_assignments.columns)}")

                # Check if assignments have the expected structure
                if 'branch' in existing_assignments.columns:
                    assigned_count = existing_assignments['branch'].notna().sum()
                    st.write(f"🔍 **Total assigned trajectories:** {assigned_count}")
                elif any(col.startswith('branch_j') for col in existing_assignments.columns):
                    branch_cols = [col for col in existing_assignments.columns if col.startswith('branch_j')]
                    st.write(f"🔍 **Found branch columns:** {branch_cols}")
                else:
                    st.warning("⚠️ **Unexpected assignments structure**")

            # If no existing assignments found, we'll need to run discover analysis
            if existing_assignments is None:
                st.info("🔍 **No existing assignments found - will run discover analysis**")
                use_existing_assignments = False
        else:
            st.info("🔍 **No existing branch assignments found - will run discover analysis**")
            use_existing_assignments = False

        # Use existing assignments if available and selected, otherwise run discover analysis
        chain_df = None

        # Use the current checkbox selections (fall back to session_state if not set)
        # CRITICAL: Respect user's choice first, then check for existing assignments
        user_wants_existing = st.session_state.get('use_existing_assignments', False)
        user_wants_custom = st.session_state.get('run_custom_discover', False)

        # If user explicitly chose custom parameters, ignore existing assignments
        if user_wants_custom:
            use_existing_assignments = False
            run_custom_discover = True
        else:
            use_existing_assignments = use_existing_assignments and user_wants_existing
            run_custom_discover = user_wants_custom

        # Debug: Show what parameters are being used
        st.info(f"🔍 **Parameter Source Debug:**")
        st.write(f"- Using existing assignments: {use_existing_assignments}")
        st.write(f"- Running custom discover: {run_custom_discover}")
        if run_custom_discover and 'custom_discover_params' in st.session_state:
            st.write(f"- Custom parameters available: {list(st.session_state.custom_discover_params.keys())}")
            st.write(f"- Custom decision mode: {st.session_state.custom_discover_params.get('decision_mode', 'not set')}")

            # Show actual parameter values being used
            st.write("🔧 **Custom Parameters Being Used:**")
            custom_params = st.session_state.custom_discover_params
            for param_name, param_value in custom_params.items():
                st.write(f"- {param_name}: {param_value}")

            # Check if parameters might be too restrictive
            st.write("🔍 **Parameter Restrictiveness Check:**")
            if custom_params.get('eps', 0.5) < 1.0:
                st.warning(f"⚠️ DBSCAN eps={custom_params.get('eps', 0.5)} might be too restrictive (try 1.0-2.0)")
            if custom_params.get('min_samples', 5) > 3:
                st.warning(f"⚠️ DBSCAN min_samples={custom_params.get('min_samples', 5)} might be too high (try 2-3)")
            if custom_params.get('path_length', 100.0) > 50.0:
                st.warning(f"⚠️ Path length={custom_params.get('path_length', 100.0)} might be too high (try 20-50)")
        else:
            st.write("- Using default parameters")

        if existing_assignments is not None and use_existing_assignments:
            # CRITICAL FIX: Ensure we're using the correct junction-specific assignments
            # If we're using chain_decisions, we need to extract the specific junction's assignments
            if "chain_decisions" in str(type(existing_assignments)) or "branch_j" in existing_assignments.columns:
                # This is the chain_decisions DataFrame - extract junction-specific assignments
                junction_index = next(i for i, j in enumerate(st.session_state.junctions)
                                    if j.cx == junction.cx and j.cz == junction.cz and j.r == junction.r)

                st.info(f"🔧 **Extracting Junction {junction_index} assignments from chain_decisions...**")

                # Create a junction-specific assignments DataFrame
                junction_assignments = existing_assignments[["trajectory"]].copy()

                # Add the junction-specific branch column
                branch_col = f"branch_j{junction_index}"
                if branch_col in existing_assignments.columns:
                    junction_assignments["branch"] = existing_assignments[branch_col]

                    # Add decision point columns if available
                    if "decision_idx" in existing_assignments.columns:
                        junction_assignments["decision_idx"] = existing_assignments["decision_idx"]
                    if "intercept_x" in existing_assignments.columns:
                        junction_assignments["intercept_x"] = existing_assignments["intercept_x"]
                    if "intercept_z" in existing_assignments.columns:
                        junction_assignments["intercept_z"] = existing_assignments["intercept_z"]

                    # Filter to only assigned trajectories
                    assigned_mask = junction_assignments["branch"].notna() & (junction_assignments["branch"] >= 0)
                    junction_assignments = junction_assignments[assigned_mask]

                    st.success(f"✅ **Extracted Junction {junction_index} assignments:** {len(junction_assignments)} trajectories")
                    st.write(f"🔍 **Branch column used:** {branch_col}")
                    st.write(f"🔍 **Sample assignments:**")
                    st.write(junction_assignments.head())

                    chain_df = junction_assignments
                else:
                    st.error(f"❌ **Branch column {branch_col} not found in chain_decisions!**")
                    st.write(f"Available branch columns: {[col for col in existing_assignments.columns if col.startswith('branch_j')]}")
                    chain_df = pd.DataFrame()  # Empty DataFrame
            else:
                # This is already a junction-specific assignments DataFrame
                chain_df = existing_assignments
                st.success(f"✅ Using existing assignments - found {len(chain_df)} assignments")

            st.info(f"🔍 **GAZE ANALYSIS PATH:** Using existing assignments from previous discover analysis")
        else:
            # Show discover analysis parameters
            st.info("🔍 Running discover analysis to get branch assignments...")
            st.info(f"🔍 **GAZE ANALYSIS PATH:** Running discover analysis (not using existing assignments)")

            # Add parameter adjustment suggestions
            st.warning("⚠️ **Low assignment rate detected!** Try these adjustments:")
            st.write("**Suggested parameter changes:**")
            st.write("- **Path length**: Try reducing from 100.0 to 20.0-50.0")
            st.write("- **Decision mode**: Try 'hybrid' instead of 'pathlen'")
            st.write("- **Junction position/radius**: Verify they match your trajectory data")
            st.write("")
            st.write("**Or run '🔍 Discover Branches' analysis first to create proper assignments!**")

            try:
                # Use custom parameters if provided
                if run_custom_discover and 'custom_discover_params' in st.session_state:
                    custom_params = st.session_state.custom_discover_params

                    # Update parameters with custom values
                    cluster_method = custom_params.get('cluster_method', 'kmeans')
                    seed = custom_params.get('seed', 0)
                    decision_mode = custom_params.get('decision_mode', 'hybrid')

                    # Decision mode specific parameters
                    if decision_mode == "radial":
                        # Don't override r_outer - use the value passed to the function
                        # r_outer = custom_params.get('r_outer', 50.0)  # REMOVED: This was overriding the correct value
                        epsilon = custom_params.get('epsilon', 0.05)
                        path_length = None
                        linger_delta = None
                    elif decision_mode == "pathlen":
                        path_length = custom_params.get('path_length', 100.0)
                        linger_delta = custom_params.get('linger_delta', 0.0)
                        # Don't override r_outer - use the value passed to the function
                        # r_outer = None  # REMOVED: This was causing the TypeError
                        epsilon = custom_params.get('epsilon', 0.05)  # Provide default epsilon for pathlen mode
                    elif decision_mode == "hybrid":
                        # Don't override r_outer - use the value passed to the function
                        # r_outer = custom_params.get('r_outer', 50.0)  # REMOVED: This was overriding the correct value
                        path_length = custom_params.get('path_length', 100.0)
                        epsilon = custom_params.get('epsilon', 0.05)  # Hybrid mode needs epsilon for DBSCAN
                        linger_delta = custom_params.get('linger_delta', 0.0)

                    # Cluster method specific parameters
                    if cluster_method == "dbscan":
                        # Use epsilon parameter for DBSCAN (not eps)
                        epsilon = custom_params.get('eps', 0.5)  # Map eps to epsilon
                        min_samples = custom_params.get('min_samples', 5)
                        angle_eps = custom_params.get('angle_eps', 15.0)
                        # DBSCAN doesn't use k parameters, but discover_decision_chain expects them
                        k = 3  # Default value for discover_decision_chain
                        k_min = 2  # Default value for discover_decision_chain
                        k_max = 6  # Default value for discover_decision_chain
                        min_sep_deg = 12.0  # Default value for discover_decision_chain

                        # Ensure no None values are passed to discover_decision_chain
                        if epsilon is None:
                            epsilon = 0.5
                        if min_samples is None:
                            min_samples = 5
                        if angle_eps is None:
                            angle_eps = 15.0

                        # Debug: Show DBSCAN parameters
                        st.write(f"**DBSCAN Parameters:** epsilon={epsilon}, min_samples={min_samples}, angle_eps={angle_eps}")
                    elif cluster_method == "kmeans":
                        k = custom_params.get('k', 3)
                        k_min = custom_params.get('k_min', 2)
                        k_max = custom_params.get('k_max', 6)
                        eps = None
                        min_samples = None
                        angle_eps = None
                        min_sep_deg = None

                        # Ensure no None values are passed to discover_decision_chain
                        if k is None:
                            k = 3
                        if k_min is None:
                            k_min = 2
                        if k_max is None:
                            k_max = 6
                    elif cluster_method == "auto":
                        k_min = custom_params.get('k_min', 2)
                        k_max = custom_params.get('k_max', 6)
                        min_sep_deg = custom_params.get('min_sep_deg', 12.0)
                        angle_eps = custom_params.get('angle_eps', 15.0)
                        eps = None
                        min_samples = None
                        k = None

                        # Ensure no None values are passed to discover_decision_chain
                        if k_min is None:
                            k_min = 2
                        if k_max is None:
                            k_max = 6
                        if min_sep_deg is None:
                            min_sep_deg = 12.0
                        if angle_eps is None:
                            angle_eps = 15.0

                    st.info(f"🔧 Using custom parameters: cluster_method={cluster_method}, decision_mode={decision_mode}, seed={seed}")
                else:
                    # Use default parameters
                    cluster_method = "kmeans"
                    seed = 0
                    decision_mode = "hybrid"
                    # Use the r_outer value defined in the junctions tab
                    # Don't override it with hardcoded logic
                    path_length = 100.0
                    epsilon = 0.05  # Default epsilon for hybrid mode
                    linger_delta = 0.0
                    # Even if not used by the selected cluster method, avoid passing None
                    eps = 0.5
                    min_samples = 5
                    angle_eps = 15.0
                    k = 3
                    k_min = 2
                    k_max = 6
                    min_sep_deg = 12.0  # Default value for discover_decision_chain

                # r_outer_list already initialized above; update if user changed r_outer
                r_outer_list = [r_outer] if r_outer is not None else [None]

                # Debug: Show parameters being used
                st.info(f"🔍 **Discover Analysis Parameters:**")
                st.write(f"- Junction: Circle(cx={junction.cx}, cz={junction.cz}, r={junction.r})")
                st.write(f"- Decision mode: {decision_mode}")
                st.write(f"🔍 **DEBUG: r_outer before debug message:** {r_outer} (type: {type(r_outer)})")
                st.write(f"- R outer: {r_outer} (from junctions tab, r_outer_list: {r_outer_list})")
                st.write(f"- Path length: {path_length}")
                st.write(f"- Cluster method: {cluster_method}, k: {k}")
                st.write(f"- Trajectories: {len(trajectories)}")

                # Additional debugging for radial mode issues
                if decision_mode == "radial":
                    st.error(f"🚨 **RADIAL MODE DETECTED - LIKELY CAUSE OF NO ASSIGNMENTS!**")
                    st.write(f"- Junction radius: {junction.r}")
                    st.write(f"- R outer: {r_outer}")
                    st.write(f"- Ratio (r_outer/junction.r): {r_outer/junction.r:.2f}")
                    if r_outer <= junction.r:
                        st.error(f"❌ **CRITICAL:** r_outer ({r_outer}) <= junction radius ({junction.r})!")
                        st.write("In radial mode, r_outer must be significantly larger than junction radius.")
                    elif r_outer/junction.r < 2.0:
                        st.warning(f"⚠️ **WARNING:** r_outer/junction.r ratio ({r_outer/junction.r:.2f}) is too low!")
                        st.write("For radial mode, r_outer should be at least 2x the junction radius for reliable detection.")

                    st.error(f"🔧 **RECOMMENDED FIXES:**")
                    st.write("1. **Change decision mode to 'hybrid'** in the gaze analysis custom parameters")
                    st.write("2. **Or increase r_outer values** in the Junctions tab to at least 2x the junction radius")
                    st.write("3. **Or use default parameters** instead of custom parameters")

                    # Show current parameter status
                    st.write(f"**Current decision mode:** {decision_mode}")
                    st.write("**To fix:** Change the decision mode to 'hybrid' in the analysis parameters")

                # Debug: Check if trajectories pass through junction
                trajectories_through_junction = 0
                for traj in trajectories[:5]:  # Check first 5 trajectories
                    # Handle NaN values in trajectory coordinates
                    valid_mask = ~(np.isnan(traj.x) | np.isnan(traj.z))
                    if np.any(valid_mask):
                        valid_x = traj.x[valid_mask]
                        valid_z = traj.z[valid_mask]
                        distances = np.sqrt((valid_x - junction.cx)**2 + (valid_z - junction.cz)**2)
                        min_distance = np.min(distances)
                        if min_distance <= junction.r:
                            trajectories_through_junction += 1

                st.write(f"- Sample trajectories through junction: {trajectories_through_junction}/5")
                if trajectories_through_junction == 0:
                    st.warning("⚠️ **Warning: No sample trajectories pass through the junction!**")
                    st.write("This might explain why no assignments are found.")
                    st.write("Check if junction coordinates match your trajectory data.")

                    # Additional debugging: Show coordinate ranges
                    st.error("🔍 **Coordinate Range Analysis:**")
                    all_x = np.concatenate([traj.x for traj in trajectories[:10]])  # Check first 10 trajectories
                    all_z = np.concatenate([traj.z for traj in trajectories[:10]])

                    # Handle NaN values in trajectory coordinates
                    valid_x = all_x[~np.isnan(all_x)]
                    valid_z = all_z[~np.isnan(all_z)]

                    if len(valid_x) > 0 and len(valid_z) > 0:
                        st.write(f"- Trajectory X range: {np.min(valid_x):.1f} to {np.max(valid_x):.1f}")
                        st.write(f"- Trajectory Z range: {np.min(valid_z):.1f} to {np.max(valid_z):.1f}")
                        st.write(f"- Junction position: ({junction.cx}, {junction.cz})")
                        st.write(f"- Junction radius: {junction.r}")

                        # Check if junction is within trajectory bounds
                        x_in_bounds = np.min(valid_x) <= junction.cx <= np.max(valid_x)
                        z_in_bounds = np.min(valid_z) <= junction.cz <= np.max(valid_z)
                        st.write(f"- Junction X in trajectory bounds: {x_in_bounds}")
                        st.write(f"- Junction Z in trajectory bounds: {z_in_bounds}")

                        if not (x_in_bounds and z_in_bounds):
                            st.error("❌ **CRITICAL:** Junction is outside trajectory coordinate bounds!")
                            st.write("**Solution:** Adjust junction coordinates in the Junctions tab to match your trajectory data.")
                    else:
                        st.error("❌ **CRITICAL:** All trajectory coordinates are NaN!")
                        st.write("**This indicates a data loading or scaling issue.**")
                        st.write("**Solutions:**")
                        st.write("1. Check if trajectory data was loaded correctly")
                        st.write("2. Check if scaling factor is appropriate")
                        st.write("3. Check if coordinate columns are mapped correctly")
                        st.write("4. Try reloading the data with different parameters")

                        # Show sample trajectory data for debugging
                        if len(trajectories) > 0:
                            sample_traj = trajectories[0]
                            st.write(f"**Sample trajectory data:**")
                            st.write(f"- X values: {sample_traj.x[:5]} (first 5)")
                            st.write(f"- Z values: {sample_traj.z[:5]} (first 5)")
                            st.write(f"- X type: {type(sample_traj.x)}")
                            st.write(f"- Z type: {type(sample_traj.z)}")

                # Debug: Show all parameters being passed to discover_decision_chain
                st.write(f"🔍 **DEBUG: Parameters being passed to discover_decision_chain:**")
                st.write(f"- path_length: {path_length} (type: {type(path_length)})")
                st.write(f"- epsilon: {epsilon} (type: {type(epsilon)})")
                st.write(f"- seed: {seed} (type: {type(seed)})")
                st.write(f"- decision_mode: {decision_mode} (type: {type(decision_mode)})")
                st.write(f"- r_outer_list: {r_outer_list} (type: {type(r_outer_list)})")
                st.write(f"- linger_delta: {linger_delta} (type: {type(linger_delta)})")
                st.write(f"- cluster_method: {cluster_method} (type: {type(cluster_method)})")
                st.write(f"- k: {k} (type: {type(k)})")
                st.write(f"- k_min: {k_min} (type: {type(k_min)})")
                st.write(f"- k_max: {k_max} (type: {type(k_max)})")
                st.write(f"- min_sep_deg: {min_sep_deg} (type: {type(min_sep_deg)})")
                st.write(f"- angle_eps: {angle_eps} (type: {type(angle_eps)})")
                st.write(f"- min_samples: {min_samples} (type: {type(min_samples)})")

                try:
                    chain_df, centers_list, decisions_chain_df = discover_decision_chain(
                        trajectories=trajectories,
                        junctions=[junction],
                        path_length=path_length,
                        epsilon=epsilon,
                        seed=seed,
                        decision_mode=decision_mode,
                        r_outer_list=r_outer_list,
                        linger_delta=linger_delta if linger_delta is not None else 0.0,
                        out_dir=out_dir,
                        cluster_method=cluster_method,
                        k=k,
                        k_min=k_min,
                        k_max=k_max,
                        min_sep_deg=min_sep_deg,
                        angle_eps=angle_eps,
                        min_samples=min_samples,
                    )
                    # Save decisions into session state for reuse by gaze
                    try:
                        if decisions_chain_df is not None and len(decisions_chain_df) > 0:
                            st.session_state.analysis_results.setdefault("branches", {})
                            st.session_state.analysis_results["branches"]["chain_decisions"] = decisions_chain_df
                    except Exception:
                        pass
                except Exception as e:
                    st.error(f"❌ **Discover analysis failed - this will prevent gaze analysis!**")
                    st.write(f"**Error type:** {type(e).__name__}")
                    st.write(f"**Error message:** {str(e)}")

                    # Show the most common issues and solutions
                    st.error("🔧 **Common Solutions:**")
                    if "int()" in str(e) and "NoneType" in str(e):
                        st.write("**Issue:** NoneType to int() conversion error")
                        st.write("**Solution:** Use 'Use existing branch assignments' or run '🔍 Discover Branches' first")
                    elif "DBSCAN" in str(e) or "eps" in str(e):
                        st.write("**Issue:** DBSCAN clustering failed")
                        st.write("**Solution:** Try 'Use existing branch assignments' or adjust DBSCAN parameters")
                    elif "trajectory" in str(e).lower():
                        st.write("**Issue:** Trajectory data problem")
                        st.write("**Solution:** Check if trajectories pass through the junction")
                    else:
                        st.write("**Solution:** Use 'Use existing branch assignments' or run '🔍 Discover Branches' first")

                    st.write("")
                    st.write("**💡 Recommended approach:**")
                    st.write("1. **Use existing assignments** (if available)")
                    st.write("2. **Or run '🔍 Discover Branches' analysis first**")
                    st.write("3. **Then return here for gaze analysis**")

                    # Fall back to empty assignments
                    import pandas as pd
                    chain_df = pd.DataFrame()
                    centers_list = []
                st.success(f"✅ Discover analysis completed - found {len(chain_df)} assignments")
                st.write(f"**Assignment Summary:**")
                st.write(f"- Total trajectories processed: {len(trajectories)}")
                st.write(f"- Trajectories with assignments: {len(chain_df)}")
                st.write(f"- Assignment rate: {len(chain_df)/len(trajectories)*100:.1f}%")

                # Show sample of assignments
                if len(chain_df) > 0:
                    st.write(f"**Sample Assignments:**")
                    st.dataframe(chain_df.head(10), width='stretch')
                else:
                    st.warning("⚠️ No trajectories were assigned to this junction!")
                    st.write("**Possible causes:**")
                    st.write("- Junction position/radius may not match trajectory data")
                    st.write("- Decision parameters may be too restrictive")
                    st.write("- Trajectories may not actually pass through this junction")

                    # Add specific parameter suggestions
                    st.info("🔧 **Parameter Adjustment Suggestions:**")
                    st.write("**Current parameters:**")
                    st.write(f"- Path length: {path_length} (try reducing to 20-50)")
                    st.write(f"- Decision mode: {decision_mode}")
                    st.write(f"- DBSCAN eps: {epsilon} (try increasing to 1.0-2.0)")
                    st.write(f"- DBSCAN min_samples: {min_samples} (try reducing to 2-3)")
                    st.write(f"- R_outer: {r_outer} (try increasing)")

                    st.write("**Suggested changes:**")
                    st.write("1. **Reduce path_length** from 100.0 to 20.0-50.0")
                    st.write("2. **Increase DBSCAN eps** from 0.5 to 1.0-2.0")
                    st.write("3. **Reduce min_samples** from 5 to 2-3")
                    st.write("4. **Check junction position** - ensure it matches your trajectory data")

                    # Debug: Show trajectory data around junction
                    st.write("🔍 **Trajectory Data Around Junction:**")
                    junction_x, junction_z = junction.cx, junction.cz
                    junction_r = junction.r

                    # Find trajectories that pass near the junction
                    nearby_trajectories = []
                    for i, traj in enumerate(trajectories[:10]):  # Check first 10 trajectories
                        distances = np.sqrt((traj.x - junction_x)**2 + (traj.z - junction_z)**2)
                        min_distance = np.min(distances)
                        if min_distance < junction_r * 3:  # Within 3x junction radius
                            nearby_trajectories.append((i, min_distance))

                    if nearby_trajectories:
                        st.write(f"- Found {len(nearby_trajectories)} trajectories within 3x junction radius")
                        for traj_idx, min_dist in nearby_trajectories[:5]:
                            st.write(f"  - Trajectory {traj_idx}: min distance = {min_dist:.1f}")
                    else:
                        st.write("- No trajectories found within 3x junction radius")
                        st.write("- This suggests the junction position may not match your data")
            except Exception as e:
                st.warning(f"⚠️ Discover analysis failed: {e}")
                st.info("🔄 Falling back to empty assignments...")
                chain_df = pd.DataFrame()

        # Preprocess trajectories to convert time values to numeric format
        processed_trajectories = []
        for traj in trajectories:
            if hasattr(traj, 't') and traj.t is not None:
                # Convert time values to numeric if they're strings
                if isinstance(traj.t[0], str):
                    try:
                        import pandas as pd
                        # Convert string time format to numeric seconds
                        numeric_times = []
                        for t_val in traj.t:
                            if isinstance(t_val, str):
                                # Parse time string like "00:00:17.425"
                                time_parts = t_val.split(':')
                                if len(time_parts) == 3:
                                    hours = float(time_parts[0])
                                    minutes = float(time_parts[1])
                                    seconds = float(time_parts[2])
                                    total_seconds = hours * 3600 + minutes * 60 + seconds
                                    numeric_times.append(total_seconds)
                                else:
                                    numeric_times.append(float(t_val))
                            else:
                                numeric_times.append(float(t_val))
                        traj.t = np.array(numeric_times)
                    except Exception as e:
                        st.warning(f"⚠️ Could not convert time values for trajectory {traj.tid}: {e}")
                        # Keep original time values
                processed_trajectories.append(traj)
            else:
                processed_trajectories.append(traj)

        # Store debug information in centralized gaze debug info
        junction_key = f"junction_{junction.cx}_{junction.cz}_{junction.r}"
        if 'gaze_debug_info' not in st.session_state:
            st.session_state['gaze_debug_info'] = {}

        st.session_state['gaze_debug_info'][junction_key] = {
            'trajectories_count': len(processed_trajectories),
            'assignments_count': len(chain_df),
            'junction': f"Circle(cx={junction.cx}, cz={junction.cz}, r={junction.r})",
            'decision_mode': decision_mode,
            'r_outer': r_outer,
            'status': 'processing'
        }

        # Check if we have any assignments
        if len(chain_df) == 0:
            st.session_state['gaze_debug_info'][junction_key]['status'] = 'no_assignments'

            st.warning("⚠️ No trajectory assignments found - cannot perform gaze analysis")
            st.info("💡 **Suggestions:**")
            st.write("1. **Adjust parameters**: Try different decision mode or parameters")
            st.write("2. **Check junction position**: Verify junction coordinates match your data")
            st.write("3. **Run discover analysis first**: Use '🔍 Discover Branches' to create assignments")
            st.write("4. **Check trajectory data**: Ensure trajectories actually pass through junctions")

            # Return empty results
            return {
                'physiological': None,
                'pupil_dilation': None,
                'head_yaw': None,
                'junction': junction,
                'r_outer': r_outer,
                'error': 'No assignments found'
            }

        # Use the actual gaze analysis functions with proper assignments
        try:
            # Update debug status
            st.session_state['gaze_debug_info'][junction_key]['status'] = 'analyzing'

            # Debug: Check what trajectories are being passed to gaze analysis
            st.info(f"🔍 **Gaze Analysis Input Debug:**")
            st.write(f"- Trajectories passed: {len(processed_trajectories)}")
            if processed_trajectories:
                sample_traj = processed_trajectories[0]
                st.write(f"- Sample trajectory type: {type(sample_traj).__name__}")
                st.write(f"- Sample trajectory ID: {sample_traj.tid}")
                st.write(f"- Has gaze_x: {hasattr(sample_traj, 'gaze_x') and sample_traj.gaze_x is not None}")
                st.write(f"- Has heart_rate: {hasattr(sample_traj, 'heart_rate') and sample_traj.heart_rate is not None}")
                st.write(f"- Has pupil_l: {hasattr(sample_traj, 'pupil_l') and sample_traj.pupil_l is not None}")

            # Debug: Check assignments DataFrame structure
            st.info(f"🔍 **Assignments DataFrame Debug:**")
            st.write(f"- Assignments shape: {chain_df.shape}")
            st.write(f"- Assignments columns: {list(chain_df.columns)}")
            if len(chain_df) > 0:
                st.write(f"- Sample assignment row:")
                st.write(chain_df.head(1))
                # Safe sorting of trajectory IDs (handle mixed types)
                try:
                    unique_ids = chain_df['trajectory'].unique()
                    # Convert to strings for safe sorting
                    unique_ids_str = [str(id) for id in unique_ids]
                    sorted_ids = sorted(unique_ids_str)[:10]
                    st.write(f"- Unique trajectory IDs in assignments: {sorted_ids}...")
                except Exception as e:
                    st.write(f"- Unique trajectory IDs in assignments: {list(unique_ids)[:10]}... (sorting failed: {e})")
                st.write(f"- Sample trajectory IDs from Trajectory objects: {[tr.tid for tr in processed_trajectories[:5]]}")

                # Check for ID mismatch (handle mixed types)
                assignment_ids = set(str(id) for id in chain_df['trajectory'].unique())
                trajectory_ids = set(str(tr.tid) for tr in processed_trajectories)
                common_ids = assignment_ids.intersection(trajectory_ids)
                st.write(f"- Common IDs between assignments and trajectories: {len(common_ids)}")
                if len(common_ids) < 10:
                    st.write(f"- Common IDs: {sorted(common_ids)}")
                else:
                    st.write(f"- Common IDs (first 10): {sorted(list(common_ids))[:10]}")

            # Debug: Check session state at the beginning of gaze analysis
            st.write(f"🔍 **Gaze Analysis Session State Check:**")
            st.write(f"- analysis_results exists: {st.session_state.analysis_results is not None}")
            if st.session_state.analysis_results is not None:
                st.write(f"- branches exists: {'branches' in st.session_state.analysis_results}")
                if 'branches' in st.session_state.analysis_results:
                    branches = st.session_state.analysis_results['branches']
                    st.write(f"- chain_decisions exists: {'chain_decisions' in branches}")
                    if 'chain_decisions' in branches:
                        decisions_chain_df = branches['chain_decisions']
                        st.write(f"- chain_decisions length: {len(decisions_chain_df)}")
                        st.write(f"- Junction indices in chain_decisions: {sorted(decisions_chain_df['junction_index'].unique())}")
                    else:
                        st.write("- chain_decisions not found in branches!")
                else:
                    st.write("- branches not found in analysis_results!")
            else:
                st.write("- analysis_results is None!")

            # Analyze physiological data
            st.info("🔬 Analyzing physiological data...")
            with st.spinner("Processing physiological data..."):
                # Debug: Test trajectory matching before calling the function
                st.write("🔍 **Pre-analysis Debug:**")
                test_traj = processed_trajectories[0]
                test_assignments = chain_df[chain_df["trajectory"] == test_traj.tid]
                st.write(f"- Test trajectory ID: {test_traj.tid}")
                st.write(f"- Test trajectory assignments: {len(test_assignments)} rows")
                if len(test_assignments) > 0:
                    st.write(f"- Test assignment: {test_assignments.iloc[0].to_dict()}")
                else:
                    st.write("- ❌ No assignments found for test trajectory!")

                # Fix: Convert single 'branch' column to junction-specific format expected by gaze functions
                current_junction_idx = next(i for i, j in enumerate(st.session_state.junctions)
                                           if j.cx == junction.cx and j.cz == junction.cz and j.r == junction.r)
                current_branch_col = f'branch_j{current_junction_idx}'

                if 'branch' in chain_df.columns:
                    if current_branch_col not in chain_df.columns:
                        st.info(f"🔧 Converting single 'branch' column to '{current_branch_col}' format for gaze analysis...")
                        chain_df_fixed = chain_df.copy()
                        chain_df_fixed[current_branch_col] = chain_df_fixed['branch']
                        st.write(f"- Converted {len(chain_df_fixed)} assignments to gaze analysis format for Junction {current_junction_idx}")
                    else:
                        chain_df_fixed = chain_df
                else:
                    chain_df_fixed = chain_df

                # CRITICAL FIX: Filter out unassigned trajectories (branch = -1 or NaN)
                original_count = len(chain_df_fixed)
                if current_branch_col in chain_df_fixed.columns:
                    # Filter out unassigned trajectories
                    assigned_mask = (chain_df_fixed[current_branch_col] != -1) & (chain_df_fixed[current_branch_col].notna())
                    chain_df_fixed = chain_df_fixed[assigned_mask]
                    filtered_count = len(chain_df_fixed)
                    unassigned_count = original_count - filtered_count

                    st.write(f"🔍 **Assignment Filtering:**")
                    st.write(f"- Original assignments: {original_count}")
                    st.write(f"- Unassigned trajectories (branch=-1 or NaN): {unassigned_count}")
                    st.write(f"- Assigned trajectories: {filtered_count}")

                    if unassigned_count > 0:
                        st.info(f"✅ Filtered out {unassigned_count} unassigned trajectories - only processing {filtered_count} assigned trajectories")
                    else:
                        st.info(f"✅ All {original_count} trajectories are assigned to branches")

                    # Debug: Show sample of filtered assignments
                    if filtered_count > 0:
                        st.write(f"🔍 **Sample filtered assignments for Junction {current_junction_idx}:**")
                        sample_assignments = chain_df_fixed[['trajectory', current_branch_col]].head()
                        st.write(sample_assignments)

                        # Show branch distribution
                        branch_counts = chain_df_fixed[current_branch_col].value_counts().sort_index()
                        st.write(f"🔍 **Branch distribution:**")
                        for branch, count in branch_counts.items():
                            st.write(f"  - Branch {branch}: {count} trajectories")

                        # Debug: Show trajectory ID matching
                        st.write(f"🔍 **Trajectory ID Matching Debug:**")
                        st.write(f"- Assignments trajectory IDs (first 10): {list(chain_df_fixed['trajectory'].astype(str))[:10]}")
                        st.write(f"- Processed trajectories IDs (first 10): {[str(getattr(t, 'tid', getattr(t, 'id', 'NA'))) for t in processed_trajectories[:10]]}")

                        # Check if trajectory ID 0 exists in assignments
                        traj_0_in_assignments = '0' in chain_df_fixed['trajectory'].astype(str).values
                        st.write(f"- Trajectory ID 0 in assignments: {traj_0_in_assignments}")

                        # ALWAYS apply trajectory ID mapping to ensure sequential IDs for gaze analysis
                        st.info("🔧 **APPLYING TRAJECTORY ID MAPPING**: Ensuring sequential trajectory IDs for gaze analysis...")

                        if True:  # Always apply mapping
                            st.write(f"🔍 **Trajectory ID Analysis for Junction {current_junction_idx}:**")
                            st.write("The gaze analysis functions expect sequential trajectory IDs [0, 1, 2, ...] for proper decision point matching.")

                            # Show the actual range of trajectory IDs in assignments
                            traj_ids = chain_df_fixed['trajectory'].astype(str).values
                            st.write(f"- Assignment trajectory IDs (first 5): {traj_ids[:5]}")
                            st.write(f"- Processed trajectory IDs (first 5): {[str(getattr(t, 'tid', getattr(t, 'id', i))) for i, t in enumerate(processed_trajectories[:5])]}")

                            # Check if trajectory IDs match between assignments and processed trajectories
                            assignment_ids = set(traj_ids)
                            processed_ids = set(str(getattr(t, 'tid', getattr(t, 'id', i))) for i, t in enumerate(processed_trajectories))
                            common_ids = assignment_ids.intersection(processed_ids)
                            st.write(f"- Common trajectory IDs: {len(common_ids)} out of {len(assignment_ids)} assignments")

                            st.error("❌ **SOLUTION NEEDED**: The trajectory ID mapping between discover and gaze analysis is broken!")
                            st.write("The discover function assigns trajectories with string IDs (filenames) but gaze analysis expects sequential integer IDs [0, 1, 2, ...]")

                            # IMPLEMENT FIX: Create trajectory ID mapping
                            st.info("🔧 **IMPLEMENTING FIX**: Creating trajectory ID mapping...")

                            # Create a mapping from assignment trajectory IDs to processed trajectory indices
                            # Map assignment IDs to sequential indices based on the order they appear in processed_trajectories
                            assignment_ids = chain_df_fixed['trajectory'].unique()
                            traj_id_to_index = {tid: i for i, tid in enumerate(sorted(assignment_ids))}

                            st.write(f"🔧 **Trajectory ID Mapping Debug:**")
                            st.write(f"- Assignment trajectory IDs (first 5): {list(chain_df_fixed['trajectory'].unique()[:5])}")
                            st.write(f"- Processed trajectory IDs (first 5): {list(traj_id_to_index.keys())[:5]}")
                            st.write(f"- Mapping dictionary (first 5): {dict(list(traj_id_to_index.items())[:5])}")

                            # Map assignment trajectory IDs to processed trajectory indices
                            chain_df_fixed['trajectory_index'] = chain_df_fixed['trajectory'].map(traj_id_to_index)

                            # Remove rows where trajectory ID couldn't be mapped
                            original_mapped_count = len(chain_df_fixed)
                            chain_df_fixed = chain_df_fixed.dropna(subset=['trajectory_index'])
                            mapped_count = len(chain_df_fixed)
                            unmapped_count = original_mapped_count - mapped_count

                            st.write(f"🔧 **Trajectory ID Mapping Results:**")
                            st.write(f"- Original assignments: {original_mapped_count}")
                            st.write(f"- Successfully mapped: {mapped_count}")
                            st.write(f"- Unmapped (removed): {unmapped_count}")

                            if mapped_count > 0:
                                st.success(f"✅ **FIX APPLIED**: Mapped {mapped_count} trajectory assignments to processed trajectory indices!")
                                st.write(f"- Sample mapping: {chain_df_fixed[['trajectory', 'trajectory_index', current_branch_col]].head()}")

                                # Update trajectory IDs to be sequential starting from 0
                                chain_df_fixed['trajectory'] = chain_df_fixed['trajectory_index'].astype(int)
                                chain_df_fixed = chain_df_fixed.drop('trajectory_index', axis=1)

                                st.write(f"🔧 **Updated assignments with sequential IDs:**")
                                st.write(f"- Sample: {chain_df_fixed[['trajectory', current_branch_col]].head()}")
                            else:
                                st.error("❌ **FIX FAILED**: No trajectory assignments could be mapped!")
                                st.write("This suggests a fundamental mismatch between discover and gaze analysis trajectory handling.")
                else:
                    st.error(f"❌ Expected column '{current_branch_col}' not found in assignments!")
                    st.write(f"Available columns: {list(chain_df_fixed.columns)}")

                # Ensure expected branch column exists for physiological analysis
                chain_df_call = chain_df_fixed.copy()

                # Ensure the junction-specific branch column exists
                if current_branch_col not in chain_df_call.columns:
                    st.error(f"❌ **Missing branch column**: {current_branch_col} not found for Junction {current_junction_idx}")
                else:
                    st.write(f"🔧 **Using junction-specific column**: {current_branch_col} for Junction {current_junction_idx}")

                    # CRITICAL FIX: Use verta_consistency.normalize_assignments for proper trajectory ID mapping
                    # This will automatically handle branch column naming and trajectory ID mapping
                    st.info(f"🔧 **Using verta_consistency.normalize_assignments for proper trajectory ID mapping...**")

                    try:
                        from verta.verta_consistency import normalize_assignments

                        # Use the proper normalization function with the fixed assignments
                        # It will automatically create branch_j{i} for single junction analysis
                        normalized_df, report = normalize_assignments(
                            assignments_df=chain_df_fixed,  # Use the fixed assignments with proper trajectory IDs
                            trajectories=processed_trajectories,
                            junctions=[junction],  # Single junction for this analysis
                            current_junction_idx=current_junction_idx,
                            prefer_decisions=False,  # We already have the assignments
                            include_outliers=False,  # Filter out negative branches
                            strict=False  # Don't fail on low coverage
                        )

                        st.write(f"🔧 **Normalization report:**")
                        st.write(f"- Input rows: {report['input_rows']}")
                        st.write(f"- Kept after ID mapping: {report['kept_after_tid_map']}")
                        st.write(f"- Dropped unmapped IDs: {report['dropped_unmapped_ids']}")
                        st.write(f"- Has decisions: {report['has_decisions']}")

                        if report['kept_after_tid_map'] > 0:
                            st.success(f"✅ **Assignment normalization successful!** {report['kept_after_tid_map']} assignments ready for gaze analysis")

                            # Show available branch columns after normalization
                            branch_cols = [col for col in normalized_df.columns if col.startswith('branch')]
                            st.write(f"🔧 **Available branch columns after normalization:** {branch_cols}")

                            # Show sample data with available branch columns
                            display_cols = ['trajectory'] + branch_cols
                            st.write(f"🔧 **Sample normalized assignments:**")
                            st.write(normalized_df[display_cols].head())

                            # Use the normalized DataFrame for gaze analysis
                            chain_df_call = normalized_df
                        else:
                            st.error(f"❌ **Assignment normalization failed!** No assignments could be mapped")
                            st.write("This suggests a fundamental mismatch between trajectory IDs and assignment IDs")

                    except Exception as e:
                        st.error(f"❌ **Error during assignment normalization:** {e}")
                        st.write("Falling back to manual trajectory ID mapping...")

                        # Fallback to manual mapping if normalization fails
                        traj_id_mapping = {}
                        for i, traj in enumerate(processed_trajectories):
                            original_id = getattr(traj, '_original_tid', traj.tid)
                            traj_id_mapping[original_id] = i

                        if 'trajectory' in chain_df_call.columns:
                            chain_df_call['trajectory'] = chain_df_call['trajectory'].map(traj_id_mapping)
                            chain_df_call = chain_df_call.dropna(subset=['trajectory'])

                # CRITICAL FIX: Ensure we're using the correct junction-specific assignments
                # If we have a single-junction assignment (single 'branch' column),
                # we need to make sure the decision points are calculated for THIS junction,
                # not some other junction's decision points.
                st.write(f"🔍 **Junction-Specific Assignment Debug:**")
                st.write(f"- Current junction index: {current_junction_idx}")
                st.write(f"- Current branch column: {current_branch_col}")
                st.write(f"- Available columns: {list(chain_df_call.columns)}")

                # Show branch column values dynamically
                branch_cols = [col for col in chain_df_call.columns if col.startswith('branch')]
                for branch_col in branch_cols:
                    st.write(f"- {branch_col} values: {chain_df_call[branch_col].value_counts().to_dict()}")

                if not branch_cols:
                    st.write("- ❌ No branch columns found!")

                # Create copies of trajectories and remap their IDs to match the assignments DataFrame
                # This ensures the analysis functions can find the correct assignments
                import copy as _copy
                trajectories_for_analysis = [_copy.copy(tr) for tr in processed_trajectories]

                # CRITICAL FIX: Update trajectory IDs to match the assignments DataFrame
                # The assignments DataFrame has original trajectory IDs, so we need to match them
                assignment_ids = chain_df_fixed['trajectory'].unique()

                # Update trajectory IDs to match assignments DataFrame IDs
                for i, _tr in enumerate(trajectories_for_analysis):
                    if i < len(assignment_ids):
                        _tr.tid = assignment_ids[i]
                    else:
                        _tr.tid = i

                # Normalize assignments using shared consistency layer (replaces ad-hoc fixes)
                try:
                    from .verta_consistency import normalize_assignments
                except Exception:
                    from verta.verta_consistency import normalize_assignments

                decisions_chain_df = st.session_state.analysis_results.get("branches", {}).get("decision_points")
                if decisions_chain_df is None:
                    # Fallback: try to load from default GUI outputs dir
                    try:
                        import os as _os
                        import pandas as _pd
                        _p = _os.path.join("gui_outputs", "branch_decisions_chain.csv")
                        if _os.path.exists(_p):
                            decisions_chain_df = _pd.read_csv(_p)
                            st.write("🔗 Loaded decisions from gui_outputs/branch_decisions_chain.csv (fallback)")
                    except Exception:
                        pass
                norm_df, norm_report = normalize_assignments(
                    chain_df_fixed,
                    trajectories=trajectories_for_analysis,
                    junctions=[junction],
                    current_junction_idx=current_junction_idx,
                    decisions_df=decisions_chain_df,
                    prefer_decisions=True,
                    include_outliers=False,
                )
                chain_df_call = norm_df  # override with normalized assignments
                st.write("🧭 Assignments normalization report:")
                st.write(norm_report)

                physio_data = analyze_physiological_at_junctions(
                    trajectories=trajectories_for_analysis,
                    junctions=[junction],
                    assignments_df=chain_df_call,  # Use the assignments with merged decision points
                    decision_mode=discover_decision_mode,  # Use discover decision mode
                    r_outer_list=r_outer_list,
                    path_length=discover_path_length,  # Use discover path length
                    epsilon=discover_epsilon,  # Use discover epsilon
                    linger_delta=discover_linger_delta,  # Use discover linger delta
                    physio_window=3.0
                )

                # DEBUG: Check what was passed to physiological analysis
                st.write(f"🔍 **DEBUG: Physiological Analysis Input Check:**")
                st.write(f"- Trajectories passed: {len(trajectories_for_analysis)}")
                st.write(f"- First trajectory ID: {trajectories_for_analysis[0].tid} (type: {type(trajectories_for_analysis[0].tid)})")
                st.write(f"- Assignments DataFrame shape: {chain_df_fixed.shape}")
                st.write(f"- Assignments trajectory column dtype: {chain_df_fixed['trajectory'].dtype}")
                st.write(f"- Assignments trajectory sample: {chain_df_fixed['trajectory'].head().tolist()}")
                st.write(f"- Junction: {junction}")
                st.write(f"- Junction index: {current_junction_idx}")

            st.success("✅ Physiological analysis completed")
            st.write(f"🔍 **Physiological Results:** {len(physio_data) if physio_data is not None else 0} rows")

            # Debug: Show physiological results details
            if physio_data is not None and len(physio_data) > 0:
                st.write(f"🔍 **Physiological Results Debug:**")
                st.write(f"- Rows: {len(physio_data)}")
                st.write(f"- Columns: {list(physio_data.columns)}")
                if 'heart_rate_change' in physio_data.columns:
                    hr_change_count = physio_data['heart_rate_change'].notna().sum()
                    st.write(f"- Heart rate change values: {hr_change_count}")
                else:
                    st.write(f"- ❌ 'heart_rate_change' column missing!")
            else:
                st.write(f"🔍 **Physiological Results Debug:** No data returned")

            # Analyze pupil dilation trajectories
            st.info("👁️ Analyzing pupil dilation trajectories...")
            with st.spinner("Processing pupil dilation data..."):
                # Ensure expected branch column exists for pupil analysis
                chain_df_call = chain_df_fixed.copy()

                # Ensure the junction-specific branch column exists
                if current_branch_col not in chain_df_call.columns:
                    st.error(f"❌ **Missing branch column**: {current_branch_col} not found for Junction {current_junction_idx}")
                else:
                    st.write(f"🔧 **Using junction-specific column**: {current_branch_col} for Junction {current_junction_idx}")

                # CRITICAL FIX: Ensure we're using the correct junction-specific assignments
                # If we have a single-junction assignment (single 'branch' column),
                # we need to make sure the decision points are calculated for THIS junction,
                # not some other junction's decision points.
                st.write(f"🔍 **Pupil Analysis - Junction-Specific Assignment Debug:**")
                st.write(f"- Current junction index: {current_junction_idx}")
                st.write(f"- Current branch column: {current_branch_col}")
                st.write(f"- Available columns: {list(chain_df_call.columns)}")
                if current_branch_col in chain_df_call.columns:
                    st.write(f"- {current_branch_col} values: {chain_df_call[current_branch_col].value_counts().to_dict()}")
                else:
                    st.write(f"- ❌ {current_branch_col} column not found!")

                # Create copies of trajectories and remap their IDs to match the assignments DataFrame
                # This ensures the analysis functions can find the correct assignments
                import copy as _copy
                trajectories_for_analysis = [_copy.copy(tr) for tr in processed_trajectories]

                # CRITICAL FIX: Update trajectory IDs to match the assignments DataFrame
                # The assignments DataFrame has original trajectory IDs, so we need to match them
                assignment_ids = chain_df_fixed['trajectory'].unique()

                # Update trajectory IDs to match assignments DataFrame IDs
                for i, _tr in enumerate(trajectories_for_analysis):
                    if i < len(assignment_ids):
                        _tr.tid = assignment_ids[i]
                    else:
                        _tr.tid = i

                # Normalize assignments using shared consistency layer (replaces ad-hoc fixes)
                try:
                    from .verta_consistency import normalize_assignments
                except Exception:
                    from verta.verta_consistency import normalize_assignments

                decisions_chain_df = st.session_state.analysis_results.get("branches", {}).get("decision_points")
                if decisions_chain_df is None:
                    # Fallback: try to load from default GUI outputs dir
                    try:
                        import os as _os
                        import pandas as _pd
                        _p = _os.path.join("gui_outputs", "branch_decisions_chain.csv")
                        if _os.path.exists(_p):
                            decisions_chain_df = _pd.read_csv(_p)
                            st.write("🔗 Loaded decisions from gui_outputs/branch_decisions_chain.csv (fallback)")
                    except Exception:
                        pass
                norm_df, norm_report = normalize_assignments(
                    chain_df_fixed,
                    trajectories=trajectories_for_analysis,
                    junctions=[junction],
                    current_junction_idx=current_junction_idx,
                    decisions_df=decisions_chain_df,
                    prefer_decisions=True,
                    include_outliers=False,
                )
                chain_df_call = norm_df  # override with normalized assignments
                st.write("🧭 Assignments normalization report:")
                st.write(norm_report)
                # Debug: Check if normalize_assignments already merged decision points
                precomputed_count = chain_df_call['decision_idx'].notna().sum() if 'decision_idx' in chain_df_call.columns else 0
                st.write(f"🔍 Decision points after normalize_assignments (pupil): {precomputed_count} trajectories have precomputed decision points")
                if precomputed_count > 0:
                    st.write(f"🔍 Sample decision points: {chain_df_call[['trajectory', 'decision_idx', 'intercept_x', 'intercept_z']].head(3).to_dict('records')}")
                else:
                    st.write("⚠️ No decision points found - this may cause analysis to fail")

                pupil_data = analyze_pupil_dilation_trajectory(
                    trajectories=trajectories_for_analysis,
                    junctions=[junction],
                    assignments_df=chain_df_call,  # Use the assignments with merged decision points
                    decision_mode=discover_decision_mode,  # Use discover decision mode
                    r_outer_list=r_outer_list,
                    path_length=discover_path_length,  # Use discover path length
                    epsilon=discover_epsilon,  # Use discover epsilon
                    linger_delta=discover_linger_delta,  # Use discover linger delta
                    physio_window=3.0
                )

            st.success("✅ Pupil dilation analysis completed")
            st.write(f"🔍 **Pupil Results:** {len(pupil_data) if pupil_data is not None else 0} rows")

            # Debug: Show pupil results details
            if pupil_data is not None and len(pupil_data) > 0:
                st.write(f"🔍 **Pupil Results Debug:**")
                st.write(f"- Rows: {len(pupil_data)}")
                st.write(f"- Columns: {list(pupil_data.columns)}")
                if 'pupil_change' in pupil_data.columns:
                    pupil_change_count = pupil_data['pupil_change'].notna().sum()
                    st.write(f"- Pupil change values: {pupil_change_count}")
                else:
                    st.write(f"- ❌ 'pupil_change' column missing!")
            else:
                st.write(f"🔍 **Pupil Results Debug:** No data returned")

            # Analyze head yaw at decisions
            st.info("🧭 Analyzing head yaw at decisions...")
            with st.spinner("Processing head yaw data..."):
                # Ensure expected branch column exists for head yaw analysis
                chain_df_call = chain_df_fixed.copy()

                # Ensure the junction-specific branch column exists
                if current_branch_col not in chain_df_call.columns:
                    st.error(f"❌ **Missing branch column**: {current_branch_col} not found for Junction {current_junction_idx}")
                else:
                    st.write(f"🔧 **Using junction-specific column**: {current_branch_col} for Junction {current_junction_idx}")

                # CRITICAL FIX: Ensure we're using the correct junction-specific assignments
                # If we have a single-junction assignment (single 'branch' column),
                # we need to make sure the decision points are calculated for THIS junction,
                # not some other junction's decision points.
                st.write(f"🔍 **Head Yaw Analysis - Junction-Specific Assignment Debug:**")
                st.write(f"- Current junction index: {current_junction_idx}")
                st.write(f"- Current branch column: {current_branch_col}")
                st.write(f"- Available columns: {list(chain_df_call.columns)}")
                if current_branch_col in chain_df_call.columns:
                    st.write(f"- {current_branch_col} values: {chain_df_call[current_branch_col].value_counts().to_dict()}")
                else:
                    st.write(f"- ❌ {current_branch_col} column not found!")

                # Create copies of trajectories and remap their IDs to match the assignments DataFrame
                # This ensures the analysis functions can find the correct assignments
                import copy as _copy
                trajectories_for_analysis = [_copy.copy(tr) for tr in processed_trajectories]

                # CRITICAL FIX: Update trajectory IDs to match the assignments DataFrame
                # The assignments DataFrame has original trajectory IDs, so we need to match them
                assignment_ids = chain_df_fixed['trajectory'].unique()

                # Update trajectory IDs to match assignments DataFrame IDs
                for i, _tr in enumerate(trajectories_for_analysis):
                    if i < len(assignment_ids):
                        _tr.tid = assignment_ids[i]
                    else:
                        _tr.tid = i

                # Normalize assignments using shared consistency layer (replaces ad-hoc fixes)
                try:
                    from .verta_consistency import normalize_assignments
                except Exception:
                    from verta.verta_consistency import normalize_assignments

                decisions_chain_df = st.session_state.analysis_results.get("branches", {}).get("decision_points")
                if decisions_chain_df is None:
                    # Fallback: try to load from default GUI outputs dir
                    try:
                        import os as _os
                        import pandas as _pd
                        _p = _os.path.join("gui_outputs", "branch_decisions_chain.csv")
                        if _os.path.exists(_p):
                            decisions_chain_df = _pd.read_csv(_p)
                            st.write("🔗 Loaded decisions from gui_outputs/branch_decisions_chain.csv (fallback)")
                    except Exception:
                        pass
                norm_df, norm_report = normalize_assignments(
                    chain_df_fixed,
                    trajectories=trajectories_for_analysis,
                    junctions=[junction],
                    current_junction_idx=current_junction_idx,
                    decisions_df=decisions_chain_df,
                    prefer_decisions=True,
                    include_outliers=False,
                )
                chain_df_call = norm_df  # override with normalized assignments
                st.write("🧭 Assignments normalization report:")
                st.write(norm_report)
                # Debug: Check if normalize_assignments already merged decision points
                precomputed_count = chain_df_call['decision_idx'].notna().sum() if 'decision_idx' in chain_df_call.columns else 0
                st.write(f"🔍 Decision points after normalize_assignments: {precomputed_count} trajectories have precomputed decision points")
                if precomputed_count > 0:
                    st.write(f"🔍 Sample decision points: {chain_df_call[['trajectory', 'decision_idx', 'intercept_x', 'intercept_z']].head(3).to_dict('records')}")
                else:
                    st.write("⚠️ No decision points found - this may cause analysis to fail")

                # Get the decision mode used by the discover analysis
                discover_decision_mode = "pathlen"  # Default to pathlen since that's what you're using
                if existing_assignments is not None:
                    # Try to get the decision mode from the junction data
                    discover_junction_key = f"junction_{current_junction_idx}"
                    if discover_junction_key in st.session_state.analysis_results.get("branches", {}):
                        junction_data = st.session_state.analysis_results["branches"][discover_junction_key]
                        discover_decision_mode = junction_data.get("decision_mode", "pathlen")

                st.write(f"🔧 **Using discover decision mode**: {discover_decision_mode} (same as discover analysis)")

                # Debug: Check what DataFrame is being passed to gaze analysis
                st.write(f"🔍 **DataFrame being passed to gaze analysis:**")
                st.write(f"- Shape: {chain_df_call.shape}")
                st.write(f"- Columns: {list(chain_df_call.columns)}")
                st.write(f"- Has decision_idx: {'decision_idx' in chain_df_call.columns}")
                st.write(f"- Has intercept_x: {'intercept_x' in chain_df_call.columns}")
                st.write(f"- Has intercept_z: {'intercept_z' in chain_df_call.columns}")
                if 'decision_idx' in chain_df_call.columns:
                    precomputed_count = chain_df_call['decision_idx'].notna().sum()
                    st.write(f"- Precomputed decision points: {precomputed_count} out of {len(chain_df_call)}")
                    st.write(f"- Sample precomputed data: {chain_df_call[['trajectory', 'decision_idx', 'intercept_x', 'intercept_z']].head(3).to_dict('records')}")

                head_yaw_data = compute_head_yaw_at_decisions(
                    trajectories=trajectories_for_analysis,
                    junctions=[junction],
                    assignments_df=chain_df_call,  # Use the assignments with merged decision points
                    decision_mode=discover_decision_mode,  # Use the same decision mode as discover analysis
                    r_outer_list=r_outer_list,
                    path_length=path_length,
                    epsilon=epsilon,
                    linger_delta=linger_delta,  # Use the same linger_delta as discover analysis
                    base_index=current_junction_idx if current_junction_idx is not None else 0,
                )

            st.success("✅ Head yaw analysis completed")
            st.write(f"🔍 **Head Yaw Results:** {len(head_yaw_data) if head_yaw_data is not None else 0} rows")

            # Debug: Show head yaw results details
            if head_yaw_data is not None and len(head_yaw_data) > 0:
                st.write(f"🔍 **Head Yaw Results Debug:**")
                st.write(f"- Rows: {len(head_yaw_data)}")
                st.write(f"- Columns: {list(head_yaw_data.columns)}")
                if 'head_yaw' in head_yaw_data.columns:
                    head_yaw_count = head_yaw_data['head_yaw'].notna().sum()
                    st.write(f"- Head yaw values: {head_yaw_count}")
                else:
                    st.write(f"- ❌ 'head_yaw' column missing!")
                if 'yaw_difference' in head_yaw_data.columns:
                    yaw_diff_count = head_yaw_data['yaw_difference'].notna().sum()
                    st.write(f"- Yaw difference values: {yaw_diff_count}")
                else:
                    st.write(f"- ❌ 'yaw_difference' column missing!")
            else:
                st.write(f"🔍 **Head Yaw Results Debug:** No data returned")

            # Create per-junction pupil dilation heatmap
            st.info("🗺️ Creating junction-specific pupil dilation heatmap...")
            with st.spinner("Generating junction heatmap..."):
                from verta.verta_gaze import create_per_junction_pupil_heatmap
                
                # Get heatmap settings from session state
                cell_size = st.session_state.get('pupil_heatmap_cell_size', 50.0)
                normalization = st.session_state.get('pupil_heatmap_normalization', 'relative')

                # Create per-junction heatmap (focused on current junction)
                # Get junction index from junction_key in session state
                junction_idx = None
                for idx, j in enumerate(st.session_state.junctions):
                    if j.cx == junction.cx and j.cz == junction.cz and j.r == junction.r:
                        junction_idx = idx
                        break

                junction_heatmaps = create_per_junction_pupil_heatmap(
                    trajectories=processed_trajectories,
                    junctions=[junction],
                    r_outer_list=[r_outer],
                    cell_size=cell_size,
                    normalization=normalization,
                    base_index=junction_idx if junction_idx is not None else 0
                )

                # Fix the junction index in the heatmap data
                # The function returns a dict with key=0 (local index), but we need the global index
                # Re-key the dictionary to use the global junction index
                if junction_idx is not None and 0 in junction_heatmaps:
                    junction_heatmaps = {junction_idx: junction_heatmaps[0]}


            # Update debug status to completed
            st.session_state['gaze_debug_info'][junction_key]['status'] = 'completed'
            st.session_state['gaze_debug_info'][junction_key]['physio_rows'] = len(physio_data) if physio_data is not None else 0
            st.session_state['gaze_debug_info'][junction_key]['pupil_rows'] = len(pupil_data) if pupil_data is not None else 0
            st.session_state['gaze_debug_info'][junction_key]['head_yaw_rows'] = len(head_yaw_data) if head_yaw_data is not None else 0

            # Save CSV files to gui_outputs
            try:
                import os
                gaze_data_dir = os.path.join("gui_outputs", "gaze_data")
                os.makedirs(gaze_data_dir, exist_ok=True)

                # Get junction index for file naming
                junction_idx = None
                for idx, j in enumerate(st.session_state.junctions):
                    if j.cx == junction.cx and j.cz == junction.cz and j.r == junction.r:
                        junction_idx = idx
                        break

                if junction_idx is not None:
                    junction_prefix = f"junction_{junction_idx}"

                    # Save physiological analysis data
                    if physio_data is not None and len(physio_data) > 0:
                        # Ensure all numpy arrays are converted to lists for CSV compatibility
                        physio_data_clean = physio_data.copy()
                        for col in physio_data_clean.columns:
                            if physio_data_clean[col].dtype == 'object':
                                # Check if column contains numpy arrays
                                physio_data_clean[col] = physio_data_clean[col].apply(
                                    lambda x: x.tolist() if hasattr(x, 'tolist') else x
                                )

                        physio_file = os.path.join(gaze_data_dir, f"{junction_prefix}_physiological_analysis.csv")
                        physio_data_clean.to_csv(physio_file, index=False)
                        st.info(f"📁 Physiological data saved to: {physio_file}")

                    # Save pupil dilation data
                    if pupil_data is not None and len(pupil_data) > 0:
                        # Ensure all numpy arrays are converted to lists for CSV compatibility
                        pupil_data_clean = pupil_data.copy()
                        for col in pupil_data_clean.columns:
                            if pupil_data_clean[col].dtype == 'object':
                                # Check if column contains numpy arrays
                                pupil_data_clean[col] = pupil_data_clean[col].apply(
                                    lambda x: x.tolist() if hasattr(x, 'tolist') else x
                                )

                        pupil_file = os.path.join(gaze_data_dir, f"{junction_prefix}_pupil_trajectory_analysis.csv")
                        pupil_data_clean.to_csv(pupil_file, index=False)
                        st.info(f"📁 Pupil trajectory data saved to: {pupil_file}")

                    # Save head yaw data
                    if head_yaw_data is not None and len(head_yaw_data) > 0:
                        # Ensure all numpy arrays are converted to lists for CSV compatibility
                        head_yaw_data_clean = head_yaw_data.copy()
                        for col in head_yaw_data_clean.columns:
                            if head_yaw_data_clean[col].dtype == 'object':
                                # Check if column contains numpy arrays
                                head_yaw_data_clean[col] = head_yaw_data_clean[col].apply(
                                    lambda x: x.tolist() if hasattr(x, 'tolist') else x
                                )

                        gaze_file = os.path.join(gaze_data_dir, f"{junction_prefix}_gaze_analysis.csv")
                        head_yaw_data_clean.to_csv(gaze_file, index=False)
                        st.info(f"📁 Gaze analysis data saved to: {gaze_file}")

                    # Save pupil heatmap data as JSON
                    if junction_heatmaps:
                        heatmap_file = os.path.join(gaze_data_dir, f"{junction_prefix}_pupil_heatmap.json")
                        import json
                        # Convert numpy arrays to lists for JSON serialization
                        def convert_numpy_to_list(obj):
                            if hasattr(obj, 'tolist'):
                                return obj.tolist()
                            elif isinstance(obj, dict):
                                return {k: convert_numpy_to_list(v) for k, v in obj.items()}
                            elif isinstance(obj, list):
                                return [convert_numpy_to_list(item) for item in obj]
                            elif hasattr(obj, 'to_dict'):  # Handle pandas DataFrames
                                return obj.to_dict('records')
                            elif hasattr(obj, '__dict__'):  # Handle dataclass objects (like Circle)
                                return {k: convert_numpy_to_list(v) for k, v in obj.__dict__.items()}
                            else:
                                return obj

                        # Use the converted version for JSON serialization
                        heatmap_data = convert_numpy_to_list(junction_heatmaps)
                        with open(heatmap_file, 'w') as f:
                            json.dump(heatmap_data, f, indent=2)
                        st.info(f"📁 Pupil heatmap data saved to: {heatmap_file}")

            except Exception as e:
                st.warning(f"⚠️ Could not save gaze data files: {e}")

            # Create comprehensive results
            results = {
                'physiological': physio_data,
                'pupil_dilation': pupil_data,
                'head_yaw': head_yaw_data,
                'pupil_heatmap_junction': junction_heatmaps,
                'junction': junction,
                'r_outer': r_outer
            }

            # Convert any numpy arrays to lists for JSON serialization
            def convert_numpy_to_list(obj):
                if hasattr(obj, 'tolist'):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_to_list(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_to_list(item) for item in obj]
                elif hasattr(obj, 'to_dict'):  # Handle pandas DataFrames
                    return obj.to_dict('records')
                elif hasattr(obj, '__dict__'):  # Handle dataclass objects (like Circle)
                    return {k: convert_numpy_to_list(v) for k, v in obj.__dict__.items()}
                else:
                    return obj

            # Apply conversion to results for JSON serialization
            results_for_json = convert_numpy_to_list(results)

            # Return original results (with DataFrames and Circle objects) for plotting functions
            return results

        except Exception as e:
            # Update debug status to error
            st.session_state['gaze_debug_info'][junction_key]['status'] = 'error'
            st.session_state['gaze_debug_info'][junction_key]['error'] = str(e)

            # More detailed error information
            import traceback
            st.error(f"❌ Comprehensive gaze analysis failed: {e}")
            st.error(f"**Error details:** {str(e)}")
            st.error(f"**Error type:** {type(e).__name__}")

            # Show traceback for debugging
            st.code(traceback.format_exc())

            if processed_trajectories:
                sample_traj = processed_trajectories[0]
                st.write(f"- Sample trajectory type: {type(sample_traj)}")
                st.write(f"- Sample trajectory attributes: {[attr for attr in dir(sample_traj) if not attr.startswith('_')]}")

            st.info("🔄 Falling back to movement pattern analysis...")
            return self._analyze_movement_patterns_optimized(
                trajectories=trajectories,
                junction=junction,
                r_outer=r_outer,
                decision_mode=decision_mode,
                path_length=path_length,
                epsilon=epsilon
            )

    def _get_filtered_trajectories_for_junction(self, trajectories, junction, r_outer):
        """Get trajectories that pass through a specific junction area, with clipped coordinates."""
        import numpy as np

        filtered_trajs = []

        for traj in trajectories:
            # Calculate distances from junction center
            rx = traj.x - junction.cx
            rz = traj.z - junction.cz
            r = np.hypot(rx, rz)

            # Keep trajectories that pass through the junction area
            if np.any(r <= r_outer):
                # Create a clipped version that only shows the junction-relevant portion
                # This helps with zoom by reducing the data extent
                junction_mask = r <= (r_outer * 1.5)  # Slightly larger than r_outer for context

                if np.any(junction_mask):
                    # Create a simple object to hold the clipped data
                    class ClippedTraj:
                        def __init__(self, tid, x, z):
                            self.tid = tid
                            self.x = x
                            self.z = z

                    clipped_traj = ClippedTraj(
                        tid=traj.tid,
                        x=traj.x[junction_mask],
                        z=traj.z[junction_mask]
                    )
                    filtered_trajs.append(clipped_traj)

        return filtered_trajs

    def _safe_get_time_value(self, trajectory, decision_idx):
        """Safely get time value from trajectory, handling different data types."""
        import numpy as np

        if not hasattr(trajectory, 't') or trajectory.t is None or decision_idx >= len(trajectory.t):
            return np.nan

        try:
            time_value = trajectory.t[decision_idx]

            # Handle string time values
            if isinstance(time_value, str):
                try:
                    import pandas as pd
                    return pd.to_timedelta(time_value).total_seconds()
                except:
                    return np.nan

            # Handle numeric values
            if isinstance(time_value, (int, float)) and not np.isnan(time_value):
                return float(time_value)

            return np.nan
        except:
            return np.nan

    def _add_pupil_dilation_analysis(self, gaze_data, trajectories, junction, r_outer):
        """Add simplified pupil dilation analysis for regular trajectories."""
        import numpy as np

        # Check if any trajectory has time data
        has_time_data = any(hasattr(traj, 't') and traj.t is not None for traj in trajectories)

        if not has_time_data:
            # Add placeholder columns
            gaze_data['pupil_baseline'] = np.nan
            gaze_data['pupil_decision'] = np.nan
            gaze_data['pupil_dilation'] = np.nan
            gaze_data['pupil_analysis_available'] = False
            return gaze_data

        # Add pupil analysis columns
        gaze_data['pupil_baseline'] = np.nan
        gaze_data['pupil_decision'] = np.nan
        gaze_data['pupil_dilation'] = np.nan
        gaze_data['pupil_analysis_available'] = True

        # For regular trajectories, we can't do real pupil analysis, but we can analyze timing patterns
        for idx, row in gaze_data.iterrows():
            traj_idx = int(row['trajectory'])
            decision_idx = int(row['decision_idx'])

            if traj_idx < len(trajectories):
                trajectory = trajectories[traj_idx]

                if hasattr(trajectory, 't') and trajectory.t is not None and decision_idx < len(trajectory.t):
                    try:
                        decision_time = self._safe_get_time_value(trajectory, decision_idx)

                        if np.isnan(decision_time):
                            continue

                        # Analyze timing patterns around decision point
                        time_window = 3.0  # 3 second window
                        time_mask = (trajectory.t >= decision_time - time_window) & (trajectory.t <= decision_time + time_window)

                        if np.any(time_mask):
                            # Calculate timing-based metrics as proxy for physiological analysis
                            pre_decision_times = trajectory.t[(trajectory.t >= decision_time - time_window) & (trajectory.t < decision_time)]
                            post_decision_times = trajectory.t[(trajectory.t > decision_time) & (trajectory.t <= decision_time + time_window)]

                            # Use timing patterns as proxy for pupil analysis
                            gaze_data.loc[idx, 'pupil_baseline'] = len(pre_decision_times) if len(pre_decision_times) > 0 else 0
                            gaze_data.loc[idx, 'pupil_decision'] = len(post_decision_times) if len(post_decision_times) > 0 else 0
                            gaze_data.loc[idx, 'pupil_dilation'] = len(post_decision_times) - len(pre_decision_times)
                    except Exception as e:
                        # Skip this trajectory if there's any error with time data
                        continue

        return gaze_data

    def _analyze_movement_patterns_all_junctions(self, trajectories, junctions, r_outer_list, decision_mode, path_length, epsilon):
        """Analyze movement patterns across all junctions in temporal order."""
        import numpy as np
        import pandas as pd

        results = []

        for traj_idx, trajectory in enumerate(trajectories):
            # Find all junction visits in temporal order for this trajectory
            junction_sequence = self._find_junction_sequence(trajectory, junctions, r_outer_list)

            # Analyze each junction visit in the sequence
            for visit_idx, junction_idx in enumerate(junction_sequence):
                junction = junctions[junction_idx]
                r_outer = r_outer_list[junction_idx]

                # Find decision point for this specific junction visit
                decision_idx = self._find_decision_point_for_junction_visit(
                    trajectory, junction, r_outer, decision_mode, path_length, epsilon, junction_sequence, visit_idx
                )

                if decision_idx is not None and decision_idx < len(trajectory.x):
                    # Calculate movement metrics for this junction visit
                    movement_yaw = np.nan
                    if decision_idx > 0 and decision_idx < len(trajectory.x) - 1:
                        dx = trajectory.x[decision_idx + 1] - trajectory.x[decision_idx - 1]
                        dz = trajectory.z[decision_idx + 1] - trajectory.z[decision_idx - 1]
                        movement_magnitude = np.hypot(dx, dz)
                        if movement_magnitude > 1e-3:
                            movement_yaw = np.degrees(np.arctan2(dx, dz))

                    # Calculate approach and exit directions
                    approach_yaw = np.nan
                    if decision_idx > 0:
                        dx_approach = trajectory.x[decision_idx] - trajectory.x[decision_idx - 1]
                        dz_approach = trajectory.z[decision_idx] - trajectory.z[decision_idx - 1]
                        approach_magnitude = np.hypot(dx_approach, dz_approach)
                        if approach_magnitude > 1e-3:
                            approach_yaw = np.degrees(np.arctan2(dx_approach, dz_approach))

                    exit_yaw = np.nan
                    if decision_idx < len(trajectory.x) - 1:
                        dx_exit = trajectory.x[decision_idx + 1] - trajectory.x[decision_idx]
                        dz_exit = trajectory.z[decision_idx + 1] - trajectory.z[decision_idx]
                        exit_magnitude = np.hypot(dx_exit, dz_exit)
                        if exit_magnitude > 1e-3:
                            exit_yaw = np.degrees(np.arctan2(dx_exit, dz_exit))

                    # Calculate distance from junction center
                    distance_from_center = np.sqrt(
                        (trajectory.x[decision_idx] - junction.cx)**2 +
                        (trajectory.z[decision_idx] - junction.cz)**2
                    )

                    # Calculate trajectory position metrics
                    trajectory_length = len(trajectory.x)
                    decision_ratio = decision_idx / trajectory_length if trajectory_length > 0 else 0

                    results.append({
                        "trajectory": traj_idx,
                        "junction": junction_idx,
                        "visit_order": visit_idx,  # Order of this junction in the trajectory
                        "decision_idx": decision_idx,
                        "trajectory_length": trajectory_length,
                        "decision_ratio": decision_ratio,
                        "movement_yaw": movement_yaw,
                        "approach_yaw": approach_yaw,
                        "exit_yaw": exit_yaw,
                        "distance_from_center": distance_from_center,
                        "decision_x": trajectory.x[decision_idx],
                        "decision_z": trajectory.z[decision_idx],
                        "time_at_decision": self._safe_get_time_value(trajectory, decision_idx)
                    })

        return pd.DataFrame(results)

    def _find_junction_sequence(self, trajectory, junctions, r_outer_list):
        """Find the temporal sequence of junction visits for a trajectory."""
        import numpy as np

        sequence = []
        current_junction = None

        for point_idx in range(len(trajectory.x)):
            x, z = trajectory.x[point_idx], trajectory.z[point_idx]

            # Check if we're at a new junction
            for junction_idx, (junction, r_outer) in enumerate(zip(junctions, r_outer_list)):
                distance = np.sqrt((x - junction.cx)**2 + (z - junction.cz)**2)

                if distance <= r_outer:
                    if current_junction != junction_idx:
                        sequence.append(junction_idx)
                        current_junction = junction_idx
                    break
            else:
                # Not at any junction
                current_junction = None

        return sequence

    def _find_decision_point_for_junction_visit(self, trajectory, junction, r_outer, decision_mode, path_length, epsilon, junction_sequence, visit_idx):
        """Find decision point for a specific junction visit in the sequence."""
        import numpy as np

        # Find the range of points where this junction was visited
        junction_points = []
        for point_idx in range(len(trajectory.x)):
            x, z = trajectory.x[point_idx], trajectory.z[point_idx]
            distance = np.sqrt((x - junction.cx)**2 + (z - junction.cz)**2)
            if distance <= r_outer:
                junction_points.append(point_idx)

        if not junction_points:
            return None

        # Use the first point of junction visit as decision point
        # This represents when the user first entered the junction
        return junction_points[0]

    def _find_radial_decision_point(self, trajectory, junction, r_outer):
        """Find decision point using radial method."""
        import numpy as np

        # Find the first point that enters the junction
        for i in range(len(trajectory.x)):
            distance = np.sqrt((trajectory.x[i] - junction.cx)**2 + (trajectory.z[i] - junction.cz)**2)
            if distance <= r_outer:
                return i
        return None

    def _find_path_length_decision_point(self, trajectory, junction, path_length, epsilon):
        """Find decision point using path length method."""
        import numpy as np

        # Find closest point to junction center
        distances = np.sqrt((trajectory.x - junction.cx)**2 + (trajectory.z - junction.cz)**2)
        closest_idx = np.argmin(distances)

        # Use a more reasonable search window
        search_window = min(int(path_length), len(trajectory.x) // 10, 50)  # Smaller, more reasonable window
        start_idx = max(0, closest_idx - search_window)
        end_idx = min(len(trajectory.x), closest_idx + search_window)

        # Look for decision point within the search window
        for i in range(start_idx, end_idx):
            distance = np.sqrt((trajectory.x[i] - junction.cx)**2 + (trajectory.z[i] - junction.cz)**2)
            if distance <= junction.r + epsilon:
                return i

        # If no point found within junction radius, return closest point
        return closest_idx

    def _find_nearest_to_center(self, trajectory, junction):
        """Find nearest point to junction center."""
        import numpy as np

        distances = np.sqrt((trajectory.x - junction.cx)**2 + (trajectory.z - junction.cz)**2)
        return np.argmin(distances)

    def render_gaze_visualizations(self):
        """Render gaze analysis visualizations"""
        st.markdown("### Gaze and Physiological Analysis Results")

        # Check if analysis results and gaze_results exist
        if (st.session_state.analysis_results is None or
            "gaze_results" not in st.session_state.analysis_results):
            st.info("No gaze analysis results available. Run gaze analysis first.")
            return

        # Display gaze results for each junction
        for junction_key, gaze_data in st.session_state.analysis_results["gaze_results"].items():
            st.markdown(f"#### {junction_key.replace('_', ' ').title()}")

            if gaze_data is None:
                st.info("No gaze analysis data available for this junction")
                continue

            # Check if we have comprehensive gaze data or fallback data
            if isinstance(gaze_data, dict):
                if 'error' in gaze_data:
                    # Show error information
                    st.error(f"❌ **Gaze Analysis Failed for {junction_key}**")
                    st.write(f"**Error:** {gaze_data['error']}")
                    st.write(f"**Error Type:** {gaze_data['error_type']}")

                    # Show suggestions based on error type
                    if "No assignments found" in gaze_data['error']:
                        st.info("💡 **Solution:** Run '🔍 Discover Branches' analysis first to create proper assignments")
                    elif "trajectory" in gaze_data['error'].lower():
                        st.info("💡 **Solution:** Check if trajectories actually pass through this junction")
                    elif "column" in gaze_data['error'].lower():
                        st.info("💡 **Solution:** Check your gaze column mappings in the Data tab")

                    # Show empty plots with error messages
                    self._render_error_gaze_results(gaze_data, junction_key)
                elif 'physiological' in gaze_data:
                    # Comprehensive gaze analysis results
                    self._render_comprehensive_gaze_results(gaze_data, junction_key)
                else:
                    # Fallback movement pattern results
                    st.warning("⚠️ Using fallback visualization")
                    self._render_fallback_gaze_results(gaze_data, junction_key)
            else:
                # Fallback movement pattern results
                st.warning("⚠️ Using fallback visualization")
                self._render_fallback_gaze_results(gaze_data, junction_key)

    def _render_fallback_gaze_results(self, gaze_data, junction_key):
        """Render fallback gaze results when no proper gaze analysis was performed."""
        st.markdown("**Gaze Analysis Results:**")

        if gaze_data is not None:
            # Check if this is actual gaze data DataFrame or movement data
            if hasattr(gaze_data, 'columns') and len(gaze_data) > 0:
                if 'analysis_type' in gaze_data.columns and gaze_data['analysis_type'].iloc[0] == 'gaze':
                    # This is actual gaze data
                    st.success("✅ Gaze analysis completed successfully!")

                    # Display gaze-specific metrics
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Total Trajectories", len(gaze_data))

                    with col2:
                        valid_head_yaw = gaze_data['head_yaw'].dropna()
                        if len(valid_head_yaw) > 0:
                            st.metric("Valid Head Directions", len(valid_head_yaw))
                        else:
                            st.metric("Valid Head Directions", 0)

                    with col3:
                        valid_pupil = gaze_data[['pupil_l', 'pupil_r']].dropna()
                        if len(valid_pupil) > 0:
                            st.metric("Valid Pupil Data", len(valid_pupil))
                        else:
                            st.metric("Valid Pupil Data", 0)

                    with col4:
                        valid_hr = gaze_data['heart_rate'].dropna()
                        if len(valid_hr) > 0:
                            st.metric("Valid Heart Rate", len(valid_hr))
                        else:
                            st.metric("Valid Heart Rate", 0)

                    # Show gaze data table
                    st.dataframe(gaze_data.head(20), width='stretch')

                    if len(gaze_data) > 20:
                        st.info(f"Showing first 20 of {len(gaze_data)} gaze records")

                    # Debug: Show what type of trajectory objects we have
                    st.markdown("**🔍 Debug Information:**")
                    sample_traj = st.session_state.trajectories[0] if st.session_state.trajectories else None
                    if sample_traj:
                        st.write(f"- Trajectory type: {type(sample_traj).__name__}")
                        st.write(f"- Available attributes: {[attr for attr in dir(sample_traj) if not attr.startswith('_')]}")
                        if hasattr(sample_traj, 'headset_gaze_x'):
                            st.write(f"- Has gaze data: ✅")
                        else:
                            st.write(f"- Has gaze data: ❌")

                    # Create gaze-specific visualizations
                    self._create_gaze_visualizations(gaze_data, junction_key)

                else:
                    # This is movement data - shouldn't happen in gaze analysis
                    st.warning("⚠️ **No Gaze Data Available**")
                    st.info("""
                    **Gaze analysis requires:**
                    - Eye tracking data (`Headset.Gaze.X`, `Headset.Gaze.Y`)
                    - Head orientation data (`Headset.Head.Forward.X`, `Headset.Head.Forward.Z`)
                    - Physiological data (`Headset.PupilDilation.L`, `Headset.PupilDilation.R`, `Headset.HeartRate`)

                    **Current data only contains position information (x, z, t).**

                    **To perform gaze analysis:**
                    1. Load data with eye tracking sensors
                    2. Ensure column mappings are correct
                    3. Use VR headset with gaze tracking capabilities
                    """)
        else:
            st.warning("⚠️ **No Gaze Analysis Performed**")
            st.info("""
            **Gaze analysis was not performed because:**
            - No eye tracking data detected
            - Column mappings not properly configured
            - Data doesn't contain required gaze/physiological fields

            **Please check:**
            1. Your data contains gaze tracking columns
            2. Column mappings are correctly specified
            3. Data format matches VR headset export format
            """)

    def _render_error_gaze_results(self, gaze_data, junction_key):
        """Render error gaze results with informative empty plots."""
        import matplotlib.pyplot as plt
        import numpy as np

        st.markdown("**Gaze Analysis Results:**")

        # Create empty plots with error messages
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'Comprehensive Gaze Analysis - {junction_key}', fontsize=16, fontweight='bold')

        # 1. Heart Rate Change Distribution
        ax1 = axes[0, 0]
        ax1.text(0.5, 0.5, f'Analysis Failed\n{gaze_data["error"]}',
                ha='center', va='center', transform=ax1.transAxes, fontsize=10)
        ax1.set_title('Heart Rate Change Distribution')
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)

        # 2. Pupil Dilation Change Distribution
        ax2 = axes[0, 1]
        ax2.text(0.5, 0.5, f'Analysis Failed\n{gaze_data["error"]}',
                ha='center', va='center', transform=ax2.transAxes, fontsize=10)
        ax2.set_title('Pupil Dilation Change Distribution')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)

        # 3. Head Yaw Distribution
        ax3 = axes[1, 0]
        ax3.text(0.5, 0.5, f'Analysis Failed\n{gaze_data["error"]}',
                ha='center', va='center', transform=ax3.transAxes, fontsize=10)
        ax3.set_title('Head Yaw Distribution')
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)

        # 4. Gaze-Movement Difference Distribution
        ax4 = axes[1, 1]
        ax4.text(0.5, 0.5, f'Analysis Failed\n{gaze_data["error"]}',
                ha='center', va='center', transform=ax4.transAxes, fontsize=10)
        ax4.set_title('Gaze-Movement Difference Distribution')
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    def _render_comprehensive_gaze_results(self, gaze_data, junction_key):
        """Render comprehensive gaze analysis results (dictionary format)."""
        import pandas as pd

        # Display physiological data
        if 'physiological' in gaze_data and gaze_data['physiological'] is not None:
            physio_df = gaze_data['physiological']
            if len(physio_df) > 0:
                st.markdown("**Physiological Analysis:**")

                col1, col2, col3 = st.columns(3)
                with col1:
                    if 'heart_rate_change' in physio_df.columns:
                        valid_hr = physio_df['heart_rate_change'].dropna()
                        if len(valid_hr) > 0:
                            st.metric("Avg Heart Rate Change", f"{valid_hr.mean():.1f} bpm")
                        else:
                            st.metric("Avg Heart Rate Change", "N/A")
                    else:
                        st.metric("Avg Heart Rate Change", "Column not found")

                with col2:
                    if 'pupil_change' in physio_df.columns:
                        valid_pupil = physio_df['pupil_change'].dropna()
                        if len(valid_pupil) > 0:
                            st.metric("Avg Pupil Change", f"{valid_pupil.mean():.2f}")
                        else:
                            st.metric("Avg Pupil Change", "N/A")
                    else:
                        st.metric("Avg Pupil Change", "Column not found")

                with col3:
                    st.metric("Total Measurements", len(physio_df))

                st.dataframe(physio_df.head(10), width='stretch')

        # Display pupil dilation data
        if 'pupil_dilation' in gaze_data and gaze_data['pupil_dilation'] is not None:
            pupil_df = gaze_data['pupil_dilation']
            if len(pupil_df) > 0:
                st.markdown("**Pupil Dilation Analysis:**")

                col1, col2, col3 = st.columns(3)
                with col1:
                    valid_baseline = pupil_df['pupil_baseline'].dropna()
                    if len(valid_baseline) > 0:
                        st.metric("Avg Pupil Baseline", f"{valid_baseline.mean():.2f}")

                with col2:
                    valid_decision = pupil_df['pupil_decision'].dropna()
                    if len(valid_decision) > 0:
                        st.metric("Avg Pupil at Decision", f"{valid_decision.mean():.2f}")

                with col3:
                    valid_change = pupil_df['pupil_change'].dropna()
                    if len(valid_change) > 0:
                        st.metric("Avg Pupil Change", f"{valid_change.mean():.2f}")

                st.dataframe(pupil_df.head(10), width='stretch')

        # Display head yaw data
        if 'head_yaw' in gaze_data and gaze_data['head_yaw'] is not None:
            head_df = gaze_data['head_yaw']
            if len(head_df) > 0:
                st.markdown("**Head Yaw Analysis:**")

                col1, col2, col3 = st.columns(3)
                with col1:
                    valid_yaw = head_df['head_yaw'].dropna()
                    if len(valid_yaw) > 0:
                        st.metric("Avg Head Yaw", f"{valid_yaw.mean():.1f}°")

                with col2:
                    valid_diff = head_df['yaw_difference'].dropna()
                    if len(valid_diff) > 0:
                        st.metric("Avg Gaze-Movement Diff", f"{valid_diff.mean():.1f}°")

                with col3:
                    st.metric("Total Measurements", len(head_df))

                st.dataframe(head_df.head(10), width='stretch')

        # Create visualizations for comprehensive gaze data
        self._create_comprehensive_gaze_visualizations(gaze_data, junction_key)

        # Add advanced gaze plotting features from CLI version
        self._create_advanced_gaze_plots(gaze_data, junction_key)

        # Display pupil dilation heatmaps
        has_global_heatmap = st.session_state.analysis_results and 'pupil_heatmap_global' in st.session_state.analysis_results

        # Check if any junction has heatmap data
        has_junction_heatmaps = False
        if st.session_state.analysis_results and 'gaze_results' in st.session_state.analysis_results:
            gaze_results = st.session_state.analysis_results['gaze_results']
            has_junction_heatmaps = any(
                isinstance(gaze_data_item, dict) and 'pupil_heatmap_junction' in gaze_data_item
                for gaze_data_item in gaze_results.values()
            )

        if has_global_heatmap or has_junction_heatmaps:
            st.markdown("---")
            st.markdown("### 🗺️ Pupil Dilation Spatial Heatmaps")
            st.info("Spatial distribution of pupil dilation changes across the map")

            # Calculate consistent scaling across all heatmaps
            from verta.verta_gaze import get_consistent_pupil_scaling
            heatmap_data_list = []
            if has_global_heatmap:
                heatmap_data_list.append(st.session_state.analysis_results['pupil_heatmap_global'])
            if has_junction_heatmaps:
                gaze_results = st.session_state.analysis_results['gaze_results']
                for gaze_data_item in gaze_results.values():
                    if isinstance(gaze_data_item, dict) and 'pupil_heatmap_junction' in gaze_data_item:
                        heatmap_data_list.extend(gaze_data_item['pupil_heatmap_junction'].values())

            normalization = st.session_state.get('pupil_heatmap_normalization', 'relative')
            vmin, vmax = get_consistent_pupil_scaling(heatmap_data_list, normalization)

            st.write(f"🎨 **Consistent Color Scaling:** {vmin:.1f}% to {vmax:.1f}% (realistic pupil dilation range)")

            # Create tabs for global and per-junction views
            tab_junction, tab_global = st.tabs(["🎯 Junction Heatmap", "🌍 Global Heatmap"])

            with tab_junction:
                if has_junction_heatmaps:
                    st.markdown("#### Junction-Specific Pupil Patterns")
                    st.caption("Focused view of pupil changes at each junction (includes approach paths)")

                    # Display heatmap for current junction only
                    current_junction_heatmaps = {}
                    if isinstance(gaze_data, dict) and 'pupil_heatmap_junction' in gaze_data:
                        current_junction_heatmaps = gaze_data['pupil_heatmap_junction']

                    if len(current_junction_heatmaps) == 0:
                        st.info("ℹ️ No junction heatmaps available")
                    else:
                        for junction_idx, heatmap_data in current_junction_heatmaps.items():
                            with st.expander(f"**Junction {junction_idx}**", expanded=True):
                                # Check for errors
                                if heatmap_data.get('error'):
                                    st.warning(f"⚠️ {heatmap_data['error']}")
                                else:
                                    # Get junction info
                                    junction = heatmap_data.get('junction')
                                    r_outer = heatmap_data.get('r_outer', 'N/A')

                                    col_info1, col_info2 = st.columns(2)
                                    with col_info1:
                                        st.caption(f"📍 Center: ({junction.cx:.1f}, {junction.cz:.1f})")
                                    with col_info2:
                                        st.caption(f"📏 Analysis radius (r_outer): {r_outer}")

                                    # Check if pre-generated plot exists first
                                    junction_key = f"junction_{junction_idx}"
                                    pre_generated_plot_path = os.path.join("gui_outputs", f"junction_{junction_idx}", "gaze_plots", f"{junction_key}_pupil_heatmap.png")

                                    # Debug: Show the path being checked
                                    st.write(f"🔍 **Debug:** Looking for junction heatmap at: `{pre_generated_plot_path}`")
                                    st.write(f"🔍 **Debug:** File exists: {os.path.exists(pre_generated_plot_path)}")

                                    if os.path.exists(pre_generated_plot_path):
                                        # Load and display pre-generated plot
                                        st.image(pre_generated_plot_path, caption=f"Junction {junction_idx} Pupil Dilation (Pre-generated)")
                                        st.caption("📁 Plot generated during analysis")
                                    else:
                                        # Generate plot on-demand (fallback)
                                        st.caption("🔄 Generating plot on-demand...")
                                        st.write(f"⚠️ **Debug:** Pre-generated plot not found at `{pre_generated_plot_path}`")

                                        # Get filtered trajectories for this junction (only those passing through)
                                        filtered_trajs_for_plot = self._get_filtered_trajectories_for_junction(
                                            st.session_state.trajectories, junction, r_outer
                                        )

                                        # Plot junction heatmap (with minimap, with filtered trajectories only)
                                        from verta.verta_gaze import plot_pupil_dilation_heatmap
                                        fig_junction = plot_pupil_dilation_heatmap(
                                            heatmap_data=heatmap_data,
                                            junctions=[junction] if junction else None,
                                            trajectories=filtered_trajs_for_plot,
                                            all_trajectories=st.session_state.trajectories,  # Pass all trajectories for minimap
                                            title=f"Junction {junction_idx} Pupil Dilation",
                                            show_sample_counts=False,
                                            show_minimap=True,
                                            vmin=vmin,
                                            vmax=vmax
                                        )
                                        st.pyplot(fig_junction)
                                        import matplotlib.pyplot as plt
                                        plt.close(fig_junction)

                                    # Show junction statistics
                                    heatmap = heatmap_data['heatmap']
                                    # Ensure heatmap is a numpy array
                                    if not isinstance(heatmap, np.ndarray):
                                        heatmap = np.array(heatmap)
                                    valid_bins = heatmap[~np.isnan(heatmap)]
                                    norm_method = heatmap_data['normalization_used']

                                    if len(valid_bins) > 0:
                                        col1, col2, col3 = st.columns(3)

                                        with col1:
                                            if norm_method == "relative":
                                                st.metric("Mean Change", f"{np.mean(valid_bins):.2f}%")
                                            else:
                                                st.metric("Mean Z-score", f"{np.mean(valid_bins):.2f}")

                                        with col2:
                                            if norm_method == "relative":
                                                st.metric("Max Absolute Change", f"±{np.max(np.abs(valid_bins)):.2f}%")
                                            else:
                                                st.metric("Max Absolute Z-score", f"{np.max(np.abs(valid_bins)):.2f}")

                                        with col3:
                                            st.metric("Trajectories", f"{heatmap_data['valid_trajectories']}")
                else:
                    st.info("ℹ️ No junction heatmaps available")

            with tab_global:
                if has_global_heatmap:
                    st.markdown("#### Global Pupil Dilation Patterns")
                    st.caption("Shows pupil changes across the entire environment")

                    global_heatmap_data = st.session_state.analysis_results['pupil_heatmap_global']

                    # Check for errors
                    if global_heatmap_data.get('error'):
                        st.warning(f"⚠️ {global_heatmap_data['error']}")
                    else:
                        # Check if pre-generated global plot exists first
                        pre_generated_global_path = os.path.join("gui_outputs", "gaze_plots", "global_pupil_heatmap.png")

                        # Debug: Show the path being checked
                        st.write(f"🔍 **Debug:** Looking for global heatmap at: `{pre_generated_global_path}`")
                        st.write(f"🔍 **Debug:** File exists: {os.path.exists(pre_generated_global_path)}")

                        if os.path.exists(pre_generated_global_path):
                            # Load and display pre-generated plot
                            st.image(pre_generated_global_path, caption="Global Pupil Dilation Heatmap (Pre-generated)")
                            st.caption("📁 Plot generated during analysis")
                        else:
                            # Generate plot on-demand (fallback)
                            st.caption("🔄 Generating global heatmap on-demand...")
                            st.write(f"⚠️ **Debug:** Pre-generated plot not found at `{pre_generated_global_path}`")

                            # Plot global heatmap (only once, without minimap)
                            from verta.verta_gaze import plot_pupil_dilation_heatmap
                            fig_global = plot_pupil_dilation_heatmap(
                                heatmap_data=global_heatmap_data,
                                junctions=st.session_state.junctions,
                                trajectories=st.session_state.trajectories,
                                title="Global Pupil Dilation Heatmap",
                                show_sample_counts=False,
                                show_minimap=False,
                                vmin=vmin,
                                vmax=vmax
                            )
                            st.pyplot(fig_global)
                            import matplotlib.pyplot as plt
                            plt.close(fig_global)

                        # Show statistics
                        heatmap = global_heatmap_data['heatmap']
                        # Ensure heatmap is a numpy array
                        if not isinstance(heatmap, np.ndarray):
                            heatmap = np.array(heatmap)
                        valid_bins = heatmap[~np.isnan(heatmap)]
                        norm_method = global_heatmap_data['normalization_used']

                        if len(valid_bins) > 0:
                            col1, col2, col3, col4 = st.columns(4)

                            with col1:
                                if norm_method == "relative":
                                    st.metric("Mean Change", f"{np.mean(valid_bins):.2f}%")
                                else:
                                    st.metric("Mean Z-score", f"{np.mean(valid_bins):.2f}")

                            with col2:
                                if norm_method == "relative":
                                    st.metric("Max Dilation", f"+{np.max(valid_bins):.2f}%")
                                else:
                                    st.metric("Max Z-score", f"{np.max(valid_bins):.2f}")

                            with col3:
                                if norm_method == "relative":
                                    st.metric("Max Constriction", f"{np.min(valid_bins):.2f}%")
                                else:
                                    st.metric("Min Z-score", f"{np.min(valid_bins):.2f}")

                            with col4:
                                st.metric("Valid Bins", f"{len(valid_bins)}/{heatmap.size}")
                else:
                    st.info("ℹ️ Global heatmap not available")

    def _create_comprehensive_gaze_visualizations(self, gaze_data, junction_key):
        """Create visualizations for comprehensive gaze analysis results."""
        import matplotlib.pyplot as plt
        import numpy as np

        # Ensure expected column names exist for plotting
        if isinstance(gaze_data, dict):
            gaze_data = self._normalize_gaze_result_frames(gaze_data)

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Comprehensive Gaze Analysis - {junction_key.replace("_", " ").title()}', fontsize=14)

        # 1. Heart Rate Distribution
        ax1 = axes[0, 0]
        if 'physiological' in gaze_data and gaze_data['physiological'] is not None:
            physio_data = gaze_data['physiological']

            # Handle both DataFrame and list formats
            if isinstance(physio_data, list):
                import pandas as pd
                physio_df = pd.DataFrame(physio_data)
            else:
                physio_df = physio_data

            if 'heart_rate_change' in physio_df.columns:
                hr_data = physio_df['heart_rate_change'].dropna()
                if len(hr_data) > 0:
                    ax1.hist(hr_data, bins=15, alpha=0.7, color='red', edgecolor='black')
                    ax1.set_title('Heart Rate Change Distribution\n(Baseline: Normal navigation 2-5s before junction entry)')
                    ax1.set_xlabel('Heart Rate Change (bpm)')
                    ax1.set_ylabel('Frequency')
                    ax1.grid(True, alpha=0.3)
                else:
                    ax1.text(0.5, 0.5, 'No heart rate data', ha='center', va='center', transform=ax1.transAxes)
                    ax1.set_title('Heart Rate Change Distribution\n(Baseline: Normal navigation 2-5s before junction entry)')
            else:
                ax1.text(0.5, 0.5, 'heart_rate_change column not found', ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('Heart Rate Change Distribution')
        else:
            ax1.text(0.5, 0.5, 'No physiological data', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Heart Rate Change Distribution')

        # 2. Pupil Dilation Change
        ax2 = axes[0, 1]
        if 'pupil_dilation' in gaze_data and gaze_data['pupil_dilation'] is not None:
            pupil_data = gaze_data['pupil_dilation']

            # Handle both DataFrame and list formats
            if isinstance(pupil_data, list):
                import pandas as pd
                pupil_df = pd.DataFrame(pupil_data)
            else:
                pupil_df = pupil_data

            if 'pupil_change' in pupil_df.columns:
                pupil_values = pupil_df['pupil_change'].dropna()
                if len(pupil_values) > 0:
                    ax2.hist(pupil_values, bins=15, alpha=0.7, color='green', edgecolor='black')
                    ax2.set_title('Pupil Dilation Change Distribution\n(Baseline: Normal navigation 2-5s before junction entry)')
                    ax2.set_xlabel('Pupil Change (mm)')
                    ax2.set_ylabel('Frequency')
                    ax2.grid(True, alpha=0.3)
                else:
                    ax2.text(0.5, 0.5, 'No pupil data', ha='center', va='center', transform=ax2.transAxes)
                    ax2.set_title('Pupil Dilation Change Distribution\n(Baseline: Normal navigation 2-5s before junction entry)')
            else:
                ax2.text(0.5, 0.5, 'pupil_change column not found', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Pupil Dilation Change Distribution')
        else:
            ax2.text(0.5, 0.5, 'No pupil dilation data', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Pupil Dilation Change Distribution')

        # 3. Head Yaw Distribution
        ax3 = axes[1, 0]
        if 'head_yaw' in gaze_data and gaze_data['head_yaw'] is not None:
            yaw_data = gaze_data['head_yaw']

            # Handle both DataFrame and list formats
            if isinstance(yaw_data, list):
                import pandas as pd
                yaw_df = pd.DataFrame(yaw_data)
            else:
                yaw_df = yaw_data

            if 'head_yaw' in yaw_df.columns:
                yaw_values = yaw_df['head_yaw'].dropna()
                if len(yaw_values) > 0:
                    ax3.hist(yaw_values, bins=20, alpha=0.7, color='blue', edgecolor='black')
                    ax3.set_title('Head Yaw Distribution')
                    ax3.set_xlabel('Head Yaw (degrees)')
                    ax3.set_ylabel('Frequency')
                    ax3.grid(True, alpha=0.3)
                else:
                    ax3.text(0.5, 0.5, 'No head yaw data', ha='center', va='center', transform=ax3.transAxes)
                    ax3.set_title('Head Yaw Distribution')
            else:
                ax3.text(0.5, 0.5, 'head_yaw column not found', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Head Yaw Distribution')
        else:
            ax3.text(0.5, 0.5, 'No head yaw data', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Head Yaw Distribution')

        # 4. Gaze-Movement Difference
        ax4 = axes[1, 1]
        if 'head_yaw' in gaze_data and gaze_data['head_yaw'] is not None:
            yaw_data = gaze_data['head_yaw']

            # Handle both DataFrame and list formats
            if isinstance(yaw_data, list):
                import pandas as pd
                yaw_df = pd.DataFrame(yaw_data)
            else:
                yaw_df = yaw_data

            if 'yaw_difference' in yaw_df.columns:
                diff_values = yaw_df['yaw_difference'].dropna()
                if len(diff_values) > 0:
                    ax4.hist(diff_values, bins=20, alpha=0.7, color='purple', edgecolor='black')
                    ax4.set_title('Gaze-Movement Difference Distribution')
                    ax4.set_xlabel('Difference (degrees)')
                    ax4.set_ylabel('Frequency')
                    ax4.grid(True, alpha=0.3)
                else:
                    ax4.text(0.5, 0.5, 'No gaze-movement data', ha='center', va='center', transform=ax4.transAxes)
                    ax4.set_title('Gaze-Movement Difference Distribution')
            else:
                ax4.text(0.5, 0.5, 'yaw_difference column not found', ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Gaze-Movement Difference Distribution')
        else:
            ax4.text(0.5, 0.5, 'No gaze-movement data', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Gaze-Movement Difference Distribution')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    def _generate_gaze_plots_during_analysis(self, gaze_data, junction_key, out_dir):
        """Generate gaze plots during analysis (not just in visualization tab)."""
        import os
        import matplotlib.pyplot as plt

        # Create gaze plots directory
        gaze_plots_dir = os.path.join("gui_outputs", f"junction_{junction_key.split('_')[1]}", "gaze_plots")
        os.makedirs(gaze_plots_dir, exist_ok=True)

        # Normalize result frames so downstream plotting functions find expected columns
        if isinstance(gaze_data, dict):
            gaze_data = self._normalize_gaze_result_frames(gaze_data)

        # Get trajectories from session state for plotting
        trajectories = st.session_state.get('trajectories', [])
        if not trajectories:
            return

        # Get junction information
        junction = gaze_data.get('junction')
        r_outer = gaze_data.get('r_outer')

        if junction is None:
            return

        # Generate gaze directions plot
        try:
            from verta.verta_gaze import plot_gaze_directions_at_junctions

            plot_path = os.path.join(gaze_plots_dir, f"{junction_key}_gaze_directions.png")

            # Use fresh head_yaw data from gaze analysis (not old cached data)
            gaze_df = gaze_data.get('head_yaw')
            if gaze_df is not None and not gaze_df.empty:
                # Debug: Show branch information in the fresh data
                unique_branches = sorted(gaze_df['branch'].unique()) if 'branch' in gaze_df.columns else []
                print(f"🔍 **Fresh head_yaw data branches**: {unique_branches}")
                print(f"🔍 **Fresh head_yaw data shape**: {gaze_df.shape}")
                print(f"🔍 **Fresh head_yaw data columns**: {list(gaze_df.columns)}")

                # Additional debugging for J3+ junctions
                junction_num = junction_key.split('_')[1] if '_' in junction_key else '0'
                if int(junction_num) >= 3:
                    print(f"🔍 **JUNCTION {junction_num} DEBUG**:")
                    print(f"- Junction: {junction}")
                    print(f"- R_outer: {r_outer}")
                    print(f"- Trajectories passed to plotting: {len(trajectories)}")
                    print(f"- Head_yaw data rows: {len(gaze_df)}")
                    print(f"- Branch distribution: {gaze_df['branch'].value_counts().to_dict() if 'branch' in gaze_df.columns else 'No branch column'}")

                # Show sample of the fresh data
                if len(gaze_df) > 0:
                    print(f"🔍 **Sample fresh head_yaw data**:")
                    print(gaze_df.head())

                plot_gaze_directions_at_junctions(
                    trajectories=trajectories,
                    junctions=[junction],
                    gaze_df=gaze_df,
                    out_path=plot_path,
                    r_outer_list=[r_outer] if r_outer else [None],
                    junction_labels=[f"Junction {junction_key.split('_')[1]}"],
                )
        except Exception as e:
            print(f"Could not generate gaze directions plot: {e}")

        # Generate physiological analysis plot
        try:
            from verta.verta_gaze import plot_physiological_by_branch

            plot_path = os.path.join(gaze_plots_dir, f"{junction_key}_physiological_analysis.png")
            physio_df = gaze_data.get('physiological')

            if physio_df is not None and not physio_df.empty:
                # Debug: Show branch information in the fresh physiological data
                unique_branches = sorted(physio_df['branch'].unique()) if 'branch' in physio_df.columns else []
                print(f"**Fresh physiological data branches**: {unique_branches}")
                print(f"**Fresh physiological data shape**: {physio_df.shape}")

                # Additional debugging for J3+ junctions
                junction_num = junction_key.split('_')[1] if '_' in junction_key else '0'
                if int(junction_num) >= 3:
                    print(f"🔍 **JUNCTION {junction_num} PHYSIO DEBUG**:")
                    print(f"- Physiological data rows: {len(physio_df)}")
                    print(f"- Branch distribution: {physio_df['branch'].value_counts().to_dict() if 'branch' in physio_df.columns else 'No branch column'}")
                    print(f"- Sample physio data:")
                    print(physio_df.head())

                plot_physiological_by_branch(
                    physio_df=physio_df,
                    out_path=plot_path,
                )
        except Exception as e:
            print(f"Could not generate physiological analysis plot: {e}")

        # Generate pupil trajectory analysis plot
        try:
            from verta.verta_gaze import plot_pupil_trajectory_analysis

            plot_path = os.path.join(gaze_plots_dir, f"{junction_key}_pupil_trajectory.png")
            pupil_df = gaze_data.get('pupil')

            if pupil_df is not None and not pupil_df.empty:
                plot_pupil_trajectory_analysis(
                    pupil_traj_df=pupil_df,
                    out_path=plot_path,
                )
        except Exception as e:
            print(f"Could not generate pupil trajectory plot: {e}")

        # Generate junction-specific heatmap plot
        try:
            from verta.verta_gaze import plot_pupil_dilation_heatmap
            import numpy as np
            import matplotlib.pyplot as plt

            # Get heatmap data from gaze_data
            junction_heatmaps = gaze_data.get('pupil_heatmap_junction')
            st.write(f"🔍 **Debug:** Junction heatmaps available: {junction_heatmaps is not None}")
            if junction_heatmaps:
                st.write(f"🔍 **Debug:** Junction heatmaps length: {len(junction_heatmaps)}")
                st.write(f"🔍 **Debug:** Junction heatmaps keys: {list(junction_heatmaps.keys())}")
            else:
                st.write(f"🔍 **Debug:** No junction heatmaps found in gaze_data")

            if junction_heatmaps and len(junction_heatmaps) > 0:
                # Get the heatmap data for this junction
                junction_idx = None
                st.write(f"🔍 **Debug:** Looking for junction: ({junction.cx}, {junction.cz}, r={junction.r})")
                for idx, junc in enumerate(st.session_state.junctions):
                    st.write(f"🔍 **Debug:** Checking junction {idx}: ({junc.cx}, {junc.cz}, r={junc.r})")
                    if (junc.cx == junction.cx and junc.cz == junction.cz and junc.r == junction.r):
                        junction_idx = idx
                        st.write(f"🔍 **Debug:** Found matching junction at index {junction_idx}")
                        break

                st.write(f"🔍 **Debug:** Junction index found: {junction_idx}")

                if junction_idx is not None and junction_idx in junction_heatmaps:
                    heatmap_data = junction_heatmaps[junction_idx]
                    st.write(f"🔍 **Debug:** Found heatmap data for junction {junction_idx}")

                    # Create heatmap plot
                    plot_path = os.path.join(gaze_plots_dir, f"{junction_key}_pupil_heatmap.png")
                    st.write(f"🔍 **Debug:** Plot path: {plot_path}")

                    # Filter trajectories for this junction (same logic as visualizations tab)
                    filtered_trajs_for_plot = []
                    for traj in trajectories:
                        # Check if trajectory passes through this junction
                        rx = traj.x - junction.cx
                        rz = traj.z - junction.cz
                        r = np.hypot(rx, rz)
                        if np.any(r <= junction.r):
                            filtered_trajs_for_plot.append(traj)

                    st.write(f"🔍 **Debug:** Filtered trajectories for plot: {len(filtered_trajs_for_plot)}")

                    # Create the heatmap plot
                    st.write(f"🔍 **Debug:** Creating heatmap plot...")
                    try:
                        fig = plot_pupil_dilation_heatmap(
                            heatmap_data=heatmap_data,
                            junctions=[junction],
                            trajectories=filtered_trajs_for_plot,
                            all_trajectories=trajectories,  # Pass all trajectories for minimap
                            title=f"Junction {junction_idx} Pupil Dilation",
                            show_sample_counts=False,
                            show_minimap=True,
                            vmin=None,  # Let the function determine scaling
                            vmax=None
                        )

                        st.write(f"🔍 **Debug:** Heatmap plot created successfully")

                        # Save the plot
                        st.write(f"🔍 **Debug:** Saving plot to {plot_path}...")
                        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
                        plt.close(fig)

                        print(f"Junction heatmap plot saved to: {plot_path}")
                        st.write(f"🔍 **Debug:** Junction plot saved to: `{plot_path}`")
                        st.write(f"🔍 **Debug:** File exists after save: {os.path.exists(plot_path)}")
                    except Exception as plot_error:
                        st.write(f"❌ **Debug:** Error creating heatmap plot: {plot_error}")
                        st.write(f"❌ **Debug:** Error type: {type(plot_error)}")
                        import traceback
                        st.write(f"❌ **Debug:** Traceback: {traceback.format_exc()}")
                        print(f"Error creating junction heatmap plot: {plot_error}")
                        print(f"Traceback: {traceback.format_exc()}")
                else:
                    st.write(f"🔍 **Debug:** No heatmap data found for junction {junction_idx}")
                    print(f"No heatmap data found for junction {junction_idx}")
            else:
                print(f"No junction heatmaps available for {junction_key}")

        except Exception as e:
            print(f"Could not generate junction heatmap plot: {e}")

    def _create_advanced_gaze_plots(self, gaze_data, junction_key):
        """Create advanced gaze plots using CLI plotting functions."""
        import matplotlib.pyplot as plt
        import tempfile
        import os

        # Normalize result frames so downstream plotting functions find expected columns
        if isinstance(gaze_data, dict):
            gaze_data = self._normalize_gaze_result_frames(gaze_data)

        st.markdown("### Advanced Gaze Analysis Plots")

        # Get trajectories from session state for plotting
        trajectories = st.session_state.get('trajectories', [])
        if not trajectories:
            st.warning("⚠️ No trajectories available for advanced plotting")
            return

        # Get junction information
        junction = gaze_data.get('junction')
        r_outer = gaze_data.get('r_outer')

        if junction is None:
            st.warning("⚠️ No junction information available for plotting")
            return

        # Check if plots were already generated during analysis
        junction_num = junction_key.split('_')[1] if '_' in junction_key else '0'
        analysis_plots_dir = os.path.join("gui_outputs", f"junction_{junction_num}", "gaze_plots")

        # Define plot paths
        gaze_directions_path = os.path.join(analysis_plots_dir, f"{junction_key}_gaze_directions.png")
        physio_path = os.path.join(analysis_plots_dir, f"{junction_key}_physiological_analysis.png")
        pupil_path = os.path.join(analysis_plots_dir, f"{junction_key}_pupil_trajectory.png")

        # If plots exist from analysis, display them instead of regenerating
        if os.path.exists(analysis_plots_dir):
            st.info(f"📊 **Displaying plots generated during analysis**")


            if os.path.exists(gaze_directions_path):
                st.markdown("#### Gaze Directions at Junction")
                st.image(gaze_directions_path, caption="👁️ Gaze directions at decision points")

            if os.path.exists(physio_path):
                st.markdown("#### Physiological Analysis")
                st.image(physio_path, caption="📊 Decision Point Analysis: Physiological changes during junction approach")

            if os.path.exists(pupil_path):
                st.markdown("#### Pupil Trajectory Analysis")
                st.image(pupil_path, caption="🗺️ Junction Area Analysis: Pupil changes across entire junction region")

            return

        # Fallback: Generate plots if they don't exist (for backward compatibility)
        st.info(f"📊 **Generating plots on-demand**")

        # Create gaze plots directory in gui_outputs
        gaze_plots_dir = os.path.join("gui_outputs", "gaze_plots")
        os.makedirs(gaze_plots_dir, exist_ok=True)

        # Only generate consistency report (plots are now generated during analysis)
        try:
            # Gaze Consistency Report
            if 'head_yaw' in gaze_data and gaze_data['head_yaw'] is not None:
                head_yaw_df = gaze_data['head_yaw']
                if len(head_yaw_df) > 0:
                    st.markdown("#### Gaze-Movement Consistency Report")

                    # Import the reporting function
                    from verta.verta_gaze import gaze_movement_consistency_report

                    consistency_report = gaze_movement_consistency_report(head_yaw_df)

                    # Display the report
                    if 'error' not in consistency_report:
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric(
                                "Mean Absolute Yaw Difference",
                                f"{consistency_report.get('mean_absolute_yaw_difference', 0):.1f}°"
                            )

                        with col2:
                            st.metric(
                                "Aligned Percentage",
                                f"{consistency_report.get('aligned_percentage', 0):.1f}%"
                            )

                        with col3:
                            st.metric(
                                "Total Decisions",
                                consistency_report.get('total_decisions', 0)
                            )

                        # Show branch-specific alignment
                        branch_metrics = {k: v for k, v in consistency_report.items()
                                       if k.startswith('branch_') and k.endswith('_alignment')}

                        if branch_metrics:
                            st.markdown("**Branch-Specific Alignment:**")
                            for branch_key, alignment in branch_metrics.items():
                                branch_num = branch_key.replace('branch_', '').replace('_alignment', '')
                                st.write(f"- Branch {branch_num}: {alignment:.1f}° average difference")

                        # Save consistency report to file
                        import json
                        consistency_file = os.path.join(gaze_plots_dir, f"{junction_key}_gaze_consistency_report.json")
                        with open(consistency_file, 'w') as f:
                            json.dump(consistency_report, f, indent=2)
                        st.info(f"📁 Consistency report saved to: {consistency_file}")

                    else:
                        st.warning(f"⚠️ {consistency_report['error']}")

        except Exception as e:
            st.error(f"❌ Error creating advanced gaze plots: {e}")
            import traceback
            st.error(f"**Error details:** {traceback.format_exc()}")

    def _create_gaze_visualizations(self, gaze_data, junction_key):
        """Create gaze-specific visualizations."""
        import matplotlib.pyplot as plt
        import numpy as np

        if len(gaze_data) == 0:
            st.info("No gaze data available for visualization")
            return

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Gaze Analysis Visualizations - {junction_key.replace("_", " ").title()}', fontsize=14)

        # 1. Head Direction Distribution
        ax1 = axes[0, 0]
        head_yaw_data = gaze_data['head_yaw'].dropna()
        if len(head_yaw_data) > 0:
            ax1.hist(head_yaw_data, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_title('Head Direction Distribution')
            ax1.set_xlabel('Head Yaw (degrees)')
            ax1.set_ylabel('Frequency')
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'No head direction data', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Head Direction Distribution')

        # 2. Pupil Dilation Comparison
        ax2 = axes[0, 1]
        pupil_l_data = gaze_data['pupil_l'].dropna()
        pupil_r_data = gaze_data['pupil_r'].dropna()
        if len(pupil_l_data) > 0 and len(pupil_r_data) > 0:
            ax2.scatter(pupil_l_data, pupil_r_data, alpha=0.6, color='green')
            ax2.set_title('Left vs Right Pupil Dilation')
            ax2.set_xlabel('Left Pupil Size')
            ax2.set_ylabel('Right Pupil Size')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No pupil data available', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Left vs Right Pupil Dilation')

        # 3. Heart Rate Distribution
        ax3 = axes[1, 0]
        hr_data = gaze_data['heart_rate'].dropna()
        if len(hr_data) > 0:
            ax3.hist(hr_data, bins=15, alpha=0.7, color='red', edgecolor='black')
            ax3.set_title('Heart Rate Distribution')
            ax3.set_xlabel('Heart Rate (bpm)')
            ax3.set_ylabel('Frequency')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No heart rate data', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Heart Rate Distribution')

        # 4. Gaze Direction Scatter
        ax4 = axes[1, 1]
        gaze_x_data = gaze_data['gaze_x'].dropna()
        gaze_y_data = gaze_data['gaze_y'].dropna()
        if len(gaze_x_data) > 0 and len(gaze_y_data) > 0:
            ax4.scatter(gaze_x_data, gaze_y_data, alpha=0.6, color='purple')
            ax4.set_title('Gaze Direction Scatter')
            ax4.set_xlabel('Gaze X')
            ax4.set_ylabel('Gaze Y')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No gaze direction data', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Gaze Direction Scatter')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    def _create_movement_visualizations(self, gaze_data, junction_key):
        """Create simple movement pattern visualizations."""
        import matplotlib.pyplot as plt
        import numpy as np

        if len(gaze_data) == 0:
            return

        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Movement direction histogram
        valid_movements = gaze_data['movement_yaw'].dropna()
        if len(valid_movements) > 0:
            ax1.hist(valid_movements, bins=20, alpha=0.7, edgecolor='black')
            ax1.set_title('Movement Direction Distribution')
            ax1.set_xlabel('Movement Yaw (degrees)')
            ax1.set_ylabel('Frequency')
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'No movement data available',
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Movement Direction Distribution')

        # 2. Junction visit order distribution (new diagnostic plot)
        if 'visit_order' in gaze_data.columns:
            visit_orders = gaze_data['visit_order'].dropna()
            if len(visit_orders) > 0:
                ax2.hist(visit_orders, bins=range(int(visit_orders.max()) + 2), alpha=0.7, edgecolor='black', color='orange')
                ax2.set_title('Junction Visit Order Distribution')
                ax2.set_xlabel('Visit Order (0=first junction, 1=second junction, etc.)')
                ax2.set_ylabel('Frequency')
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'No visit order data',
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Junction Visit Order Distribution')
        else:
            ax2.text(0.5, 0.5, 'No visit order data available',
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Junction Visit Order Distribution')

        # 3. Distance from center distribution
        distances = gaze_data['distance_from_center'].dropna()
        if len(distances) > 0:
            ax3.hist(distances, bins=20, alpha=0.7, edgecolor='black', color='green')
            ax3.set_title('Distance from Junction Center')
            ax3.set_xlabel('Distance (units)')
            ax3.set_ylabel('Frequency')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No distance data available',
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Distance from Junction Center')

        # 4. Decision points scatter plot
        ax4.scatter(gaze_data['decision_x'], gaze_data['decision_z'], alpha=0.6, s=20)
        ax4.set_title('Decision Points Location')
        ax4.set_xlabel('X Position')
        ax4.set_ylabel('Z Position')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save the plot
        import os
        junction_num = junction_key.split('_')[1]
        junction_dir = os.path.join("gui_outputs", f"junction_{junction_num}")
        os.makedirs(junction_dir, exist_ok=True)

        plot_path = os.path.join(junction_dir, "Movement_Patterns.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Display the plot
        st.image(plot_path, caption=f"Movement Pattern Analysis - {junction_key}", width='stretch')

    def generate_cli_command(self, analysis_type: str, results: dict, cluster_method: str = "dbscan", cluster_params: dict = None, decision_mode: str = "hybrid", decision_params: dict = None):
        """Generate CLI command for easy copying"""
        st.markdown("### 📋 Command Line Output")
        st.markdown("Copy and paste this command to run the same analysis in the terminal:")

        if analysis_type == "discover":
            # Generate discover commands for each junction
            for junction_key, branch_data in results.items():
                # Skip non-junction keys like "chain_decisions"
                if junction_key == "chain_decisions" or not junction_key.startswith("junction_"):
                    continue

                junction_num = junction_key.split('_')[1]
                junction = st.session_state.junctions[int(junction_num)]
                r_outer = st.session_state.junction_r_outer.get(int(junction_num), 50.0)

                st.markdown(f"#### {junction_key.replace('_', ' ').title()}")

                # Generate the CLI command with current cluster method and parameters
                cli_command = f"""verta discover \\
  --input ./data \\
  --columns x=Headset.Head.Position.X,z=Headset.Head.Position.Z,t=Time \\
  --scale 0.2 \\
  --junction {junction.cx:.1f} {junction.cz:.1f} \\
  --radius {junction.r:.1f} \\
  --r_outer {r_outer:.1f} \\
  --distance 100.0 \\
  --epsilon 0.05 \\
  --k 3 \\
  --decision_mode {decision_mode} \\
  --cluster_method {cluster_method}"""

                # Add cluster method specific parameters
                if cluster_method == "dbscan" and cluster_params:
                    cli_command += f" \\\n  --min_samples {cluster_params.get('min_samples', 5)} \\\n  --angle_eps {cluster_params.get('angle_eps', 15.0)}"
                elif cluster_method == "kmeans" and cluster_params:
                    cli_command += f" \\\n  --k_min {cluster_params.get('k_min', 2)} \\\n  --k_max {cluster_params.get('k_max', 6)}"
                elif cluster_method == "auto" and cluster_params:
                    cli_command += f" \\\n  --k_min {cluster_params.get('k_min', 2)} \\\n  --k_max {cluster_params.get('k_max', 6)} \\\n  --min_sep_deg {cluster_params.get('min_sep_deg', 12.0)} \\\n  --angle_eps {cluster_params.get('angle_eps', 15.0)}"

                # Add decision mode specific parameters
                if decision_mode == "radial" and decision_params:
                    cli_command += f" \\\n  --r_outer {decision_params.get('r_outer', 50.0)} \\\n  --epsilon {decision_params.get('epsilon', 0.05)}"
                elif decision_mode == "pathlen" and decision_params:
                    cli_command += f" \\\n  --distance {decision_params.get('path_length', 100.0)} \\\n  --linger_delta {decision_params.get('linger_delta', 0.0)}"
                elif decision_mode == "hybrid" and decision_params:
                    cli_command += f" \\\n  --r_outer {decision_params.get('r_outer', 50.0)} \\\n  --distance {decision_params.get('path_length', 100.0)}"

                cli_command += f" \\\n  --out ./outputs/{junction_key}"

                st.code(cli_command, language="bash")

                # Show branch statistics
                if "summary" in branch_data and branch_data["summary"] is not None:
                    st.markdown("**Branch Statistics:**")
                    summary_df = branch_data["summary"]
                    for _, row in summary_df.iterrows():
                        st.write(f"- Branch {int(row['branch'])}: {int(row['count'])} trajectories ({row['percent']:.1f}%)")

        elif analysis_type == "assign":
            # Generate assign commands for each junction
            for junction_key, assignment_data in results.items():
                junction_num = junction_key.split('_')[1]

                # Get junction info from assignment data or session state
                if "junction" in assignment_data:
                    junction = assignment_data["junction"]
                    # Try to get r_outer from assignment data, then session state, then default
                    r_outer = assignment_data.get("r_outer",
                                                st.session_state.junction_r_outer.get(int(junction_num), 50.0))
                else:
                    junction = st.session_state.junctions[int(junction_num)]
                    r_outer = st.session_state.junction_r_outer.get(int(junction_num), 50.0)

                st.markdown(f"#### {junction_key.replace('_', ' ').title()}")

                # Get parameters from assignment data
                path_length = assignment_data.get("path_length", 100.0)
                epsilon = assignment_data.get("epsilon", 0.05)
                assign_scale = assignment_data.get("assign_scale", 0.2)  # Get assign-specific scale factor

                # Generate the CLI command (requires centers file from discover)
                cli_command = f"""verta assign \\
  --input ./data \\
  --columns x=Headset.Head.Position.X,z=Headset.Head.Position.Z,t=Time \\
  --scale {assign_scale:.1f} \\
  --junction {junction.cx:.1f} {junction.cz:.1f} \\
  --radius {junction.r:.1f} \\
  --r_outer {r_outer:.1f} \\
  --distance {path_length:.1f} \\
  --epsilon {epsilon:.3f} \\
  --decision_mode pathlen \\
  --centers ./outputs/{junction_key}/branch_centers.npy \\
  --out ./outputs/{junction_key}_assign"""

                st.code(cli_command, language="bash")

                # Show assignment statistics
                if "assignments" in assignment_data:
                    assignments_df = assignment_data["assignments"]
                    st.markdown("**Assignment Statistics:**")
                    branch_counts = assignments_df['branch'].value_counts().sort_index()
                    total = len(assignments_df)
                    for branch, count in branch_counts.items():
                        percentage = (count / total * 100) if total > 0 else 0
                        st.write(f"- Branch {int(branch)}: {int(count)} trajectories ({percentage:.1f}%)")

        elif analysis_type == "predict":
            # Generate predict command for all junctions
            junctions_str = " ".join([f"{j.cx:.1f} {j.cz:.1f} {j.r:.1f}" for j in st.session_state.junctions])
            r_outer_str = " ".join([str(st.session_state.junction_r_outer.get(i, 50.0)) for i in range(len(st.session_state.junctions))])

            cli_command = f"""verta predict \\
  --input ./data \\
  --columns x=Headset.Head.Position.X,z=Headset.Head.Position.Z,t=Time \\
  --scale 0.2 \\
  --junctions {junctions_str} \\
  --r_outer_list {r_outer_str} \\
  --distance 100.0 \\
  --decision_mode {decision_mode} \\
  --cluster_method {cluster_method}"""

            # Add cluster method specific parameters
            if cluster_method == "dbscan" and cluster_params:
                cli_command += f" \\\n  --min_samples {cluster_params.get('min_samples', 5)} \\\n  --angle_eps {cluster_params.get('angle_eps', 15.0)}"
            elif cluster_method == "kmeans" and cluster_params:
                cli_command += f" \\\n  --k_min {cluster_params.get('k_min', 2)} \\\n  --k_max {cluster_params.get('k_max', 6)}"
            elif cluster_method == "auto" and cluster_params:
                cli_command += f" \\\n  --k_min {cluster_params.get('k_min', 2)} \\\n  --k_max {cluster_params.get('k_max', 6)} \\\n  --min_sep_deg {cluster_params.get('min_sep_deg', 12.0)} \\\n  --angle_eps {cluster_params.get('angle_eps', 15.0)}"

            # Add decision mode specific parameters
            if decision_mode == "radial" and decision_params:
                cli_command += f" \\\n  --r_outer_list {decision_params.get('r_outer', 50.0)} \\\n  --epsilon {decision_params.get('epsilon', 0.05)}"
            elif decision_mode == "pathlen" and decision_params:
                cli_command += f" \\\n  --distance {decision_params.get('path_length', 100.0)} \\\n  --linger_delta {decision_params.get('linger_delta', 0.0)}"
            elif decision_mode == "hybrid" and decision_params:
                cli_command += f" \\\n  --r_outer_list {decision_params.get('r_outer', 50.0)} \\\n  --distance {decision_params.get('path_length', 100.0)}"

            cli_command += f" \\\n  --out ./outputs/prediction"

            st.code(cli_command, language="bash")

        elif analysis_type == "metrics":
            # Generate metrics commands for each junction
            for junction_key, metrics_data in results.items():
                if not junction_key.startswith("junction_"):
                    continue

                junction_num = junction_key.split('_')[1]
                junction = metrics_data.get("junction")
                if junction is None:
                    continue

                r_outer = metrics_data.get("r_outer_value", metrics_data.get("r_outer", 50.0))
                decision_mode = metrics_data.get("decision_mode", "pathlen")
                distance = metrics_data.get("distance", 100.0)
                trend_window = metrics_data.get("trend_window", 5)
                min_outward = metrics_data.get("min_outward", 0.0)

                st.markdown(f"#### {junction_key.replace('_', ' ').title()}")

                # Generate the CLI command
                cli_command = f"""verta metrics \\
  --input ./data \\
  --columns x=Headset.Head.Position.X,z=Headset.Head.Position.Z,t=Time \\
  --scale 0.2 \\
  --junction {junction.cx:.1f} {junction.cz:.1f} \\
  --radius {junction.r:.1f} \\
  --decision_mode {decision_mode} \\
  --distance {distance:.1f}"""

                if decision_mode in ["radial", "hybrid"]:
                    cli_command += f" \\\n  --r_outer {r_outer:.1f}"

                if decision_mode == "radial":
                    cli_command += f" \\\n  --trend_window {trend_window} \\\n  --min_outward {min_outward:.1f}"

                cli_command += f" \\\n  --out ./outputs/{junction_key}_metrics"

                st.code(cli_command, language="bash")

        elif analysis_type == "gaze":
            # Generate gaze command for all junctions
            if len(st.session_state.junctions) == 1:
                # Single junction
                junction = st.session_state.junctions[0]
                r_outer = st.session_state.junction_r_outer.get(0, 50.0)
                gaze_data = list(results.values())[0] if results else {}
                decision_mode = gaze_data.get("decision_mode", "hybrid")
                path_length = gaze_data.get("path_length", 100.0)
                epsilon = gaze_data.get("epsilon", 0.05)
                linger_delta = gaze_data.get("linger_delta", 5.0)

                cli_command = f"""verta gaze \\
  --input ./data \\
  --columns x=Headset.Head.Position.X,z=Headset.Head.Position.Z,t=Time \\
  --scale 0.2 \\
  --junction {junction.cx:.1f} {junction.cz:.1f} \\
  --radius {junction.r:.1f} \\
  --r_outer {r_outer:.1f} \\
  --distance {path_length:.1f} \\
  --epsilon {epsilon:.3f} \\
  --decision_mode {decision_mode} \\
  --linger_delta {linger_delta:.1f} \\
  --cluster_method {cluster_method}"""
            else:
                # Multiple junctions
                junctions_str = " ".join([f"{j.cx:.1f} {j.cz:.1f} {j.r:.1f}" for j in st.session_state.junctions])
                r_outer_list = [st.session_state.junction_r_outer.get(i, 50.0) for i in range(len(st.session_state.junctions))]
                r_outer_str = " ".join([str(r) for r in r_outer_list])
                gaze_data = list(results.values())[0] if results else {}
                decision_mode = gaze_data.get("decision_mode", "hybrid")
                path_length = gaze_data.get("path_length", 100.0)
                epsilon = gaze_data.get("epsilon", 0.05)
                linger_delta = gaze_data.get("linger_delta", 5.0)

                cli_command = f"""verta gaze \\
  --input ./data \\
  --columns x=Headset.Head.Position.X,z=Headset.Head.Position.Z,t=Time \\
  --scale 0.2 \\
  --junctions {junctions_str} \\
  --r_outer_list {r_outer_str} \\
  --distance {path_length:.1f} \\
  --epsilon {epsilon:.3f} \\
  --decision_mode {decision_mode} \\
  --linger_delta {linger_delta:.1f} \\
  --cluster_method {cluster_method}"""

            # Add cluster method specific parameters
            if cluster_method == "dbscan" and cluster_params:
                cli_command += f" \\\n  --min_samples {cluster_params.get('min_samples', 5)} \\\n  --angle_eps {cluster_params.get('angle_eps', 15.0)}"
            elif cluster_method == "kmeans" and cluster_params:
                cli_command += f" \\\n  --k {cluster_params.get('k', 3)}"
            elif cluster_method == "auto" and cluster_params:
                cli_command += f" \\\n  --k_min {cluster_params.get('k_min', 2)} \\\n  --k_max {cluster_params.get('k_max', 6)} \\\n  --min_sep_deg {cluster_params.get('min_sep_deg', 12.0)} \\\n  --angle_eps {cluster_params.get('angle_eps', 15.0)}"

            cli_command += f" \\\n  --out ./outputs/gaze_analysis"

            st.code(cli_command, language="bash")

        elif analysis_type == "intent":
            # Generate intent command for all junctions
            if len(st.session_state.junctions) == 1:
                # Single junction
                junction = st.session_state.junctions[0]
                intent_data = list(results.values())[0] if results else {}
                decision_mode = intent_data.get("decision_mode", "hybrid")
                path_length = intent_data.get("path_length", 100.0)
                epsilon = intent_data.get("epsilon", 0.05)
                linger_delta = intent_data.get("linger_delta", 5.0)
                prediction_distances = intent_data.get("prediction_distances", [100.0, 75.0, 50.0, 25.0])
                model_type = intent_data.get("model_type", "random_forest")
                cv_folds = intent_data.get("cv_folds", 5)
                test_split = intent_data.get("test_split", 0.2)

                cli_command = f"""verta intent \\
  --input ./data \\
  --columns x=Headset.Head.Position.X,z=Headset.Head.Position.Z,t=Time \\
  --scale 0.2 \\
  --junction {junction.cx:.1f} {junction.cz:.1f} {junction.r:.1f} \\
  --distance {path_length:.1f} \\
  --epsilon {epsilon:.3f} \\
  --decision_mode {decision_mode} \\
  --linger_delta {linger_delta:.1f} \\
  --cluster_method {cluster_method}"""
            else:
                # Multiple junctions
                junctions_str = " ".join([f"{j.cx:.1f} {j.cz:.1f} {j.r:.1f}" for j in st.session_state.junctions])
                intent_data = list(results.values())[0] if results else {}
                decision_mode = intent_data.get("decision_mode", "hybrid")
                path_length = intent_data.get("path_length", 100.0)
                epsilon = intent_data.get("epsilon", 0.05)
                linger_delta = intent_data.get("linger_delta", 5.0)
                prediction_distances = intent_data.get("prediction_distances", [100.0, 75.0, 50.0, 25.0])
                model_type = intent_data.get("model_type", "random_forest")
                cv_folds = intent_data.get("cv_folds", 5)
                test_split = intent_data.get("test_split", 0.2)

                cli_command = f"""verta intent \\
  --input ./data \\
  --columns x=Headset.Head.Position.X,z=Headset.Head.Position.Z,t=Time \\
  --scale 0.2 \\
  --junctions {junctions_str} \\
  --distance {path_length:.1f} \\
  --epsilon {epsilon:.3f} \\
  --decision_mode {decision_mode} \\
  --linger_delta {linger_delta:.1f} \\
  --cluster_method {cluster_method}"""

            # Add cluster method specific parameters
            if cluster_method == "dbscan" and cluster_params:
                cli_command += f" \\\n  --min_samples {cluster_params.get('min_samples', 5)} \\\n  --angle_eps {cluster_params.get('angle_eps', 15.0)}"
            elif cluster_method == "kmeans" and cluster_params:
                cli_command += f" \\\n  --k {cluster_params.get('k', 3)}"
            elif cluster_method == "auto" and cluster_params:
                cli_command += f" \\\n  --k_min {cluster_params.get('k_min', 2)} \\\n  --k_max {cluster_params.get('k_max', 6)} \\\n  --min_sep_deg {cluster_params.get('min_sep_deg', 12.0)} \\\n  --angle_eps {cluster_params.get('angle_eps', 15.0)}"

            # Add intent-specific parameters
            prediction_distances_str = " ".join([str(d) for d in prediction_distances])
            cli_command += f" \\\n  --prediction_distances {prediction_distances_str} \\\n  --model_type {model_type} \\\n  --cv_folds {cv_folds} \\\n  --test_split {test_split}"

            cli_command += f" \\\n  --out ./outputs/intent_recognition"

            st.code(cli_command, language="bash")

    def render_conditional_probabilities(self):
        """Render conditional probabilities"""
        st.markdown("### Conditional Probabilities")

        if "conditional_probabilities" in st.session_state.analysis_results:
            cond_probs = st.session_state.analysis_results["conditional_probabilities"]

            # Create a DataFrame for better display
            df_data = []
            for junction_key, probs in cond_probs.items():
                junction_num = junction_key.split('_')[1] if '_' in junction_key else junction_key[1:]
                for origin, dest_probs in probs.items():
                    for dest, prob in dest_probs.items():
                        df_data.append({
                            'Junction': f'J{junction_num}',
                            'From': origin,
                            'To': dest,
                            'Probability': f"{prob:.1%}"
                        })

            if df_data:
                df = pd.DataFrame(df_data)
                st.dataframe(df, width='stretch')
            else:
                st.info("No conditional probabilities available")

    def render_pattern_analysis(self):
        """Render pattern analysis results"""
        st.markdown("### Pattern Analysis")

        if "choice_patterns" in st.session_state.analysis_results:
            patterns = st.session_state.analysis_results["choice_patterns"]

            # Display pattern statistics
            st.markdown("#### Pattern Statistics")
            for junction_key, pattern_data in patterns.items():
                junction_num = junction_key.split('_')[1] if '_' in junction_key else junction_key[1:]
                st.markdown(f"**Junction {junction_num}:**")

                if "total_trajectories" in pattern_data:
                    st.write(f"- Total trajectories: {pattern_data['total_trajectories']}")

                if "choice_counts" in pattern_data:
                    st.write(f"- Choice counts: {pattern_data['choice_counts']}")

    def render_intent_visualizations(self):
        """Render intent recognition analysis visualizations"""
        st.markdown("### 🧠 Intent Recognition Results")

        if (st.session_state.analysis_results is None or
            "intent_recognition" not in st.session_state.analysis_results):
            st.info("No intent recognition results available.")
            return

        intent_data = st.session_state.analysis_results["intent_recognition"]

        # Get successful junctions
        successful_junctions = {k: v for k, v in intent_data.items() if 'error' not in v}

        if not successful_junctions:
            st.warning("⚠️ No successful intent recognition results to visualize")
            return

        # Junction selector
        junction_keys = list(successful_junctions.keys())
        if len(junction_keys) > 1:
            selected_junction = st.selectbox(
                "Select Junction:",
                junction_keys,
                format_func=lambda x: f"Junction {x.replace('junction_', '')}"
            )
        else:
            selected_junction = junction_keys[0]

        junction_results = successful_junctions[selected_junction]
        junction_num = selected_junction.replace('junction_', '')

        # Summary metrics
        st.markdown(f"#### Junction {junction_num} Summary")

        models_trained = junction_results['training_results'].get('models_trained', {})

        if models_trained:
            # Create metrics row
            cols = st.columns(len(models_trained))
            for idx, (dist, model_info) in enumerate(sorted(models_trained.items())):
                with cols[idx]:
                    st.metric(
                        f"{dist} units",
                        f"{model_info['cv_mean_accuracy']:.1%}",
                        f"n={model_info['n_samples']}"
                    )

            # Overall accuracy
            avg_acc = np.mean([m['cv_mean_accuracy'] for m in models_trained.values()])
            st.markdown(f"**Average Accuracy:** {avg_acc:.1%}")

            # Interpretation
            if avg_acc > 0.85:
                st.success("🟢 Excellent Predictability")
            elif avg_acc > 0.70:
                st.info("🟡 Good Predictability")
            else:
                st.warning("🔴 Moderate Predictability")

        # Feature Importance Plot
        st.markdown("#### Feature Importance")
        feature_importance_path = os.path.join("gui_outputs", "intent_recognition",
                                               f"junction_{junction_num}",
                                               "intent_feature_importance.png")
        if os.path.exists(feature_importance_path):
            st.image(feature_importance_path, width='stretch')
        else:
            st.info("Feature importance plot not available")

        # Accuracy Analysis Plot
        st.markdown("#### Prediction Accuracy vs Distance")
        accuracy_path = os.path.join("gui_outputs", "intent_recognition",
                                     f"junction_{junction_num}",
                                     "intent_accuracy_analysis.png")
        if os.path.exists(accuracy_path):
            st.image(accuracy_path, width='stretch')
            st.caption("This shows how prediction accuracy improves as users get closer to the junction")
        else:
            st.info("Accuracy analysis plot not available")

        # Test Predictions
        if 'test_predictions' in junction_results:
            st.markdown("#### Sample Predictions")

            test_preds = junction_results['test_predictions']

            # Show a few example predictions
            example_count = min(5, len(test_preds))

            for traj_id in list(test_preds.keys())[:example_count]:
                pred_info = test_preds[traj_id]
                actual = pred_info['actual_branch']

                with st.expander(f"Trajectory: {traj_id} (Actual: Branch {actual})"):
                    predictions = pred_info['predictions_by_distance']

                    # Create visualization
                    distances = []
                    predicted_branches = []
                    confidences = []
                    correct_flags = []

                    for dist in sorted(predictions.keys(), reverse=True):
                        p = predictions[dist]
                        distances.append(f"{dist}u")
                        predicted_branches.append(f"Branch {p['predicted_branch']}")
                        confidences.append(p['confidence'])
                        correct_flags.append("✓" if p['correct'] else "✗")

                    # Create DataFrame
                    pred_df = pd.DataFrame({
                        'Distance Before': distances,
                        'Predicted': predicted_branches,
                        'Confidence': [f"{c:.1%}" for c in confidences],
                        'Correct': correct_flags
                    })

                    st.dataframe(pred_df, width='stretch')

                    # Confidence chart
                    import plotly.graph_objects as go

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=[float(d.replace('u', '')) for d in distances],
                        y=confidences,
                        mode='lines+markers',
                        name='Confidence',
                        line=dict(color='blue', width=3),
                        marker=dict(size=10)
                    ))
                    fig.update_layout(
                        title="Prediction Confidence Over Distance",
                        xaxis_title="Distance to Junction (units)",
                        yaxis_title="Confidence",
                        yaxis_range=[0, 1],
                        height=300
                    )
                    st.plotly_chart(fig, width='stretch', key=f"intent_confidence_{junction_num}_{traj_id}")

        # Feature importance table
        if 'feature_importance' in junction_results['training_results']:
            st.markdown("#### Feature Importance (Detailed)")

            with st.expander("View Feature Importance by Distance"):
                feature_imp = junction_results['training_results']['feature_importance']

                for dist in sorted(feature_imp.keys()):
                    st.markdown(f"**{dist} units before junction:**")

                    importance_dict = feature_imp[dist]
                    sorted_features = sorted(importance_dict.items(),
                                           key=lambda x: x[1], reverse=True)

                    feat_df = pd.DataFrame(sorted_features[:10],
                                          columns=['Feature', 'Importance'])
                    feat_df['Importance'] = feat_df['Importance'].apply(lambda x: f"{x:.3f}")

                    st.dataframe(feat_df, width='stretch')
                    st.markdown("---")

        # Download results
        st.markdown("#### Download Results")

        results_path = os.path.join("gui_outputs", "intent_recognition",
                                    f"junction_{junction_num}",
                                    "intent_training_results.json")

        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                results_json = f.read()

            st.download_button(
                label="📥 Download Training Results (JSON)",
                data=results_json,
                file_name=f"intent_recognition_junction_{junction_num}.json",
                mime="application/json"
            )

        # Explanation
        with st.expander("ℹ️ Understanding Intent Recognition"):
            st.markdown("""
            **Intent Recognition** predicts which route users will choose **before** they reach decision points.

            **Key Insights:**
            - **Higher accuracy at closer distances**: Predictions improve as users approach junctions
            - **Feature importance**: Shows which trajectory features best predict choices
            - **Early prediction**: Enables proactive systems that respond before users act

            **Applications:**
            - 🗺️ Proactive wayfinding and navigation hints
            - 🎨 Adaptive UI that highlights likely options
            - 🚦 Congestion prediction and traffic management
            - ⚠️ Anomaly detection (unexpected behavior)
            - ⚡ Performance optimization (asset preloading)

            **Accuracy Interpretation:**
            - **>85%**: Excellent - Highly predictable behavior
            - **70-85%**: Good - Clear patterns exist
            - **<70%**: Moderate - Variable or exploratory behavior
            """)

    def render_enhanced_visualizations(self):
        """Render enhanced analysis visualizations"""
        st.markdown("### 🚨 Enhanced Analysis Results")

        # Check if enhanced analysis results exist
        if (st.session_state.analysis_results is None or
            "enhanced" not in st.session_state.analysis_results):
            st.info("No enhanced analysis results available. Run enhanced analysis first.")
            return

        enhanced_data = st.session_state.analysis_results["enhanced"]

        # Create tabs for different analysis components
        tab1, tab2, tab3, tab4 = st.tabs(["🚨 Evacuation Analysis", "💡 Recommendations", "⚠️ Risk Assessment", "📊 Efficiency Metrics"])

        with tab1:
            self._render_evacuation_analysis(enhanced_data["evacuation_analysis"])

        with tab2:
            self._render_recommendations(enhanced_data["recommendations"])

        with tab3:
            self._render_risk_assessment(enhanced_data["risk_assessment"])

        with tab4:
            self._render_efficiency_metrics(enhanced_data["efficiency_metrics"])

    def _render_evacuation_analysis(self, evacuation_data):
        """Render evacuation analysis visualizations"""
        st.markdown("#### Evacuation Flow Analysis")

        # Add explanation
        st.info("""
        **🚨 Evacuation Analysis Explanation:**
        - **Bottlenecks**: Junctions where >60% of traffic uses the same route (HIGH risk: >80%)
        - **Optimal Routes**: Junctions with balanced traffic distribution (balance ratio >0.7)
        - **Balance Ratio**: Measures how evenly traffic is distributed across branches (0.0=all traffic in one route, 1.0=perfectly balanced)
        - **Entropy**: Information theory measure of traffic distribution diversity
        """)

        # Bottlenecks
        if evacuation_data["bottlenecks"]:
            st.markdown("##### 🚧 Identified Bottlenecks")
            for bottleneck in evacuation_data["bottlenecks"]:
                risk_color = "🔴" if bottleneck["risk_level"] == "HIGH" else "🟡"
                st.markdown(f"""
                {risk_color} **Junction {bottleneck['junction']}, Branch {int(bottleneck['branch'])}**
                - Concentration: {bottleneck['concentration']:.1%}
                - Trajectories: {bottleneck['trajectory_count']}
                - Risk Level: {bottleneck['risk_level']}
                """)
        else:
            st.success("✅ No significant bottlenecks detected")

        # Optimal routes
        if evacuation_data["optimal_routes"]:
            st.markdown("##### ✅ Optimal Routes")
            st.info("**Optimal Routes**: Junctions with well-balanced traffic distribution (balance ratio >0.7) - these are good for evacuation as traffic spreads evenly across multiple routes.")
            for route in evacuation_data["optimal_routes"]:
                st.markdown(f"""
                **Junction {route['junction']}**
                - Balance Ratio: {route['balance_ratio']:.2f} (higher = more balanced)
                - Entropy: {route['entropy']:.2f} (higher = more diverse routes)
                - Branch Count: {route['branch_count']} (number of available routes)
                """)
        else:
            st.info("ℹ️ No optimal routes identified (all junctions have concentrated traffic)")

        # Flow analysis chart
        if evacuation_data["flow_analysis"]:
            st.markdown("##### 📊 Flow Distribution")
            import pandas as pd
            import plotly.express as px

            flow_data = []
            for junction_key, data in evacuation_data["flow_analysis"].items():
                junction_num = junction_key.split('_')[1]
                for branch, count in data["branch_distribution"].items():
                    flow_data.append({
                        "Junction": f"J{junction_num}",
                        "Branch": f"Branch {int(branch)}",
                        "Trajectory Count": count,
                        "Percentage": count / data["total_trajectories"] * 100
                    })

            if flow_data:
                df = pd.DataFrame(flow_data)
                fig = px.bar(df, x="Junction", y="Trajectory Count", color="Branch",
                           title="Trajectory Distribution by Junction and Branch",
                           hover_data=["Percentage"])
                st.plotly_chart(fig, width='stretch')

    def _render_recommendations(self, recommendations):
        """Render recommendations"""
        st.markdown("#### 💡 Actionable Recommendations")

        # Add explanation
        st.info("""
        **💡 Recommendations Explanation:**
        - **HIGH Priority**: Critical issues requiring immediate attention (bottlenecks >80% concentration)
        - **MEDIUM Priority**: System-wide issues or moderate bottlenecks (60-80% concentration)
        - **LOW Priority**: Maintenance recommendations for well-performing junctions
        - **Signage**: Directional signs to distribute traffic away from bottlenecks
        - **Route Modification**: Physical changes like widening or adding alternative routes
        """)

        if not recommendations:
            st.info("No specific recommendations generated")
            return

        # Group by priority
        high_priority = [r for r in recommendations if r["priority"] == "HIGH"]
        medium_priority = [r for r in recommendations if r["priority"] == "MEDIUM"]
        low_priority = [r for r in recommendations if r["priority"] == "LOW"]

        if high_priority:
            st.markdown("##### 🔴 High Priority")
            for rec in high_priority:
                st.markdown(f"""
                **{rec['type']}** - Junction {rec['junction']}
                {rec['message']}
                """)

        if medium_priority:
            st.markdown("##### 🟡 Medium Priority")
            for rec in medium_priority:
                st.markdown(f"""
                **{rec['type']}** - Junction {rec['junction']}
                {rec['message']}
                """)

        if low_priority:
            st.markdown("##### 🟢 Low Priority")
            for rec in low_priority:
                st.markdown(f"""
                **{rec['type']}** - Junction {rec['junction']}
                {rec['message']}
                """)

    def _render_risk_assessment(self, risk_data):
        """Render risk assessment visualizations"""
        st.markdown("#### ⚠️ Risk Assessment")

        # Add explanation
        st.info("""
        **⚠️ Unified Risk Assessment Explanation:**
        - **Overall Risk Score**: 0.0-1.0 scale (0.0=Low Risk, 1.0=High Risk) - normalized across all junctions
        - **Risk Factors**: Each junction assessed on 3 dimensions:
          • **Concentration Risk**: Traffic concentration in single route (>70% = risk)
          • **Diversity Risk**: Number of available routes (<2 routes = high risk, 2 routes = moderate risk)
          • **Crowding Risk**: Traffic volume (>50 trajectories = moderate, >100 = high)
        - **Risk Levels**: HIGH (≥0.7), MEDIUM (≥0.4), LOW (<0.4)
        - **Unified Score**: All risk factors combined and normalized to 0-1 scale
        """)

        # Overall risk score
        overall_score = risk_data["overall_risk_score"]
        risk_level = "HIGH" if overall_score > 0.7 else "MEDIUM" if overall_score > 0.3 else "LOW"
        risk_color = "🔴" if risk_level == "HIGH" else "🟡" if risk_level == "MEDIUM" else "🟢"

        st.markdown(f"""
        ##### Overall Risk Score: {risk_color} {risk_level}
        **Score: {overall_score:.2f}** (0.0 = Low Risk, 1.0 = High Risk)
        """)

        # High risk junctions (now includes all risk levels)
        if risk_data["high_risk_junctions"]:
            st.markdown("##### 🚨 Risk Assessment by Junction")

            # Group by risk level
            high_risk = [j for j in risk_data["high_risk_junctions"] if j["risk_level"] == "HIGH"]
            medium_risk = [j for j in risk_data["high_risk_junctions"] if j["risk_level"] == "MEDIUM"]

            if high_risk:
                st.markdown("###### 🔴 HIGH Risk Junctions")
                for junction in high_risk:
                    st.markdown(f"""
                    **Junction {junction['junction']}**
                    - Risk Score: {junction['risk_score']:.2f}
                    - Trajectory Count: {junction['trajectory_count']}
                    - Concentration: {junction['concentration']:.1%}
                    - Route Count: {junction['route_count']}
                    - Risk Factors: {', '.join([f[0] for f in junction['risk_factors']])}
                    """)

            if medium_risk:
                st.markdown("###### 🟡 MEDIUM Risk Junctions")
                for junction in medium_risk:
                    st.markdown(f"""
                    **Junction {junction['junction']}**
                    - Risk Score: {junction['risk_score']:.2f}
                    - Trajectory Count: {junction['trajectory_count']}
                    - Concentration: {junction['concentration']:.1%}
                    - Route Count: {junction['route_count']}
                    - Risk Factors: {', '.join([f[0] for f in junction['risk_factors']])}
                    """)
        else:
            st.success("✅ No significant risks identified")

        # Risk visualization
        if risk_data["high_risk_junctions"]:
            import plotly.express as px
            import pandas as pd

            risk_chart_data = []
            for junction in risk_data["high_risk_junctions"]:
                risk_chart_data.append({
                    "Junction": f"J{junction['junction']}",
                    "Risk Score": junction['risk_score'],
                    "Risk Level": junction['risk_level'],
                    "Trajectory Count": junction['trajectory_count'],
                    "Concentration": junction['concentration'],
                    "Route Count": junction['route_count']
                })

            if risk_chart_data:
                df = pd.DataFrame(risk_chart_data)
                fig = px.bar(df, x="Junction", y="Risk Score", color="Risk Level",
                           title="Unified Risk Assessment by Junction",
                           color_discrete_map={"HIGH": "red", "MEDIUM": "orange", "LOW": "green"},
                           hover_data=["Trajectory Count", "Concentration", "Route Count"])
                st.plotly_chart(fig, width='stretch')

    def _render_efficiency_metrics(self, efficiency_data):
        """Render efficiency metrics visualizations"""
        st.markdown("#### 📊 Efficiency Metrics")

        # Add explanation
        st.info("""
        **📊 Efficiency Metrics Explanation:**
        - **Route Efficiency**: Entropy-based measure of traffic distribution quality (0.0=all traffic in one route, 1.0=perfectly distributed)
        - **Capacity Utilization**: How well junctions handle their traffic load (trajectories/100, capped at 100%)
        - **Overall Efficiency**: Average route efficiency across all junctions
        - **Higher values = better evacuation performance**
        """)

        # Overall efficiency
        overall_efficiency = efficiency_data["overall_efficiency"]
        efficiency_level = "HIGH" if overall_efficiency > 0.7 else "MEDIUM" if overall_efficiency > 0.4 else "LOW"
        efficiency_color = "🟢" if efficiency_level == "HIGH" else "🟡" if efficiency_level == "MEDIUM" else "🔴"

        st.markdown(f"""
        ##### Overall Efficiency: {efficiency_color} {efficiency_level}
        **Score: {overall_efficiency:.2f}** (0.0 = Low Efficiency, 1.0 = High Efficiency)
        """)

        # Route efficiency by junction
        if efficiency_data["route_efficiency"]:
            st.markdown("##### 🛣️ Route Efficiency by Junction")
            import plotly.express as px
            import pandas as pd

            efficiency_chart_data = []
            for junction_key, efficiency in efficiency_data["route_efficiency"].items():
                junction_num = junction_key.split('_')[1]
                efficiency_chart_data.append({
                    "Junction": f"J{junction_num}",
                    "Route Efficiency": efficiency,
                    "Capacity Utilization": efficiency_data["capacity_utilization"].get(junction_key, 0)
                })

            if efficiency_chart_data:
                df = pd.DataFrame(efficiency_chart_data)

                # Route efficiency chart
                fig1 = px.bar(df, x="Junction", y="Route Efficiency",
                            title="Route Efficiency by Junction",
                            color="Route Efficiency",
                            color_continuous_scale="RdYlGn")
                st.plotly_chart(fig1, width='stretch')

                # Capacity utilization chart
                fig2 = px.bar(df, x="Junction", y="Capacity Utilization",
                            title="Capacity Utilization by Junction",
                            color="Capacity Utilization",
                            color_continuous_scale="RdYlGn")
                st.plotly_chart(fig2, width='stretch')

        # Efficiency summary
        st.markdown("##### 📈 Efficiency Summary")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Overall Route Efficiency", f"{overall_efficiency:.2f}")

        with col2:
            avg_capacity = sum(efficiency_data["capacity_utilization"].values()) / len(efficiency_data["capacity_utilization"]) if efficiency_data["capacity_utilization"] else 0
            st.metric("Average Capacity Utilization", f"{avg_capacity:.2f}")

    def render_export(self):
        """Render the export interface"""
        st.markdown('<h2 class="section-header">💾 Export Results</h2>', unsafe_allow_html=True)

        if not st.session_state.analysis_results:
            st.warning("⚠️ Please run an analysis first")
            return

        st.markdown("### Export Options")

        # Export format selection
        export_format = st.selectbox(
            "Export Format:",
            ["JSON", "CSV", "ZIP Archive"],
            help="Select the format for exporting results"
        )

        if st.button("📥 Export Results"):
            self.export_results(export_format)

    def export_results(self, format: str):
        """Export analysis results"""
        try:
            if format == "JSON":
                # Export as JSON
                json_str = json.dumps(st.session_state.analysis_results, indent=2, default=str)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name="analysis_results.json",
                    mime="application/json"
                )

            elif format == "CSV":
                # Export as CSV (if applicable)
                if "metrics" in st.session_state.analysis_results:
                    # Export metrics as CSV
                    import pandas as pd
                    df = pd.DataFrame(st.session_state.analysis_results["metrics"])
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        label="Download Metrics CSV",
                        data=csv_data,
                        file_name="metrics_results.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("CSV export available for metrics data")

            elif format == "ZIP Archive":
                # Create comprehensive ZIP archive with all files from gui_outputs
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Save analysis results JSON
                    results_file = os.path.join(temp_dir, "analysis_results.json")
                    with open(results_file, 'w') as f:
                        json.dump(st.session_state.analysis_results, f, indent=2, default=str)

                    # Create ZIP
                    zip_path = os.path.join(temp_dir, "analysis_results.zip")
                    with zipfile.ZipFile(zip_path, 'w') as zipf:
                        # Add analysis results JSON
                        zipf.write(results_file, "analysis_results.json")

                        # Add all files from gui_outputs directory
                        gui_outputs_dir = "gui_outputs"
                        if os.path.exists(gui_outputs_dir):
                            for root, dirs, files in os.walk(gui_outputs_dir):
                                for file in files:
                                    file_path = os.path.join(root, file)
                                    # Create relative path within ZIP
                                    rel_path = os.path.relpath(file_path, gui_outputs_dir)
                                    zipf.write(file_path, rel_path)

                    # Download ZIP
                    with open(zip_path, 'rb') as f:
                        zip_data = f.read()

                    st.download_button(
                        label="Download Complete Analysis Package",
                        data=zip_data,
                        file_name="complete_analysis_results.zip",
                        mime="application/zip"
                    )

                    # Show what's included in the ZIP
                    st.info("📦 **Complete Analysis Package includes:**")
                    st.write("• Analysis results (JSON)")
                    st.write("• All visualizations (PNG files)")
                    st.write("• All data tables (CSV files)")
                    st.write("• Gaze analysis results")
                    st.write("• Physiological analysis data")
                    st.write("• Pupil trajectory data")
                    st.write("• Consistency reports")
                    st.write("• Branch assignments")
                    st.write("• Decision points")
                    st.write("• Metrics results")
                    st.write("• Enhanced analysis results")
                    st.write("• Risk assessment data")
                    st.write("• Efficiency metrics")
                    st.write("• Evacuation analysis")
                    st.write("• Recommendations")
                    st.write("• Intent recognition models")
                    st.write("• Feature importance analysis")
                    st.write("• ML prediction results")

            st.success("✅ Export ready!")

        except Exception as e:
            st.error(f"❌ Export failed: {str(e)}")

    def run_quick_analysis(self):
        """Run a quick analysis with default parameters"""
        if not st.session_state.trajectories or not st.session_state.junctions:
            st.warning("⚠️ Please load data and define junctions first")
            return

        st.session_state.current_step = "analysis"
        st.rerun()

    def clear_all_data(self):
        """Clear all data and reset the application"""
        st.session_state.trajectories = []
        st.session_state.junctions = []
        st.session_state.junction_r_outer = {}
        st.session_state.analysis_results = None
        st.session_state.current_step = "data_upload"
        st.success("✅ All data cleared!")
        st.rerun()

    def run(self):
        """Main GUI run method"""
        # Add custom CSS for image aspect ratio preservation
        st.markdown("""
        <style>
        .stImage > img {
            object-fit: contain !important;
            max-width: 100% !important;
            height: auto !important;
        }
        .stImage > div {
            display: flex !important;
            justify-content: center !important;
        }
        </style>
        """, unsafe_allow_html=True)

        self.render_header()
        self.render_navigation()

        # Render current step
        if st.session_state.current_step == "data_upload":
            self.render_data_upload()
        elif st.session_state.current_step == "junction_editor":
            self.render_junction_editor()
        elif st.session_state.current_step == "analysis":
            self.render_analysis()
        elif st.session_state.current_step == "visualization":
            self.render_visualization()
        elif st.session_state.current_step == "export":
            self.render_export()

def main():
    """Main entry point"""
    gui = VERTAGUI()
    gui.run()

if __name__ == "__main__":
    main()
