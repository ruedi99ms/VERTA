"""
Progress tracking utilities for Route Analyzer.

Provides progress bars and time predictions for long-running analysis functions.
Supports both terminal (tqdm) and GUI (streamlit) progress tracking.
"""

import time
from typing import Optional, List, Dict, Any, Callable, Union
from contextlib import contextmanager
import streamlit as st
from tqdm import tqdm
import numpy as np


class ProgressTracker:
    """Unified progress tracker for both terminal and GUI environments."""
    
    def __init__(self, total: int, description: str = "Processing", 
                 gui_mode: bool = False, show_time: bool = True):
        self.total = total
        self.description = description
        self.gui_mode = gui_mode
        self.show_time = show_time
        self.current = 0
        self.start_time = time.time()
        self.last_update_time = self.start_time
        
        if gui_mode:
            self._init_gui_progress()
        else:
            self._init_terminal_progress()
    
    def _init_gui_progress(self):
        """Initialize GUI progress bar."""
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
        self.time_text = st.empty()
    
    def _init_terminal_progress(self):
        """Initialize terminal progress bar."""
        self.progress_bar = tqdm(
            total=self.total,
            desc=self.description,
            unit="item",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
    
    def update(self, increment: int = 1, status: str = None):
        """Update progress by increment."""
        self.current += increment
        self.current = min(self.current, self.total)
        
        if self.gui_mode:
            self._update_gui_progress(status)
        else:
            self._update_terminal_progress(status)
    
    def _update_gui_progress(self, status: str = None):
        """Update GUI progress bar."""
        progress = self.current / self.total
        self.progress_bar.progress(progress)
        
        if status:
            self.status_text.text(status)
        
        if self.show_time:
            elapsed = time.time() - self.start_time
            if self.current > 0:
                eta = (elapsed / self.current) * (self.total - self.current)
                time_str = f"Elapsed: {self._format_time(elapsed)} | ETA: {self._format_time(eta)}"
            else:
                time_str = f"Elapsed: {self._format_time(elapsed)}"
            self.time_text.text(time_str)
    
    def _update_terminal_progress(self, status: str = None):
        """Update terminal progress bar."""
        if status:
            self.progress_bar.set_postfix_str(status)
        self.progress_bar.update(self.current - self.progress_bar.n)
    
    def set_status(self, status: str):
        """Set status text without updating progress."""
        if self.gui_mode:
            self.status_text.text(status)
        else:
            self.progress_bar.set_postfix_str(status)
    
    def _format_time(self, seconds: float) -> str:
        """Format time in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"
    
    def close(self):
        """Close the progress bar."""
        if not self.gui_mode:
            self.progress_bar.close()


class AnalysisProgressManager:
    """Manages progress across multiple analysis steps."""
    
    def __init__(self, gui_mode: bool = False):
        self.gui_mode = gui_mode
        self.steps = []
        self.current_step = 0
        self.overall_progress = None
        self.step_progress = None
    
    def add_step(self, name: str, weight: float = 1.0):
        """Add an analysis step with relative weight."""
        self.steps.append({"name": name, "weight": weight})
    
    def start_analysis(self, total_trajectories: int):
        """Start the overall analysis progress tracking."""
        total_weight = sum(step["weight"] for step in self.steps)
        self.start_time = time.time()  # Initialize start time
        
        if self.gui_mode:
            st.subheader("Analysis Progress")
            self.overall_progress = st.progress(0)
            self.overall_status = st.empty()
            self.step_status = st.empty()
            self.time_status = st.empty()
        else:
            print(f"\n=== Starting Analysis ===")
            print(f"Total trajectories: {total_trajectories}")
            print(f"Analysis steps: {len(self.steps)}")
            print("=" * 50)
    
    def start_step(self, step_name: str, total_items: int):
        """Start tracking progress for a specific step."""
        self.current_step += 1
        step_info = f"Step {self.current_step}/{len(self.steps)}: {step_name}"
        
        if self.gui_mode:
            self.step_status.text(step_info)
            self.step_progress = st.progress(0)
        else:
            print(f"\n{step_info}")
        
        return ProgressTracker(
            total=total_items,
            description=step_name,
            gui_mode=self.gui_mode
        )
    
    def update_overall_progress(self, step_progress: float):
        """Update overall progress based on step completion."""
        if not self.gui_mode:
            return
        
        # Calculate weighted progress
        completed_weight = sum(step["weight"] for step in self.steps[:self.current_step-1])
        current_step_weight = self.steps[self.current_step-1]["weight"]
        progress = (completed_weight + current_step_weight * step_progress) / sum(step["weight"] for step in self.steps)
        
        self.overall_progress.progress(progress)
        
        # Update time estimates
        elapsed = time.time() - self.start_time
        if progress > 0:
            eta = (elapsed / progress) * (1 - progress)
            time_str = f"Elapsed: {self._format_time(elapsed)} | ETA: {self._format_time(eta)}"
        else:
            time_str = f"Elapsed: {self._format_time(elapsed)}"
        self.time_status.text(time_str)
    
    def _format_time(self, seconds: float) -> str:
        """Format time in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"
    
    def finish_analysis(self):
        """Mark analysis as complete."""
        if self.gui_mode:
            self.overall_progress.progress(1.0)
            self.overall_status.text("✅ Analysis Complete!")
            self.step_status.text("")
            self.time_status.text("")
        else:
            print("\n" + "=" * 50)
            print("✅ Analysis Complete!")
            print("=" * 50)


@contextmanager
def progress_context(total: int, description: str = "Processing", gui_mode: bool = False):
    """Context manager for progress tracking."""
    tracker = ProgressTracker(total, description, gui_mode)
    try:
        yield tracker
    finally:
        tracker.close()


def track_trajectory_processing(trajectories: List, func: Callable, 
                               description: str = "Processing trajectories",
                               gui_mode: bool = False, **kwargs):
    """Process trajectories with progress tracking."""
    tracker = ProgressTracker(len(trajectories), description, gui_mode)
    
    results = []
    for i, trajectory in enumerate(trajectories):
        result = func(trajectory, **kwargs)
        results.append(result)
        tracker.update(1, f"Processing trajectory {i+1}/{len(trajectories)}")
    
    tracker.close()
    return results


def estimate_analysis_time(trajectories: int, junctions: int, analysis_type: str) -> Dict[str, float]:
    """Estimate analysis time based on data size and analysis type."""
    # Base time estimates (in seconds per trajectory)
    base_times = {
        "discover": 0.1,
        "predict": 0.2,
        "gaze": 0.3,
        "physio": 0.4,
        "enhanced": 0.5
    }
    
    # Junction complexity factor
    junction_factor = 1 + (junctions - 1) * 0.2
    
    # Base time for analysis type
    base_time = base_times.get(analysis_type, 0.2)
    
    # Calculate estimated time
    estimated_time = trajectories * base_time * junction_factor
    
    return {
        "estimated_seconds": estimated_time,
        "estimated_minutes": estimated_time / 60,
        "estimated_hours": estimated_time / 3600,
        "complexity_factor": junction_factor
    }


def show_analysis_estimate(trajectories: int, junctions: int, analysis_type: str, gui_mode: bool = False):
    """Show analysis time estimate to user."""
    estimate = estimate_analysis_time(trajectories, junctions, analysis_type)
    
    if estimate["estimated_minutes"] < 1:
        time_str = f"{estimate['estimated_seconds']:.1f} seconds"
    elif estimate["estimated_hours"] < 1:
        time_str = f"{estimate['estimated_minutes']:.1f} minutes"
    else:
        time_str = f"{estimate['estimated_hours']:.1f} hours"
    
    if gui_mode:
        st.info(f"📊 **Analysis Estimate**: {time_str} for {trajectories} trajectories and {junctions} junctions")
    else:
        print(f"📊 Analysis Estimate: {time_str} for {trajectories} trajectories and {junctions} junctions")
    
    return estimate
