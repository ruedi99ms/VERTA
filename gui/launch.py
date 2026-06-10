#!/usr/bin/env python3
"""
VERTA GUI Launcher
==================

Launch the VERTA (Virtual Environment Route and Trajectory Analyzer) web UI from the repository root.

Usage:
  python gui/launch.py
  # or
  streamlit run src/verta/verta_gui.py
"""

import os
import sys
import subprocess
from pathlib import Path


def main() -> int:
    # Determine repository root (parent of gui/ folder)
    this_file = Path(__file__).resolve()
    repo_root = this_file.parent.parent

    # Path to GUI app (in src/ layout)
    gui_path = repo_root / "src" / "verta" / "verta_gui.py"
    if not gui_path.exists():
        print(f"GUI entry not found at: {gui_path}")
        return 1

    # Ensure Streamlit is available
    try:
        import streamlit  # noqa: F401
    except Exception:
        print("Streamlit is not installed.")
        print("Install GUI deps first: pip install verta[gui]")
        return 1

    # Build streamlit command
    cmd = [sys.executable, "-m", "streamlit", "run", str(gui_path)]

    # Run with repository root as CWD so absolute imports work
    cwd = repo_root

    print("Launching VERTA GUI...")
    print("If the browser doesn't open automatically, visit: http://localhost:8501\n")

    # Prefer local src/ package over an older pip install
    env = os.environ.copy()
    src_path = str(repo_root / "src")
    env["PYTHONPATH"] = src_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    try:
        result = subprocess.run(cmd, cwd=str(cwd), env=env)
        return result.returncode
    except KeyboardInterrupt:
        return 0
    except Exception as e:
        print(f"Failed to launch GUI: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
