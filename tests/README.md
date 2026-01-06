# VERTA Test Suite

This directory contains pytest tests for VERTA (Virtual Environment Route and Trajectory Analyzer).

## Running Tests

### Install Dependencies

First, install the project with test dependencies:

```bash
pip install -e ".[test]"
```

Or install pytest separately:

```bash
pip install pytest>=7.0.0
```

### Run All Tests

```bash
pytest
```

### Run Specific Test Files

```bash
pytest tests/test_geometry.py
pytest tests/test_commands.py
pytest tests/test_data_loader.py
```

### Run Specific Test Classes or Functions

```bash
pytest tests/test_geometry.py::TestCircle
pytest tests/test_geometry.py::TestCircle::test_circle_contains_inside
```

### Run with Verbose Output

```bash
pytest -v
```

### Run with Coverage

```bash
pip install pytest-cov
pytest --cov=verta --cov-report=html
```

This generates an HTML coverage report in `htmlcov/index.html` showing which code is covered by tests.

### Run with Markers

```bash
# Run only fast tests (if markers are defined)
pytest -m "not slow"

# Run integration tests
pytest -m integration
```

## Test Structure

The test suite currently includes:

- `conftest.py`: Shared pytest fixtures for all tests (trajectories, circles, rectangles, CSV files, etc.)
- `test_geometry.py`: Tests for geometry classes (Circle, Rect) and functions (entered_junction_idx)
- `test_commands.py`: Tests for command classes (BaseCommand, DiscoverCommand, AssignCommand, MetricsCommand, GazeCommand)
- `test_data_loader.py`: Tests for data loading functionality (Trajectory, ColumnMapping, has_gaze_data, has_physio_data)

## Adding New Tests

When adding new test files:

1. Follow the naming convention: `test_*.py`
2. Use descriptive test class names: `TestClassName`
3. Use descriptive test function names: `test_function_name`
4. Use fixtures from `conftest.py` when possible
5. Add docstrings to test functions explaining what they test

## Example Test

```python
import pytest
from verta.verta_geometry import Circle

class TestCircle:
    def test_circle_creation(self):
        """Test creating a circle."""
        circle = Circle(cx=1.0, cz=2.0, r=3.0)
        assert circle.cx == 1.0
        assert circle.cz == 2.0
        assert circle.r == 3.0
```

## Troubleshooting

**Tests fail with import errors:**
```bash
# Install the package in development mode
pip install -e .

# Or install with test dependencies
pip install -e ".[test]"
```

**Tests skip with "package not available":**
- Ensure you're running tests from the repository root
- Install the package: `pip install -e .`
- Verify installation: `python -c "import verta; print('OK')"`

