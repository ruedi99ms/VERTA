# Route Analyzer Test Suite

This directory contains pytest tests for the route_analyzer project.

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
pytest --cov=. --cov-report=html
```

## Test Structure

- `conftest.py`: Shared pytest fixtures for all tests
- `test_geometry.py`: Tests for geometry classes (Circle, Rect) and functions
- `test_commands.py`: Tests for command classes (BaseCommand, DiscoverCommand, etc.)
- `test_data_loader.py`: Tests for data loading functionality (Trajectory, ColumnMapping)

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
from ra_geometry import Circle

class TestCircle:
    def test_circle_creation(self):
        """Test creating a circle."""
        circle = Circle(cx=1.0, cz=2.0, r=3.0)
        assert circle.cx == 1.0
        assert circle.cz == 2.0
        assert circle.r == 3.0
```

