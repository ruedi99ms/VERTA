"""
Tests for geometry module (Circle, Rect, and related functions).
"""
import pytest
import numpy as np

from route_analyzer.ra_geometry import Circle, Rect, entered_junction_idx


class TestCircle:
    """Test cases for Circle class."""
    
    def test_circle_creation(self):
        """Test creating a circle."""
        circle = Circle(cx=1.0, cz=2.0, r=3.0)
        assert circle.cx == 1.0
        assert circle.cz == 2.0
        assert circle.r == 3.0
    
    def test_circle_contains_inside(self):
        """Test contains method for points inside circle."""
        circle = Circle(cx=0.0, cz=0.0, r=2.0)
        x = np.array([0.0, 1.0, -1.0])
        z = np.array([0.0, 1.0, -1.0])
        result = circle.contains(x, z)
        assert np.all(result == True)
    
    def test_circle_contains_outside(self):
        """Test contains method for points outside circle."""
        circle = Circle(cx=0.0, cz=0.0, r=1.0)
        x = np.array([2.0, 3.0, -2.0])
        z = np.array([2.0, 3.0, -2.0])
        result = circle.contains(x, z)
        assert np.all(result == False)
    
    def test_circle_contains_mixed(self):
        """Test contains method for mixed inside/outside points."""
        circle = Circle(cx=0.0, cz=0.0, r=1.5)
        x = np.array([0.0, 1.0, 2.0])
        z = np.array([0.0, 1.0, 2.0])
        result = circle.contains(x, z)
        assert result[0] == True  # Center point
        assert result[1] == True  # Inside
        assert result[2] == False  # Outside
    
    def test_circle_contains_on_boundary(self):
        """Test contains method for points on circle boundary."""
        circle = Circle(cx=0.0, cz=0.0, r=1.0)
        x = np.array([1.0, 0.0])
        z = np.array([0.0, 1.0])
        result = circle.contains(x, z)
        assert np.all(result == True)  # Boundary points are included


class TestRect:
    """Test cases for Rect class."""
    
    def test_rect_creation(self):
        """Test creating a rectangle."""
        rect = Rect(xmin=0.0, xmax=2.0, zmin=1.0, zmax=3.0)
        assert rect.xmin == 0.0
        assert rect.xmax == 2.0
        assert rect.zmin == 1.0
        assert rect.zmax == 3.0
    
    def test_rect_contains_inside(self):
        """Test contains method for points inside rectangle."""
        rect = Rect(xmin=0.0, xmax=2.0, zmin=0.0, zmax=2.0)
        x = np.array([0.5, 1.0, 1.5])
        z = np.array([0.5, 1.0, 1.5])
        result = rect.contains(x, z)
        assert np.all(result == True)
    
    def test_rect_contains_outside(self):
        """Test contains method for points outside rectangle."""
        rect = Rect(xmin=0.0, xmax=2.0, zmin=0.0, zmax=2.0)
        x = np.array([-1.0, 3.0, 1.0])
        z = np.array([1.0, 1.0, 3.0])
        result = rect.contains(x, z)
        assert np.all(result == False)
    
    def test_rect_contains_on_boundary(self):
        """Test contains method for points on rectangle boundary."""
        rect = Rect(xmin=0.0, xmax=2.0, zmin=0.0, zmax=2.0)
        x = np.array([0.0, 2.0, 1.0])
        z = np.array([1.0, 1.0, 0.0])
        result = rect.contains(x, z)
        assert np.all(result == True)  # Boundary points are included
    
    def test_rect_contains_mixed(self):
        """Test contains method for mixed inside/outside points."""
        rect = Rect(xmin=0.0, xmax=2.0, zmin=0.0, zmax=2.0)
        x = np.array([0.5, 1.0, 3.0, -1.0])
        z = np.array([0.5, 1.0, 1.0, 1.0])
        result = rect.contains(x, z)
        assert result[0] == True
        assert result[1] == True
        assert result[2] == False
        assert result[3] == False


class TestEnteredJunctionIdx:
    """Test cases for entered_junction_idx function."""
    
    def test_entered_junction_true(self):
        """Test when trajectory enters junction."""
        circle = Circle(cx=2.0, cz=2.0, r=1.0)
        # Trajectory that starts outside and enters
        x = np.array([0.0, 1.0, 2.0, 3.0])
        z = np.array([0.0, 1.0, 2.0, 3.0])
        entered, idx = entered_junction_idx(x, z, circle)
        assert entered == True
        assert idx == 2  # First point inside
    
    def test_entered_junction_false(self):
        """Test when trajectory never enters junction."""
        circle = Circle(cx=10.0, cz=10.0, r=1.0)
        # Trajectory that stays far away
        x = np.array([0.0, 1.0, 2.0, 3.0])
        z = np.array([0.0, 1.0, 2.0, 3.0])
        entered, idx = entered_junction_idx(x, z, circle)
        assert entered == False
        assert 0 <= idx < len(x)  # Index of nearest approach
    
    def test_entered_junction_starts_inside(self):
        """Test when trajectory starts inside junction."""
        circle = Circle(cx=1.0, cz=1.0, r=2.0)
        # Trajectory that starts inside
        x = np.array([1.0, 2.0, 3.0, 4.0])
        z = np.array([1.0, 2.0, 3.0, 4.0])
        entered, idx = entered_junction_idx(x, z, circle)
        assert entered == True
        assert idx == 0  # First point is inside
    
    def test_entered_junction_exact_boundary(self):
        """Test when trajectory touches junction boundary."""
        circle = Circle(cx=0.0, cz=0.0, r=1.0)
        # Point exactly on boundary
        x = np.array([1.0, 2.0])
        z = np.array([0.0, 0.0])
        entered, idx = entered_junction_idx(x, z, circle)
        assert entered == True
        assert idx == 0

