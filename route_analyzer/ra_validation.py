# ------------------------------
# Validation Framework
# ------------------------------

from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional
import argparse


class ValidationRule(ABC):
    """Base class for validation rules"""
    
    @abstractmethod
    def validate(self, args: Any) -> Tuple[bool, str]:
        """
        Validate arguments and return (is_valid, error_message)
        
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        pass


class PositiveNumberRule(ValidationRule):
    """Validate that a field is a positive number"""
    
    def __init__(self, field_name: str):
        self.field_name = field_name
    
    def validate(self, args: Any) -> Tuple[bool, str]:
        value = getattr(args, self.field_name, None)
        if value is not None and (not isinstance(value, (int, float)) or value <= 0):
            return False, f"--{self.field_name} must be > 0"
        return True, ""


class NonNegativeNumberRule(ValidationRule):
    """Validate that a field is a non-negative number"""
    
    def __init__(self, field_name: str):
        self.field_name = field_name
    
    def validate(self, args: Any) -> Tuple[bool, str]:
        value = getattr(args, self.field_name, None)
        if value is not None and (not isinstance(value, (int, float)) or value < 0):
            return False, f"--{self.field_name} must be >= 0"
        return True, ""


class IntegerRule(ValidationRule):
    """Validate that a field is an integer"""
    
    def __init__(self, field_name: str, min_value: int = None):
        self.field_name = field_name
        self.min_value = min_value
    
    def validate(self, args: Any) -> Tuple[bool, str]:
        value = getattr(args, self.field_name, None)
        if value is not None:
            if not isinstance(value, int):
                return False, f"--{self.field_name} must be an integer"
            if self.min_value is not None and value < self.min_value:
                return False, f"--{self.field_name} must be >= {self.min_value}"
        return True, ""


class RadiusOuterRule(ValidationRule):
    """Validate r_outer requirements for radial decision mode"""
    
    def validate(self, args: Any) -> Tuple[bool, str]:
        if getattr(args, "decision_mode", None) == "radial":
            # Check for single-junction case (r_outer)
            r_outer = getattr(args, "r_outer", None)
            radius = getattr(args, "radius", 0.0)
            
            # Check for multi-junction case (r_outer_list)
            r_outer_list = getattr(args, "r_outer_list", None)
            junctions = getattr(args, "junctions", None)
            
            if r_outer is not None:
                # Single-junction validation
                if r_outer <= radius:
                    return False, "--decision_mode radial requires --r_outer > --radius"
            elif r_outer_list is not None and junctions is not None:
                # Multi-junction validation
                if len(r_outer_list) == 0:
                    return False, "--decision_mode radial requires --r_outer_list with values > junction radii"
                
                # Parse junctions (triples: x z r)
                if len(junctions) % 3 != 0:
                    return False, "--junctions must be triples: x z r ..."
                
                junction_radii = [junctions[i+2] for i in range(0, len(junctions), 3)]
                
                # Check if we have enough r_outer values
                if len(r_outer_list) != len(junction_radii):
                    return False, f"--r_outer_list length ({len(r_outer_list)}) must match number of junctions ({len(junction_radii)})"
                
                # Check each r_outer > radius
                for i, (r_outer_val, radius_val) in enumerate(zip(r_outer_list, junction_radii)):
                    if r_outer_val <= radius_val:
                        return False, f"--decision_mode radial requires r_outer_list[{i}] ({r_outer_val}) > junction radius ({radius_val})"
            else:
                # Neither r_outer nor r_outer_list provided
                return False, "--decision_mode radial requires --r_outer or --r_outer_list"
        
        return True, ""


class LingerDeltaRule(ValidationRule):
    """Validate linger_delta vs r_outer relationship"""
    
    def validate(self, args: Any) -> Tuple[bool, str]:
        has_ld = getattr(args, "linger_delta", None) is not None
        has_ro = getattr(args, "r_outer", None) is not None
        
        if has_ld and has_ro:
            min_radial = float(args.radius) + float(args.linger_delta)
            r_outer = float(args.r_outer)
            if min_radial >= r_outer:
                return False, (
                    f"--linger_delta too large: radius + linger_delta = {min_radial:.3f} "
                    f"must be < r_outer = {r_outer:.3f}"
                )
        return True, ""


class JunctionTriplesRule(ValidationRule):
    """Validate that junctions are specified as triples"""
    
    def validate(self, args: Any) -> Tuple[bool, str]:
        junctions = getattr(args, "junctions", None)
        if junctions is not None:
            if len(junctions) % 3 != 0:
                return False, "--junctions must be triples: x z r ..."
        return True, ""


class NumberRangeRule(ValidationRule):
    """Rule for validating numeric values within a range."""
    
    def __init__(self, field_name: str, min_value: Optional[float] = None, max_value: Optional[float] = None, 
                 exclusive_min: bool = False, exclusive_max: bool = False):
        self.field_name = field_name
        self.min_value = min_value
        self.max_value = max_value
        self.exclusive_min = exclusive_min
        self.exclusive_max = exclusive_max
    
    def validate(self, args: Any) -> Tuple[bool, str]:
        value = getattr(args, self.field_name, None)
        if value is None:
            return True, ""
        
        try:
            num_value = float(value)
        except (ValueError, TypeError):
            return False, f"--{self.field_name} must be a number"
        
        if self.min_value is not None:
            if self.exclusive_min:
                if num_value <= self.min_value:
                    return False, f"--{self.field_name} must be greater than {self.min_value}"
            else:
                if num_value < self.min_value:
                    return False, f"--{self.field_name} must be at least {self.min_value}"
        
        if self.max_value is not None:
            if self.exclusive_max:
                if num_value >= self.max_value:
                    return False, f"--{self.field_name} must be less than {self.max_value}"
            else:
                if num_value > self.max_value:
                    return False, f"--{self.field_name} must be at most {self.max_value}"
        
        return True, ""


class RequiredFieldRule(ValidationRule):
    """Validate that required fields are present"""
    
    def __init__(self, field_name: str):
        self.field_name = field_name
    
    def validate(self, args: Any) -> Tuple[bool, str]:
        value = getattr(args, self.field_name, None)
        if value is None:
            return False, f"--{self.field_name} is required"
        return True, ""


class ValidationEngine:
    """Engine for running validation rules"""
    
    def __init__(self):
        self.rules: list[ValidationRule] = []
    
    def add_rule(self, rule: ValidationRule) -> 'ValidationEngine':
        """Add a validation rule"""
        self.rules.append(rule)
        return self
    
    def validate(self, args: Any, parser: argparse.ArgumentParser, strict: bool = False) -> Any:
        """
        Validate arguments using all registered rules
        
        Args:
            args: Parsed arguments
            parser: Argument parser for error reporting
            strict: If True, fail on validation errors. If False, try to fix them.
        
        Returns:
            Validated (and possibly modified) arguments
        """
        for rule in self.rules:
            is_valid, error_msg = rule.validate(args)
            if not is_valid:
                if strict:
                    parser.error(error_msg)
                else:
                    # Try to fix common issues
                    if "linger_delta too large" in error_msg:
                        # Auto-fix linger_delta
                        r_outer = float(args.r_outer)
                        radius = float(args.radius)
                        new_ld = max(0.0, r_outer - radius - 1e-6)
                        args.linger_delta = new_ld
                        print(f"[warn] Auto-fixed linger_delta: {new_ld:.6f}")
                    else:
                        parser.error(error_msg)
        return args


def create_default_validator() -> ValidationEngine:
    """Create a validator with common validation rules"""
    return (ValidationEngine()
            .add_rule(PositiveNumberRule("radius"))
            .add_rule(PositiveNumberRule("epsilon"))
            .add_rule(PositiveNumberRule("distance"))
            .add_rule(PositiveNumberRule("r_outer"))
            .add_rule(NonNegativeNumberRule("linger_delta"))
            .add_rule(IntegerRule("k", min_value=1))
            .add_rule(IntegerRule("k_min", min_value=2))
            .add_rule(IntegerRule("k_max", min_value=2))
            .add_rule(RadiusOuterRule())
            .add_rule(LingerDeltaRule())
            .add_rule(JunctionTriplesRule()))


def validate_args(args: Any, parser: argparse.ArgumentParser, strict: bool = False) -> Any:
    """
    Convenience function to validate arguments using default rules
    
    Args:
        args: Parsed arguments
        parser: Argument parser
        strict: Whether to use strict validation
    
    Returns:
        Validated arguments
    """
    validator = create_default_validator()
    return validator.validate(args, parser, strict=strict)
