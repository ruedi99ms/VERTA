# ------------------------------
# Logging and Error Handling
# ------------------------------

import logging
import sys
import time
from contextlib import contextmanager
from typing import Optional


class RouteAnalyzerLogger:
    """Centralized logging for route analyzer operations"""
    
    def __init__(self, level: str = "INFO"):
        self.logger = logging.getLogger("route_analyzer")
        self.logger.setLevel(getattr(logging, level.upper()))
        
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '[%(levelname)s] %(name)s: %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def info(self, message: str) -> None:
        """Log an info message"""
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log a warning message"""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log an error message"""
        self.logger.error(message)
    
    def debug(self, message: str) -> None:
        """Log a debug message"""
        self.logger.debug(message)
    
    @contextmanager
    def operation(self, operation_name: str):
        """Context manager for operations with timing and error handling"""
        self.info(f"Starting {operation_name}")
        start_time = time.time()
        try:
            yield
        except Exception as e:
            self.error(f"{operation_name} failed: {e}")
            raise
        finally:
            duration = time.time() - start_time
            self.info(f"Completed {operation_name} in {duration:.2f}s")
    
    def set_level(self, level: str) -> None:
        """Set the logging level"""
        self.logger.setLevel(getattr(logging, level.upper()))


# Global logger instance
logger = RouteAnalyzerLogger()


def get_logger() -> RouteAnalyzerLogger:
    """Get the global logger instance"""
    return logger
