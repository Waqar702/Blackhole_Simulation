"""
Numerical integration methods for geodesic equations.

This module provides various integration methods for solving the
geodesic equations, including RK4 and adaptive RK45.
"""

import numpy as np
import logging
from typing import Callable, Tuple, Optional

logger = logging.getLogger(__name__)


class RK4Integrator:
    """Runge-Kutta 4th order integrator."""
    
    def __init__(self, step_size: float = 1e7):
        """
        Initialize RK4 integrator.
        
        Args:
            step_size: Integration step size
        """
        self.step_size = step_size
    
    def integrate(
        self,
        rhs_func: Callable[[np.ndarray], np.ndarray],
        y0: np.ndarray,
        steps: int,
    ) -> np.ndarray:
        """
        Integrate system of ODEs using RK4 method.
        
        Args:
            rhs_func: Right-hand side function f(t, y) -> dy/dt
            y0: Initial conditions
            steps: Number of integration steps
            
        Returns:
            Final state vector
        """
        y = y0.copy()
        
        for _ in range(steps):
            # RK4 stages
            k1 = self.step_size * rhs_func(y)
            k2 = self.step_size * rhs_func(y + 0.5 * k1)
            k3 = self.step_size * rhs_func(y + 0.5 * k2)
            k4 = self.step_size * rhs_func(y + k3)
            
            # Update state
            y += (k1 + 2*k2 + 2*k3 + k4) / 6.0
        
        return y
    
    def step(
        self,
        rhs_func: Callable[[np.ndarray], np.ndarray],
        y: np.ndarray,
    ) -> np.ndarray:
        """
        Take a single RK4 step.
        
        Args:
            rhs_func: Right-hand side function
            y: Current state
            
        Returns:
            Updated state
        """
        k1 = self.step_size * rhs_func(y)
        k2 = self.step_size * rhs_func(y + 0.5 * k1)
        k3 = self.step_size * rhs_func(y + 0.5 * k2)
        k4 = self.step_size * rhs_func(y + k3)
        
        return y + (k1 + 2*k2 + 2*k3 + k4) / 6.0
    
    def set_step_size(self, step_size: float) -> None:
        """
        Set integration step size.
        
        Args:
            step_size: New step size
        """
        self.step_size = step_size


class RK45Integrator:
    """Adaptive Runge-Kutta-Fehlberg 4th/5th order integrator."""
    
    def __init__(
        self,
        initial_step_size: float = 1e7,
        rel_tol: float = 1e-6,
        abs_tol: float = 1e-8,
        min_step_size: float = 1e3,
        max_step_size: float = 1e9,
    ):
        """
        Initialize RK45 integrator.
        
        Args:
            initial_step_size: Initial step size
            rel_tol: Relative tolerance
            abs_tol: Absolute tolerance
            min_step_size: Minimum allowed step size
            max_step_size: Maximum allowed step size
        """
        self.step_size = initial_step_size
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol
        self.min_step_size = min_step_size
        self.max_step_size = max_step_size
        
        # RK45 coefficients
        self.a = np.array([
            [0, 0, 0, 0, 0, 0],
            [1/4, 0, 0, 0, 0, 0],
            [3/32, 9/32, 0, 0, 0, 0],
            [1932/2197, -7200/2197, 7296/2197, 0, 0, 0],
            [439/216, -8, 3680/513, -845/4104, 0, 0],
            [-8/27, 2, -3544/2565, 1859/4104, -11/40, 0],
        ])
        
        self.b4 = np.array([25/216, 0, 1408/2565, 2197/4104, -1/5, 0])  # 4th order
        self.b5 = np.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55])  # 5th order
        
        self.c = np.array([0, 1/4, 3/8, 12/13, 1, 1/2])
    
    def integrate(
        self,
        rhs_func: Callable[[np.ndarray, float], np.ndarray],
        y0: np.ndarray,
        t0: float = 0.0,
        t_end: float = 1e8,
        max_steps: int = 10000,
    ) -> Tuple[np.ndarray, float, bool]:
        """
        Integrate system of ODEs using adaptive RK45 method.
        
        Args:
            rhs_func: Right-hand side function f(t, y) -> dy/dt
            y0: Initial conditions
            t0: Initial time
            t_end: End time
            max_steps: Maximum number of steps
            
        Returns:
            Tuple of (final_state, final_time, success)
        """
        y = y0.copy()
        t = t0
        steps = 0
        
        while t < t_end and steps < max_steps:
            success, new_y, new_t, new_step_size = self._step(rhs_func, y, t)
            
            if success:
                y = new_y
                t = new_t
                self.step_size = new_step_size
            else:
                # Step failed, reduce step size and try again
                self.step_size *= 0.5
                if self.step_size < self.min_step_size:
                    logger.warning("Step size too small, stopping integration")
                    break
            
            steps += 1
        
        return y, t, t >= t_end
    
    def _step(
        self,
        rhs_func: Callable[[np.ndarray, float], np.ndarray],
        y: np.ndarray,
        t: float,
    ) -> Tuple[bool, np.ndarray, float, float]:
        """
        Take a single adaptive RK45 step.
        
        Args:
            rhs_func: Right-hand side function
            y: Current state
            t: Current time
            
        Returns:
            Tuple of (success, new_state, new_time, new_step_size)
        """
        # Calculate k values
        k = np.zeros((6, len(y)))
        
        for i in range(6):
            if i == 0:
                k[i] = self.step_size * rhs_func(y, t)
            else:
                sum_a = np.sum(self.a[i, :i] * k[:i], axis=0)
                k[i] = self.step_size * rhs_func(y + sum_a, t + self.c[i] * self.step_size)
        
        # Calculate 4th and 5th order solutions
        y4 = y + np.sum(self.b4.reshape(-1, 1) * k, axis=0)
        y5 = y + np.sum(self.b5.reshape(-1, 1) * k, axis=0)
        
        # Estimate error
        error = np.abs(y5 - y4)
        tolerance = self.rel_tol * np.abs(y5) + self.abs_tol
        
        # Check if step is acceptable
        if np.all(error <= tolerance):
            # Calculate optimal step size for next step
            max_error_ratio = np.max(error / tolerance)
            if max_error_ratio > 0:
                safety_factor = 0.9
                optimal_step = safety_factor * self.step_size * (1.0 / max_error_ratio) ** 0.2
                new_step_size = np.clip(optimal_step, self.min_step_size, self.max_step_size)
            else:
                new_step_size = self.step_size * 2.0  # Increase step size
            
            return True, y5, t + self.step_size, new_step_size
        else:
            # Step rejected
            return False, y, t, self.step_size * 0.5
    
    def set_tolerances(self, rel_tol: float, abs_tol: float) -> None:
        """
        Set integration tolerances.
        
        Args:
            rel_tol: Relative tolerance
            abs_tol: Absolute tolerance
        """
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol
    
    def set_step_size_bounds(self, min_step: float, max_step: float) -> None:
        """
        Set step size bounds.
        
        Args:
            min_step: Minimum step size
            max_step: Maximum step size
        """
        self.min_step_size = min_step
        self.max_step_size = max_step


def create_integrator(
    method: str = "rk4",
    **kwargs
) -> RK4Integrator | RK45Integrator:
    """
    Create an integrator instance.
    
    Args:
        method: Integration method ("rk4" or "rk45")
        **kwargs: Additional arguments for the integrator
        
    Returns:
        Integrator instance
    """
    if method.lower() == "rk4":
        return RK4Integrator(**kwargs)
    elif method.lower() == "rk45":
        return RK45Integrator(**kwargs)
    else:
        raise ValueError(f"Unknown integration method: {method}")
