#!/usr/bin/env python3
"""
Shared utilities and constants for Stewart Platform simulators.
"""

# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================

MAX_TILT_ANGLE_DEG = 15.0
MAX_SERVO_ANGLE_DEG = 40.0
PLATFORM_SIZE_MM = 200.0
PLATFORM_HALF_SIZE_MM = 100.0

# ============================================================================
# TIMING CONSTANTS
# ============================================================================

class ControlLoopConfig:
    """Configuration for real-time control loop."""
    FREQUENCY_HZ = 100
    INTERVAL_S = 1.0 / FREQUENCY_HZ
    IK_TIMEOUT_S = 0.008
    MAX_LOOP_TIME_S = 0.01


class GUIConfig:
    """Configuration for GUI updates."""
    UPDATE_HZ = 5
    UPDATE_INTERVAL_MS = 200
    LOG_INTERVAL_S = 2.0


class SimulationConfig:
    """Configuration for simulation."""
    UPDATE_RATE_MS = 20
    DEFAULT_SERVO_TAU = 0.1
    DEFAULT_SERVO_DELAY = 0.0


# ============================================================================
# FORMATTING UTILITIES
# ============================================================================

def format_vector_2d(vec, units="mm", decimals=1):
    """Format 2D vector for display."""
    return f"({vec[0]:.{decimals}f}, {vec[1]:.{decimals}f}) {units}"


def format_time(seconds, decimals=2):
    """Format time with consistent precision."""
    return f"{seconds:.{decimals}f}s"


def format_angle(degrees, decimals=2):
    """Format angle with degree symbol."""
    return f"{degrees:.{decimals}f}°"


def format_error_context(sim_time, ball_pos, ball_vel, error_msg):
    """Format error message with full context."""
    return (
        f"Error at t={format_time(sim_time)}: {error_msg}\n"
        f"Ball state: pos={format_vector_2d(ball_pos[:2])}, "
        f"vel={format_vector_2d(ball_vel[:2], 'mm/s')}"
    )


# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def is_position_in_bounds(x, y, max_size=PLATFORM_HALF_SIZE_MM):
    """Check if position is within platform bounds."""
    return abs(x) <= max_size and abs(y) <= max_size


def is_tilt_magnitude_valid(rx, ry, max_tilt=MAX_TILT_ANGLE_DEG):
    """Check if tilt magnitude is within limits."""
    import numpy as np
    magnitude = np.sqrt(rx**2 + ry**2)
    return magnitude <= max_tilt


def is_servo_angle_valid(angle, max_angle=MAX_SERVO_ANGLE_DEG):
    """Check if servo angle is within limits."""
    return abs(angle) <= max_angle