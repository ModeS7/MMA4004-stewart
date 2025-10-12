#!/usr/bin/env python3
"""
Stewart Platform Control Core

- PIDController: PID controller for ball balancing
"""
import numpy as np


class PIDController:
    """
    2D PID Controller for ball position control.

    Controls platform tilt (rx, ry) to keep ball at target position.
    Separate PID for X and Y axes.
    """

    def __init__(self, kp=1.0, ki=0.0, kd=0.5, output_limit=15.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit

        # State for X axis
        self.integral_x = 0.0
        self.prev_error_x = 0.0

        # State for Y axis
        self.integral_y = 0.0
        self.prev_error_y = 0.0

        self.integral_limit = 100.0  # Anti-windup

    def update(self, ball_pos_mm, target_pos_mm, dt):
        """
        Compute PID control output.

        Args:
            ball_pos_mm: (x, y) current ball position in mm
            target_pos_mm: (x, y) target ball position in mm
            dt: time step in seconds

        Returns:
            (rx, ry) platform tilt angles in degrees
        """
        if dt <= 0:
            return 0.0, 0.0

        # Compute errors (ball position - target position)
        error_x = ball_pos_mm[0] - target_pos_mm[0]
        error_y = ball_pos_mm[1] - target_pos_mm[1]

        # X axis PID
        self.integral_x += error_x * dt
        self.integral_x = np.clip(self.integral_x, -self.integral_limit, self.integral_limit)
        derivative_x = (error_x - self.prev_error_x) / dt if dt > 0 else 0.0
        output_x = self.kp * error_x + self.ki * self.integral_x + self.kd * derivative_x
        self.prev_error_x = error_x

        # Y axis PID
        self.integral_y += error_y * dt
        self.integral_y = np.clip(self.integral_y, -self.integral_limit, self.integral_limit)
        derivative_y = (error_y - self.prev_error_y) / dt if dt > 0 else 0.0
        output_y = self.kp * error_y + self.ki * self.integral_y + self.kd * derivative_y
        self.prev_error_y = error_y

        # Map PID output to platform tilt angles
        # Ball at +X needs platform to tilt to bring it back
        # Ball at +Y needs platform to tilt to bring it back
        rx = np.clip(output_y, -self.output_limit, self.output_limit)
        ry = np.clip(output_x, -self.output_limit, self.output_limit)

        return rx, ry

    def reset(self):
        """Reset PID state."""
        self.integral_x = 0.0
        self.prev_error_x = 0.0
        self.integral_y = 0.0
        self.prev_error_y = 0.0

    def set_gains(self, kp, ki, kd):
        """Update PID gains."""
        self.kp = kp
        self.ki = ki
        self.kd = kd
