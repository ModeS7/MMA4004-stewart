#!/usr/bin/env python3
"""
Stewart Platform Control Core

Controllers:
- PIDController: PID control for ball balancing
- LQRController: Linear Quadratic Regulator for optimal control
- BallPositionFilter: EMA filter for camera noise reduction
"""
import numpy as np
from scipy import linalg

from core.utils import MAX_TILT_ANGLE_DEG


def clip_tilt_vector(rx, ry, max_magnitude=MAX_TILT_ANGLE_DEG):
    """
    Clip tilt vector to maximum magnitude.

    Treats (rx, ry) as a 2D vector and scales proportionally if magnitude exceeds limit.
    Prevents servo constraint violations when both rx and ry are large.

    Example: (11, 11) → magnitude 15.56° → scaled to (10.6, 10.6) at 15°

    Args:
        rx: Roll angle in degrees
        ry: Pitch angle in degrees
        max_magnitude: Maximum allowed tilt magnitude in degrees

    Returns:
        (rx_clipped, ry_clipped, actual_magnitude)
    """
    magnitude = np.sqrt(rx ** 2 + ry ** 2)

    if magnitude > max_magnitude:
        scale = max_magnitude / magnitude
        return rx * scale, ry * scale, magnitude

    return rx, ry, magnitude


class PIDController:
    """
    2D PID Controller for ball position control.

    Controls platform tilt (rx, ry) to maintain ball at target position.
    Separate PID loops for X and Y axes with vector-based output limiting.
    """

    def __init__(self, kp=1.0, ki=0.0, kd=0.5,
                 output_limit=MAX_TILT_ANGLE_DEG,
                 derivative_filter_alpha=0.0):
        """
        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            output_limit: Maximum tilt angle (vector magnitude)
            derivative_filter_alpha: Low-pass filter coefficient (0=none, 0.1=light, 0.5=heavy)
                                    Use >0 for hardware to reduce camera noise
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.derivative_filter_alpha = derivative_filter_alpha

        self.integral_x = 0.0
        self.prev_error_x = 0.0
        self.filtered_derivative_x = 0.0

        self.integral_y = 0.0
        self.prev_error_y = 0.0
        self.filtered_derivative_y = 0.0

        self.integral_limit = 100.0

    def update(self, ball_pos_mm, target_pos_mm, dt):
        """
        Compute PID control output.

        Args:
            ball_pos_mm: (x, y) current ball position in mm
            target_pos_mm: (x, y) target position in mm
            dt: timestep in seconds

        Returns:
            (rx, ry): platform tilt angles in degrees
        """
        if dt <= 0:
            return 0.0, 0.0

        error_x = ball_pos_mm[0] - target_pos_mm[0]
        error_y = ball_pos_mm[1] - target_pos_mm[1]

        output_x = self._compute_pid_axis(error_x, dt, 'x')
        output_y = self._compute_pid_axis(error_y, dt, 'y')

        # Map to platform tilt (axes swapped)
        rx_raw = output_y
        ry_raw = output_x

        rx, ry, _ = clip_tilt_vector(rx_raw, ry_raw, self.output_limit)
        return rx, ry

    def _compute_pid_axis(self, error, dt, axis):
        """Compute PID output for single axis with anti-windup and optional filtering."""
        integral = getattr(self, f'integral_{axis}')
        integral = np.clip(integral + error * dt, -self.integral_limit, self.integral_limit)
        setattr(self, f'integral_{axis}', integral)

        prev_error = getattr(self, f'prev_error_{axis}')
        raw_derivative = (error - prev_error) / dt

        if self.derivative_filter_alpha > 0:
            filtered = getattr(self, f'filtered_derivative_{axis}')
            filtered = (self.derivative_filter_alpha * raw_derivative +
                        (1 - self.derivative_filter_alpha) * filtered)
            setattr(self, f'filtered_derivative_{axis}', filtered)
            derivative = filtered
        else:
            derivative = raw_derivative

        setattr(self, f'prev_error_{axis}', error)

        return self.kp * error + self.ki * integral + self.kd * derivative

    def reset(self):
        """Reset PID state."""
        self.integral_x = 0.0
        self.prev_error_x = 0.0
        self.filtered_derivative_x = 0.0
        self.integral_y = 0.0
        self.prev_error_y = 0.0
        self.filtered_derivative_y = 0.0

    def set_gains(self, kp, ki, kd):
        """Update PID gains."""
        self.kp = kp
        self.ki = ki
        self.kd = kd


class LQRController:
    """
    Linear Quadratic Regulator (LQR) for ball position control.

    Uses optimal control theory to minimize position error and control effort.

    State: [pos_x, pos_y, vel_x, vel_y]
    Control: [tilt_ry, tilt_rx]

    Linearized dynamics around equilibrium (ball centered, zero velocity):
        dx/dt = A*x + B*u
    """

    def __init__(self, Q_pos=1.0, Q_vel=1.0, R=0.01,
                 output_limit=MAX_TILT_ANGLE_DEG,
                 ball_physics_params=None):
        """
        Args:
            Q_pos: Position error cost weight (higher = tighter tracking)
            Q_vel: Velocity cost weight (higher = more damping)
            R: Control effort cost weight (higher = smoother, less aggressive)
            output_limit: Maximum tilt angle in degrees (vector magnitude)
            ball_physics_params: Dict with 'radius', 'mass', 'gravity', 'mass_factor'
        """
        self.Q_pos = Q_pos
        self.Q_vel = Q_vel
        self.R_weight = R
        self.output_limit = output_limit

        if ball_physics_params is None:
            ball_physics_params = {
                'radius': 0.04,
                'mass': 0.0027,
                'gravity': 9.81,
                'mass_factor': 1.667
            }

        self.ball_radius = ball_physics_params['radius']
        self.ball_mass = ball_physics_params['mass']
        self.g = ball_physics_params['gravity']
        self.mass_factor = ball_physics_params['mass_factor']

        self.K = None
        self.compute_lqr_gain()

    def compute_lqr_gain(self):
        """
        Compute LQR gain matrix by solving algebraic Riccati equation.

        Linearization: acceleration = (g / mass_factor) * tilt_radians
        """
        k = (self.g / self.mass_factor) * (np.pi / 180.0)

        A = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])

        B = np.array([
            [0, 0],
            [0, 0],
            [-k, 0],
            [0, -k]
        ])

        Q = np.diag([self.Q_pos, self.Q_pos, self.Q_vel, self.Q_vel])
        R = np.eye(2) * self.R_weight

        try:
            P = linalg.solve_continuous_are(A, B, Q, R)
            self.K = np.linalg.inv(R) @ B.T @ P

            eig_vals = np.linalg.eigvals(A - B @ self.K)
            max_real = np.max(np.real(eig_vals))

            if max_real >= 0:
                print(f"Warning: LQR may be unstable (max eigenvalue: {max_real:.4f})")

        except np.linalg.LinAlgError as e:
            print(f"Error solving Riccati equation: {e}")
            self.K = np.array([[1.0, 0.0, 1.0, 0.0],
                               [0.0, 1.0, 0.0, 1.0]])

    def update(self, ball_pos_mm, ball_vel_mm_s, target_pos_mm=(0.0, 0.0)):
        """
        Compute LQR control output.

        Args:
            ball_pos_mm: (x, y) current position in mm
            ball_vel_mm_s: (vx, vy) current velocity in mm/s
            target_pos_mm: (x, y) target position in mm

        Returns:
            (rx, ry): platform tilt angles in degrees
        """
        x_error = (ball_pos_mm[0] - target_pos_mm[0]) / 1000.0
        y_error = (ball_pos_mm[1] - target_pos_mm[1]) / 1000.0
        vx = ball_vel_mm_s[0] / 1000.0
        vy = ball_vel_mm_s[1] / 1000.0

        state = np.array([x_error, y_error, vx, vy])
        u = -self.K @ state

        rx, ry, _ = clip_tilt_vector(u[1], u[0], self.output_limit)
        return rx, ry

    def reset(self):
        """Reset controller state (LQR is stateless)."""
        pass

    def set_weights(self, Q_pos=None, Q_vel=None, R=None):
        """Update cost weights and recompute gain matrix."""
        if Q_pos is not None:
            self.Q_pos = Q_pos
        if Q_vel is not None:
            self.Q_vel = Q_vel
        if R is not None:
            self.R_weight = R

        self.compute_lqr_gain()

    def get_weights(self):
        """Get current cost weights."""
        return {
            'Q_pos': self.Q_pos,
            'Q_vel': self.Q_vel,
            'R': self.R_weight
        }

    def get_gain_matrix(self):
        """Get current LQR gain matrix."""
        return self.K.copy() if self.K is not None else None


class BallPositionFilter:
    """
    Exponential Moving Average (EMA) filter for ball position.

    Reduces camera noise and vibration while maintaining responsiveness.

    Formula:
        filtered = alpha * raw + (1 - alpha) * filtered_prev

    Where alpha (0 to 1):
        0.0 = no filtering (all old value)
        0.3 = light filtering (responsive)
        0.5 = moderate filtering (balanced)
        0.7 = heavy filtering (smooth but laggy)
        1.0 = no filtering (all new value)
    """

    def __init__(self, alpha=0.3):
        """
        Args:
            alpha: Filter coefficient (0.0 to 1.0). Default 0.3 for responsiveness.
        """
        self.alpha = np.clip(float(alpha), 0.0, 1.0)
        self.x_filtered = 0.0
        self.y_filtered = 0.0
        self.initialized = False

    def update(self, x_raw, y_raw):
        """
        Apply EMA filter to raw ball position.

        Args:
            x_raw: Raw X position (mm)
            y_raw: Raw Y position (mm)

        Returns:
            (x_filtered, y_filtered): Filtered position in mm
        """
        if not self.initialized:
            # First measurement: initialize with raw values
            self.x_filtered = x_raw
            self.y_filtered = y_raw
            self.initialized = True
        else:
            # Apply EMA: blend new measurement with previous filtered value
            self.x_filtered = self.alpha * x_raw + (1.0 - self.alpha) * self.x_filtered
            self.y_filtered = self.alpha * y_raw + (1.0 - self.alpha) * self.y_filtered

        return self.x_filtered, self.y_filtered

    def set_alpha(self, alpha):
        """
        Update filter coefficient on-the-fly.

        Args:
            alpha: New filter coefficient (0.0 to 1.0)
        """
        self.alpha = np.clip(float(alpha), 0.0, 1.0)

    def get_alpha(self):
        """Get current filter coefficient."""
        return self.alpha

    def reset(self):
        """Reset filter state."""
        self.initialized = False
        self.x_filtered = 0.0
        self.y_filtered = 0.0