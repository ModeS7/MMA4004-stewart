#!/usr/bin/env python3
"""
Stewart Platform Control Core

- PIDController: PID controller for ball balancing
- LQRController: Linear Quadratic Regulator for optimal control
"""
import numpy as np
from scipy import linalg


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


class LQRController:
    """
    Linear Quadratic Regulator (LQR) for ball position control.

    Controls platform tilt (rx, ry) to keep ball at target position.
    Uses optimal control theory to minimize both position error and control effort.

    State space model:
        x = [pos_x, pos_y, vel_x, vel_y]
        u = [tilt_ry, tilt_rx]

    Linearized dynamics around equilibrium (ball at center, zero velocity):
        dx/dt = A*x + B*u
    """

    def __init__(self,
                 Q_pos=1.0,
                 Q_vel=1.0,
                 R=0.01,
                 output_limit=15.0,
                 ball_physics_params=None):
        """
        Initialize LQR controller.

        Args:
            Q_pos: State cost weight for position error (higher = more aggressive position correction)
            Q_vel: State cost weight for velocity (higher = more damping)
            R: Control cost weight (higher = less aggressive control, smoother motion)
            output_limit: Maximum tilt angle output in degrees
            ball_physics_params: Dict with 'radius', 'mass', 'gravity', 'mass_factor'
        """
        # Cost matrix weights
        self.Q_pos = Q_pos
        self.Q_vel = Q_vel
        self.R_weight = R
        self.output_limit = output_limit

        # Ball physics parameters (needed for linearization)
        if ball_physics_params is None:
            # Default values matching SimpleBallPhysics2D defaults
            ball_physics_params = {
                'radius': 0.04,
                'mass': 0.0027,
                'gravity': 9.81,
                'mass_factor': 1.667  # For hollow sphere
            }

        self.ball_radius = ball_physics_params['radius']
        self.ball_mass = ball_physics_params['mass']
        self.g = ball_physics_params['gravity']
        self.mass_factor = ball_physics_params['mass_factor']

        # Compute gain matrix
        self.K = None
        self.compute_lqr_gain()

    def compute_lqr_gain(self):
        """
        Compute LQR gain matrix by solving the algebraic Riccati equation.

        System linearization:
        For a ball on a tilted surface with small angles:
            acceleration = (g / mass_factor) * sin(tilt) ≈ (g / mass_factor) * tilt_radians

        State space:
            x = [x_pos(m), y_pos(m), x_vel(m/s), y_vel(m/s)]
            u = [ry(deg), rx(deg)]  # Platform roll and pitch

            dx/dt = A*x + B*u
        """
        # Linearization coefficient: acceleration per degree of tilt
        # For small angles: sin(θ) ≈ θ (in radians)
        # k = (g / mass_factor) * (π/180)  [m/s² per degree]
        k = (self.g / self.mass_factor) * (np.pi / 180.0)

        # State transition matrix A (4x4)
        # States: [x, y, vx, vy]
        # Dynamics: dx = vx, dy = vy, dvx = 0, dvy = 0 (with no control)
        A = np.array([
            [0, 0, 1, 0],  # dx/dt = vx
            [0, 0, 0, 1],  # dy/dt = vy
            [0, 0, 0, 0],  # dvx/dt = 0 (without control)
            [0, 0, 0, 0]  # dvy/dt = 0 (without control)
        ])

        # Control input matrix B (4x2)
        # Controls: [ry, rx]
        # Effect: tilt in +ry causes -x acceleration, tilt in +rx causes -y acceleration
        B = np.array([
            [0, 0],  # Position not directly affected
            [0, 0],  # Position not directly affected
            [-k, 0],  # dvx/dt = -k * ry (tilt in ry moves ball in -x)
            [0, -k]  # dvy/dt = -k * rx (tilt in rx moves ball in -y)
        ])

        # State cost matrix Q (4x4) - penalizes deviation from target
        Q = np.diag([
            self.Q_pos,  # x position error cost
            self.Q_pos,  # y position error cost
            self.Q_vel,  # x velocity cost (damping)
            self.Q_vel  # y velocity cost (damping)
        ])

        # Control cost matrix R (2x2) - penalizes control effort
        R = np.eye(2) * self.R_weight

        # Solve continuous-time algebraic Riccati equation
        try:
            P = linalg.solve_continuous_are(A, B, Q, R)

            # Compute optimal gain: K = R^-1 * B^T * P
            self.K = np.linalg.inv(R) @ B.T @ P

            # Verify stability (closed-loop eigenvalues should have negative real parts)
            eig_vals = np.linalg.eigvals(A - B @ self.K)
            max_real = np.max(np.real(eig_vals))

            if max_real >= 0:
                print(f"Warning: LQR controller may be unstable (max eigenvalue real part: {max_real:.4f})")

        except np.linalg.LinAlgError as e:
            print(f"Error solving Riccati equation: {e}")
            # Fallback to simple gain
            self.K = np.array([[1.0, 0.0, 1.0, 0.0],
                               [0.0, 1.0, 0.0, 1.0]])

    def update(self, ball_pos_mm, ball_vel_mm_s, target_pos_mm=(0.0, 0.0)):
        """
        Compute LQR control output.

        Args:
            ball_pos_mm: (x, y) current ball position in mm
            ball_vel_mm_s: (vx, vy) current ball velocity in mm/s
            target_pos_mm: (x, y) target ball position in mm

        Returns:
            (rx, ry) platform tilt angles in degrees
        """
        # Convert to meters for state vector
        x_error = (ball_pos_mm[0] - target_pos_mm[0]) / 1000.0  # m
        y_error = (ball_pos_mm[1] - target_pos_mm[1]) / 1000.0  # m
        vx = ball_vel_mm_s[0] / 1000.0  # m/s
        vy = ball_vel_mm_s[1] / 1000.0  # m/s

        # State vector: [x, y, vx, vy]
        state = np.array([x_error, y_error, vx, vy])

        # Optimal control: u = -K * x
        u = -self.K @ state

        # u = [ry, rx] in degrees
        ry = np.clip(u[0], -self.output_limit, self.output_limit)
        rx = np.clip(u[1], -self.output_limit, self.output_limit)

        return rx, ry

    def reset(self):
        """Reset controller state (LQR is stateless, so this is a no-op)."""
        pass

    def set_weights(self, Q_pos=None, Q_vel=None, R=None):
        """
        Update cost matrix weights and recompute gain.

        Args:
            Q_pos: State cost weight for position
            Q_vel: State cost weight for velocity
            R: Control cost weight
        """
        if Q_pos is not None:
            self.Q_pos = Q_pos
        if Q_vel is not None:
            self.Q_vel = Q_vel
        if R is not None:
            self.R_weight = R

        self.compute_lqr_gain()

    def get_weights(self):
        """Get current cost matrix weights."""
        return {
            'Q_pos': self.Q_pos,
            'Q_vel': self.Q_vel,
            'R': self.R_weight
        }

    def get_gain_matrix(self):
        """Get the current LQR gain matrix."""
        return self.K.copy() if self.K is not None else None