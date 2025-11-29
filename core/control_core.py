#!/usr/bin/env python3
"""
Stewart Platform Control Core

Controllers:
- PIDController: PID control for ball balancing
- LQRController: Linear Quadratic Regulator for optimal control
- MPCController: Model Predictive Control for optimal control with constraints
- RLController: Reinforcement Learning (SAC) policy for ball balancing
- KalmanFilter: Linear Kalman Filter for ball state estimation
- OrientationKalmanFilter: EKF for IMU-based orientation estimation
"""
import time
from typing import Optional, Tuple, Dict
import numpy as np
from scipy import linalg

from core.utils import (
    MAX_TILT_ANGLE_DEG, MAX_CONTROLLER_OUTPUT_DEG, MAX_IMU_CORRECTION_DEG,
    IMUKalmanConfig, IMUCalibrationConfig,
    PID_INTEGRAL_LIMIT_DEG, MAX_CALIBRATION_TILT_DEG,
    GIMBAL_LOCK_THRESHOLD, DIVISION_BY_ZERO_EPSILON,
    KALMAN_POSITION_PROCESS_NOISE, KALMAN_VELOCITY_PROCESS_NOISE,
    CAMERA_PIXEL_SIZE_M, Pixy2CameraConfig
)


def clip_tilt_vector(rx: float, ry: float, max_magnitude: float = MAX_TILT_ANGLE_DEG) -> Tuple[float, float, float]:
    """
    Clip tilt vector to maximum magnitude while preserving direction.

    Treats (rx, ry) as 2D vector and scales proportionally if magnitude exceeds limit.
    Prevents servo constraint violations when both rx and ry are large.

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

    def __init__(self, kp: float = 1.0, ki: float = 0.0, kd: float = 0.5,
                 output_limit: float = MAX_CONTROLLER_OUTPUT_DEG,
                 derivative_filter_alpha: float = 0.0) -> None:
        """
        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            output_limit: Maximum tilt angle (vector magnitude)
            derivative_filter_alpha: Low-pass filter coefficient (0=none, >0=filtering)
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

        self.integral_limit = PID_INTEGRAL_LIMIT_DEG

    def update(self, ball_pos_mm: Tuple[float, float], target_pos_mm: Tuple[float, float], dt: float) -> Tuple[float, float]:
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

        rx_raw = output_y
        ry_raw = -output_x

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

    def reset(self) -> None:
        """Reset PID state."""
        self.integral_x = 0.0
        self.prev_error_x = 0.0
        self.filtered_derivative_x = 0.0
        self.integral_y = 0.0
        self.prev_error_y = 0.0
        self.filtered_derivative_y = 0.0

    def set_gains(self, kp: float, ki: float, kd: float) -> None:
        """Update PID gains."""
        self.kp = kp
        self.ki = ki
        self.kd = kd


class LQRController:
    """
    Linear Quadratic Regulator (LQR) for ball position control.

    Uses optimal control theory to minimize position error and control effort.
    Linearized dynamics around equilibrium (ball centered, zero velocity).

    State: [pos_x, pos_y, vel_x, vel_y]
    Control: [tilt_ry, tilt_rx]
    """

    def __init__(self, Q_pos: float = 1.0, Q_vel: float = 1.0, R: float = 0.01,
                 output_limit: float = MAX_CONTROLLER_OUTPUT_DEG,
                 ball_physics_params: Optional[dict] = None) -> None:
        """
        Args:
            Q_pos: Position error cost weight
            Q_vel: Velocity cost weight
            R: Control effort cost weight
            output_limit: Maximum tilt angle in degrees (vector magnitude)
            ball_physics_params: Dict with 'radius', 'mass', 'gravity', 'mass_factor'
        """
        self.Q_pos = Q_pos
        self.Q_vel = Q_vel
        self.R_weight = R
        self.output_limit = output_limit

        if ball_physics_params is None:
            ball_physics_params = {
                'radius': 0.02,
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
        """Compute LQR gain matrix by solving algebraic Riccati equation."""
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
            [k, 0],
            [0, -k]
        ])

        Q = np.diag([self.Q_pos, self.Q_pos, self.Q_vel, self.Q_vel])
        R = np.eye(2) * self.R_weight

        try:
            P = linalg.solve_continuous_are(A, B, Q, R)
            self.K = np.linalg.solve(R, B.T @ P)

            # Verify closed-loop stability
            eig_vals = np.linalg.eigvals(A - B @ self.K)
            max_real = np.max(np.real(eig_vals))

            if max_real >= 0:
                import warnings
                warnings.warn(f"LQR: Closed-loop system may be unstable (max eigenvalue real part: {max_real:.4f})")

        except np.linalg.LinAlgError as e:
            import warnings
            warnings.warn(f"LQR: Riccati equation solver failed: {e}. Using fallback gains.")
            # Fallback to simple proportional+derivative gains
            self.K = np.array([[1.0, 0.0, 1.0, 0.0],
                               [0.0, 1.0, 0.0, 1.0]])

    def update(self, ball_pos_mm: Tuple[float, float], ball_vel_mm_s: Tuple[float, float],
               target_pos_mm: Tuple[float, float] = (0.0, 0.0)) -> Tuple[float, float]:
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

    def reset(self) -> None:
        """Reset controller state (LQR is stateless)."""
        pass

    def set_weights(self, Q_pos: Optional[float] = None, Q_vel: Optional[float] = None, R: Optional[float] = None) -> None:
        """Update cost weights and recompute gain matrix."""
        if Q_pos is not None:
            self.Q_pos = Q_pos
        if Q_vel is not None:
            self.Q_vel = Q_vel
        if R is not None:
            self.R_weight = R

        self.compute_lqr_gain()

    def get_weights(self) -> dict:
        """Get current cost weights."""
        return {
            'Q_pos': self.Q_pos,
            'Q_vel': self.Q_vel,
            'R': self.R_weight
        }

    def get_gain_matrix(self):
        """Get current LQR gain matrix."""
        return self.K.copy() if self.K is not None else None


class RLController:
    """
    Reinforcement Learning Controller using trained SAC policy.

    Uses frame stacking (12 frames × 7 values = 84 inputs) with physics estimation.
    Matches the architecture trained in RL/train.py.
    """

    def __init__(self,
                 model_path: str = "RL/checkpoints/sac_final.pt",
                 output_limit: float = MAX_CONTROLLER_OUTPUT_DEG,
                 device: str = "cpu") -> None:
        """
        Args:
            model_path: Path to trained SAC model checkpoint
            output_limit: Maximum tilt angle (degrees)
            device: "cpu" or "cuda" for inference
        """
        import torch
        import torch.nn as nn

        self.output_limit = output_limit
        self.device = torch.device(device)

        # Frame stacking config (must match training)
        self.num_frames = 12
        self.obs_per_frame = 7  # [ball_x, ball_y, platform_rx, platform_ry, dt, target_x, target_y]
        self.obs_dim = self.num_frames * self.obs_per_frame  # 84

        # Normalization constants (must match training)
        self.platform_radius_mm = 150.0
        self.max_tilt_deg = 10.0

        # Frame history buffer
        self.frame_history = np.zeros((self.num_frames, self.obs_per_frame), dtype=np.float32)

        # Build actor network (same architecture as training)
        class Actor(nn.Module):
            def __init__(self, input_dim=84, action_dim=2, hidden_dim=256, physics_dim=3):
                super().__init__()
                # Shared backbone
                self.backbone = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU()
                )
                # Action head
                self.action_mean = nn.Linear(hidden_dim, action_dim)
                self.action_log_std = nn.Linear(hidden_dim, action_dim)
                # Physics head (not used for inference, but needed for loading weights)
                self.physics_head = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, physics_dim),
                    nn.Sigmoid()
                )

            def forward(self, obs):
                features = self.backbone(obs)
                action_mean = torch.tanh(self.action_mean(features))
                return action_mean

        self.actor = Actor(input_dim=self.obs_dim).to(self.device)
        self.actor.eval()

        # Load trained weights
        self._load_model(model_path)

        # Track current platform tilt
        self.current_rx = 0.0
        self.current_ry = 0.0

        # Track dt for observation
        self.last_time = None

    def _load_model(self, model_path: str) -> None:
        """Load trained actor weights."""
        import torch
        import glob
        import os

        # If path doesn't exist, try to find latest checkpoint
        if not os.path.exists(model_path):
            # Try timestamped folders
            checkpoints = glob.glob('RL/checkpoints/*/sac_*.pt')
            if checkpoints:
                model_path = max(checkpoints, key=os.path.getmtime)
                print(f"[RLController] Auto-selected latest checkpoint: {model_path}")
            else:
                # Try root folder
                root_checkpoints = glob.glob('RL/checkpoints/sac_*.pt')
                if root_checkpoints:
                    model_path = max(root_checkpoints, key=os.path.getmtime)
                    print(f"[RLController] Auto-selected checkpoint: {model_path}")

        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            if 'actor' in checkpoint:
                self.actor.load_state_dict(checkpoint['actor'], strict=False)
            else:
                self.actor.load_state_dict(checkpoint, strict=False)
            print(f"[RLController] Loaded model from {model_path}")
        except Exception as e:
            print(f"[RLController] Warning: Could not load model: {e}")
            print("[RLController] Using random initialization")

    def _get_frame(self, ball_pos_mm: Tuple[float, float],
                   platform_tilt_deg: Tuple[float, float],
                   dt: float,
                   target_pos_mm: Tuple[float, float]) -> np.ndarray:
        """Build single frame observation (normalized)."""
        frame = np.array([
            ball_pos_mm[0] / self.platform_radius_mm,  # ball_x normalized
            ball_pos_mm[1] / self.platform_radius_mm,  # ball_y normalized
            platform_tilt_deg[0] / self.max_tilt_deg,  # platform_rx normalized
            platform_tilt_deg[1] / self.max_tilt_deg,  # platform_ry normalized
            dt / 0.01,  # dt normalized (10ms = 1.0)
            target_pos_mm[0] / self.platform_radius_mm,  # target_x normalized
            target_pos_mm[1] / self.platform_radius_mm,  # target_y normalized
        ], dtype=np.float32)
        return frame

    def update(self, ball_pos_mm: Tuple[float, float], ball_vel_mm_s: Tuple[float, float],
               target_pos_mm: Tuple[float, float] = (0.0, 0.0),
               platform_tilt_deg: Optional[Tuple[float, float]] = None,
               dt: Optional[float] = None) -> Tuple[float, float]:
        """
        Compute RL control output.

        Args:
            ball_pos_mm: (x, y) current position in mm
            ball_vel_mm_s: (vx, vy) current velocity in mm/s (not directly used, implicit in frame history)
            target_pos_mm: (x, y) target position in mm
            platform_tilt_deg: (rx, ry) current platform tilt in degrees
            dt: Time step in seconds (optional, estimated if not provided)

        Returns:
            (rx, ry): platform tilt angles in degrees
        """
        import torch
        import time

        # Use provided platform tilt or tracked values
        if platform_tilt_deg is not None:
            rx_current, ry_current = platform_tilt_deg
        else:
            rx_current, ry_current = self.current_rx, self.current_ry

        # Estimate dt if not provided
        if dt is None:
            current_time = time.time()
            if self.last_time is not None:
                dt = current_time - self.last_time
            else:
                dt = 0.01  # Default 10ms
            self.last_time = current_time
        dt = np.clip(dt, 0.001, 0.1)  # Clamp to reasonable range

        # Build new frame
        new_frame = self._get_frame(ball_pos_mm, (rx_current, ry_current), dt, target_pos_mm)

        # Shift history and add new frame
        self.frame_history[:-1] = self.frame_history[1:]
        self.frame_history[-1] = new_frame

        # Flatten for network input
        obs = self.frame_history.flatten()

        # Run inference
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action = self.actor(obs_tensor).cpu().numpy()[0]

        # Scale action from [-1, 1] to degrees
        rx_target = action[0] * self.max_tilt_deg
        ry_target = action[1] * self.max_tilt_deg

        # Apply output limit
        rx, ry, _ = clip_tilt_vector(rx_target, ry_target, self.output_limit)

        # Track for next iteration
        self.current_rx = rx
        self.current_ry = ry

        return rx, ry

    def reset(self) -> None:
        """Reset controller state."""
        self.current_rx = 0.0
        self.current_ry = 0.0
        self.frame_history = np.zeros((self.num_frames, self.obs_per_frame), dtype=np.float32)
        self.last_time = None

    def set_platform_tilt(self, rx: float, ry: float) -> None:
        """Update tracked platform tilt (call with IMU/FK feedback)."""
        self.current_rx = rx
        self.current_ry = ry


class MPCController:
    """
    Model Predictive Controller (MPC) for ball position control.

    Uses OSQP to solve the constrained QP at each timestep. Provides proper
    constraint handling and rate penalties that LQR cannot offer.

    Advantages over LQR:
    - Constraints are part of optimization (not just clipping)
    - Rate penalty properly integrated into cost
    - Can preview future reference trajectories (not implemented yet)

    State: [pos_x, pos_y, vel_x, vel_y] (meters, m/s)
    Control: [tilt_ry, tilt_rx] (degrees)

    Requires: pip install osqp scipy
    """

    def __init__(self, N: int = 40, dt: float = 0.04,
                 Q_pos: float = 1.0, Q_vel: float = 1.0,
                 R_pos: float = 0.01, R_vel: float = 0.01,
                 output_limit: float = MAX_CONTROLLER_OUTPUT_DEG,
                 ball_physics_params: Optional[Dict] = None) -> None:
        """
        Initialize MPC controller.

        Args:
            N: Prediction horizon (number of steps)
            dt: Time step for MPC (seconds)
            Q_pos: Position error cost weight
            Q_vel: Velocity cost weight
            R_pos: Control effort cost weight
            R_vel: Control rate (smoothness) cost weight
            output_limit: Maximum tilt angle in degrees
            ball_physics_params: Physics parameters dict
        """
        try:
            import osqp
            import scipy.sparse as sparse
            self._osqp = osqp
            self._sparse = sparse
        except ImportError:
            raise ImportError("OSQP required for MPC controller. Install with: pip install osqp")

        self.dt = dt
        self.N = N
        self.output_limit = output_limit

        self.Q_pos = Q_pos
        self.Q_vel = Q_vel
        self.R_pos = R_pos
        self.R_vel = R_vel

        # Ball physics (same as LQR)
        if ball_physics_params is None:
            ball_physics_params = {
                'radius': 0.02,
                'mass': 0.0027,
                'gravity': 9.81,
                'mass_factor': 1.667
            }

        self.g = ball_physics_params['gravity']
        self.mass_factor = ball_physics_params['mass_factor']

        # Linearized gain: acceleration = k * tilt_angle (degrees)
        self.k = (self.g / self.mass_factor) * (np.pi / 180.0)

        # Previous control for rate penalty
        self.u_prev = np.zeros(2)

        # For adaptive dt: rebuild QP if dt changes by more than this fraction
        self._dt_tolerance = 0.15  # 15% tolerance
        self._last_build_dt = dt
        self._qp_initialized = False
        self.solver = None

        # Build QP matrices
        self._build_qp()

    def _build_qp(self):
        """Build condensed QP matrices for MPC."""
        try:
            self._build_qp_internal()
        except Exception as e:
            print(f"[MPC] QP build failed: {e}")
            self._qp_initialized = False

    def _build_qp_internal(self):
        """Internal QP matrix construction."""
        sparse = self._sparse
        N = self.N
        nx = 4  # State dimension
        nu = 2  # Control dimension
        dt = self.dt
        k = self.k

        # Discrete-time dynamics: x_{k+1} = A @ x_k + B @ u_k
        A = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        B = np.array([
            [0.5 * k * dt**2, 0],
            [0, -0.5 * k * dt**2],
            [k * dt, 0],
            [0, -k * dt]
        ])

        self.A = A
        self.B = B

        # Build prediction matrices: X = Sx @ x0 + Su @ U
        # where X = [x_1; x_2; ...; x_N] and U = [u_0; u_1; ...; u_{N-1}]
        Sx = np.zeros((N * nx, nx))
        Su = np.zeros((N * nx, N * nu))

        A_pow = np.eye(nx)
        for i in range(N):
            A_pow = A_pow @ A
            Sx[i*nx:(i+1)*nx, :] = A_pow

            for j in range(i + 1):
                A_pow_j = np.linalg.matrix_power(A, i - j)
                Su[i*nx:(i+1)*nx, j*nu:(j+1)*nu] = A_pow_j @ B

        self.Sx = Sx
        self.Su = Su

        # Cost matrices
        Q = np.diag([self.Q_pos, self.Q_pos, self.Q_vel, self.Q_vel])
        R = np.eye(nu) * self.R_pos

        # Build block diagonal Q and R for full horizon
        Q_bar = np.kron(np.eye(N), Q)
        R_bar = np.kron(np.eye(N), R)

        # Rate penalty: penalize (u_k - u_{k-1})^2
        # Build difference matrix D such that D @ U = [u_0 - u_prev; u_1 - u_0; ...]
        D = np.eye(N * nu)
        for i in range(1, N):
            D[i*nu:(i+1)*nu, (i-1)*nu:i*nu] = -np.eye(nu)
        R_rate = np.eye(N * nu) * self.R_vel

        # QP cost: 0.5 * U' @ H @ U + f' @ U
        # J = (Sx @ x0 + Su @ U)' @ Q_bar @ (Sx @ x0 + Su @ U) + U' @ R_bar @ U + (D @ U)' @ R_rate @ (D @ U)
        # Expanding: J = 0.5 * U' @ H @ U + f' @ U + const
        # H = Su' @ Q_bar @ Su + R_bar + D' @ R_rate @ D
        # f = Su' @ Q_bar @ Sx @ x0 - D' @ R_rate @ [u_prev; 0; 0; ...]

        self.H_base = Su.T @ Q_bar @ Su + R_bar + D.T @ R_rate @ D
        self.f_coef = Su.T @ Q_bar @ Sx  # f = f_coef @ x0 + f_rate_term

        # Rate penalty term for u_prev (only affects first control)
        self.D = D
        self.R_rate = R_rate

        # Control bounds
        u_max = self.output_limit
        self.u_min = -u_max * np.ones(N * nu)
        self.u_max = u_max * np.ones(N * nu)

        # Setup OSQP solver
        H_sparse = sparse.csc_matrix(self.H_base)

        # Constraint matrix for bounds: I @ U (identity)
        A_constr = sparse.eye(N * nu, format='csc')

        # Create solver
        self.solver = self._osqp.OSQP()
        self.solver.setup(
            P=H_sparse,
            q=np.zeros(N * nu),  # Will be updated each solve
            A=A_constr,
            l=self.u_min,
            u=self.u_max,
            verbose=False,
            warm_start=True,
            eps_abs=1e-4,
            eps_rel=1e-4,
            max_iter=100
        )

        self._qp_initialized = True

    def update(self, ball_pos_mm: Tuple[float, float], ball_vel_mm_s: Tuple[float, float],
               target_pos_mm: Tuple[float, float] = (0.0, 0.0),
               actual_dt: Optional[float] = None) -> Tuple[float, float]:
        """
        Compute MPC control output by solving QP.

        Args:
            ball_pos_mm: Current ball position (x, y) in mm
            ball_vel_mm_s: Current ball velocity (vx, vy) in mm/s
            target_pos_mm: Target position (x, y) in mm
            actual_dt: Actual time step (if different from nominal, QP is rebuilt)

        Returns:
            (rx, ry): Platform tilt angles in degrees
        """
        # Safety checks for None inputs
        if ball_pos_mm is None or ball_vel_mm_s is None:
            return 0.0, 0.0
        if target_pos_mm is None:
            target_pos_mm = (0.0, 0.0)

        # Check solver is initialized
        if not self._qp_initialized or self.solver is None:
            return 0.0, 0.0

        # Adaptive dt: rebuild QP if actual dt differs significantly
        if actual_dt is not None and actual_dt > 0:
            dt_ratio = abs(actual_dt - self._last_build_dt) / self._last_build_dt
            if dt_ratio > self._dt_tolerance:
                self.dt = actual_dt
                try:
                    self._build_qp()
                    self._last_build_dt = actual_dt
                except Exception:
                    # If rebuild fails, continue with old QP
                    pass

        # Convert to meters
        x_error = (ball_pos_mm[0] - target_pos_mm[0]) / 1000.0
        y_error = (ball_pos_mm[1] - target_pos_mm[1]) / 1000.0
        vx = ball_vel_mm_s[0] / 1000.0
        vy = ball_vel_mm_s[1] / 1000.0

        x0 = np.array([x_error, y_error, vx, vy])

        # Compute linear term: f = f_coef @ x0 + rate_term
        f = self.f_coef @ x0

        # Add rate penalty for u_prev (penalize u_0 - u_prev)
        if self.R_vel > 0:
            # The rate term affects the linear cost for u_0
            f[:2] -= self.R_vel * self.u_prev

        # Update QP and solve
        try:
            self.solver.update(q=f)
            result = self.solver.solve()
        except Exception:
            # Solver error - return previous control
            return self.u_prev[1], self.u_prev[0]

        if result.info.status not in ['solved', 'solved_inaccurate']:
            # Fallback to previous control if QP fails
            return self.u_prev[1], self.u_prev[0]  # rx, ry

        # Extract first control action
        U_opt = result.x
        if U_opt is None:
            return self.u_prev[1], self.u_prev[0]

        u0 = U_opt[:2]  # [tilt_ry, tilt_rx]

        ry_deg = u0[0]
        rx_deg = u0[1]

        # Store for next rate penalty
        self.u_prev = np.array([ry_deg, rx_deg])

        return rx_deg, ry_deg

    def reset(self) -> None:
        """Reset MPC state."""
        self.u_prev = np.zeros(2)

    def set_weights(self, Q_pos: Optional[float] = None, Q_vel: Optional[float] = None,
                    R_pos: Optional[float] = None, R_vel: Optional[float] = None) -> None:
        """Update cost weights and rebuild QP."""
        if Q_pos is not None:
            self.Q_pos = Q_pos
        if Q_vel is not None:
            self.Q_vel = Q_vel
        if R_pos is not None:
            self.R_pos = R_pos
        if R_vel is not None:
            self.R_vel = R_vel

        # Rebuild QP with new weights
        self._build_qp()

    def set_horizon(self, N: int) -> None:
        """Update prediction horizon and rebuild QP."""
        self.N = N
        self._build_qp()

    def get_weights(self) -> Dict[str, float]:
        """Get current cost weights."""
        return {
            'Q_pos': self.Q_pos,
            'Q_vel': self.Q_vel,
            'R_pos': self.R_pos,
            'R_vel': self.R_vel
        }


# ============================================================================


class KalmanFilter:
    """
    Linear Kalman Filter for ball position and velocity estimation.

    State: [x, y, vx, vy] (position and velocity in meters)
    Control: [rx, ry] (platform tilt angles in degrees)
    Measurement: [x, y] (camera position in meters)

    Features:
    - Handles asynchronous measurements (prediction at control rate, update at camera rate)
    - Handles detection dropouts (prediction-only mode)
    - Uses platform dynamics for accurate prediction
    - Smooths quantized camera measurements
    """

    def __init__(self,
                 process_noise_scale: float = 1.0,
                 measurement_noise_scale: float = 1.0,
                 ball_physics_params: Optional[dict] = None,
                 dt: float = 0.01,
                 include_damping: bool = True,
                 camera_type: str = 'PIXY2') -> None:
        """
        Args:
            process_noise_scale: Scaling factor for process noise Q (tunable)
            measurement_noise_scale: Scaling factor for measurement noise R (tunable)
            ball_physics_params: Dict with 'radius', 'mass', 'gravity', 'mass_factor'
            dt: Time step for prediction (control loop period)
            include_damping: If True, include velocity-dependent damping (rolling friction)
            camera_type: Camera type ('PIXY2' or 'STEREO') for noise characteristics
        """
        self.camera_type = camera_type
        # Default ball physics parameters
        if ball_physics_params is None:
            ball_physics_params = {
                'radius': 0.02,
                'mass': 0.0027,
                'gravity': 9.81,
                'mass_factor': 1.667
            }

        self.g = ball_physics_params['gravity']
        self.mass_factor = ball_physics_params['mass_factor']
        self.mu_roll = ball_physics_params.get('rolling_friction', 0.01)
        self.include_damping = include_damping
        self.dt = dt

        # Acceleration constant: a = k * tilt_angle
        # where k = (g / mass_factor) * (pi/180) converts degrees to m/s²
        self.k = (self.g / self.mass_factor) * (np.pi / 180.0)

        # State: [x, y, vx, vy] in meters and m/s
        self.x = np.zeros(4)

        # State covariance matrix
        self.P = np.eye(4) * 0.01  # Initial uncertainty

        # Store scaling factors
        self.process_noise_scale = process_noise_scale
        self.measurement_noise_scale = measurement_noise_scale

        # Build system matrices
        self._build_system_matrices()

        # Statistics
        self.prediction_count = 0
        self.update_count = 0
        self.last_measurement_time = -1.0

    def _build_system_matrices(self):
        """Build state transition and noise covariance matrices."""
        dt = self.dt

        # State transition matrix (constant velocity model + control input)
        # x_k+1 = A*x_k + B*u_k
        self.A = np.array([
            [1, 0, dt, 0],  # x = x + vx*dt
            [0, 1, 0, dt],  # y = y + vy*dt
            [0, 0, 1, 0],  # vx = vx + ax*dt (ax from control)
            [0, 0, 0, 1]  # vy = vy + ay*dt (ay from control)
        ])

        # Control input matrix: converts [rx, ry] to acceleration
        # ax = k * ry (tilt in Y causes acceleration in X)
        # ay = -k * rx (tilt in X causes acceleration in Y, inverted)
        self.B = np.array([
            [0, 0],
            [0, 0],
            [0, dt * self.k],  # vx += dt * k * ry
            [-dt * self.k, 0]  # vy += dt * (-k) * rx
        ])

        # Measurement matrix: we observe [x, y]
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # Process noise covariance Q
        # Accounts for modeling errors, unmodeled dynamics, disturbances
        # Use continuous white noise model
        q_pos = KALMAN_POSITION_PROCESS_NOISE  # Position process noise (m²)
        q_vel = KALMAN_VELOCITY_PROCESS_NOISE  # Velocity process noise (m²/s²)

        self.Q_base = np.array([
            [q_pos * dt ** 3 / 3, 0, q_pos * dt ** 2 / 2, 0],
            [0, q_pos * dt ** 3 / 3, 0, q_pos * dt ** 2 / 2],
            [q_pos * dt ** 2 / 2, 0, q_vel * dt, 0],
            [0, q_pos * dt ** 2 / 2, 0, q_vel * dt]
        ])

        # Measurement noise covariance R
        # Based on camera characteristics (platform-specific):
        if self.camera_type == 'STEREO':
            # Stereo camera: Use conservative operational noise estimates
            # Stereo triangulation introduces noise from both cameras and depth estimation
            # Use values that give stable filtering similar to well-tuned Pixy2 performance
            noise_std_x = 1.0 / 1000.0  # 1.0mm operational noise in X (m)
            noise_std_y = 1.0 / 1000.0  # 1.0mm operational noise in Y (m)

            # Conservative operational noise for stereo triangulation
            self.R_base = np.diag([noise_std_x ** 2, noise_std_y ** 2])
        else:  # PIXY2
            # Pixy2 camera: Pixel quantization + sub-pixel noise
            from core.utils import Pixy2CameraConfig
            pixel_size = Pixy2CameraConfig.PIXEL_SIZE_MM / 1000.0  # Convert mm to meters
            subpixel_noise = Pixy2CameraConfig.SUBPIXEL_NOISE_STD_MM / 1000.0  # Convert mm to meters

            # Quantization noise variance (uniform distribution)
            quantization_var = (pixel_size ** 2) / 12
            # Sub-pixel noise variance (Gaussian)
            subpixel_var = subpixel_noise ** 2

            # Total measurement noise (same for X and Y)
            measurement_std = np.sqrt(quantization_var + subpixel_var)
            self.R_base = np.eye(2) * (measurement_std ** 2)

        # Apply scaling factors
        self._update_noise_matrices()

    def _update_noise_matrices(self):
        """Update Q and R matrices with current scaling factors."""
        self.Q = self.Q_base * self.process_noise_scale
        self.R = self.R_base * self.measurement_noise_scale

    def set_process_noise(self, scale):
        """
        Set process noise scaling factor.

        Args:
            scale: Scaling factor (0.1 to 10.0 typical range)
                  - Lower: trust model more, smoother but may lag
                  - Higher: trust measurements more, faster response but noisier
        """
        self.process_noise_scale = scale
        self._update_noise_matrices()

    def set_measurement_noise(self, scale):
        """
        Set measurement noise scaling factor.

        Args:
            scale: Scaling factor (0.1 to 10.0 typical range)
                  - Lower: trust measurements more
                  - Higher: smooth measurements more aggressively
        """
        self.measurement_noise_scale = scale
        self._update_noise_matrices()

    def set_dt(self, dt):
        """Update time step and rebuild system matrices."""
        self.dt = dt
        self._build_system_matrices()

    def predict(self, control_input):
        """
        Prediction step: propagate state using dynamics.

        Args:
            control_input: [rx, ry] platform tilt angles in degrees

        Returns:
            Predicted state [x, y, vx, vy] in meters and m/s
        """
        u = np.array(control_input)

        # State prediction: x = A*x + B*u
        self.x = self.A @ self.x + self.B @ u

        # Apply velocity-dependent damping (rolling friction)
        if self.include_damping:
            vx, vy = self.x[2], self.x[3]
            vel_mag = np.sqrt(vx**2 + vy**2) + DIVISION_BY_ZERO_EPSILON

            # Rolling friction: a_friction = -μ * g * v_hat
            damping_accel_x = -self.mu_roll * self.g * (vx / vel_mag)
            damping_accel_y = -self.mu_roll * self.g * (vy / vel_mag)

            # Apply damping to velocity: dv = a_damping * dt
            self.x[2] += damping_accel_x * self.dt
            self.x[3] += damping_accel_y * self.dt

        # Covariance prediction: P = A*P*A' + Q
        self.P = self.A @ self.P @ self.A.T + self.Q

        self.prediction_count += 1

        return self.x.copy()

    def update(self, measurement: Tuple[float, float], current_time: Optional[float] = None) -> np.ndarray:
        """
        Update step: correct prediction with measurement.

        Args:
            measurement: [x, y] measured position in METERS (required)
            current_time: Current timestamp in seconds (optional, for statistics)

        Returns:
            Updated state [x, y, vx, vy] in meters and m/s
        """
        z = np.array(measurement)

        # Innovation: y = z - H*x
        y = z - self.H @ self.x

        # Innovation covariance: S = H*P*H' + R
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain: K = P*H' * inv(S) - using solve for numerical stability
        K = np.linalg.solve(S, self.H @ self.P).T

        # State update: x = x + K*y
        self.x = self.x + K @ y

        # Covariance update: P = (I - K*H) * P
        I_KH = np.eye(4) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T  # Joseph form for numerical stability

        self.update_count += 1
        if current_time is not None:
            self.last_measurement_time = current_time

        return self.x.copy()

    def get_state(self) -> Tuple[Tuple[float, float], Tuple[float, float], np.ndarray]:
        """
        Get current filter state.

        Returns:
            position: (x, y) in mm
            velocity: (vx, vy) in mm/s
            state_vector: [x, y, vx, vy] in meters and m/s
        """
        x_mm = self.x[0] * 1000.0
        y_mm = self.x[1] * 1000.0
        vx_mm = self.x[2] * 1000.0
        vy_mm = self.x[3] * 1000.0

        return (x_mm, y_mm), (vx_mm, vy_mm), self.x.copy()

    def get_position_mm(self):
        """Get filtered position in mm."""
        return self.x[0] * 1000.0, self.x[1] * 1000.0

    def get_velocity_mm_s(self):
        """Get estimated velocity in mm/s."""
        return self.x[2] * 1000.0, self.x[3] * 1000.0

    def get_covariance(self):
        """Get state covariance matrix (for diagnostics)."""
        return self.P.copy()

    def get_position_uncertainty(self):
        """
        Get position uncertainty (standard deviation) in mm.

        Returns:
            (std_x, std_y) in mm
        """
        std_x = np.sqrt(self.P[0, 0]) * 1000.0
        std_y = np.sqrt(self.P[1, 1]) * 1000.0
        return std_x, std_y

    def get_velocity_uncertainty(self):
        """
        Get velocity uncertainty (standard deviation) in mm/s.

        Returns:
            (std_vx, std_vy) in mm/s
        """
        std_vx = np.sqrt(self.P[2, 2]) * 1000.0
        std_vy = np.sqrt(self.P[3, 3]) * 1000.0
        return std_vx, std_vy

    def reset(self, initial_position: Optional[Tuple[float, float]] = None) -> None:
        """
        Reset filter state.

        Args:
            initial_position: Optional (x, y) in mm to initialize position
        """
        if initial_position is not None:
            x_m = initial_position[0] / 1000.0
            y_m = initial_position[1] / 1000.0
            self.x = np.array([x_m, y_m, 0.0, 0.0])
        else:
            self.x = np.zeros(4)

        self.P = np.eye(4) * 0.01
        self.prediction_count = 0
        self.update_count = 0
        self.last_measurement_time = -1.0

    def get_statistics(self):
        """Get filter statistics for monitoring."""
        return {
            'predictions': self.prediction_count,
            'updates': self.update_count,
            'update_ratio': self.update_count / max(1, self.prediction_count),
            'last_measurement_time': self.last_measurement_time,
            'process_noise_scale': self.process_noise_scale,
            'measurement_noise_scale': self.measurement_noise_scale
        }


# ============================================================================
# IMU ORIENTATION ESTIMATION
# ============================================================================

# Measured gravity vector (m/s²) - from stationary IMU data
GRAVITY_VECTOR = np.array([-0.2725, -0.1496, -9.8283])
GRAVITY_MAGNITUDE = np.linalg.norm(GRAVITY_VECTOR)  # 9.8332 m/s²


def apply_imu_transforms(raw_data, axis_flip, rotation_matrix, scale):
    """Apply axis flip, rotation, and scaling to raw IMU data.

    Args:
        raw_data: Raw sensor values [LSB] (can be single sample or array)
        axis_flip: Axis orientation multipliers [±1, ±1, ±1]
        rotation_matrix: 3x3 rotation matrix or None
        scale: Scaling factor to convert to physical units

    Returns:
        Transformed data in physical units
    """
    # Handle both single samples and arrays
    is_single = (raw_data.ndim == 1)
    data = raw_data.reshape(1, -1) if is_single else raw_data

    # Apply scaling
    scaled = data * scale

    # Apply axis flip
    scaled = scaled * axis_flip

    # Apply rotation if provided
    if rotation_matrix is not None:
        scaled = scaled @ rotation_matrix.T

    return scaled[0] if is_single else scaled


class OrientationKalmanFilter:
    """Extended Kalman Filter for full 3D orientation (roll, pitch, yaw) from IMU.

    State vector: [roll, pitch, yaw, gyro_bias_x, gyro_bias_y, gyro_bias_z]

    Features:
        - Full 6-DOF orientation estimation with yaw tracking
        - Gyroscope bias estimation (3-axis)
        - Magnetometer-based yaw measurement with tilt compensation
        - Accelerometer-based roll/pitch measurement
        - Automatic gravity vector zeroing at initialization
        - Supports axis transformations and gyro scale calibration
    """

    def __init__(self, accel_noise=1.0, gyro_noise=1.0, mag_noise=0.1, process_noise_angle=0.0, process_noise_bias=0.0,
                 accel_axis_flip=None, gyro_axis_flip=None, accel_rotation=None, gyro_rotation=None,
                 initial_bias_x=0.0, initial_bias_y=0.0, initial_bias_z=0.0, gyro_scale_multiplier=1.0,
                 enable_yaw_tracking=True):
        # IMU scaling
        self.accel_scale = 0.001 * 9.81  # LSM303: 1mg/LSB -> m/s²
        self.gyro_scale_multiplier = gyro_scale_multiplier  # Store for later updates
        self.gyro_scale = 0.00875 * np.pi / 180 * gyro_scale_multiplier  # L3GD20: 8.75 mdps/LSB -> rad/s (with calibration multiplier)

        # Magnetometer parameters
        self.enable_yaw_tracking = enable_yaw_tracking  # Enable full 6-DOF with yaw tracking
        self.use_magnetometer = enable_yaw_tracking  # Auto-enable magnetometer if yaw tracking enabled
        self.mag_offset = np.array([0.0, 0.0, 0.0])  # Hard-iron calibration offset
        self.mag_noise = mag_noise  # Magnetometer measurement noise
        self.mag_inclination = np.radians(IMUKalmanConfig.DEFAULT_MAG_INCLINATION_DEG)  # Magnetic inclination for Trondheim
        self.mag_update_count = 0  # Statistics
        self.yaw_offset = 0.0  # Yaw reference offset (radians)

        # Axis transformations
        self.accel_axis_flip = accel_axis_flip if accel_axis_flip is not None else np.array([1, 1, 1])
        self.gyro_axis_flip = gyro_axis_flip if gyro_axis_flip is not None else np.array([1, 1, 1])
        self.accel_rotation = accel_rotation
        self.gyro_rotation = gyro_rotation

        # Transform initial bias from raw sensor frame to transformed frame
        # 1. Scale by gyro_scale_multiplier (since bias was measured with wrong scale)
        # 2. Apply axis flips and rotation
        bias_vec = np.array([initial_bias_x * gyro_scale_multiplier,
                            initial_bias_y * gyro_scale_multiplier,
                            initial_bias_z * gyro_scale_multiplier])

        # Apply axis flip
        bias_vec = bias_vec * self.gyro_axis_flip

        # Apply rotation if provided
        if self.gyro_rotation is not None:
            bias_vec = bias_vec @ self.gyro_rotation.T

        # State: [roll, pitch, yaw, gyro_bias_x, gyro_bias_y, gyro_bias_z]
        if self.enable_yaw_tracking:
            self.state = np.array([0.0, 0.0, 0.0, bias_vec[0], bias_vec[1], bias_vec[2]])
            self.P = np.eye(6) * 0.1
        else:
            # Backward compatibility: 4D state without yaw
            self.state = np.array([0.0, 0.0, bias_vec[0], bias_vec[1]])
            self.P = np.eye(4) * 0.1

        # Store noise parameters for process noise calculation
        self.accel_noise = accel_noise
        self.gyro_noise = gyro_noise
        self.process_noise_angle = process_noise_angle
        self.process_noise_bias = process_noise_bias

        # Process noise covariance (will be computed dynamically in predict())
        if self.enable_yaw_tracking:
            self.Q = np.diag([
                process_noise_angle,
                process_noise_angle,
                process_noise_angle,  # yaw
                process_noise_bias,
                process_noise_bias,
                process_noise_bias    # z-axis bias
            ])
        else:
            self.Q = np.diag([
                process_noise_angle,
                process_noise_angle,
                process_noise_bias,
                process_noise_bias
            ])

        # Measurement noise covariance (accelerometer)
        self.R_accel = np.diag([
            accel_noise ** 2,
            accel_noise ** 2,
            accel_noise ** 2
        ])

        # Measurement noise covariance (magnetometer yaw)
        self.R_mag = mag_noise ** 2

        self.initialized = False
        self.initial_accel = None

    def initialize(self, accel_raw, mag_raw=None, calibrated_gravity=None):
        """Initialize filter state from first accelerometer reading (raw LSB).

        Args:
            accel_raw: Initial acceleration measurement [LSB]
            mag_raw: Optional magnetometer reading [LSB] for yaw initialization
            calibrated_gravity: Optional pre-calibrated gravity vector [m/s²] to use as zero reference
        """
        if not self.initialized:
            # Apply transformations and convert to m/s²
            accel = apply_imu_transforms(accel_raw, self.accel_axis_flip, self.accel_rotation, self.accel_scale)
            ax, ay, az = accel

            # Validate gravity magnitude
            gravity_mag = np.linalg.norm(accel)
            expected_gravity = GRAVITY_MAGNITUDE
            gravity_error = abs(gravity_mag - expected_gravity)
            if gravity_error > 1.0:  # More than 1 m/s² error
                warnings.warn(
                    f"Measured gravity magnitude {gravity_mag:.2f} m/s² differs from expected "
                    f"{expected_gravity:.2f} m/s² by {gravity_error:.2f} m/s². "
                    f"Check IMU calibration or mounting orientation.",
                    UserWarning
                )

            roll0 = np.arctan2(ay, az)
            pitch0 = np.arctan2(-ax, np.sqrt(ay ** 2 + az ** 2))

            self.state[0] = roll0
            self.state[1] = pitch0

            # Initialize yaw from magnetometer if enabled
            if self.enable_yaw_tracking and mag_raw is not None:
                mag = apply_imu_transforms(mag_raw, self.accel_axis_flip, self.accel_rotation, scale=1.0)
                mag = mag - self.mag_offset
                mag_norm = mag / (np.linalg.norm(mag) + DIVISION_BY_ZERO_EPSILON)

                # Tilt-compensated heading
                mx, my, mz = mag_norm
                n_y = my * np.cos(roll0) + mz * np.sin(roll0)
                n_x = (mx * np.cos(pitch0) + my * np.sin(pitch0) * np.sin(roll0) -
                       mz * np.sin(pitch0) * np.cos(roll0))
                yaw0 = np.arctan2(n_y, n_x)
                self.state[2] = yaw0

            # Use calibrated gravity if provided, otherwise use current reading
            if calibrated_gravity is not None:
                self.initial_accel = calibrated_gravity.copy()
            else:
                self.initial_accel = accel.copy()

            self.initialized = True

    def predict(self, gyro_raw, dt):
        """Prediction step using gyroscope measurements (raw LSB).

        Args:
            gyro_raw: Angular velocity measurement [LSB]
            dt: Time step [s]
        """
        # Validate dt
        if dt <= 0:
            return  # Skip prediction for invalid timestep

        # Apply transformations and convert to rad/s
        gyro = apply_imu_transforms(gyro_raw, self.gyro_axis_flip, self.gyro_rotation, self.gyro_scale)

        if self.enable_yaw_tracking:
            # Full 6-DOF prediction with yaw
            gx, gy, gz = gyro[0], gyro[1], gyro[2]

            # Bias-corrected angular velocity (body rates)
            p = gx - self.state[3]  # roll rate
            q = gy - self.state[4]  # pitch rate
            r = gz - self.state[5]  # yaw rate

            # Current orientation
            phi = self.state[0]
            theta = self.state[1]

            # Kinematic transformation matrix M(φ,θ): converts body rates to Euler rates
            # [φ̇]   [1  sin(φ)tan(θ)   cos(φ)tan(θ)  ] [p]
            # [θ̇] = [0      cos(φ)         -sin(φ)    ] [q]
            # [ψ̇]   [0  sin(φ)/cos(θ)   cos(φ)/cos(θ)] [r]
            cos_phi = np.cos(phi)
            sin_phi = np.sin(phi)
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            tan_theta = np.tan(theta)

            # Avoid singularity at θ = ±90°
            if abs(cos_theta) < GIMBAL_LOCK_THRESHOLD:
                cos_theta = np.sign(cos_theta) * GIMBAL_LOCK_THRESHOLD

            M = np.array([
                [1, sin_phi * tan_theta, cos_phi * tan_theta],
                [0, cos_phi, -sin_phi],
                [0, sin_phi / cos_theta, cos_phi / cos_theta]
            ])

            # Euler angle rates
            euler_rates = M @ np.array([p, q, r])

            # State propagation
            self.state[0] += euler_rates[0] * dt  # roll
            self.state[1] += euler_rates[1] * dt  # pitch
            self.state[2] += euler_rates[2] * dt  # yaw
            # Bias states remain constant (random walk model)

            # Wrap yaw to [-π, π]
            self.state[2] = np.arctan2(np.sin(self.state[2]), np.cos(self.state[2]))

            # Jacobian of state transition F = ∂f/∂x
            # State: [φ, θ, ψ, bx, by, bz]
            F = np.eye(6)
            F[0, 3] = -dt  # ∂φ/∂bx = -M[0,0]*dt = -dt
            F[0, 4] = -sin_phi * tan_theta * dt  # ∂φ/∂by = -M[0,1]*dt
            F[0, 5] = -cos_phi * tan_theta * dt  # ∂φ/∂bz = -M[0,2]*dt
            F[1, 3] = 0    # ∂θ/∂bx = 0
            F[1, 4] = -cos_phi * dt  # ∂θ/∂by = -M[1,1]*dt
            F[1, 5] = sin_phi * dt   # ∂θ/∂bz = -M[1,2]*dt
            F[2, 3] = 0    # ∂ψ/∂bx = 0
            F[2, 4] = -sin_phi / cos_theta * dt  # ∂ψ/∂by = -M[2,1]*dt
            F[2, 5] = -cos_phi / cos_theta * dt  # ∂ψ/∂bz = -M[2,2]*dt

            # Process noise
            Q = np.diag([
                self.process_noise_angle + (self.gyro_noise * dt) ** 2,
                self.process_noise_angle + (self.gyro_noise * dt) ** 2,
                self.process_noise_angle + (self.gyro_noise * dt) ** 2,
                self.process_noise_bias,
                self.process_noise_bias,
                self.process_noise_bias
            ])
        else:
            # Backward compatibility: 4D prediction (roll/pitch only)
            gx, gy = gyro[0], gyro[1]
            self.last_gyro_magnitude = np.sqrt(gx**2 + gy**2)

            gx_corrected = gx - self.state[2]
            gy_corrected = gy - self.state[3]

            self.state[0] += gx_corrected * dt
            self.state[1] += gy_corrected * dt

            F = np.array([
                [1, 0, -dt, 0],
                [0, 1, 0, -dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])

            Q = np.diag([
                self.process_noise_angle + (self.gyro_noise * dt) ** 2,
                self.process_noise_angle + (self.gyro_noise * dt) ** 2,
                self.process_noise_bias,
                self.process_noise_bias
            ])

        # Covariance propagation
        self.P = F @ self.P @ F.T + Q

    def update(self, accel_raw, mag_raw=None):
        """Update step using accelerometer measurements (raw LSB).

        Args:
            accel_raw: Acceleration measurement [LSB]
            mag_raw: Optional magnetometer measurement [LSB]
        """
        # Apply transformations and convert to m/s²
        accel = apply_imu_transforms(accel_raw, self.accel_axis_flip, self.accel_rotation, self.accel_scale)
        ax, ay, az = accel

        if self.enable_yaw_tracking:
            # Simplified accelerometer update for 6-DOF mode
            # Uses direct angle measurement (same as 4-DOF) for stability
            roll_meas = np.arctan2(ay, az)
            pitch_meas = np.arctan2(-ax, np.sqrt(ay ** 2 + az ** 2))

            # Remove initial gravity offset
            if self.initial_accel is not None:
                roll_init = np.arctan2(self.initial_accel[1], self.initial_accel[2])
                pitch_init = np.arctan2(-self.initial_accel[0],
                                        np.sqrt(self.initial_accel[1] ** 2 + self.initial_accel[2] ** 2))
                roll_meas -= roll_init
                pitch_meas -= pitch_init

            # Measurement vector (roll, pitch only - yaw from magnetometer)
            z = np.array([roll_meas, pitch_meas])

            # Measurement Jacobian (2x6): only observe roll and pitch
            H = np.zeros((2, 6))
            H[0, 0] = 1  # roll measurement
            H[1, 1] = 1  # pitch measurement
            # Yaw, biases not observable from accelerometer

            # Measurement noise (2x2)
            R = np.diag([self.R_accel[0,0], self.R_accel[1,1]])

            # Kalman update
            y = z - H @ self.state
            S = H @ self.P @ H.T + R
            K = np.linalg.solve(S, H @ self.P).T
            self.state = self.state + K @ y
            # Covariance update using Joseph form for numerical stability
            I_KH = np.eye(6) - K @ H
            self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

            # Update magnetometer if available (for yaw)
            if mag_raw is not None:
                self.update_magnetometer_yaw(mag_raw)

        else:
            # Backward compatibility: direct roll/pitch measurement
            roll_meas = np.arctan2(ay, az)
            pitch_meas = np.arctan2(-ax, np.sqrt(ay ** 2 + az ** 2))

            # Remove initial gravity offset
            if self.initial_accel is not None:
                roll_init = np.arctan2(self.initial_accel[1], self.initial_accel[2])
                pitch_init = np.arctan2(-self.initial_accel[0],
                                        np.sqrt(self.initial_accel[1] ** 2 + self.initial_accel[2] ** 2))
                roll_meas -= roll_init
                pitch_meas -= pitch_init

            z = np.array([roll_meas, pitch_meas])

            H = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0]
            ])

            # Use simple R for 4D mode
            R = np.diag([self.R_accel[0,0], self.R_accel[1,1]])

            y = z - H @ self.state
            S = H @ self.P @ H.T + R
            K = np.linalg.solve(S, H @ self.P).T
            self.state = self.state + K @ y
            # Covariance update using Joseph form for numerical stability
            I_KH = np.eye(4) - K @ H
            self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

    def update_magnetometer_yaw(self, mag_raw):
        """Update yaw using magnetometer with tilt compensation.

        Args:
            mag_raw: Magnetometer measurement [LSB]

        Note:
            Implements tilt-compensated heading calculation.
            Uses measurement Jacobian for proper EKF update.
        """
        # Apply transformations (no scaling needed - direction only)
        mag = apply_imu_transforms(mag_raw, self.accel_axis_flip, self.accel_rotation, scale=1.0)

        # Remove hard-iron offset
        mag = mag - self.mag_offset

        # Check magnitude
        mag_magnitude = np.linalg.norm(mag)
        if mag_magnitude < 1e-6:
            return

        # Get current orientation estimate
        phi = self.state[0]
        theta = self.state[1]
        psi_pred = self.state[2]

        mx, my, mz = mag

        # Tilt-compensated heading calculation
        # n_y = my·cos(φ) + mz·sin(φ)
        # n_x = mx·cos(θ) + my·sin(θ)·sin(φ) - mz·sin(θ)·cos(φ)
        # ψ_mag = atan2(n_y, n_x)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        n_y = my * cos_phi + mz * sin_phi
        n_x = mx * cos_theta + my * sin_theta * sin_phi - mz * sin_theta * cos_phi

        # Measured yaw from magnetometer
        psi_mag = np.arctan2(n_y, n_x)

        # Innovation (handle angle wrapping)
        innov = psi_mag - psi_pred
        # Wrap to [-π, π]
        innov = np.arctan2(np.sin(innov), np.cos(innov))

        # Measurement Jacobian H = ∂ψ_mag/∂x
        # State: [φ, θ, ψ, bx, by, bz]
        # ψ_mag depends on φ and θ (not ψ or biases)
        d = np.sqrt(n_y**2 + n_x**2) + DIVISION_BY_ZERO_EPSILON  # Avoid division by zero

        # ∂ψ_mag/∂φ
        dn_y_dphi = -my * sin_phi + mz * cos_phi
        dn_x_dphi = my * sin_theta * cos_phi + mz * sin_theta * sin_phi
        dpsi_dphi = (n_x * dn_y_dphi - n_y * dn_x_dphi) / (d**2)

        # ∂ψ_mag/∂θ
        dn_y_dtheta = 0
        dn_x_dtheta = -mx * sin_theta + my * cos_theta * sin_phi - mz * cos_theta * cos_phi
        dpsi_dtheta = (n_x * dn_y_dtheta - n_y * dn_x_dtheta) / (d**2)

        # Fully decoupled yaw-only update
        # Update only yaw state and yaw variance, no coupling to roll/pitch
        # This prevents magnetometer errors from affecting roll/pitch through covariance

        # Simple scalar Kalman update for yaw only
        yaw_var = self.P[2, 2]  # Current yaw variance
        S = yaw_var + self.R_mag  # Innovation covariance (R_mag is scalar)
        K_yaw = yaw_var / S  # Kalman gain for yaw

        # Update yaw state only
        self.state[2] = self.state[2] + K_yaw * innov

        # Update yaw variance only (no cross-covariances)
        self.P[2, 2] = (1 - K_yaw) * yaw_var

        # Wrap yaw to [-π, π]
        self.state[2] = np.arctan2(np.sin(self.state[2]), np.cos(self.state[2]))

        self.mag_update_count += 1

    def update_with_magnetometer(self, mag_raw):
        """Backward compatibility: Update step using magnetometer when accelerometer is corrupted.

        For 4D mode (without yaw tracking), uses magnetometer to estimate tilt.
        For 6D mode, this is replaced by update_magnetometer_yaw().
        """
        if self.enable_yaw_tracking:
            # In 6D mode, magnetometer updates yaw only
            self.update_magnetometer_yaw(mag_raw)
            return

        # 4D mode: Use magnetometer for tilt estimation (old behavior)
        mag = apply_imu_transforms(mag_raw, self.accel_axis_flip, self.accel_rotation, scale=1.0)
        mag = mag - self.mag_offset

        mag_magnitude = np.linalg.norm(mag)
        if mag_magnitude < 1.0:
            return

        mag_norm = mag / mag_magnitude
        mx, my, mz = mag_norm

        # Simple tilt estimation from mag direction
        roll_meas = np.arctan2(-my, mz) * np.cos(self.mag_inclination)
        pitch_meas = np.arctan2(mx, np.sqrt(my ** 2 + mz ** 2)) * np.cos(self.mag_inclination)

        if self.initial_accel is not None:
            roll_init = np.arctan2(self.initial_accel[1], self.initial_accel[2])
            pitch_init = np.arctan2(-self.initial_accel[0],
                                    np.sqrt(self.initial_accel[1] ** 2 + self.initial_accel[2] ** 2))
            roll_meas -= roll_init
            pitch_meas -= pitch_init

        z = np.array([roll_meas, pitch_meas])
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        R = np.diag([self.R_accel[0,0], self.R_accel[1,1]]) * 4.0

        y = z - H @ self.state
        S = H @ self.P @ H.T + R
        K = np.linalg.solve(S, H @ self.P).T
        self.state = self.state + K @ y
        # Covariance update using Joseph form for numerical stability
        I_KH = np.eye(4) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

        self.mag_update_count += 1

    def get_orientation(self):
        """Return current orientation estimate.

        Returns:
            - 6D mode: Tuple of (roll, pitch, yaw) in radians
            - 4D mode: Tuple of (roll, pitch) in radians
        """
        if self.enable_yaw_tracking:
            return self.state[0], self.state[1], self.state[2]
        else:
            return self.state[0], self.state[1]

    def get_linear_acceleration(self, accel_raw):
        """Extract linear acceleration by removing gravity component.

        Args:
            accel_raw: Raw accelerometer reading [LSB]

        Returns:
            Linear acceleration in world frame [ax, ay, az] in m/s²

        Note:
            Since Kalman filter tracks RELATIVE orientation (zeroed at calibration),
            we must subtract the calibrated gravity in SENSOR frame before rotation.
        """
        # Convert to m/s² and apply transformations
        accel = apply_imu_transforms(accel_raw, self.accel_axis_flip, self.accel_rotation, self.accel_scale)

        # Remove calibrated gravity in sensor frame (this gives acceleration relative to calibration position)
        if self.initial_accel is not None:
            accel_no_gravity = accel - self.initial_accel
        else:
            # Fallback: assume gravity is [0, 0, -9.83] in sensor frame
            accel_no_gravity = accel - np.array([0, 0, -GRAVITY_MAGNITUDE])

        # Get current RELATIVE orientation estimate
        roll, pitch = self.state[0], self.state[1]

        # Rotation matrix from sensor frame to world frame (Z-X-Y Euler)
        # Uses relative angles, so world frame is relative to calibration position
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)

        # Rotation matrix for roll and pitch only (no yaw)
        R = np.array([
            [cp, sr*sp, cr*sp],
            [0, cr, -sr],
            [-sp, sr*cp, cr*cp]
        ])

        # Rotate gravity-compensated acceleration to relative world frame
        linear_accel = R @ accel_no_gravity

        return linear_accel

    def set_gyro_scale_multiplier(self, multiplier):
        """Update gyro scale multiplier and recompute gyro_scale.

        This recalibrates the gyroscope sensitivity in real-time.
        Gyro bias estimates are rescaled to maintain consistency.

        Args:
            multiplier: New gyro scale multiplier
        """
        old_multiplier = self.gyro_scale_multiplier
        self.gyro_scale_multiplier = multiplier
        self.gyro_scale = 0.00875 * np.pi / 180 * multiplier

        # Rescale bias estimates (they were calibrated with old scale)
        if old_multiplier > 0:
            scale_ratio = multiplier / old_multiplier
            self.state[2] *= scale_ratio  # bias_x
            self.state[3] *= scale_ratio  # bias_y


class IMUControllerMixin:
    """Mixin for IMU orientation tracking and tilt correction.

    Provides IMU integration for hardware controllers. This mixin adds:
    - IMU orientation Kalman filtering
    - Calibration sequences (initialization + calibration)
    - Tilt correction/compensation
    - Real-time IMU data processing

    Requirements (must be provided by parent class):
    - self.serial_controller: SerialController instance with IMU data queues
    - self.control_interval: float, control loop timestep (seconds)
    - self.log(message): logging method

    Usage:
        class MyController(IMUControllerMixin, HardwareControllerBase):
            def __init__(self, ...):
                self._init_imu_system()
                super().__init__(...)
    """

    def _init_imu_system(self):
        """Initialize IMU state variables and Kalman filter.

        Must be called BEFORE parent __init__ to set up IMU attributes.
        """
        # Initialize orientation Kalman filter with default configuration
        self.orientation_kalman = OrientationKalmanFilter(
            accel_noise=IMUKalmanConfig.DEFAULT_ACCEL_NOISE,
            gyro_noise=IMUKalmanConfig.DEFAULT_GYRO_NOISE,
            mag_noise=IMUKalmanConfig.DEFAULT_MAG_NOISE,
            process_noise_angle=IMUKalmanConfig.DEFAULT_PROCESS_NOISE_ANGLE,
            process_noise_bias=IMUKalmanConfig.DEFAULT_PROCESS_NOISE_BIAS,
            accel_axis_flip=IMUKalmanConfig.DEFAULT_ACCEL_AXIS_FLIP,
            gyro_axis_flip=IMUKalmanConfig.DEFAULT_GYRO_AXIS_FLIP,
            accel_rotation=IMUKalmanConfig.DEFAULT_ACCEL_ROTATION,
            gyro_rotation=IMUKalmanConfig.DEFAULT_GYRO_ROTATION,
            initial_bias_x=IMUKalmanConfig.CALIBRATED_GYRO_BIAS_X,
            initial_bias_y=IMUKalmanConfig.CALIBRATED_GYRO_BIAS_Y,
            initial_bias_z=IMUKalmanConfig.CALIBRATED_GYRO_BIAS_Z,
            gyro_scale_multiplier=IMUKalmanConfig.DEFAULT_GYRO_SCALE_MULTIPLIER,
            enable_yaw_tracking=IMUKalmanConfig.ENABLE_YAW_TRACKING
        )

        # Set magnetometer calibration
        self.orientation_kalman.mag_offset = np.array([
            IMUKalmanConfig.MAG_OFFSET_X,
            IMUKalmanConfig.MAG_OFFSET_Y,
            IMUKalmanConfig.MAG_OFFSET_Z
        ])
        self.orientation_kalman.mag_inclination = np.radians(IMUKalmanConfig.MAG_INCLINATION_DEG)

        # Current IMU orientation state
        self.current_rx_imu = 0.0
        self.current_ry_imu = 0.0
        self.current_yaw_imu = 0.0
        self.imu_tilt_correction_enabled = False
        self.imu_compensation_gain = 1.0

        # Initialization phase (3 seconds - stabilization)
        self.imu_initializing = False
        self.initialization_duration = IMUCalibrationConfig.INITIALIZATION_DURATION_S
        self.initialization_start_time = None
        self.initialization_time_remaining = 0.0

        # Calibration phase (10 seconds - data collection)
        self.imu_calibrating = False
        self.calibration_duration = IMUCalibrationConfig.CALIBRATION_DURATION_S
        self.calibration_start_time = None
        self.calibration_time_remaining = 0.0
        self.calibration_raw_data = {'gyro': [], 'accel': [], 'mag': []}
        self.calibrated_gravity = None

        # Debug tracking
        self._imu_data_debug_count = 0
        self._imu_debug_count = 0

        # Yaw auto-calibration flag
        self._auto_calibrate_yaw_on_enable = False

    def start_imu_initialization(self):
        """Begin IMU initialization sequence (3 seconds).

        During initialization:
        - IMU queues are drained (sensors stabilize)
        - Platform should remain stationary
        - Automatically transitions to calibration
        """
        self.imu_initializing = True
        self.initialization_start_time = time.time()
        self.initialization_time_remaining = self.initialization_duration
        self.log("IMU initialization started: stabilizing for 3 seconds")

    def start_imu_calibration(self):
        """Begin IMU calibration sequence (10 seconds).

        During calibration:
        - Raw IMU data is collected
        - Gyro bias is measured
        - Gravity vector is established
        - Platform MUST remain stationary and level
        """
        self.imu_calibrating = True
        self.calibration_start_time = time.time()
        self.calibration_time_remaining = self.calibration_duration
        self.calibration_raw_data = {'gyro': [], 'accel': [], 'mag': []}
        self.log("IMU calibration started: maintain platform stationary for 10 seconds")

    def finish_imu_calibration(self):
        """Process calibration data and initialize Kalman filter.

        Calibration processing:
        1. Calculates mean gyro values (bias estimation)
        2. Calculates mean accel values (gravity reference)
        3. Applies sensor transformations
        4. Initializes orientation Kalman filter
        5. Validates platform levelness (warns if >5° tilt)
        """
        gyro_data = self.calibration_raw_data['gyro']
        accel_data = self.calibration_raw_data['accel']

        if not gyro_data or not accel_data:
            self.log("Warning: Insufficient calibration data collected")
            self.imu_calibrating = False
            return

        # Convert to arrays
        accel_samples = np.array([sample[1] for sample in accel_data])
        gyro_samples = np.array([sample[1] for sample in gyro_data])

        # Calculate mean raw values
        accel_mean_raw = np.mean(accel_samples, axis=0)
        gyro_mean_raw = np.mean(gyro_samples, axis=0)

        # Apply sensor-specific transformations
        # LSM303 accelerometer: 1mg/LSB → m/s²
        accel_scale = 0.001 * 9.81
        # L3GD20 gyroscope: 8.75 mdps/LSB → rad/s (with calibration multiplier)
        gyro_scale = 0.00875 * np.pi / 180 * self.orientation_kalman.gyro_scale_multiplier

        accel_mean = apply_imu_transforms(accel_mean_raw,
                                         self.orientation_kalman.accel_axis_flip,
                                         self.orientation_kalman.accel_rotation,
                                         accel_scale)
        gyro_mean = apply_imu_transforms(gyro_mean_raw,
                                        self.orientation_kalman.gyro_axis_flip,
                                        self.orientation_kalman.gyro_rotation,
                                        gyro_scale)

        # Store calibrated gravity reference
        self.calibrated_gravity = accel_mean.copy()

        # Initialize orientation Kalman filter with calibration data
        self.orientation_kalman.initialize(accel_mean_raw, calibrated_gravity=self.calibrated_gravity)

        # Validate calibration quality (check platform levelness)
        ax, ay, az = accel_mean
        tilt_x = np.arctan2(ay, az)
        tilt_y = np.arctan2(-ax, np.sqrt(ay**2 + az**2))
        tilt_x_deg = np.degrees(tilt_x)
        tilt_y_deg = np.degrees(tilt_y)

        if abs(tilt_x_deg) > MAX_CALIBRATION_TILT_DEG or abs(tilt_y_deg) > MAX_CALIBRATION_TILT_DEG:
            self.log(f"Warning: Platform tilted during calibration - RX={tilt_x_deg:.1f}°, RY={tilt_y_deg:.1f}°")
        else:
            self.log(f"Platform level verified: RX={tilt_x_deg:.1f}°, RY={tilt_y_deg:.1f}°")

        self.log(f"Gyroscope bias (deg/s): X={np.degrees(gyro_mean[0]):.4f}, Y={np.degrees(gyro_mean[1]):.4f}")
        self.log("IMU calibration completed successfully")
        self.imu_calibrating = False

    def _process_imu_calibration_phase(self):
        """Process IMU during initialization/calibration phases.

        Call this from control thread BEFORE normal operation.

        Returns:
            bool: True if still calibrating (skip rest of control loop), False if ready
        """
        # Handle initialization phase (3 seconds)
        if self.imu_initializing:
            elapsed = time.time() - self.initialization_start_time
            self.initialization_time_remaining = max(0.0, self.initialization_duration - elapsed)

            if elapsed >= self.initialization_duration:
                self.imu_initializing = False
                self.initialization_time_remaining = 0.0
                self.start_imu_calibration()
            else:
                # Drain queues during initialization
                self.serial_controller.get_imu_data_batch()

            return True  # Skip rest of control loop

        # Handle calibration phase (10 seconds)
        if self.imu_calibrating:
            elapsed = time.time() - self.calibration_start_time
            self.calibration_time_remaining = max(0.0, self.calibration_duration - elapsed)

            # Collect calibration data
            gyro_batch, accel_batch, mag_batch = self.serial_controller.get_imu_data_batch()
            self.calibration_raw_data['gyro'].extend(gyro_batch)
            self.calibration_raw_data['accel'].extend(accel_batch)
            self.calibration_raw_data['mag'].extend(mag_batch)

            if elapsed >= self.calibration_duration:
                self.finish_imu_calibration()
                self.calibration_time_remaining = 0.0

            return True  # Skip rest of control loop

        return False  # Calibration complete, proceed with normal operation

    def _update_imu_orientation(self):
        """Update IMU orientation estimate from sensor data.

        Call this from control thread during normal operation.

        Processing:
        1. Fetches latest IMU sample from serial queues
        2. Predict step (gyro data → integrate orientation)
        3. Update step (accel + optional mag → correct drift)
        4. Updates self.current_rx_imu and self.current_ry_imu
        """
        # Get latest IMU sample (non-blocking)
        gyro_data, accel_data, mag_data = self.serial_controller.get_single_imu_sample()

        # Debug logging (first 5 samples only)
        if self._imu_data_debug_count < 5:
            if gyro_data or accel_data:
                self.log(f"IMU data: Gyro={'Yes' if gyro_data else 'No'} "
                        f"Accel={'Yes' if accel_data else 'No'} "
                        f"Mag={'Yes' if mag_data else 'No'}")
                self._imu_data_debug_count += 1

        # Kalman filter predict step (gyroscope integration)
        if gyro_data is not None:
            timestamp_us, gyro_raw = gyro_data
            self.orientation_kalman.predict(gyro_raw, self.control_interval)

        # Kalman filter update step (accelerometer + magnetometer correction)
        if accel_data is not None:
            timestamp_us, accel_raw = accel_data
            mag_raw = mag_data[1] if mag_data else None
            self.orientation_kalman.update(accel_raw, mag_raw=mag_raw)

        # Extract current orientation estimate (radians → degrees)
        self.current_rx_imu = np.degrees(self.orientation_kalman.state[0])
        self.current_ry_imu = np.degrees(self.orientation_kalman.state[1])
        if self.orientation_kalman.enable_yaw_tracking:
            # Auto-calibrate yaw reference on first update after enabling 6-DOF
            if self._auto_calibrate_yaw_on_enable:
                self.orientation_kalman.yaw_offset = self.orientation_kalman.state[2]
                self.log(f"Yaw reference initialized (offset={np.degrees(self.orientation_kalman.yaw_offset):.2f}°)")
                self._auto_calibrate_yaw_on_enable = False

            # Apply yaw offset for reference frame correction
            yaw_corrected = self.orientation_kalman.state[2] - self.orientation_kalman.yaw_offset
            # Wrap to [-π, π]
            yaw_corrected = np.arctan2(np.sin(yaw_corrected), np.cos(yaw_corrected))
            self.current_yaw_imu = np.degrees(yaw_corrected)

    def _apply_imu_tilt_correction(self, rx_ctrl, ry_ctrl):
        """Apply IMU compensation to controller output.

        Args:
            rx_ctrl: Controller X tilt output (degrees)
            ry_ctrl: Controller Y tilt output (degrees)

        Returns:
            (rx_final, ry_final): Combined tilt with IMU correction (degrees)

        IMU Correction Logic:
        - Opposes platform tilt: compensation = -imu_angle * gain
        - Clipped to ±15° to prevent excessive compensation
        - Combined with controller output (no additional clipping)
        - If disabled, returns controller output unchanged
        """
        if not self.imu_tilt_correction_enabled:
            return rx_ctrl, ry_ctrl

        # Calculate IMU compensation (oppose platform tilt)
        rx_imu_comp = -self.current_rx_imu * self.imu_compensation_gain
        ry_imu_comp = -self.current_ry_imu * self.imu_compensation_gain

        # Clip compensation to safe range
        rx_imu_comp, ry_imu_comp, _ = clip_tilt_vector(rx_imu_comp, ry_imu_comp, MAX_IMU_CORRECTION_DEG)

        # Combine controller output + IMU compensation
        rx_final = rx_ctrl + rx_imu_comp
        ry_final = ry_ctrl + ry_imu_comp

        # Debug logging (first 10 samples only)
        if self._imu_debug_count < 10:
            self.log(f"IMU correction: RX_IMU={self.current_rx_imu:.2f}° RY_IMU={self.current_ry_imu:.2f}° | "
                    f"Comp=({rx_imu_comp:.2f}°, {ry_imu_comp:.2f}°) | "
                    f"Ctrl=({rx_ctrl:.2f}°, {ry_ctrl:.2f}°) → Final=({rx_final:.2f}°, {ry_final:.2f}°)")
            self._imu_debug_count += 1

        return rx_final, ry_final

    # ============================================================================
    # GUI CALLBACKS
    # ============================================================================

    def on_imu_tilt_correction_toggle(self, enabled):
        """Handle IMU tilt correction enable/disable."""
        self.imu_tilt_correction_enabled = enabled
        self.log(f"IMU tilt correction: {'ENABLED' if enabled else 'DISABLED'}")

    def on_imu_kalman_param_change(self, param_name, value):
        """Handle IMU Kalman filter parameter change.

        Args:
            param_name: Parameter to update ('accel_noise', 'gyro_noise', etc.)
            value: New parameter value (float)
        """
        if param_name == 'gyro_scale':
            # Special handling for gyro scale multiplier (uses setter method)
            self.orientation_kalman.set_gyro_scale_multiplier(value)
            self.log(f"Gyro scale multiplier: {value:.2f}")
        else:
            param_map = {
                'accel_noise': 'accel_noise',
                'gyro_noise': 'gyro_noise',
                'process_noise_angle': 'process_noise_angle',
                'process_noise_bias': 'process_noise_bias'
            }

            if param_name in param_map:
                attr_name = param_map[param_name]
                setattr(self.orientation_kalman, attr_name, value)
                self.log(f"IMU Kalman {param_name}: {value:.4f}")

    def on_imu_mag_toggle(self, enabled):
        """Handle magnetometer enable/disable."""
        self.orientation_kalman.use_magnetometer = enabled
        self.log(f"Magnetometer backup: {'ENABLED' if enabled else 'DISABLED'}")

    def on_imu_yaw_tracking_toggle(self, enabled):
        """Handle yaw tracking (6-DOF) enable/disable.

        Note: This requires re-initializing the Kalman filter as the state
        dimension changes between 4-DOF and 6-DOF modes.
        """
        old_mode = "6-DOF" if self.orientation_kalman.enable_yaw_tracking else "4-DOF"
        new_mode = "6-DOF" if enabled else "4-DOF"

        if old_mode == new_mode:
            return

        self.log(f"Switching IMU mode: {old_mode} → {new_mode}")
        self.log("Warning: Re-initializing IMU Kalman filter (state dimension changed)")

        # If enabling 6-DOF, mark for auto-calibration on first mag update
        if enabled:
            self._auto_calibrate_yaw_on_enable = True

        # Store current state and covariance
        old_state = self.orientation_kalman.state.copy()
        old_P = self.orientation_kalman.P.copy()

        # Debug: log current orientation before switch
        self.log(f"Before switch - RX: {np.degrees(old_state[0]):.2f}°, RY: {np.degrees(old_state[1]):.2f}°")

        # Store current noise parameters
        accel_noise = self.orientation_kalman.accel_noise
        gyro_noise = self.orientation_kalman.gyro_noise
        mag_noise = self.orientation_kalman.mag_noise
        process_noise_angle = self.orientation_kalman.process_noise_angle
        process_noise_bias = self.orientation_kalman.process_noise_bias

        # Re-initialize with new yaw tracking setting
        self.orientation_kalman.enable_yaw_tracking = enabled
        self.orientation_kalman.use_magnetometer = enabled  # Auto-enable mag for yaw

        # Reinitialize state vector and covariance
        if enabled:
            # Switch to 6-DOF mode - preserve current roll, pitch, and covariances
            self.orientation_kalman.state = np.array([
                old_state[0],  # preserve roll
                old_state[1],  # preserve pitch
                0.0,  # yaw (start at 0, will converge with magnetometer)
                old_state[2] if len(old_state) > 2 else 0.0,  # bx
                old_state[3] if len(old_state) > 3 else 0.0,  # by
                0.0  # bz
            ])

            # Build new covariance matrix with no cross-coupling to yaw
            self.orientation_kalman.P = np.eye(6) * 0.01
            # Copy roll, pitch covariances from old filter (2x2 block)
            self.orientation_kalman.P[0:2, 0:2] = old_P[0:2, 0:2]
            # Copy bias covariances from old filter (2x2 block)
            self.orientation_kalman.P[3:5, 3:5] = old_P[2:4, 2:4]
            # Yaw has high initial uncertainty (fully decoupled)
            self.orientation_kalman.P[2, 2] = 10.0
            # Z-bias has moderate initial uncertainty (decoupled)
            self.orientation_kalman.P[5, 5] = 0.1

            self.orientation_kalman.Q = np.diag([
                process_noise_angle, process_noise_angle, process_noise_angle,
                process_noise_bias, process_noise_bias, process_noise_bias
            ])

            # Debug: log orientation after switch
            self.log(f"After switch to 6-DOF - RX: {np.degrees(self.orientation_kalman.state[0]):.2f}°, "
                    f"RY: {np.degrees(self.orientation_kalman.state[1]):.2f}°, "
                    f"Yaw: {np.degrees(self.orientation_kalman.state[2]):.2f}°")
        else:
            # Switch to 4-DOF mode - preserve current roll, pitch, and covariances
            self.orientation_kalman.state = np.array([
                old_state[0],  # preserve roll
                old_state[1],  # preserve pitch
                old_state[3] if len(old_state) > 3 else 0.0,  # bx
                old_state[4] if len(old_state) > 4 else 0.0   # by
            ])

            # Preserve covariance for existing states
            self.orientation_kalman.P = np.eye(4) * 0.1
            # Copy roll, pitch covariances from old filter
            self.orientation_kalman.P[0:2, 0:2] = old_P[0:2, 0:2]
            # Copy bias covariances from old filter
            self.orientation_kalman.P[2:4, 2:4] = old_P[3:5, 3:5]

            self.orientation_kalman.Q = np.diag([
                process_noise_angle, process_noise_angle,
                process_noise_bias, process_noise_bias
            ])

            # Debug: log orientation after switch
            self.log(f"After switch to 4-DOF - RX: {np.degrees(self.orientation_kalman.state[0]):.2f}°, "
                    f"RY: {np.degrees(self.orientation_kalman.state[1]):.2f}°")

        # Reset yaw state when switching modes
        if not enabled:
            self.current_yaw_imu = 0.0

        self.log(f"Yaw tracking: {'ENABLED (6-DOF with magnetometer)' if enabled else 'DISABLED (4-DOF)'}")

    def on_imu_set_yaw_reference(self):
        """Set current yaw orientation as reference zero point.

        Captures current yaw estimate and applies offset to make current
        orientation the zero reference for yaw measurements.
        """
        if not self.orientation_kalman.enable_yaw_tracking:
            self.log("Warning: Yaw tracking not enabled, cannot set yaw reference")
            return

        current_yaw = self.orientation_kalman.state[2]  # Current yaw in radians
        self.orientation_kalman.yaw_offset = current_yaw
        self.log(f"Yaw reference set: current orientation is now 0° (offset={np.degrees(current_yaw):.2f}°)")

    def on_imu_yaw_offset_change(self, offset_deg):
        """Adjust yaw reference offset.

        Args:
            offset_deg: Yaw offset in degrees (float)
        """
        if not self.orientation_kalman.enable_yaw_tracking:
            return

        self.orientation_kalman.yaw_offset = np.radians(offset_deg)