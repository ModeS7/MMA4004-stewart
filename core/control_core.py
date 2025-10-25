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

    def __init__(self, kp=1.0, ki=0.0, kd=0.5,
                 output_limit=MAX_TILT_ANGLE_DEG,
                 derivative_filter_alpha=0.0):
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
    Linearized dynamics around equilibrium (ball centered, zero velocity).

    State: [pos_x, pos_y, vel_x, vel_y]
    Control: [tilt_ry, tilt_rx]
    """

    def __init__(self, Q_pos=1.0, Q_vel=1.0, R=0.01,
                 output_limit=MAX_TILT_ANGLE_DEG,
                 ball_physics_params=None):
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
            self.K = np.linalg.inv(R) @ B.T @ P

            eig_vals = np.linalg.eigvals(A - B @ self.K)
            max_real = np.max(np.real(eig_vals))

            if max_real >= 0:
                print(f"LQR eigenvalue check: max real part = {max_real:.4f}")

        except np.linalg.LinAlgError as e:
            print(f"Riccati equation solver failed: {e}")
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
                 process_noise_scale=1.0,
                 measurement_noise_scale=1.0,
                 ball_physics_params=None,
                 dt=0.01):
        """
        Args:
            process_noise_scale: Scaling factor for process noise Q (tunable)
            measurement_noise_scale: Scaling factor for measurement noise R (tunable)
            ball_physics_params: Dict with 'radius', 'mass', 'gravity', 'mass_factor'
            dt: Time step for prediction (control loop period)
        """
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
        q_pos = 0.0001  # Position process noise (m²)
        q_vel = 0.001  # Velocity process noise (m²/s²)

        self.Q_base = np.array([
            [q_pos * dt ** 3 / 3, 0, q_pos * dt ** 2 / 2, 0],
            [0, q_pos * dt ** 3 / 3, 0, q_pos * dt ** 2 / 2],
            [q_pos * dt ** 2 / 2, 0, q_vel * dt, 0],
            [0, q_pos * dt ** 2 / 2, 0, q_vel * dt]
        ])

        # Measurement noise covariance R
        # Based on camera characteristics:
        # - Pixel quantization: 1.4mm
        # - Sub-pixel noise: 0.4mm std
        # Combined uncertainty ≈ sqrt(1.4²/12 + 0.4²) ≈ 0.58mm
        pixel_size = 0.0014  # 1.4mm in meters
        subpixel_noise = 0.0004  # 0.4mm in meters

        # Quantization noise variance (uniform distribution)
        quantization_var = (pixel_size ** 2) / 12
        # Sub-pixel noise variance (Gaussian)
        subpixel_var = subpixel_noise ** 2

        # Total measurement noise
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

        # Covariance prediction: P = A*P*A' + Q
        self.P = self.A @ self.P @ self.A.T + self.Q

        self.prediction_count += 1

        return self.x.copy()

    def update(self, measurement, current_time=None):
        """
        Update step: correct prediction with measurement.

        Args:
            measurement: [x, y] measured position in meters (or mm, will be converted)
            current_time: Current timestamp in seconds (optional, for statistics)

        Returns:
            Updated state [x, y, vx, vy] in meters and m/s
        """
        z = np.array(measurement)

        # Convert mm to meters if needed
        if np.abs(z[0]) > 1.0 or np.abs(z[1]) > 1.0:
            z = z / 1000.0

        # Innovation: y = z - H*x
        y = z - self.H @ self.x

        # Innovation covariance: S = H*P*H' + R
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain: K = P*H' * inv(S)
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update: x = x + K*y
        self.x = self.x + K @ y

        # Covariance update: P = (I - K*H) * P
        I_KH = np.eye(4) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T  # Joseph form for numerical stability

        self.update_count += 1
        if current_time is not None:
            self.last_measurement_time = current_time

        return self.x.copy()

    def get_state(self):
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

    def reset(self, initial_position=None):
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


class IMUKalmanFilter:
    """
    Extended Kalman Filter for IMU orientation estimation.

    State: [rx, ry, bias_gx, bias_gy, bias_gz]
    - rx, ry: Roll and pitch angles in radians
    - bias_gx, bias_gy, bias_gz: Gyro biases in rad/s

    Sensors:
    - Gyroscope: Provides angular velocity (prediction)
    - Accelerometer: Provides gravity direction (measurement update when quasi-static)

    Features:
    - Dual-rate processing (gyro ~759 Hz, accel ~1265 Hz)
    - Automatic rejection of dynamic acceleration
    - Gyro bias estimation for long-term stability
    """

    def __init__(self,
                 accel_noise_std=None,
                 gyro_noise_std=None,
                 bias_process_noise=1e-6,
                 gyro_scale=0.00875,
                 accel_scale=0.001,
                 dt_gyro=1.0/759.0,
                 dynamic_accel_threshold=2.0):
        """
        Args:
            accel_noise_std: (ax, ay, az) noise std in m/s² (from analyze_imu.py)
            gyro_noise_std: (gx, gy, gz) noise std in rad/s (from analyze_imu.py)
            bias_process_noise: Gyro bias random walk std (rad/s/sqrt(s))
            gyro_scale: Gyro scaling factor (deg/s per raw unit), default 0.00875 for L3GD20
            accel_scale: Accel scaling factor (g per raw unit), default 0.001 for LSM303
            dt_gyro: Gyro sampling period (seconds)
            dynamic_accel_threshold: Accel magnitude threshold for quasi-static detection (m/s²)
        """
        # State: [rx(rad), ry(rad), bias_gx(rad/s), bias_gy(rad/s), bias_gz(rad/s)]
        # Initialize with measured gyro biases from analyze_imu.py
        self.x = np.array([
            0.0,        # rx (rad)
            0.0,        # ry (rad)
            0.065968,   # bias_gx (rad/s) - measured mean
            0.050949,   # bias_gy (rad/s) - measured mean
            0.018238    # bias_gz (rad/s) - measured mean
        ])

        # State covariance (units: rad² for angles, (rad/s)² for biases)
        self.P = np.diag([0.01, 0.01, 0.001, 0.001, 0.001])  # Lower bias uncertainty since we used measured values

        # Sensor scaling
        self.gyro_scale = gyro_scale * (np.pi / 180.0)  # Convert to rad/s per raw unit
        self.accel_scale = accel_scale * 9.81  # Convert to m/s² per raw unit

        # Time step
        self.dt_gyro = dt_gyro

        # Noise parameters (measured from analyze_imu.py)
        # Using physical units: accel in m/s², gyro in rad/s
        if gyro_noise_std is None:
            # Measured values from IMU characterization
            gyro_noise_std = [0.017550, 0.017716, 0.004335]  # rad/s
        if accel_noise_std is None:
            # Measured values from IMU characterization
            accel_noise_std = [0.0701, 0.0651, 0.0889]  # m/s²

        self.gyro_noise_std = np.array(gyro_noise_std)
        self.accel_noise_std = np.array(accel_noise_std)

        self.bias_process_noise = bias_process_noise

        # Dynamic acceleration threshold
        self.dynamic_threshold = dynamic_accel_threshold
        self.g = 9.81

        # Build noise matrices
        self._build_noise_matrices()

        # Statistics
        self.prediction_count = 0
        self.update_count = 0
        self.rejected_update_count = 0
        self.last_accel_magnitude = 0.0

    def _build_noise_matrices(self):
        """Build process and measurement noise covariance matrices."""
        dt = self.dt_gyro

        # Process noise Q: uncertainty in gyro integration + bias random walk
        # State: [rx, ry, bias_gx, bias_gy, bias_gz]
        q_angle = (self.gyro_noise_std[0] * dt) ** 2  # Angle uncertainty from gyro noise
        q_bias = (self.bias_process_noise * dt) ** 2  # Bias random walk (very small to keep biases stable)

        # Use extremely small bias process noise to prevent bias drift
        # Biases are initialized with measured values and should remain nearly constant
        self.Q = np.diag([q_angle, q_angle, q_bias * 0.001, q_bias * 0.001, q_bias * 0.001])

        # Measurement noise R: accelerometer noise
        # Measurement: [rx_meas, ry_meas] from atan2(accel components)
        # Noise propagates through nonlinear function - approximate with accel noise
        self.R = np.diag([
            (self.accel_noise_std[1] / self.g) ** 2,  # rx uncertainty from ay, az noise
            (self.accel_noise_std[0] / self.g) ** 2   # ry uncertainty from ax, az noise
        ])

    def predict(self, gyro_raw):
        """
        Prediction step using gyroscope measurement.

        Args:
            gyro_raw: [gx, gy, gz] raw gyro values

        Returns:
            Predicted state [rx_deg, ry_deg, bias_gx, bias_gy, bias_gz]
        """
        gx, gy, gz = gyro_raw

        # Convert raw gyro to rad/s
        gx_rad_s = gx * self.gyro_scale
        gy_rad_s = gy * self.gyro_scale
        gz_rad_s = gz * self.gyro_scale

        # Current state (biases are in rad/s)
        rx, ry, bias_gx, bias_gy, bias_gz = self.x

        # Saturate biases to reasonable limits (±10°/s = ±0.174 rad/s)
        max_bias = 0.174  # rad/s
        bias_gx = np.clip(bias_gx, -max_bias, max_bias)
        bias_gy = np.clip(bias_gy, -max_bias, max_bias)
        bias_gz = np.clip(bias_gz, -max_bias, max_bias)

        # Gyro integration (bias-corrected)
        rx_new = rx + (gy_rad_s - bias_gy) * self.dt_gyro
        ry_new = ry + (gx_rad_s - bias_gx) * self.dt_gyro

        # Wrap angles to [-pi, +pi] to prevent infinite accumulation
        rx_new = np.arctan2(np.sin(rx_new), np.cos(rx_new))
        ry_new = np.arctan2(np.sin(ry_new), np.cos(ry_new))

        # Update state with wrapped angles and saturated biases
        self.x = np.array([rx_new, ry_new, bias_gx, bias_gy, bias_gz])

        # Jacobian of state transition (F = ∂f/∂x)
        F = np.array([
            [1, 0, 0, -self.dt_gyro, 0],
            [0, 1, -self.dt_gyro, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1]
        ])

        # Covariance prediction: P = F*P*F' + Q
        self.P = F @ self.P @ F.T + self.Q

        # Ensure covariance stays positive definite
        self.P = 0.5 * (self.P + self.P.T)  # Force symmetry
        min_var = 1e-8
        for i in range(5):
            if self.P[i, i] < min_var:
                self.P[i, i] = min_var

        self.prediction_count += 1

        # Return orientation in degrees
        rx_deg = rx_new * (180.0 / np.pi)
        ry_deg = ry_new * (180.0 / np.pi)
        return np.array([rx_deg, ry_deg, bias_gx, bias_gy, bias_gz])

    def update(self, accel_raw, force_update=False):
        """
        Update step using accelerometer measurement.

        Only updates if acceleration magnitude is close to gravity (quasi-static).
        Uses accelerometer to measure absolute tilt angles from gravity direction.

        Args:
            accel_raw: [ax, ay, az] raw accelerometer values
            force_update: If True, skip dynamic acceleration check

        Returns:
            (updated, updated_state)
            - updated: True if measurement was used (quasi-static)
            - updated_state: Current state [rx_deg, ry_deg, bias_gx, bias_gy, bias_gz]
        """
        ax, ay, az = accel_raw

        # Convert to physical units
        ax_ms2 = ax * self.accel_scale
        ay_ms2 = ay * self.accel_scale
        az_ms2 = az * self.accel_scale

        # Check if quasi-static
        accel_magnitude = np.sqrt(ax_ms2**2 + ay_ms2**2 + az_ms2**2)
        self.last_accel_magnitude = accel_magnitude

        if not force_update and abs(accel_magnitude - self.g) > self.dynamic_threshold:
            # Dynamic acceleration detected - skip update
            self.rejected_update_count += 1
            rx_deg = self.x[0] * (180.0 / np.pi)
            ry_deg = self.x[1] * (180.0 / np.pi)
            return False, np.array([rx_deg, ry_deg, self.x[2], self.x[3], self.x[4]])

        # Measurement model: extract tilt from gravity direction
        # rx = atan2(-ay, az)  (roll: rotation around X axis)
        # ry = atan2(ax, az)   (pitch: rotation around Y axis)
        z_meas = np.array([
            np.arctan2(-ay_ms2, az_ms2),
            np.arctan2(ax_ms2, az_ms2)
        ])

        # Predicted measurement
        rx, ry = self.x[0], self.x[1]
        z_pred = np.array([rx, ry])

        # Innovation
        y = z_meas - z_pred

        # Wrap angles to [-pi, pi]
        y[0] = np.arctan2(np.sin(y[0]), np.cos(y[0]))
        y[1] = np.arctan2(np.sin(y[1]), np.cos(y[1]))

        # Measurement Jacobian H = ∂h/∂x
        # h(x) = [rx, ry] (measurement model is just extracting first two states)
        # But bias states don't affect measurement
        H = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0]
        ])

        # Innovation covariance: S = H*P*H' + R
        S = H @ self.P @ H.T + self.R

        # Kalman gain: K = P*H' * inv(S)
        K = self.P @ H.T @ np.linalg.inv(S)

        # State update: x = x + K*y
        self.x = self.x + K @ y

        # Wrap angles to [-pi, +pi] after update
        self.x[0] = np.arctan2(np.sin(self.x[0]), np.cos(self.x[0]))
        self.x[1] = np.arctan2(np.sin(self.x[1]), np.cos(self.x[1]))

        # Saturate biases to reasonable limits (±10°/s = ±0.174 rad/s)
        max_bias = 0.174  # rad/s
        self.x[2] = np.clip(self.x[2], -max_bias, max_bias)
        self.x[3] = np.clip(self.x[3], -max_bias, max_bias)
        self.x[4] = np.clip(self.x[4], -max_bias, max_bias)

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(5) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        # Ensure covariance stays positive definite and symmetric
        self.P = 0.5 * (self.P + self.P.T)  # Force symmetry
        min_var = 1e-8
        for i in range(5):
            if self.P[i, i] < min_var:
                self.P[i, i] = min_var

        self.update_count += 1

        rx_deg = self.x[0] * (180.0 / np.pi)
        ry_deg = self.x[1] * (180.0 / np.pi)
        return True, np.array([rx_deg, ry_deg, self.x[2], self.x[3], self.x[4]])

    def get_orientation_deg(self):
        """
        Get current orientation estimate in degrees.

        Returns:
            (rx, ry): Roll and pitch angles in degrees
        """
        rx_deg = self.x[0] * (180.0 / np.pi)
        ry_deg = self.x[1] * (180.0 / np.pi)
        return rx_deg, ry_deg

    def get_gyro_biases(self):
        """
        Get estimated gyro biases in rad/s.

        Returns:
            (bias_gx, bias_gy, bias_gz) in rad/s
        """
        return self.x[2], self.x[3], self.x[4]

    def get_state(self):
        """
        Get full filter state.

        Returns:
            [rx_deg, ry_deg, bias_gx, bias_gy, bias_gz]
        """
        rx_deg = self.x[0] * (180.0 / np.pi)
        ry_deg = self.x[1] * (180.0 / np.pi)
        return np.array([rx_deg, ry_deg, self.x[2], self.x[3], self.x[4]])

    def calibrate_biases(self, gyro_samples, duration=5.0):
        """
        Calibrate gyro biases from stationary measurements.

        Args:
            gyro_samples: List of [gx, gy, gz] raw gyro samples
            duration: Duration of calibration in seconds

        Returns:
            (bias_gx, bias_gy, bias_gz) estimated biases in rad/s
        """
        gyro_array = np.array(gyro_samples)

        # Compute mean in raw units, then convert to rad/s
        bias_gx_raw = np.mean(gyro_array[:, 0])
        bias_gy_raw = np.mean(gyro_array[:, 1])
        bias_gz_raw = np.mean(gyro_array[:, 2])

        # Convert to rad/s (consistent with state units)
        bias_gx = bias_gx_raw * self.gyro_scale
        bias_gy = bias_gy_raw * self.gyro_scale
        bias_gz = bias_gz_raw * self.gyro_scale

        # Update filter state
        self.x[2] = bias_gx
        self.x[3] = bias_gy
        self.x[4] = bias_gz

        # Reduce bias uncertainty after calibration (units: (rad/s)^2)
        self.P[2, 2] = (0.01 * self.gyro_scale) ** 2
        self.P[3, 3] = (0.01 * self.gyro_scale) ** 2
        self.P[4, 4] = (0.01 * self.gyro_scale) ** 2

        return bias_gx, bias_gy, bias_gz

    def reset(self, initial_orientation_deg=None, reset_biases=False):
        """
        Reset filter state.

        Args:
            initial_orientation_deg: Optional (rx, ry) in degrees
            reset_biases: If True, reset biases to measured defaults
        """
        if initial_orientation_deg is not None:
            rx_rad = initial_orientation_deg[0] * (np.pi / 180.0)
            ry_rad = initial_orientation_deg[1] * (np.pi / 180.0)
            self.x[0] = rx_rad
            self.x[1] = ry_rad
        else:
            self.x[0] = 0.0
            self.x[1] = 0.0

        # Reset biases to measured defaults if requested
        if reset_biases:
            self.x[2] = 0.065968  # bias_gx (rad/s)
            self.x[3] = 0.050949  # bias_gy (rad/s)
            self.x[4] = 0.018238  # bias_gz (rad/s)

        self.P = np.diag([0.01, 0.01, 0.001, 0.001, 0.001])
        self.prediction_count = 0
        self.update_count = 0
        self.rejected_update_count = 0

    def get_statistics(self):
        """Get filter statistics for monitoring."""
        total_updates = self.update_count + self.rejected_update_count
        return {
            'predictions': self.prediction_count,
            'updates': self.update_count,
            'rejected_updates': self.rejected_update_count,
            'update_rate': self.update_count / max(1, total_updates),
            'last_accel_mag': self.last_accel_magnitude,
            'gyro_biases': self.get_gyro_biases()
        }