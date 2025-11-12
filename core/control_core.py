#!/usr/bin/env python3
"""
Stewart Platform Control Core

Controllers:
- PIDController: PID control for ball balancing
- LQRController: Linear Quadratic Regulator for optimal control
- KalmanFilter: Linear Kalman Filter for ball state estimation
- OrientationKalmanFilter: EKF for IMU-based orientation estimation
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
    """Extended Kalman Filter for roll and pitch estimation from IMU.

    State vector: [roll, pitch, gyro_bias_x, gyro_bias_y]

    Features:
        - Automatic gravity vector zeroing at initialization
        - Gyroscope bias estimation
        - Removes initial gravity offset to make orientation relative
        - Supports axis transformations and gyro scale calibration
    """

    def __init__(self, accel_noise=1.0, gyro_noise=1.0, process_noise_angle=0.0, process_noise_bias=0.0,
                 accel_axis_flip=None, gyro_axis_flip=None, accel_rotation=None, gyro_rotation=None,
                 initial_bias_x=0.0, initial_bias_y=0.0, gyro_scale_multiplier=1.0,
                 accel_magnitude_threshold=2.0, gyro_magnitude_threshold=0.5):
        # IMU scaling
        self.accel_scale = 0.001 * 9.81  # LSM303: 1mg/LSB -> m/s²
        self.gyro_scale_multiplier = gyro_scale_multiplier  # Store for later updates
        self.gyro_scale = 0.00875 * np.pi / 180 * gyro_scale_multiplier  # L3GD20: 8.75 mdps/LSB -> rad/s (with calibration multiplier)

        # Motion detection thresholds
        self.accel_magnitude_threshold = accel_magnitude_threshold  # m/s² deviation from gravity
        self.gyro_magnitude_threshold = gyro_magnitude_threshold  # rad/s - high rotation rate
        self.enable_rejection = False  # Default OFF - can be toggled on/off via GUI
        self.rejected_accel_count = 0  # Statistics
        self.total_accel_count = 0

        # Magnetometer parameters
        self.use_magnetometer = False  # Enable mag-based tilt during accel rejection
        self.mag_offset = np.array([0.0, 0.0, 0.0])  # Hard-iron calibration offset
        from core.utils import IMUKalmanConfig
        self.mag_inclination = np.radians(IMUKalmanConfig.DEFAULT_MAG_INCLINATION_DEG)  # Magnetic inclination for Trondheim
        self.mag_update_count = 0  # Statistics

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
                            0.0])

        # Apply axis flip
        bias_vec = bias_vec * self.gyro_axis_flip

        # Apply rotation if provided
        if self.gyro_rotation is not None:
            bias_vec = bias_vec @ self.gyro_rotation.T

        # State: [roll, pitch, gyro_bias_x, gyro_bias_y]
        self.state = np.array([0.0, 0.0, bias_vec[0], bias_vec[1]])
        self.P = np.eye(4) * 0.1

        # Store noise parameters for process noise calculation
        self.gyro_noise = gyro_noise
        self.process_noise_angle = process_noise_angle
        self.process_noise_bias = process_noise_bias

        # Process noise covariance (will be computed dynamically in predict())
        self.Q = np.diag([
            process_noise_angle,
            process_noise_angle,
            process_noise_bias,
            process_noise_bias
        ])

        # Measurement noise covariance
        self.R = np.diag([
            accel_noise ** 2,
            accel_noise ** 2
        ])

        self.initialized = False
        self.initial_accel = None
        self.last_gyro_magnitude = 0.0  # Track gyro magnitude for motion detection

    def initialize(self, accel_raw, calibrated_gravity=None):
        """Initialize filter state from first accelerometer reading (raw LSB).

        Args:
            accel_raw: Initial acceleration measurement [LSB]
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
                print(f"WARNING: Measured gravity magnitude {gravity_mag:.2f} differs from expected {expected_gravity:.2f} by {gravity_error:.2f} m/s²")

            roll0 = np.arctan2(ay, az)
            pitch0 = np.arctan2(-ax, np.sqrt(ay ** 2 + az ** 2))

            self.state[0] = roll0
            self.state[1] = pitch0

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
        # Apply transformations and convert to rad/s
        gyro = apply_imu_transforms(gyro_raw, self.gyro_axis_flip, self.gyro_rotation, self.gyro_scale)
        gx, gy = gyro[0], gyro[1]

        # Store gyro magnitude for motion detection (used in update step)
        self.last_gyro_magnitude = np.sqrt(gx**2 + gy**2)

        # Bias-corrected angular velocity
        gx_corrected = gx - self.state[2]
        gy_corrected = gy - self.state[3]

        # State propagation
        self.state[0] += gx_corrected * dt
        self.state[1] += gy_corrected * dt

        # Jacobian of state transition
        F = np.array([
            [1, 0, -dt, 0],
            [0, 1, 0, -dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Process noise covariance (gyro noise affects angle uncertainty)
        # Angle uncertainty grows with gyro noise * dt
        # Bias uncertainty grows with process_noise_bias
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
            mag_raw: Optional magnetometer measurement [LSB] - used when accel is rejected

        Note:
            Skips update if motion is detected (prevents accelerometer corruption during impacts).
            If magnetometer is available and enabled, uses it as backup tilt reference during impacts.
        """
        self.total_accel_count += 1

        # Apply transformations and convert to m/s²
        accel = apply_imu_transforms(accel_raw, self.accel_axis_flip, self.accel_rotation, self.accel_scale)
        ax, ay, az = accel

        # Motion detection: Check acceleration magnitude
        accel_magnitude = np.linalg.norm(accel)
        accel_deviation = abs(accel_magnitude - GRAVITY_MAGNITUDE)

        # Motion detection: Check gyroscope magnitude (high rotation rate)
        gyro_is_high = self.last_gyro_magnitude > self.gyro_magnitude_threshold

        # Check if we should reject accelerometer update
        motion_detected = accel_deviation > self.accel_magnitude_threshold or gyro_is_high

        # If motion detected and rejection enabled
        if self.enable_rejection and motion_detected:
            self.rejected_accel_count += 1

            # Try to use magnetometer as backup if available
            if self.use_magnetometer and mag_raw is not None:
                self.update_with_magnetometer(mag_raw)
                return
            else:
                return  # Skip update - rely on gyro prediction only

        # Tilt angles from accelerometer
        roll_meas = np.arctan2(ay, az)
        pitch_meas = np.arctan2(-ax, np.sqrt(ay ** 2 + az ** 2))

        # Remove initial gravity offset (KEY: makes orientation relative to start)
        if self.initial_accel is not None:
            roll_init = np.arctan2(self.initial_accel[1], self.initial_accel[2])
            pitch_init = np.arctan2(-self.initial_accel[0],
                                    np.sqrt(self.initial_accel[1] ** 2 + self.initial_accel[2] ** 2))
            roll_meas -= roll_init
            pitch_meas -= pitch_init

        z = np.array([roll_meas, pitch_meas])

        # Measurement matrix
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # Innovation
        y = z - H @ self.state
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        # State and covariance update
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

    def update_with_magnetometer(self, mag_raw):
        """Update step using magnetometer when accelerometer is corrupted.

        Args:
            mag_raw: Magnetometer measurement [LSB]

        Note:
            Uses magnetic field direction to estimate tilt angles.
            Less accurate than accelerometer but immune to linear acceleration.
        """
        # Apply transformations (no scaling needed - we use direction only)
        mag = apply_imu_transforms(mag_raw, self.accel_axis_flip, self.accel_rotation, scale=1.0)

        # Remove hard-iron offset
        mag = mag - self.mag_offset

        # Normalize (we only care about direction)
        mag_magnitude = np.linalg.norm(mag)
        if mag_magnitude < 1.0:  # Avoid division by zero
            return

        mag_norm = mag / mag_magnitude
        mx, my, mz = mag_norm

        # Calculate tilt from magnetic field direction
        # Assumes Earth's magnetic field has known inclination (dip angle)
        # Roll primarily affects my/mz ratio
        # Pitch affects mx and horizontal components

        # For steep inclination (Norway ~75°), magnetic field is mostly vertical
        # When level: mag ≈ [Bh, 0, Bv] where Bv >> Bh
        # Roll rotates around X: affects my and mz
        # Pitch rotates around Y: affects mx and mz

        # Tilt-compensated magnetometer equations
        roll_meas = np.arctan2(-my, mz)
        pitch_meas = np.arctan2(mx, np.sqrt(my ** 2 + mz ** 2))

        # Adjust for magnetic inclination (field not horizontal)
        # This is approximate - works well for small tilts
        roll_meas = roll_meas * np.cos(self.mag_inclination)
        pitch_meas = pitch_meas * np.cos(self.mag_inclination)

        # Remove initial offset (same as accelerometer)
        if self.initial_accel is not None:
            roll_init = np.arctan2(self.initial_accel[1], self.initial_accel[2])
            pitch_init = np.arctan2(-self.initial_accel[0],
                                    np.sqrt(self.initial_accel[1] ** 2 + self.initial_accel[2] ** 2))
            roll_meas -= roll_init
            pitch_meas -= pitch_init

        z = np.array([roll_meas, pitch_meas])

        # Measurement matrix
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # Use higher measurement noise for magnetometer (less accurate than accel)
        R_mag = self.R * 4.0  # 2x higher noise covariance

        # Innovation
        y = z - H @ self.state
        S = H @ self.P @ H.T + R_mag
        K = self.P @ H.T @ np.linalg.inv(S)

        # State and covariance update
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

        self.mag_update_count += 1

    def get_orientation(self):
        """Return current orientation estimate.

        Returns:
            Tuple of (roll, pitch) in radians
        """
        return self.state[0], self.state[1]

    def get_rejection_stats(self):
        """Return accelerometer rejection statistics.

        Returns:
            Tuple of (rejected_count, total_count, rejection_rate_percent, mag_update_count)
        """
        if self.total_accel_count > 0:
            rejection_rate = 100.0 * self.rejected_accel_count / self.total_accel_count
        else:
            rejection_rate = 0.0
        return self.rejected_accel_count, self.total_accel_count, rejection_rate, self.mag_update_count

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
        from core.utils import IMUKalmanConfig, IMUCalibrationConfig

        # Initialize orientation Kalman filter with default configuration
        self.orientation_kalman = OrientationKalmanFilter(
            accel_noise=IMUKalmanConfig.DEFAULT_ACCEL_NOISE,
            gyro_noise=IMUKalmanConfig.DEFAULT_GYRO_NOISE,
            process_noise_angle=IMUKalmanConfig.DEFAULT_PROCESS_NOISE_ANGLE,
            process_noise_bias=IMUKalmanConfig.DEFAULT_PROCESS_NOISE_BIAS,
            accel_axis_flip=IMUKalmanConfig.DEFAULT_ACCEL_AXIS_FLIP,
            gyro_axis_flip=IMUKalmanConfig.DEFAULT_GYRO_AXIS_FLIP,
            accel_rotation=IMUKalmanConfig.DEFAULT_ACCEL_ROTATION,
            gyro_rotation=IMUKalmanConfig.DEFAULT_GYRO_ROTATION,
            initial_bias_x=IMUKalmanConfig.CALIBRATED_GYRO_BIAS_X,
            initial_bias_y=IMUKalmanConfig.CALIBRATED_GYRO_BIAS_Y,
            gyro_scale_multiplier=IMUKalmanConfig.DEFAULT_GYRO_SCALE_MULTIPLIER,
            accel_magnitude_threshold=IMUKalmanConfig.DEFAULT_ACCEL_THRESHOLD,
            gyro_magnitude_threshold=IMUKalmanConfig.DEFAULT_GYRO_THRESHOLD
        )

        # Current IMU orientation state
        self.current_rx_imu = 0.0
        self.current_ry_imu = 0.0
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

    def start_imu_initialization(self):
        """Begin IMU initialization sequence (3 seconds).

        During initialization:
        - IMU queues are drained (sensors stabilize)
        - Platform should remain stationary
        - Automatically transitions to calibration
        """
        import time
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
        import time
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
        import numpy as np

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
        # L3GD20 gyroscope: 8.75 mdps/LSB → rad/s
        gyro_scale = 0.00875 * np.pi / 180

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

        if abs(tilt_x_deg) > 5 or abs(tilt_y_deg) > 5:
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
        import time

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
        import numpy as np

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

        import numpy as np

        # Calculate IMU compensation (oppose platform tilt)
        rx_imu_comp = -self.current_rx_imu * self.imu_compensation_gain
        ry_imu_comp = -self.current_ry_imu * self.imu_compensation_gain

        # Clip compensation to safe range
        rx_imu_comp, ry_imu_comp, _ = clip_tilt_vector(rx_imu_comp, ry_imu_comp, 15.0)

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
            param_name: Parameter to update ('accel_noise', 'gyro_noise', 'gyro_scale', etc.)
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

    def on_imu_motion_param_change(self, param_name, value):
        """Handle IMU motion detection parameter change.

        Args:
            param_name: 'accel_threshold' or 'gyro_threshold'
            value: New threshold value (float)
        """
        if param_name == 'accel_threshold':
            self.orientation_kalman.accel_magnitude_threshold = value
            self.log(f"IMU accel threshold: {value:.2f} m/s²")
        elif param_name == 'gyro_threshold':
            self.orientation_kalman.gyro_magnitude_threshold = value
            self.log(f"IMU gyro threshold: {value:.2f} rad/s")

    def on_imu_detection_toggle(self, enabled):
        """Handle IMU motion detection enable/disable."""
        self.orientation_kalman.enable_rejection = enabled
        self.log(f"IMU motion detection: {'ENABLED' if enabled else 'DISABLED'}")

    def on_imu_mag_toggle(self, enabled):
        """Handle magnetometer enable/disable."""
        self.orientation_kalman.use_magnetometer = enabled
        self.log(f"Magnetometer backup: {'ENABLED' if enabled else 'DISABLED'}")