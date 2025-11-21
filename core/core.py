#!/usr/bin/env python3
"""
Stewart Platform Core Components

Shared classes for Stewart platform simulation and control:
- FirstOrderServo: Servo dynamics model
- StewartPlatformIK: Inverse and forward kinematics
- SimpleBallPhysics2D: 2D ball physics with rolling motion
"""

import logging
from typing import Optional, Tuple, Dict
import numpy as np
from collections import deque

from scipy.optimize import brentq, minimize_scalar

from core.utils import (
    MAX_SERVO_ANGLE_DEG, PLATFORM_HALF_SIZE_MM, PLATFORM_RADIUS_MM,
    SimulationConfig, DIVISION_BY_ZERO_EPSILON,
    IK_TAU_EPSILON, IK_SQRT_TERM_EPSILON, IK_DAMPING_INITIAL, IK_DAMPING_MAX,
    IK_JACOBIAN_DELTA, IK_Z_MIN_MM, IK_Z_MAX_MM, IK_MAX_ROLL_PITCH_DEG,
    IK_MAX_YAW_DEG, IK_PENALTY_VALUE, IK_VALID_THRESHOLD,
    IK_Z_SAMPLES_STANDARD, IK_Z_SAMPLES_EXTENDED,
    IK_Z_SEARCH_MIN_MM, IK_Z_SEARCH_EXTEND_MM,
    BALL_GEFF_MIN_FACTOR, BALL_GEFF_MAX_FACTOR
)

# Set up logger for this module
logger = logging.getLogger(__name__)


class FirstOrderServo:
    """First-order servo model with command delay."""

    def __init__(self, K: float = 1.0, tau: float = 0.05, delay: float = 0.0,
                 max_velocity: float = SimulationConfig.DEFAULT_SERVO_MAX_VELOCITY) -> None:
        self.K = K
        self.tau = tau
        self.delay = delay
        self.max_velocity = max_velocity
        self.current_angle = 0.0
        self.target_angle = 0.0
        self.command_queue = deque()

    def send_command(self, angle: float, current_time: float) -> None:
        delivery_time = current_time + self.delay
        self.command_queue.append((delivery_time, angle))

    def update(self, dt: float, current_time: float) -> None:
        while self.command_queue and self.command_queue[0][0] <= current_time:
            _, angle = self.command_queue.popleft()
            self.target_angle = angle

        # Compute ideal response using analytical solution
        # For τẏ + y = K·u, solution is: y(t+dt) = K·u + (y(t) - K·u)·e^(-dt/τ)
        if self.tau > IK_TAU_EPSILON:
            decay = np.exp(-dt / self.tau)
            steady_state = self.K * self.target_angle
            ideal_angle = steady_state + (self.current_angle - steady_state) * decay
        else:
            # Instantaneous response when time constant approaches zero
            ideal_angle = self.K * self.target_angle

        # Apply velocity constraints
        angle_change = ideal_angle - self.current_angle
        max_change = self.max_velocity * dt

        # Enforce maximum velocity limit
        if abs(angle_change) > max_change:
            angle_change = np.sign(angle_change) * max_change

        self.current_angle += angle_change

    def get_angle(self) -> float:
        return self.current_angle

    def reset(self) -> None:
        self.current_angle = 0.0
        self.target_angle = 0.0
        self.command_queue.clear()


class StewartPlatformIK:
    """Stewart Platform inverse and forward kinematics."""

    def __init__(self, horn_length: float = 31.75, rod_length: float = 145.0, base: float = 73.025,
                 base_anchors: float = 36.8893, platform: float = 67.775, platform_anchors: float = 12.7,
                 top_surface_offset: float = 26.0) -> None:
        self.horn_length = horn_length
        self.rod_length = rod_length
        self.base = base
        self.base_anchors = base_anchors
        self.platform = platform
        self.platform_anchors = platform_anchors
        self.top_surface_offset = top_surface_offset

        base_angles = np.array([-np.pi / 2, np.pi / 6, np.pi * 5 / 6])
        platform_angles = np.array([-np.pi * 5 / 6, -np.pi / 6, np.pi / 2])

        self.base_anchors = self.calculate_home_coordinates(self.base, self.base_anchors, base_angles)
        platform_anchors_out_of_phase = self.calculate_home_coordinates(self.platform, self.platform_anchors,
                                                                        platform_angles)
        self.platform_anchors = np.roll(platform_anchors_out_of_phase, shift=-1, axis=0)

        self.beta_angles = self._calculate_beta_angles()

        base_pos = self.base_anchors[0]
        platform_pos = self.platform_anchors[0]
        horn_end_x = base_pos[0] + self.horn_length * np.cos(self.beta_angles[0])
        horn_end_y = base_pos[1] + self.horn_length * np.sin(self.beta_angles[0])
        dx = platform_pos[0] - horn_end_x
        dy = platform_pos[1] - horn_end_y
        horiz_dist_sq = dx ** 2 + dy ** 2
        self.home_height = np.sqrt(self.rod_length ** 2 - horiz_dist_sq)
        self.home_height_top_surface = self.home_height + self.top_surface_offset

    def calculate_home_coordinates(self, l, d, phi):
        """
        Calculate anchor point coordinates for platform or base.

        Args:
            l: Radius of the main circle (mm)
            d: Offset distance from main circle to anchor points (mm)
            phi: Array of angular positions for anchor pairs (radians)

        Returns:
            (6, 3) array of anchor coordinates [x, y, z] in mm
        """
        angles = np.array([-np.pi / 2, np.pi / 2])
        xy = np.zeros((6, 3))
        for i in range(len(phi)):
            for j in range(len(angles)):
                x = l * np.cos(phi[i]) + d * np.cos(phi[i] + angles[j])
                y = l * np.sin(phi[i]) + d * np.sin(phi[i] + angles[j])
                xy[i * 2 + j] = np.array([x, y, 0])
        return xy

    def _calculate_beta_angles(self):
        """
        Calculate servo horn orientation angles (beta) for each servo.

        Beta angles define the direction each servo horn extends from its base anchor.
        These are fixed geometric properties of the Stewart platform design.

        Returns:
            Array of 6 beta angles in radians
        """
        beta_angles = np.zeros(6)
        beta_angles[0] = 0
        beta_angles[1] = np.pi
        dx_23 = self.base_anchors[3, 0] - self.base_anchors[2, 0]
        dy_23 = self.base_anchors[3, 1] - self.base_anchors[2, 1]
        angle_23 = np.arctan2(dy_23, dx_23)
        beta_angles[2] = angle_23
        beta_angles[3] = angle_23 + np.pi
        dx_54 = self.base_anchors[4, 0] - self.base_anchors[5, 0]
        dy_54 = self.base_anchors[4, 1] - self.base_anchors[5, 1]
        angle_54 = np.arctan2(dy_54, dx_54)
        beta_angles[5] = angle_54
        beta_angles[4] = angle_54 + np.pi
        return beta_angles

    def update_offset(self, new_offset: float) -> None:
        """
        Update the top surface offset (distance from platform anchors to ball surface).

        Args:
            new_offset: New offset value in mm
        """
        self.top_surface_offset = new_offset
        self.home_height_top_surface = self.home_height + self.top_surface_offset

    def calculate_servo_angles(self, translation: np.ndarray, rotation: np.ndarray,
                               use_top_surface_offset: bool = True) -> Optional[np.ndarray]:
        quat = self._euler_to_quaternion(np.radians(rotation))

        if use_top_surface_offset:
            offset_platform_frame = np.array([0, 0, -self.top_surface_offset])
            offset_world_frame = self._rotate_vector(offset_platform_frame, quat)
            anchor_center_translation = translation + offset_world_frame
        else:
            anchor_center_translation = translation

        angles = np.zeros(6)
        for k in range(6):
            p_world = anchor_center_translation + self._rotate_vector(self.platform_anchors[k], quat)
            leg = p_world - self.base_anchors[k]
            leg_length_sq = np.dot(leg, leg)

            e_k = 2 * self.horn_length * leg[2]
            f_k = 2 * self.horn_length * (
                    np.cos(self.beta_angles[k]) * leg[0] +
                    np.sin(self.beta_angles[k]) * leg[1]
            )
            g_k = leg_length_sq - (self.rod_length ** 2 - self.horn_length ** 2)

            sqrt_term = e_k ** 2 + f_k ** 2
            if sqrt_term < IK_SQRT_TERM_EPSILON:
                return None

            ratio = g_k / np.sqrt(sqrt_term)
            if abs(ratio) > 1.0:
                return None

            alpha_k = np.arcsin(ratio) - np.arctan2(f_k, e_k)
            angles[k] = np.degrees(alpha_k)

            if abs(angles[k]) > MAX_SERVO_ANGLE_DEG:
                return None

        return -angles

    def calculate_forward_kinematics(self, servo_angles: np.ndarray,
                                     initial_guess: Optional[tuple] = None,
                                     use_top_surface_offset: bool = True,
                                     max_iterations: int = 50,
                                     tolerance: float = 1e-6) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool, int]:
        if initial_guess is None:
            if use_top_surface_offset:
                translation = np.array([0.0, 0.0, self.home_height_top_surface])
            else:
                translation = np.array([0.0, 0.0, self.home_height])
            rotation = np.array([0.0, 0.0, 0.0])
        else:
            translation, rotation = initial_guess
            translation = np.array(translation, dtype=float)
            rotation = np.array(rotation, dtype=float)

        target_angles = servo_angles
        damping = IK_DAMPING_INITIAL

        for iteration in range(max_iterations):
            calculated_angles = self.calculate_servo_angles(
                translation, rotation, use_top_surface_offset
            )

            if calculated_angles is None:
                return None, None, False, iteration

            error = target_angles - calculated_angles
            error_magnitude = np.linalg.norm(error)

            if error_magnitude < tolerance:
                return translation.copy(), rotation.copy(), True, iteration

            delta = IK_JACOBIAN_DELTA
            jacobian = np.zeros((6, 6))

            for i in range(3):
                trans_perturbed = translation.copy()
                trans_perturbed[i] += delta
                angles_perturbed = self.calculate_servo_angles(
                    trans_perturbed, rotation, use_top_surface_offset
                )
                if angles_perturbed is not None:
                    jacobian[:, i] = (angles_perturbed - calculated_angles) / delta

            for i in range(3):
                rot_perturbed = rotation.copy()
                rot_perturbed[i] += delta
                angles_perturbed = self.calculate_servo_angles(
                    translation, rot_perturbed, use_top_surface_offset
                )
                if angles_perturbed is not None:
                    jacobian[:, i + 3] = (angles_perturbed - calculated_angles) / delta

            JTJ = jacobian.T @ jacobian
            JTJ_damped = JTJ + damping * np.eye(6)
            JT_error = jacobian.T @ error

            try:
                pose_update = np.linalg.solve(JTJ_damped, JT_error)
            except np.linalg.LinAlgError:
                damping *= 10
                if damping > IK_DAMPING_MAX:
                    return None, None, False, iteration
                continue

            translation += pose_update[:3]
            rotation += pose_update[3:]

            if np.linalg.norm(translation[:2]) > PLATFORM_HALF_SIZE_MM:
                return None, None, False, iteration
            if translation[2] < IK_Z_MIN_MM or translation[2] > IK_Z_MAX_MM:
                return None, None, False, iteration
            if abs(rotation[0]) > IK_MAX_ROLL_PITCH_DEG or abs(rotation[1]) > IK_MAX_ROLL_PITCH_DEG:  # Check roll/pitch
                return None, None, False, iteration
            if abs(rotation[2]) > IK_MAX_YAW_DEG:  # Separate yaw check
                return None, None, False, iteration

        return None, None, False, max_iterations

    def _euler_to_quaternion(self, euler: np.ndarray) -> np.ndarray:
        """
        Convert Euler angles (ZYX convention) to quaternion.

        Args:
            euler: [rx, ry, rz] Euler angles in radians (roll, pitch, yaw)

        Returns:
            Quaternion [w, x, y, z] representing the same rotation
        """
        rx, ry, rz = euler
        cy = np.cos(rz * 0.5)
        sy = np.sin(rz * 0.5)
        cp = np.cos(ry * 0.5)
        sp = np.sin(ry * 0.5)
        cr = np.cos(rx * 0.5)
        sr = np.sin(rx * 0.5)
        return np.array([
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy
        ])

    def _rotate_vector(self, v: np.ndarray, q: np.ndarray) -> np.ndarray:
        """
        Rotate a 3D vector using a quaternion.

        Uses the formula: v' = q * v * q^(-1) in quaternion multiplication.

        Args:
            v: 3D vector [x, y, z] to rotate
            q: Quaternion [w, x, y, z] representing rotation

        Returns:
            Rotated vector [x', y', z']
        """
        w, x, y, z = q
        vx, vy, vz = v
        return np.array([
            vx * (w * w + x * x - y * y - z * z) + vy * (2 * x * y - 2 * w * z) + vz * (2 * x * z + 2 * w * y),
            vx * (2 * x * y + 2 * w * z) + vy * (w * w - x * x + y * y - z * z) + vz * (2 * y * z - 2 * w * x),
            vx * (2 * x * z - 2 * w * y) + vy * (2 * y * z + 2 * w * x) + vz * (w * w - x * x - y * y + z * z)
        ])

    def optimize_z_offset(self, translation: np.ndarray, rotation: np.ndarray,
                          use_top_surface_offset: bool = True,
                          z_search_range: float = 30.0,
                          max_iterations: int = 20,
                          tolerance: float = 0.1,
                          verbose: bool = False,
                          ik_cache=None) -> Tuple[np.ndarray, Optional[np.ndarray], bool]:
        """
        Dynamically adjust Z position to balance servo angles around neutral (0°).

        Uses Brent's method for fast convergence (typically 3-5 iterations).
        If standard optimization fails, automatically expands search range to find
        ANY valid Z height that allows IK to succeed.

        Args:
            translation: [x, y, z] position in mm
            rotation: [rx, ry, rz] angles in degrees
            use_top_surface_offset: Use top surface as reference
            z_search_range: Maximum Z adjustment range (±mm) for initial search
            max_iterations: Maximum optimization iterations for Brent's method
            tolerance: Convergence tolerance (degrees)
            verbose: Enable debug output
            ik_cache: Optional IKCache instance to speed up repeated IK calculations

        Returns:
            (optimized_translation, servo_angles, success)
        """
        x, y, z_initial = translation
        z_min = z_initial - z_search_range
        z_max = z_initial + z_search_range

        # Track best solution for fallback
        best_imbalance = float('inf')
        best_z = z_initial
        best_angles = None
        samples_evaluated = 0

        def imbalance_at_z(z):
            """Compute servo angle imbalance at given Z height."""
            nonlocal best_imbalance, best_z, best_angles, samples_evaluated

            trans = np.array([x, y, z])

            # Check cache for existing solution
            if ik_cache is not None:
                angles = ik_cache.get(trans, rotation)
                if angles is None:
                    # Compute and cache result
                    angles = self.calculate_servo_angles(trans, rotation, use_top_surface_offset)
                    if angles is not None:
                        ik_cache.put(trans, rotation, angles)
            else:
                # Compute directly without caching
                angles = self.calculate_servo_angles(trans, rotation, use_top_surface_offset)

            samples_evaluated += 1

            if angles is None:
                # Return penalty value indicating IK failure direction
                return IK_PENALTY_VALUE if z > z_initial else -IK_PENALTY_VALUE

            max_angle = np.max(angles)
            min_angle = np.min(angles)
            imbalance = max_angle + min_angle

            # Update best solution tracker
            if abs(imbalance) < abs(best_imbalance):
                best_imbalance = imbalance
                best_z = z
                best_angles = angles.copy()

            if verbose and samples_evaluated <= 5:
                logger.debug(f"  Sample {samples_evaluated}: z={z:.1f}mm, imbalance={imbalance:.2f}°")

            return imbalance

        # Identify valid Z range through dense sampling
        # Dense sampling ensures narrow valid ranges are detected at extreme angles
        z_samples = np.linspace(z_min, z_max, IK_Z_SAMPLES_STANDARD)
        valid_z = []
        valid_imbalances = []

        for z_test in z_samples:
            imb_test = imbalance_at_z(z_test)
            if abs(imb_test) < IK_VALID_THRESHOLD:  # Valid IK solution exists
                valid_z.append(z_test)
                valid_imbalances.append(imb_test)

        if len(valid_z) < 2:
            # Insufficient valid solutions in standard search range
            if best_angles is not None:
                # Return single valid solution found
                if verbose:
                    logger.debug(f"  Only {len(valid_z)} valid Z found - using best: z={best_z:.1f}mm")
                return np.array([x, y, best_z]), best_angles, True

            # Extend search to full mechanical range
            if verbose:
                logger.debug(f"  No valid Z found in range [{z_min:.1f}, {z_max:.1f}]mm")
                logger.debug(f"  Expanding search to full mechanical range")

            # Search full mechanical range
            z_extreme_min = max(IK_Z_SEARCH_MIN_MM, z_initial - IK_Z_SEARCH_EXTEND_MM)
            z_extreme_max = min(IK_Z_MAX_MM, z_initial + IK_Z_SEARCH_EXTEND_MM)

            # Coarse sampling across extended range
            z_extreme_samples = np.linspace(z_extreme_min, z_extreme_max, IK_Z_SAMPLES_EXTENDED)

            for z_test in z_extreme_samples:
                trans_test = np.array([x, y, z_test])

                # Check cache for extended range samples
                if ik_cache is not None:
                    angles_test = ik_cache.get(trans_test, rotation)
                    if angles_test is None:
                        angles_test = self.calculate_servo_angles(trans_test, rotation, use_top_surface_offset)
                        if angles_test is not None:
                            ik_cache.put(trans_test, rotation, angles_test)
                else:
                    angles_test = self.calculate_servo_angles(trans_test, rotation, use_top_surface_offset)

                samples_evaluated += 1

                if angles_test is not None:
                    # Valid solution found in extended range
                    max_angle = np.max(angles_test)
                    min_angle = np.min(angles_test)
                    imbalance = max_angle + min_angle

                    if verbose:
                        logger.debug(f"  Extended search successful: z={z_test:.1f}mm, imbalance={imbalance:.2f}° ({samples_evaluated} samples)")

                    return trans_test, angles_test, True

            # Extended range search unsuccessful
            if verbose:
                logger.debug(f"  Extended search failed: no valid Z in [{z_extreme_min:.1f}, {z_extreme_max:.1f}]mm ({samples_evaluated} samples)")
            return translation, None, False

        # Refine search to valid Z region
        z_min = min(valid_z)
        z_max = max(valid_z)
        imb_min = valid_imbalances[0]
        imb_max = valid_imbalances[-1]

        if verbose:
            logger.debug(f"  Valid range: [{z_min:.1f}, {z_max:.1f}]mm ({len(valid_z)}/41 samples)")
            logger.debug(f"  Imbalance at bounds: [{imb_min:.2f}°, {imb_max:.2f}°]")

        # Check for zero crossing within valid solutions
        has_valid_sign_change = (
            abs(imb_min) < IK_VALID_THRESHOLD and abs(imb_max) < IK_VALID_THRESHOLD and  # Both values are valid
            imb_min * imb_max < 0  # Opposite signs indicate zero crossing
        )

        if has_valid_sign_change:
            # Apply Brent's method for root finding
            try:
                z_opt = brentq(imbalance_at_z, z_min, z_max, xtol=tolerance / 10.0, maxiter=max_iterations)
                trans_opt = np.array([x, y, z_opt])
                angles_opt = self.calculate_servo_angles(trans_opt, rotation, use_top_surface_offset)

                if angles_opt is not None:
                    if verbose:
                        final_imb = np.max(angles_opt) + np.min(angles_opt)
                        logger.debug(f"  Brent converged: z={z_opt:.1f}mm, imbalance={final_imb:.4f}° ({samples_evaluated} samples)")
                    return trans_opt, angles_opt, True
            except (ValueError, RuntimeError) as e:
                if verbose:
                    logger.debug(f"  Brent failed: {e}")
        else:
            if verbose:
                logger.debug(f"  No sign change - using minimize_scalar")

            # Apply minimization without zero crossing
            result = minimize_scalar(
                lambda z: abs(imbalance_at_z(z)),
                bounds=(z_min, z_max),
                method='bounded',
                options={'xatol': tolerance / 10.0}
            )

            if result.success:
                z_opt = result.x
                trans_opt = np.array([x, y, z_opt])
                angles_opt = self.calculate_servo_angles(trans_opt, rotation, use_top_surface_offset)

                if angles_opt is not None:
                    if verbose:
                        final_imb = np.max(angles_opt) + np.min(angles_opt)
                        logger.debug(f"  Minimizer converged: z={z_opt:.1f}mm, imbalance={final_imb:.2f}° ({samples_evaluated} samples)")
                    return trans_opt, angles_opt, True

        # Return best solution from sampling
        if best_angles is not None:
            trans_best = np.array([x, y, best_z])
            if verbose:
                logger.debug(f"  Using best: z={best_z:.1f}mm, imbalance={best_imbalance:.2f}° ({samples_evaluated} samples)")
            return trans_best, best_angles, True

        # Fallback case: valid Z range found but optimization failed
        if verbose:
            logger.debug(f"  Warning: no valid solution after finding {len(valid_z)} valid Z values")
        return translation, None, False


class SimpleBallPhysics2D:
    """
    2D Ball Physics with rolling motion.

    Physics model:
    - Ball rolls (not slides) on tilted surface
    - Rolling constraint: v = omega × r
    - Acceleration: a = (g * sin(theta)) / mass_factor
      where mass_factor = 1 + I/(m*r²)
    - Rolling resistance for energy dissipation
    """

    def __init__(self,
                 ball_radius: float = 0.02,
                 ball_mass: float = 0.0027,
                 gravity: float = 9.81,
                 rolling_friction: float = 0.0225,
                 sphere_type: str = 'hollow',
                 air_density: float = 1.225,
                 drag_coefficient: float = 0.47) -> None:
        self.radius = ball_radius
        self.mass = ball_mass
        self.g = gravity
        self.mu_roll = rolling_friction
        self.sphere_type = sphere_type

        # Air resistance parameters
        self.air_density = air_density
        self.drag_coefficient = drag_coefficient
        self.cross_section_area = np.pi * self.radius ** 2

        self.update_sphere_type(sphere_type)

    def update_sphere_type(self, sphere_type: str) -> None:
        """Update moment of inertia based on sphere type."""
        self.sphere_type = sphere_type

        if sphere_type == 'solid':
            self.I = (2.0 / 5.0) * self.mass * self.radius ** 2
        elif sphere_type == 'hollow':
            self.I = (2.0 / 3.0) * self.mass * self.radius ** 2
        else:
            raise ValueError(f"Unknown sphere_type: {sphere_type}. Use 'solid' or 'hollow'")

        self.mass_factor = 1.0 + self.I / (self.mass * self.radius ** 2)

    def step(self, ball_pos, ball_vel, ball_omega, platform_pose, dt, platform_angular_accel=None):
        """
        Step physics using RK4 integration with rolling motion.

        Args:
            ball_pos: (batch, 3) - only X and Y matter, Z is computed
            ball_vel: (batch, 3) - only X and Y matter
            ball_omega: (batch, 3) - angular velocity [wx, wy, wz] (rad/s)
            platform_pose: (batch, 6) [x, y, z, rx, ry, rz]
            dt: timestep
            platform_angular_accel: optional dict with 'rx' and 'ry' angular accelerations

        Returns:
            new_pos, new_vel, new_omega, contact_info
        """
        batch_size = ball_pos.shape[0]
        device = ball_pos.device

        xy_pos = ball_pos[:, :2]
        xy_vel = ball_vel[:, :2]
        xy_omega = ball_omega[:, :2]

        state = (xy_pos, xy_vel, xy_omega)
        new_xy_pos, new_xy_vel, new_xy_omega = rk4_step(
            state,
            self._compute_derivatives,
            dt,
            platform_pose,
            platform_angular_accel
        )

        # Check if ball fell off circular platform
        max_radius = PLATFORM_RADIUS_MM / 1000.0
        distance_from_center = np.linalg.norm(new_xy_pos, axis=1)
        fell_off = distance_from_center > max_radius

        contact_info = {'fell_off': False}

        if fell_off.any():
            for i in range(batch_size):
                if fell_off[i]:
                    new_xy_pos[i] = 0.0
                    new_xy_vel[i] = 0.0
                    new_xy_omega[i] = 0.0
            contact_info['fell_off'] = True

        platform_z = self._compute_platform_height(new_xy_pos, platform_pose)

        new_ball_pos = np.zeros((batch_size, 3), dtype=np.float32)
        new_ball_pos[:, :2] = new_xy_pos
        new_ball_pos[:, 2] = platform_z + self.radius

        new_ball_vel = np.zeros((batch_size, 3), dtype=np.float32)
        new_ball_vel[:, :2] = new_xy_vel

        new_ball_omega = np.zeros((batch_size, 3), dtype=np.float32)
        new_ball_omega[:, :2] = new_xy_omega

        contact_info['in_contact'] = np.ones(batch_size, dtype=bool)
        contact_info['rolling_speed'] = np.linalg.norm(new_xy_omega, axis=1) * self.radius

        return new_ball_pos, new_ball_vel, new_ball_omega, contact_info

    def _compute_derivatives(self, state, platform_pose, platform_angular_accel=None):
        """Compute derivatives for RK4 with rolling physics."""
        xy_pos, xy_vel, xy_omega = state

        d_pos = xy_vel
        d_vel, d_omega = self._compute_accelerations(xy_pos, xy_vel, xy_omega,
                                                     platform_pose, platform_angular_accel)

        return d_pos, d_vel, d_omega

    def _compute_accelerations(self, xy_pos, xy_vel, xy_omega, platform_pose, platform_angular_accel=None):
        """Compute accelerations for rolling ball on tilted surface."""
        batch_size = xy_pos.shape[0]

        # Extract platform orientation
        rx = np.radians(platform_pose[:, 3])  # Roll angle (rotation about X)
        ry = np.radians(platform_pose[:, 4])  # Pitch angle (rotation about Y)

        ball_x = xy_pos[:, 0]
        ball_y = xy_pos[:, 1]

        # Start with standard gravity
        g_eff = self.g

        # Account for platform angular acceleration (creates fictitious forces)
        if platform_angular_accel is not None:
            alpha_rx_rad = platform_angular_accel['rx'] * (np.pi / 180.0)
            alpha_ry_rad = platform_angular_accel['ry'] * (np.pi / 180.0)

            # Fictitious vertical acceleration due to platform rotation
            # a_z = r × α (cross product of position and angular acceleration)
            a_z_platform = ball_x * alpha_ry_rad - ball_y * alpha_rx_rad
            g_eff = self.g - a_z_platform
            # Clamp to reasonable range to prevent numerical instability
            g_eff = np.clip(g_eff, BALL_GEFF_MIN_FACTOR * self.g, BALL_GEFF_MAX_FACTOR * self.g)

        cos_rx = np.cos(rx)
        cos_ry = np.cos(ry)
        sin_rx = np.sin(rx)
        sin_ry = np.sin(ry)

        # Project gravity onto tilted surface to get downslope acceleration
        # For a tilted plane, gravity component along surface = g * sin(θ) * cos(φ)
        # where θ is tilt angle and φ is perpendicular tilt
        gx = g_eff * sin_ry * cos_ry * cos_rx  # X-direction component
        gy = -g_eff * sin_rx * cos_rx * cos_ry  # Y-direction component

        # Apply rolling ball dynamics: a = F/(m·mass_factor)
        # mass_factor accounts for rotational inertia (1 + I/(m·r²))
        ax = gx / self.mass_factor
        ay = gy / self.mass_factor

        accel_linear = np.stack([ax, ay], axis=1)

        vel_magnitude = np.linalg.norm(xy_vel, axis=1, keepdims=True)
        # Rolling resistance: F_roll = μ_roll * N * v_hat (opposes motion)
        rolling_resistance = -self.mu_roll * g_eff * xy_vel / (vel_magnitude + DIVISION_BY_ZERO_EPSILON)

        # Air resistance (quadratic drag): F_drag = -½ρC_dAv²v̂
        # Increases with v² (dominant at high speeds)
        vel_squared = vel_magnitude ** 2
        drag_magnitude = 0.5 * self.air_density * self.drag_coefficient * self.cross_section_area * vel_squared / self.mass
        air_resistance = -drag_magnitude * xy_vel / (vel_magnitude + DIVISION_BY_ZERO_EPSILON)

        # Total linear acceleration
        accel_linear = accel_linear + rolling_resistance + air_resistance

        # Angular acceleration from rolling constraint: α = a/r
        # For rolling without slipping: v = ω × r, so a = α × r
        accel_angular = accel_linear / self.radius

        # Angular damping (energy dissipation in rotation)
        omega_damping = -self.mu_roll * xy_omega
        accel_angular = accel_angular + omega_damping

        return accel_linear, accel_angular

    def _compute_platform_height(self, xy_pos, platform_pose):
        """Compute platform Z height at given XY position."""
        px = platform_pose[:, 0] / 1000
        py = platform_pose[:, 1] / 1000
        pz = platform_pose[:, 2] / 1000

        rx = np.radians(platform_pose[:, 3])
        ry = np.radians(platform_pose[:, 4])

        dx = xy_pos[:, 0] - px
        dy = xy_pos[:, 1] - py

        height = pz - dx * np.tan(ry) - dy * np.tan(rx)

        return height

    def set_air_resistance(self, air_density: Optional[float] = None, drag_coefficient: Optional[float] = None) -> None:
        """
        Update air resistance parameters.

        Args:
            air_density: Air density in kg/m³ (None to keep current)
            drag_coefficient: Drag coefficient (None to keep current)
        """
        if air_density is not None:
            self.air_density = air_density
        if drag_coefficient is not None:
            self.drag_coefficient = drag_coefficient


def rk4_step(state, derivative_fn, dt, *args):
    """
    Generic RK4 (Runge-Kutta 4th order) integration step.

    Args:
        state: tuple of tensors representing system state
        derivative_fn: function computing derivatives
        dt: timestep
        *args: additional arguments for derivative_fn

    Returns:
        tuple of new state tensors
    """
    k1 = derivative_fn(state, *args)

    state_k2 = tuple(s + 0.5 * dt * k for s, k in zip(state, k1))
    k2 = derivative_fn(state_k2, *args)

    state_k3 = tuple(s + 0.5 * dt * k for s, k in zip(state, k2))
    k3 = derivative_fn(state_k3, *args)

    state_k4 = tuple(s + dt * k for s, k in zip(state, k3))
    k4 = derivative_fn(state_k4, *args)

    new_state = tuple(
        s + (dt / 6.0) * (k1_i + 2 * k2_i + 2 * k3_i + k4_i)
        for s, k1_i, k2_i, k3_i, k4_i in zip(state, k1, k2, k3, k4)
    )

    return new_state


class TrajectoryPattern:
    """Base class for trajectory patterns. All patterns return positions in mm and velocities in mm/s."""

    def get_position(self, t):
        """Get (x, y) position at time t in seconds."""
        raise NotImplementedError

    def get_velocity(self, t):
        """Get (vx, vy) velocity at time t in seconds."""
        raise NotImplementedError

    def reset(self):
        """Reset pattern state."""
        pass


class CirclePattern(TrajectoryPattern):
    """Circular trajectory pattern."""

    def __init__(self, radius: float = 50.0, period: float = 10.0, clockwise: bool = True) -> None:
        """
        Create a circular trajectory pattern.

        Args:
            radius: Circle radius in mm
            period: Time to complete one circle in seconds
            clockwise: True for clockwise motion, False for counter-clockwise
        """
        self.radius = radius
        self.period = period
        self.omega = 2 * np.pi / period
        self.direction = -1 if clockwise else 1

    def get_position(self, t: float) -> Tuple[float, float]:
        """Get (x, y) position in mm at time t."""
        angle = self.direction * self.omega * t
        x = self.radius * np.cos(angle)
        y = self.radius * np.sin(angle)
        return x, y

    def get_velocity(self, t: float) -> Tuple[float, float]:
        """Get (vx, vy) velocity in mm/s at time t."""
        angle = self.direction * self.omega * t
        vx = -self.direction * self.radius * self.omega * np.sin(angle)
        vy = self.direction * self.radius * self.omega * np.cos(angle)
        return vx, vy


class FigureEightPattern(TrajectoryPattern):
    """Figure-8 (lemniscate) trajectory pattern."""

    def __init__(self, width: float = 60.0, height: float = 40.0, period: float = 12.0) -> None:
        """
        Create a figure-8 (lemniscate) trajectory pattern.

        Args:
            width: Pattern width in mm
            height: Pattern height in mm
            period: Time to complete one figure-8 in seconds
        """
        self.period = period
        self.a = width / 2.0  # Semi-width
        self.b = height / 2.0  # Semi-height
        self.omega = 2 * np.pi / period

    def get_position(self, t: float) -> Tuple[float, float]:
        """Get (x, y) position in mm at time t."""
        angle = self.omega * t
        x = self.a * np.cos(angle)
        y = self.b * np.sin(angle) * np.cos(angle)  # Lemniscate formula
        return x, y

    def get_velocity(self, t: float) -> Tuple[float, float]:
        """Get (vx, vy) velocity in mm/s at time t."""
        angle = self.omega * t
        vx = -self.a * self.omega * np.sin(angle)
        vy = self.b * self.omega * np.cos(2 * angle)  # Derivative of lemniscate
        return vx, vy


class StarPattern(TrajectoryPattern):
    """Five-pointed star with instant jumps between corners."""

    def __init__(self, radius: float = 60.0, period: float = 15.0, dwell_time: float = 0.5) -> None:
        """
        Create a star pattern where target instantly jumps between star points.

        The ball jumps to corners in order: 1→3→5→2→4→1, creating a star pattern.
        Stays at each corner for period/5 seconds before jumping to the next.

        Args:
            radius: Distance from center to star points in mm
            period: Time to complete one full star cycle in seconds
            dwell_time: Time to stay at each corner before jumping (seconds) - deprecated, not used
        """
        self.radius = radius
        self.period = period
        self.dwell_time = dwell_time
        self.num_vertices = 5
        self.visit_order = [0, 2, 4, 1, 3]  # Jump order: point 1→3→5→2→4
        self.vertex_positions = self._compute_vertex_positions()
        self.path_positions = [self.vertex_positions[i] for i in self.visit_order]

    def _compute_vertex_positions(self) -> list:
        """
        Compute (x, y) positions of 5 vertices around circle.

        Vertices are evenly spaced at 72° intervals (360°/5).
        Starting from top (90°) going counter-clockwise.
        """
        positions = []
        for i in range(self.num_vertices):
            angle = np.pi / 2 - i * (2 * np.pi / self.num_vertices)  # Start at top, go CCW
            x = self.radius * np.cos(angle)
            y = self.radius * np.sin(angle)
            positions.append((x, y))
        return positions

    def get_position(self, t: float) -> Tuple[float, float]:
        """
        Get (x, y) position in mm at time t (instant jump between corners).

        The target stays at each corner, then instantly jumps to next corner.
        """
        t_in_cycle = t % self.period

        # Determine which corner we're at based on time
        time_per_point = self.period / len(self.path_positions)
        point_idx = int(t_in_cycle / time_per_point)
        point_idx = min(point_idx, len(self.path_positions) - 1)

        # Return the corner position (no interpolation - instant jump)
        return self.path_positions[point_idx]

    def get_velocity(self, t: float) -> Tuple[float, float]:
        """
        Get (vx, vy) velocity in mm/s at time t.

        Velocity is zero when dwelling at a corner, and infinite (approximated as zero)
        during the instant jump. For practical purposes, returns zero velocity.
        """
        # Since we instantly jump between points, velocity is effectively zero
        # (infinite during jump, zero while dwelling)
        return 0.0, 0.0


class StaticPattern(TrajectoryPattern):
    """Static position (no movement)."""

    def __init__(self, x: float = 0.0, y: float = 0.0) -> None:
        """
        Create a static position pattern (ball stays at fixed location).

        Args:
            x: X position in mm
            y: Y position in mm
        """
        self.x = x
        self.y = y

    def get_position(self, t: float) -> Tuple[float, float]:
        """Get (x, y) position in mm (constant regardless of time)."""
        return self.x, self.y

    def get_velocity(self, t: float) -> Tuple[float, float]:
        """Get (vx, vy) velocity in mm/s (always zero)."""
        return 0.0, 0.0


class PatternFactory:
    """Factory for creating trajectory patterns."""

    @staticmethod
    def create(pattern_type: str, **kwargs) -> TrajectoryPattern:
        """
        Create a trajectory pattern.

        Args:
            pattern_type: 'static', 'circle', 'figure8', 'star'
            **kwargs: pattern-specific parameters

        Returns:
            TrajectoryPattern instance

        Example:
            >>> circle = PatternFactory.create('circle', radius=50.0, period=10.0)
            >>> star = PatternFactory.create('star', radius=60.0, period=15.0)
        """
        patterns = {
            'static': StaticPattern,
            'circle': CirclePattern,
            'figure8': FigureEightPattern,
            'star': StarPattern
        }

        if pattern_type not in patterns:
            raise ValueError(f"Unknown pattern type: {pattern_type}. "
                             f"Available: {list(patterns.keys())}")

        return patterns[pattern_type](**kwargs)

    @staticmethod
    def list_patterns():
        """Get list of available pattern types."""
        return ['static', 'circle', 'figure8', 'star']


class CameraModel:
    """
    Camera sensor model with realistic measurement noise characteristics.

    Models camera behavior based on configured camera type (CAMERA_TYPE):
    - Pixy2: Pixel grid quantization + sub-pixel noise
    - ZED: Sub-pixel CNN detection with measured noise characteristics

    Features:
    - Configurable noise model (quantization and/or Gaussian noise)
    - Detection dropout modeling
    - Realistic sample rate limiting
    """

    def __init__(self,
                 pixel_size_mm=None,
                 noise_std_x_mm=None,
                 noise_std_y_mm=None,
                 detection_rate=None,
                 sample_rate_hz=None,
                 camera_type=None):
        """
        Args:
            pixel_size_mm: Physical size of one pixel for quantization (None = no quantization)
            noise_std_x_mm: Gaussian noise std dev in X-axis (mm)
            noise_std_y_mm: Gaussian noise std dev in Y-axis (mm)
            detection_rate: Probability of detecting ball (0.0 to 1.0)
            sample_rate_hz: Camera update rate in Hz (0 = sample every call)
            camera_type: Camera type ('PIXY2' or 'ZED'), uses CAMERA_TYPE from config if None
        """
        # Import here to avoid circular import
        from core.utils import CAMERA_TYPE, Pixy2CameraConfig, ZEDCameraConfig

        # Use configured camera type if not specified
        if camera_type is None:
            camera_type = CAMERA_TYPE

        # Load defaults based on camera type
        if camera_type == 'ZED':
            self.pixel_size = pixel_size_mm if pixel_size_mm is not None else 0.0  # No quantization
            self.noise_std_x = noise_std_x_mm if noise_std_x_mm is not None else ZEDCameraConfig.NOISE_STD_X_MM
            self.noise_std_y = noise_std_y_mm if noise_std_y_mm is not None else ZEDCameraConfig.NOISE_STD_Y_MM
            self.detection_rate = detection_rate if detection_rate is not None else ZEDCameraConfig.DETECTION_RATE
            sample_rate_default = ZEDCameraConfig.DEFAULT_SAMPLE_RATE_HZ
        else:  # PIXY2
            self.pixel_size = pixel_size_mm if pixel_size_mm is not None else Pixy2CameraConfig.PIXEL_SIZE_MM
            self.noise_std_x = noise_std_x_mm if noise_std_x_mm is not None else Pixy2CameraConfig.SUBPIXEL_NOISE_STD_MM
            self.noise_std_y = noise_std_y_mm if noise_std_y_mm is not None else Pixy2CameraConfig.SUBPIXEL_NOISE_STD_MM
            self.detection_rate = detection_rate if detection_rate is not None else Pixy2CameraConfig.DEFAULT_DETECTION_RATE
            sample_rate_default = Pixy2CameraConfig.DEFAULT_SAMPLE_RATE_HZ

        self.camera_type = camera_type

        # Setup sample rate
        if sample_rate_hz is not None:
            self.sample_period = 1.0 / sample_rate_hz if sample_rate_hz > 0 else 0.0
        else:
            self.sample_period = 1.0 / sample_rate_default if sample_rate_default > 0 else 0.0

        # Timing state for sample rate
        self.last_sample_time = -float('inf')  # Force first sample
        self.cached_measurement = (None, None)
        self.cached_detected = False

    def measure(self, true_position_mm, current_time):
        """
        Take a camera measurement with realistic noise and sample rate.

        Args:
            true_position_mm: (x, y) tuple or array of true ball position in mm
            current_time: Current simulation time in seconds

        Returns:
            (x_measured, y_measured, detected, is_new_sample):
                - x_measured, y_measured: Measured position in mm (None if not detected)
                - detected: Boolean, True if ball was detected
                - is_new_sample: Boolean, True if this is a fresh measurement
        """
        # Check if enough time has passed for new sample
        time_since_last = current_time - self.last_sample_time

        if self.sample_period > 0 and time_since_last < self.sample_period:
            # Return cached measurement (camera hasn't updated yet)
            return (self.cached_measurement[0],
                    self.cached_measurement[1],
                    self.cached_detected,
                    False)

        # Time for new measurement
        self.last_sample_time = current_time

        # Detection dropout
        detected = np.random.random() < self.detection_rate

        if not detected:
            self.cached_measurement = (None, None)
            self.cached_detected = False
            return None, None, False, True

        # Add measurement noise (sub-pixel for Pixy2, CNN detection noise for ZED)
        x_true, y_true = true_position_mm
        x_noisy = x_true + np.random.normal(0, self.noise_std_x)
        y_noisy = y_true + np.random.normal(0, self.noise_std_y)

        # Quantize to pixel grid (only if pixel size > 0, i.e., Pixy2)
        if self.pixel_size > 0:
            x_measured = np.round(x_noisy / self.pixel_size) * self.pixel_size
            y_measured = np.round(y_noisy / self.pixel_size) * self.pixel_size
        else:
            # No quantization (ZED camera has sub-pixel accuracy)
            x_measured = x_noisy
            y_measured = y_noisy

        self.cached_measurement = (x_measured, y_measured)
        self.cached_detected = True

        return x_measured, y_measured, True, True

    def measure_batch(self, true_positions_mm, current_time=None):
        """
        Batch measurement for PyTorch tensors (for vectorized simulation).

        Note: Batch mode ignores sample rate timing - all measurements are fresh.
        For sample rate modeling, use single measure() calls.

        Args:
            true_positions_mm: (batch, 2) tensor of true positions
            current_time: Ignored in batch mode

        Returns:
            measured_positions: (batch, 2) tensor
            detected: (batch,) boolean tensor
        """
        batch_size = true_positions_mm.shape[0]

        # Detection mask
        detection_probs = np.random.rand(batch_size)
        detected = detection_probs < self.detection_rate

        # Measurement noise (different for X and Y axes)
        noise_x = np.random.randn(batch_size) * self.noise_std_x
        noise_y = np.random.randn(batch_size) * self.noise_std_y
        noise = np.stack([noise_x, noise_y], axis=1)
        noisy_positions = true_positions_mm + noise

        # Quantize to pixel grid (only if pixel size > 0)
        if self.pixel_size > 0:
            measured = np.round(noisy_positions / self.pixel_size) * self.pixel_size
        else:
            # No quantization (ZED camera has sub-pixel accuracy)
            measured = noisy_positions

        # Set undetected measurements to zero
        measured[~detected] = 0.0

        return measured, detected

    def reset(self) -> None:
        """Reset camera state (timing and cached measurements)."""
        self.last_sample_time = -float('inf')
        self.cached_measurement = (None, None)
        self.cached_detected = False

    def set_sample_rate(self, sample_rate_hz: float) -> None:
        """Update camera sample rate on the fly."""
        if sample_rate_hz > 0:
            self.sample_period = 1.0 / sample_rate_hz
        else:
            self.sample_period = 0.0

    def get_sample_rate(self) -> float:
        """Get current sample rate in Hz."""
        if self.sample_period > 0:
            return 1.0 / self.sample_period
        return float('inf')

    def set_noise_level(self, noise_std: float) -> None:
        """Update noise level on the fly (sets both X and Y to same value)."""
        self.noise_std_x = noise_std
        self.noise_std_y = noise_std

    def get_noise_level(self) -> tuple:
        """Get current noise standard deviations (X, Y)."""
        return (self.noise_std_x, self.noise_std_y)


# Backward compatibility alias
Pixy2Camera = CameraModel