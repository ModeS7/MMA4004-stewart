#!/usr/bin/env python3
"""
Stewart Platform Core Components

Shared classes for Stewart platform simulation and control:
- FirstOrderServo: Servo dynamics model
- StewartPlatformIK: Inverse and forward kinematics
- SimpleBallPhysics2D: 2D ball physics with ROLLING
"""

import numpy as np
import torch
from collections import deque


class FirstOrderServo:
    """First-order servo model with command delay."""

    def __init__(self, K=1.0, tau=0.1, delay=0.35):
        self.K = K
        self.tau = tau
        self.delay = delay
        self.current_angle = 0.0
        self.target_angle = 0.0
        self.command_queue = deque()

    def send_command(self, angle, current_time):
        delivery_time = current_time + self.delay
        self.command_queue.append((delivery_time, angle))

    def update(self, dt, current_time):
        while self.command_queue and self.command_queue[0][0] <= current_time:
            _, angle = self.command_queue.popleft()
            self.target_angle = angle

        error = self.target_angle - self.current_angle
        d_angle = (self.K * error / self.tau) * dt
        self.current_angle += d_angle

    def get_angle(self):
        return self.current_angle

    def reset(self):
        self.current_angle = 0.0
        self.target_angle = 0.0
        self.command_queue.clear()


class StewartPlatformIK:
    """Stewart Platform inverse and forward kinematics."""

    def __init__(self, horn_length=31.75, rod_length=145.0, base=73.025,
                 base_anchors=36.8893, platform=67.775, platform_anchors=12.7,
                 top_surface_offset=26.0):
        self.horn_length = horn_length
        self.rod_length = rod_length
        self.base = base
        self.base_anchors = base_anchors
        self.platform = platform
        self.platform_anchors = platform_anchors
        self.top_surface_offset = top_surface_offset

        base_angels = np.array([-np.pi / 2, np.pi / 6, np.pi * 5 / 6])
        platform_angels = np.array([-np.pi * 5 / 6, -np.pi / 6, np.pi / 2])

        self.base_anchors = self.calculate_home_coordinates(self.base, self.base_anchors, base_angels)
        platform_anchors_out_of_fase = self.calculate_home_coordinates(self.platform, self.platform_anchors,
                                                                       platform_angels)
        self.platform_anchors = np.roll(platform_anchors_out_of_fase, shift=-1, axis=0)

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
        angels = np.array([-np.pi / 2, np.pi / 2])
        xy = np.zeros((6, 3))
        for i in range(len(phi)):
            for j in range(len(angels)):
                x = l * np.cos(phi[i]) + d * np.cos(phi[i] + angels[j])
                y = l * np.sin(phi[i]) + d * np.sin(phi[i] + angels[j])
                xy[i * 2 + j] = np.array([x, y, 0])
        return xy

    def _calculate_beta_angles(self):
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

    def update_offset(self, new_offset):
        self.top_surface_offset = new_offset
        self.home_height_top_surface = self.home_height + self.top_surface_offset

    def calculate_servo_angles(self, translation: np.ndarray, rotation: np.ndarray,
                               use_top_surface_offset: bool = True):
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
            if sqrt_term < 1e-6:
                return None

            ratio = g_k / np.sqrt(sqrt_term)
            if abs(ratio) > 1.0:
                return None

            alpha_k = np.arcsin(ratio) - np.arctan2(f_k, e_k)
            angles[k] = np.degrees(alpha_k)

            if abs(angles[k]) > 40:
                return None

        return -angles

    def calculate_forward_kinematics(self, servo_angles: np.ndarray,
                                     initial_guess: tuple = None,
                                     use_top_surface_offset: bool = True,
                                     max_iterations: int = 50,
                                     tolerance: float = 1e-6):
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
        damping = 0.01

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

            delta = 0.01
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
                if damping > 1.0:
                    return None, None, False, iteration
                continue

            translation += pose_update[:3]
            rotation += pose_update[3:]

            if np.linalg.norm(translation[:2]) > 100:
                return None, None, False, iteration
            if translation[2] < 0 or translation[2] > 300:
                return None, None, False, iteration
            if np.any(np.abs(rotation) > 45):
                return None, None, False, iteration

        return None, None, False, max_iterations

    def _euler_to_quaternion(self, euler: np.ndarray) -> np.ndarray:
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
        w, x, y, z = q
        vx, vy, vz = v
        return np.array([
            vx * (w * w + x * x - y * y - z * z) + vy * (2 * x * y - 2 * w * z) + vz * (2 * x * z + 2 * w * y),
            vx * (2 * x * y + 2 * w * z) + vy * (w * w - x * x + y * y - z * z) + vz * (2 * y * z - 2 * w * x),
            vx * (2 * x * z - 2 * w * y) + vy * (2 * y * z + 2 * w * x) + vz * (w * w - x * x - y * y + z * z)
        ])


class SimpleBallPhysics2D:
    """
    2D Ball Physics with PROPER ROLLING (not sliding)

    Physics model:
    - Ball rolls (not slides) on tilted surface
    - Angular velocity (omega) tracked and coupled to linear velocity
    - Rolling constraint: v = omega × r
    - For solid sphere rolling down slope: a = (5/7) * g * sin(theta)
    - For hollow sphere rolling down slope: a = (3/5) * g * sin(theta)
    - Rolling resistance for realistic energy dissipation
    """

    def __init__(self,
                 ball_radius=0.01,
                 ball_mass=0.0027,
                 gravity=9.81,
                 rolling_friction=0.01,
                 sphere_type='hollow'):  # ← Add this parameter

        self.radius = ball_radius
        self.mass = ball_mass
        self.g = gravity
        self.mu_roll = rolling_friction
        self.sphere_type = sphere_type

        # Calculate moment of inertia based on sphere type
        self.update_sphere_type(sphere_type)

    def update_sphere_type(self, sphere_type):
        """Update moment of inertia and mass factor based on sphere type."""
        self.sphere_type = sphere_type

        if sphere_type == 'solid':
            # Solid sphere: I = (2/5) * m * r²
            self.I = (2.0 / 5.0) * self.mass * self.radius ** 2
        elif sphere_type == 'hollow':
            # Hollow sphere (thin shell): I = (2/3) * m * r²
            self.I = (2.0 / 3.0) * self.mass * self.radius ** 2
        else:
            raise ValueError(f"Unknown sphere_type: {sphere_type}. Use 'solid' or 'hollow'")

        # For rolling motion, effective mass factor
        # a = F / m_eff, where m_eff = m * (1 + I/(m*r²))
        # Solid: 1 + 2/5 = 7/5 = 1.4
        # Hollow: 1 + 2/3 = 5/3 ≈ 1.667
        self.mass_factor = 1.0 + self.I / (self.mass * self.radius ** 2)

    def step(self, ball_pos, ball_vel, ball_omega, platform_pose, dt, platform_angular_accel=None):
        """
        Step physics using RK4 integration with ROLLING.

        Args:
            ball_pos: (batch, 3) - only X and Y matter, Z is computed
            ball_vel: (batch, 3) - only X and Y matter
            ball_omega: (batch, 3) - angular velocity [wx, wy, wz] (rad/s)
            platform_pose: (batch, 6) [x, y, z, rx, ry, rz]
            dt: timestep

        Returns:
            new_pos, new_vel, new_omega, contact_info
        """
        batch_size = ball_pos.shape[0]
        device = ball_pos.device

        # Extract 2D state
        xy_pos = ball_pos[:, :2]
        xy_vel = ball_vel[:, :2]
        xy_omega = ball_omega[:, :2]  # Only X and Y components matter for 2D rolling

        # Use generic RK4 integration
        state = (xy_pos, xy_vel, xy_omega)
        new_xy_pos, new_xy_vel, new_xy_omega = rk4_step(
            state,
            self._compute_derivatives,
            dt,
            platform_pose
        )

        # Check boundary (square platform ±100mm)
        max_xy = 0.1  # 100mm = 0.1 meters
        fell_off = (torch.abs(new_xy_pos[:, 0]) > max_xy) | (torch.abs(new_xy_pos[:, 1]) > max_xy)

        contact_info = {'fell_off': False}

        if fell_off.any():
            for i in range(batch_size):
                if fell_off[i]:
                    new_xy_pos[i] = 0.0
                    new_xy_vel[i] = 0.0
                    new_xy_omega[i] = 0.0
            contact_info['fell_off'] = True

        # Compute Z position
        platform_z = self._compute_platform_height(new_xy_pos, platform_pose)

        # Reconstruct 3D vectors
        new_ball_pos = torch.zeros((batch_size, 3), device=device)
        new_ball_pos[:, :2] = new_xy_pos
        new_ball_pos[:, 2] = platform_z + self.radius

        new_ball_vel = torch.zeros((batch_size, 3), device=device)
        new_ball_vel[:, :2] = new_xy_vel

        new_ball_omega = torch.zeros((batch_size, 3), device=device)
        new_ball_omega[:, :2] = new_xy_omega

        contact_info['in_contact'] = torch.ones(batch_size, dtype=torch.bool)
        contact_info['rolling_speed'] = torch.norm(new_xy_omega, dim=1) * self.radius

        return new_ball_pos, new_ball_vel, new_ball_omega, contact_info

    def _compute_derivatives(self, state, platform_pose):
        """
        Compute derivatives for RK4 with ROLLING physics.

        Args:
            state: tuple of (xy_pos, xy_vel, xy_omega)
            platform_pose: platform pose tensor

        Returns:
            tuple of (d_pos, d_vel, d_omega)
        """
        xy_pos, xy_vel, xy_omega = state

        # Position derivative is velocity
        d_pos = xy_vel

        # Compute accelerations (linear and angular)
        d_vel, d_omega = self._compute_accelerations(xy_pos, xy_vel, xy_omega, platform_pose)

        return d_pos, d_vel, d_omega

    def _compute_accelerations(self, xy_pos, xy_vel, xy_omega, platform_pose, platform_angular_accel=None):
        """
        Compute accelerations for a ROLLING ball on tilted surface.

        Accounts for:
        - Gravitational component down the slope
        - Vertical acceleration of platform at ball position (changes effective gravity)
        - Rolling dynamics (moment of inertia)
        """
        batch_size = xy_pos.shape[0]
        device = xy_pos.device

        # Extract rotation angles
        rx = torch.deg2rad(platform_pose[:, 3])  # roll
        ry = torch.deg2rad(platform_pose[:, 4])  # pitch

        # Ball position relative to platform center (in meters)
        ball_x = xy_pos[:, 0]
        ball_y = xy_pos[:, 1]

        # Compute vertical acceleration of platform at ball's position
        # This comes from platform_angular_accel if we track it
        # For now, we'll use a simplified model based on current angles
        # (assumes quasi-static: angular velocity ≈ 0)

        # Compute effective gravity accounting for platform vertical acceleration
        g_eff = self.g

        if platform_angular_accel is not None:
            # Vertical acceleration at ball position due to platform rotation
            # a_z = x·(d²ry/dt²) - y·(d²rx/dt²)
            # Convert angular accel from deg/s² to rad/s²
            alpha_rx_rad = platform_angular_accel['rx'] * (np.pi / 180.0)
            alpha_ry_rad = platform_angular_accel['ry'] * (np.pi / 180.0)

            # Vertical acceleration at ball position (simplified, assuming small angles)
            a_z_platform = ball_x * alpha_ry_rad - ball_y * alpha_rx_rad

            # Modify effective gravity
            g_eff = self.g - a_z_platform

            # Clamp to reasonable range (platform can't create negative gravity)
            g_eff = torch.clamp(g_eff, 0.1 * self.g, 2.0 * self.g)

        # Rest of function uses g_eff instead of self.g
        gx = -g_eff * torch.sin(ry)
        gy = -g_eff * torch.sin(rx)

        # For rolling motion, acceleration is reduced by rotational inertia
        ax = gx / self.mass_factor
        ay = gy / self.mass_factor

        accel_linear = torch.stack([ax, ay], dim=1)

        # Rolling resistance (opposes motion)
        vel_magnitude = torch.norm(xy_vel, dim=1, keepdim=True)
        rolling_resistance = -self.mu_roll * g_eff * xy_vel / (vel_magnitude + 1e-8)

        accel_linear = accel_linear + rolling_resistance

        # Angular acceleration from rolling constraint
        accel_angular = accel_linear / self.radius

        # Additional damping on angular velocity
        omega_damping = -self.mu_roll * xy_omega
        accel_angular = accel_angular + omega_damping

        return accel_linear, accel_angular

    def _compute_platform_height(self, xy_pos, platform_pose):
        """Compute platform Z height at given XY position."""
        px = platform_pose[:, 0] / 1000  # mm to m
        py = platform_pose[:, 1] / 1000
        pz = platform_pose[:, 2] / 1000

        rx = torch.deg2rad(platform_pose[:, 3])
        ry = torch.deg2rad(platform_pose[:, 4])

        dx = xy_pos[:, 0] - px
        dy = xy_pos[:, 1] - py

        height = pz + dx * torch.tan(ry) - dy * torch.tan(rx)

        return height


def rk4_step(state, derivative_fn, dt, *args):
    """
    Generic RK4 (Runge-Kutta 4th order) integration step.

    Args:
        state: tuple of tensors representing the system state
        derivative_fn: function that computes derivatives, signature: (state, *args) -> derivatives
        dt: timestep
        *args: additional arguments passed to derivative_fn

    Returns:
        tuple of new state tensors
    """
    # k1 = f(t, y)
    k1 = derivative_fn(state, *args)

    # k2 = f(t + dt/2, y + k1*dt/2)
    state_k2 = tuple(s + 0.5 * dt * k for s, k in zip(state, k1))
    k2 = derivative_fn(state_k2, *args)

    # k3 = f(t + dt/2, y + k2*dt/2)
    state_k3 = tuple(s + 0.5 * dt * k for s, k in zip(state, k2))
    k3 = derivative_fn(state_k3, *args)

    # k4 = f(t + dt, y + k3*dt)
    state_k4 = tuple(s + dt * k for s, k in zip(state, k3))
    k4 = derivative_fn(state_k4, *args)

    # y_new = y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    new_state = tuple(
        s + (dt / 6.0) * (k1_i + 2 * k2_i + 2 * k3_i + 4 * k4_i)
        for s, k1_i, k2_i, k3_i, k4_i in zip(state, k1, k2, k3, k4)
    )

    return new_state