#!/usr/bin/env python3
"""
Stewart Platform Core Components

Shared classes for Stewart platform simulation and control:
- FirstOrderServo: Servo dynamics model
- StewartPlatformIK: Inverse and forward kinematics
- SimpleBallPhysics2D: 2D ball physics with rolling motion
"""

import numpy as np
import torch
from collections import deque

from core.utils import MAX_SERVO_ANGLE_DEG, PLATFORM_HALF_SIZE_MM


class FirstOrderServo:
    """First-order servo model with command delay."""

    def __init__(self, K=1.0, tau=0.1, delay=0.0):
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
        angles = np.array([-np.pi / 2, np.pi / 2])
        xy = np.zeros((6, 3))
        for i in range(len(phi)):
            for j in range(len(angles)):
                x = l * np.cos(phi[i]) + d * np.cos(phi[i] + angles[j])
                y = l * np.sin(phi[i]) + d * np.sin(phi[i] + angles[j])
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

            if abs(angles[k]) > MAX_SERVO_ANGLE_DEG:
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

            if np.linalg.norm(translation[:2]) > PLATFORM_HALF_SIZE_MM:
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
    2D Ball Physics with rolling motion.

    Physics model:
    - Ball rolls (not slides) on tilted surface
    - Rolling constraint: v = omega × r
    - Acceleration: a = (g * sin(theta)) / mass_factor
      where mass_factor = 1 + I/(m*r²)
    - Rolling resistance for energy dissipation
    """

    def __init__(self,
                 ball_radius=0.01,
                 ball_mass=0.0027,
                 gravity=9.81,
                 rolling_friction=0.01,
                 sphere_type='hollow',
                 air_density=1.225,
                 drag_coefficient=0.47):
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

    def update_sphere_type(self, sphere_type):
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

        max_xy = PLATFORM_HALF_SIZE_MM / 1000.0
        fell_off = (torch.abs(new_xy_pos[:, 0]) > max_xy) | (torch.abs(new_xy_pos[:, 1]) > max_xy)

        contact_info = {'fell_off': False}

        if fell_off.any():
            for i in range(batch_size):
                if fell_off[i]:
                    new_xy_pos[i] = 0.0
                    new_xy_vel[i] = 0.0
                    new_xy_omega[i] = 0.0
            contact_info['fell_off'] = True

        platform_z = self._compute_platform_height(new_xy_pos, platform_pose)

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

        rx = torch.deg2rad(platform_pose[:, 3])
        ry = torch.deg2rad(platform_pose[:, 4])

        ball_x = xy_pos[:, 0]
        ball_y = xy_pos[:, 1]

        g_eff = self.g

        if platform_angular_accel is not None:
            alpha_rx_rad = platform_angular_accel['rx'] * (np.pi / 180.0)
            alpha_ry_rad = platform_angular_accel['ry'] * (np.pi / 180.0)

            a_z_platform = ball_x * alpha_ry_rad - ball_y * alpha_rx_rad
            g_eff = self.g - a_z_platform
            g_eff = torch.clamp(g_eff, 0.1 * self.g, 2.0 * self.g)

        cos_rx = torch.cos(rx)
        cos_ry = torch.cos(ry)
        sin_rx = torch.sin(rx)
        sin_ry = torch.sin(ry)

        # Gravity vector in tilted frame
        gx = g_eff * sin_ry * cos_ry * cos_rx
        gy = -g_eff * sin_rx * cos_rx * cos_ry

        ax = gx / self.mass_factor
        ay = gy / self.mass_factor

        accel_linear = torch.stack([ax, ay], dim=1)

        vel_magnitude = torch.norm(xy_vel, dim=1, keepdim=True)
        rolling_resistance = -self.mu_roll * g_eff * xy_vel / (vel_magnitude + 1e-8)

        # Air resistance (quadratic drag)
        # F_drag = -1/2 * ρ * C_d * A * v² * v_hat
        # a_drag = F_drag / m
        vel_squared = vel_magnitude ** 2
        drag_magnitude = 0.5 * self.air_density * self.drag_coefficient * self.cross_section_area * vel_squared / self.mass
        air_resistance = -drag_magnitude * xy_vel / (vel_magnitude + 1e-8)

        accel_linear = accel_linear + rolling_resistance + air_resistance

        accel_angular = accel_linear / self.radius

        omega_damping = -self.mu_roll * xy_omega
        accel_angular = accel_angular + omega_damping

        return accel_linear, accel_angular

    def _compute_platform_height(self, xy_pos, platform_pose):
        """Compute platform Z height at given XY position."""
        px = platform_pose[:, 0] / 1000
        py = platform_pose[:, 1] / 1000
        pz = platform_pose[:, 2] / 1000

        rx = torch.deg2rad(platform_pose[:, 3])
        ry = torch.deg2rad(platform_pose[:, 4])

        dx = xy_pos[:, 0] - px
        dy = xy_pos[:, 1] - py

        height = pz - dx * torch.tan(ry) - dy * torch.tan(rx)

        return height

    def set_air_resistance(self, air_density=None, drag_coefficient=None):
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
        s + (dt / 6.0) * (k1_i + 2 * k2_i + 2 * k3_i + 4 * k4_i)
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

    def __init__(self, radius=50.0, period=10.0, clockwise=True):
        self.radius = radius
        self.period = period
        self.omega = 2 * np.pi / period
        self.direction = -1 if clockwise else 1

    def get_position(self, t):
        angle = self.direction * self.omega * t
        x = self.radius * np.cos(angle)
        y = self.radius * np.sin(angle)
        return x, y

    def get_velocity(self, t):
        angle = self.direction * self.omega * t
        vx = -self.direction * self.radius * self.omega * np.sin(angle)
        vy = self.direction * self.radius * self.omega * np.cos(angle)
        return vx, vy


class FigureEightPattern(TrajectoryPattern):
    """Figure-8 (lemniscate) trajectory pattern."""

    def __init__(self, width=60.0, height=40.0, period=12.0):
        self.a = width / 2.0
        self.b = height / 2.0
        self.omega = 2 * np.pi / period

    def get_position(self, t):
        angle = self.omega * t
        x = self.a * np.cos(angle)
        y = self.b * np.sin(angle) * np.cos(angle)
        return x, y

    def get_velocity(self, t):
        angle = self.omega * t
        vx = -self.a * self.omega * np.sin(angle)
        vy = self.b * self.omega * np.cos(2 * angle)
        return vx, vy


class StarPattern(TrajectoryPattern):
    """Five-pointed star trajectory pattern."""

    def __init__(self, radius=60.0, period=15.0):
        self.radius = radius
        self.period = period
        self.num_vertices = 5
        self.visit_order = [0, 2, 4, 1, 3, 0]
        self.vertex_positions = self._compute_vertex_positions()
        self.path_positions = [self.vertex_positions[i] for i in self.visit_order]

    def _compute_vertex_positions(self):
        """Compute (x, y) positions of 5 vertices around circle."""
        positions = []
        for i in range(self.num_vertices):
            angle = np.pi / 2 - i * (2 * np.pi / self.num_vertices)
            x = self.radius * np.cos(angle)
            y = self.radius * np.sin(angle)
            positions.append((x, y))
        return positions

    def get_position(self, t):
        t_norm = (t % self.period) / self.period
        num_segments = len(self.path_positions) - 1
        segment_idx = min(int(t_norm * num_segments), num_segments - 1)
        segment_progress = (t_norm * num_segments) % 1.0

        current_pos = self.path_positions[segment_idx]
        next_pos = self.path_positions[segment_idx + 1]

        x = current_pos[0] + segment_progress * (next_pos[0] - current_pos[0])
        y = current_pos[1] + segment_progress * (next_pos[1] - current_pos[1])

        return x, y

    def get_velocity(self, t):
        t_norm = (t % self.period) / self.period
        num_segments = len(self.path_positions) - 1
        segment_idx = min(int(t_norm * num_segments), num_segments - 1)

        current_pos = self.path_positions[segment_idx]
        next_pos = self.path_positions[segment_idx + 1]

        dx = next_pos[0] - current_pos[0]
        dy = next_pos[1] - current_pos[1]

        segment_time = self.period / num_segments

        vx = dx / segment_time
        vy = dy / segment_time

        return vx, vy


class StaticPattern(TrajectoryPattern):
    """Static position (no movement)."""

    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y

    def get_position(self, t):
        return self.x, self.y

    def get_velocity(self, t):
        return 0.0, 0.0


class PatternFactory:
    """Factory for creating trajectory patterns."""

    @staticmethod
    def create(pattern_type, **kwargs):
        """
        Create a trajectory pattern.

        Args:
            pattern_type: 'static', 'circle', 'figure8', 'star'
            **kwargs: pattern-specific parameters

        Returns:
            TrajectoryPattern instance
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