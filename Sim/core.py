#!/usr/bin/env python3
"""
Stewart Platform Core Components

Shared classes for Stewart platform simulation and control:
- FirstOrderServo: Servo dynamics model
- StewartPlatformIK: Inverse and forward kinematics
- SimpleBallPhysics2D: Simplified 2D ball physics with RK4
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
    SIMPLIFIED 2D Ball Physics with RK4 Integration

    Key simplifications:
    1. Ball is always on platform surface (no Z dynamics)
    2. Only XY motion matters
    3. Gravity component projected onto tilted surface
    4. RK4 integration for numerical stability
    5. Simple linear friction
    """

    def __init__(self,
                 ball_radius=0.01,
                 gravity=9.81,
                 friction_coef=0.1):

        self.radius = ball_radius
        self.g = gravity
        self.friction = friction_coef

    def step(self, ball_pos, ball_vel, platform_pose, dt):
        """
        Step physics using RK4 integration.

        Args:
            ball_pos: (batch, 3) - only X and Y matter, Z is computed
            ball_vel: (batch, 3) - only X and Y matter
            platform_pose: (batch, 6) [x, y, z, rx, ry, rz]
            dt: timestep

        Returns:
            new_pos, new_vel, contact_info
        """
        batch_size = ball_pos.shape[0]
        device = ball_pos.device

        # Extract 2D state (X, Y positions and velocities)
        xy_pos = ball_pos[:, :2]  # (batch, 2)
        xy_vel = ball_vel[:, :2]  # (batch, 2)

        # RK4 integration in 2D
        k1_vel, k1_pos = self._compute_derivatives(xy_pos, xy_vel, platform_pose)

        k2_vel, k2_pos = self._compute_derivatives(
            xy_pos + 0.5 * dt * k1_pos,
            xy_vel + 0.5 * dt * k1_vel,
            platform_pose
        )

        k3_vel, k3_pos = self._compute_derivatives(
            xy_pos + 0.5 * dt * k2_pos,
            xy_vel + 0.5 * dt * k2_vel,
            platform_pose
        )

        k4_vel, k4_pos = self._compute_derivatives(
            xy_pos + dt * k3_pos,
            xy_vel + dt * k3_vel,
            platform_pose
        )

        # Combine RK4 stages
        new_xy_pos = xy_pos + (dt / 6.0) * (k1_pos + 2 * k2_pos + 2 * k3_pos + k4_pos)
        new_xy_vel = xy_vel + (dt / 6.0) * (k1_vel + 2 * k2_vel + 2 * k3_vel + k4_vel)

        # Check boundary (square platform ±100mm)
        max_xy = 0.1  # 100mm = 0.1 meters
        fell_off = (torch.abs(new_xy_pos[:, 0]) > max_xy) | (torch.abs(new_xy_pos[:, 1]) > max_xy)

        contact_info = {'fell_off': False}

        if fell_off.any():
            # Reset to center
            for i in range(batch_size):
                if fell_off[i]:
                    new_xy_pos[i] = 0.0
                    new_xy_vel[i] = 0.0
            contact_info['fell_off'] = True

        # Compute Z position (always on surface)
        platform_z = self._compute_platform_height(new_xy_pos, platform_pose)

        # Reconstruct 3D position
        new_ball_pos = torch.zeros((batch_size, 3), device=device)
        new_ball_pos[:, :2] = new_xy_pos
        new_ball_pos[:, 2] = platform_z + self.radius

        # Reconstruct 3D velocity (Z velocity is zero)
        new_ball_vel = torch.zeros((batch_size, 3), device=device)
        new_ball_vel[:, :2] = new_xy_vel

        # No angular velocity in this simplified model
        new_ball_omega = torch.zeros((batch_size, 3), device=device)

        contact_info['in_contact'] = torch.ones(batch_size, dtype=torch.bool)

        return new_ball_pos, new_ball_vel, new_ball_omega, contact_info

    def _compute_derivatives(self, xy_pos, xy_vel, platform_pose):
        """
        Compute derivatives for RK4.

        Returns:
            d(velocity)/dt, d(position)/dt
        """
        # Position derivative is just velocity
        d_pos = xy_vel

        # Velocity derivative is acceleration from gravity and friction
        d_vel = self._compute_acceleration(xy_pos, xy_vel, platform_pose)

        return d_vel, d_pos

    def _compute_acceleration(self, xy_pos, xy_vel, platform_pose):
        """
        Compute 2D acceleration on tilted platform.

        For a tilted plane:
        - Pitch (ry) tilts platform in X direction
        - Roll (rx) tilts platform in Y direction

        Gravity component down the slope:
        - ax = -g * sin(pitch)
        - ay = -g * sin(roll)
        """
        batch_size = xy_pos.shape[0]
        device = xy_pos.device

        # Extract rotation angles (degrees -> radians)
        rx = torch.deg2rad(platform_pose[:, 3])  # roll
        ry = torch.deg2rad(platform_pose[:, 4])  # pitch

        # Gravity component projected onto slope
        # Positive pitch -> ball rolls backward (negative x)
        # Positive roll -> ball rolls left (negative y)
        ax = -self.g * torch.sin(ry)
        ay = -self.g * torch.sin(rx)

        accel_gravity = torch.stack([ax, ay], dim=1)

        # Friction opposes motion (simple linear friction)
        vel_magnitude = torch.norm(xy_vel, dim=1, keepdim=True)
        friction_force = -self.friction * xy_vel / (vel_magnitude + 1e-8)

        # Total acceleration
        total_accel = accel_gravity + friction_force

        return total_accel

    def _compute_platform_height(self, xy_pos, platform_pose):
        """
        Compute platform Z height at given XY position.
        """
        # Platform center
        px = platform_pose[:, 0] / 1000  # mm to m
        py = platform_pose[:, 1] / 1000
        pz = platform_pose[:, 2] / 1000

        # Rotation angles
        rx = torch.deg2rad(platform_pose[:, 3])
        ry = torch.deg2rad(platform_pose[:, 4])

        # Distance from platform center
        dx = xy_pos[:, 0] - px
        dy = xy_pos[:, 1] - py

        # Height change due to tilt (small angle approximation)
        height = pz + dx * torch.tan(ry) - dy * torch.tan(rx)

        return height