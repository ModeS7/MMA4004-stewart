"""
Stewart Platform RL Environment
"""

import numpy as np
import torch
from typing import Tuple, Dict, Any, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.core import SimpleBallPhysics2D, FirstOrderServo, StewartPlatformIK
from core.utils import (
    SimulationConfig, StewartPlatformConfig, BallPhysicsConfig,
    PLATFORM_RADIUS_MM, MAX_CONTROLLER_OUTPUT_DEG
)
from RL.rl_config import EnvConfig, RewardConfig


class StewartEnv:
    """
    Stewart platform environment for RL training.
    Single environment only (no vectorization).
    """

    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        reward_config: Optional[RewardConfig] = None,
        device: str = "cuda"
    ):
        self.cfg = config or EnvConfig()
        self.reward_cfg = reward_config or RewardConfig()
        self.device = torch.device(device)

        # Platform parameters
        self.platform_radius_mm = PLATFORM_RADIUS_MM
        self.platform_radius_m = PLATFORM_RADIUS_MM / 1000.0
        self.max_tilt_deg = MAX_CONTROLLER_OUTPUT_DEG

        # Initialize Stewart Platform IK
        platform_params = StewartPlatformConfig.as_dict()
        self.ik = StewartPlatformIK(**platform_params)

        # Initialize 6 servos
        self.servos = [
            FirstOrderServo(
                K=1.0,
                tau=SimulationConfig.DEFAULT_SERVO_TAU,
                delay=SimulationConfig.DEFAULT_SERVO_DELAY,
                max_velocity=SimulationConfig.DEFAULT_SERVO_MAX_VELOCITY
            )
            for _ in range(6)
        ]

        # Initialize ball physics
        self.ball_physics = SimpleBallPhysics2D(**BallPhysicsConfig.for_physics_sim())

        # State variables
        self.ball_pos = np.zeros((1, 3), dtype=np.float32)  # [x, y, z] in meters
        self.ball_vel = np.zeros((1, 3), dtype=np.float32)
        self.ball_omega = np.zeros((1, 3), dtype=np.float32)

        # Platform state
        self.platform_rx = 0.0  # degrees
        self.platform_ry = 0.0  # degrees
        self.home_z = self.ik.home_height_top_surface

        # For FK warm-start
        self.last_fk_translation = None
        self.last_fk_rotation = None

        # Platform angular acceleration (for advanced physics)
        self.prev_platform_angles = {'rx': 0.0, 'ry': 0.0}
        self.platform_angular_vel = {'rx': 0.0, 'ry': 0.0}
        self.platform_angular_accel = {'rx': 0.0, 'ry': 0.0}

        # Gravity offset (simulates unlevel surface / gravity vector misalignment)
        self.gravity_offset_rx = 0.0  # degrees
        self.gravity_offset_ry = 0.0  # degrees
        self.gravity_offset_max = self.cfg.platform_offset_max_deg

        # Simulation time
        self.sim_time = 0.0
        self.dt = self.cfg.dt
        self._base_dt = self.dt     # Base dt for this episode (randomized per episode)
        self._current_dt = self.dt  # Current step dt (may vary with randomization)
        self.step_count = 0

        # Frame history for observations
        self.num_frames = self.cfg.num_frames
        self.obs_per_frame = self.cfg.obs_per_frame
        self.frame_history = np.zeros((self.num_frames, self.obs_per_frame), dtype=np.float32)

        # Target position (randomized on reset if enabled)
        self.target_x = 0.0  # mm
        self.target_y = 0.0  # mm

        # Reward scales
        self.dist_scale = self.reward_cfg.dist_scale
        self.base_speed_scale = self.reward_cfg.base_speed_scale
        self.center_bonus_radius = self.reward_cfg.center_bonus_radius
        self.center_bonus_max = self.reward_cfg.center_bonus_max
        self.center_bonus_speed_scale = self.reward_cfg.center_bonus_speed_scale
        self.approach_scale = self.reward_cfg.approach_scale
        self.fall_penalty = self.reward_cfg.fall_penalty

        # Physics ground truth for auxiliary loss
        self.physics_gt = np.array([
            self.ball_physics.mu_roll,
            SimulationConfig.DEFAULT_SERVO_TAU,
            self.ball_physics.mass_factor
        ], dtype=np.float32)

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to random initial state."""
        # Reset simulation time
        self.sim_time = 0.0
        self.step_count = 0

        # Reset servos
        for servo in self.servos:
            servo.reset()

        # Reset platform state
        self.platform_rx = 0.0
        self.platform_ry = 0.0
        self.prev_platform_angles = {'rx': 0.0, 'ry': 0.0}
        self.platform_angular_vel = {'rx': 0.0, 'ry': 0.0}
        self.platform_angular_accel = {'rx': 0.0, 'ry': 0.0}

        # Randomize gravity offset (random direction with random magnitude)
        if self.cfg.use_platform_offset:
            magnitude = np.random.uniform(0, self.gravity_offset_max)
            direction = np.random.uniform(0, 2 * np.pi)
            self.gravity_offset_rx = magnitude * np.sin(direction)
            self.gravity_offset_ry = magnitude * np.cos(direction)
        else:
            self.gravity_offset_rx = 0.0
            self.gravity_offset_ry = 0.0

        # Domain randomization: randomize physics parameters
        if self.cfg.use_domain_randomization:
            # Randomize rolling friction
            friction = np.random.uniform(*self.cfg.friction_range)
            self.ball_physics.mu_roll = friction

            # Randomize servo time constant
            servo_tau = np.random.uniform(*self.cfg.servo_tau_range)
            for servo in self.servos:
                servo.tau = servo_tau

            # Randomize ball mass factor (effective inertia)
            mass_factor = np.random.uniform(*self.cfg.mass_factor_range)
            self.ball_physics.mass_factor = mass_factor

            # Update physics ground truth
            self.physics_gt = np.array([friction, servo_tau, mass_factor], dtype=np.float32)

        # Randomize base dt for this episode
        if self.cfg.use_dt_randomization:
            self._base_dt = np.random.uniform(*self.cfg.dt_range)
        else:
            self._base_dt = self.dt

        # Randomize target position within specified radius
        if self.cfg.randomize_target:
            target_radius_mm = self.cfg.target_range_mm
            r = np.sqrt(np.random.random()) * target_radius_mm
            theta = np.random.random() * 2 * np.pi
            self.target_x = r * np.cos(theta)
            self.target_y = r * np.sin(theta)
        else:
            self.target_x = 0.0
            self.target_y = 0.0

        # Random initial ball position (uniform in circle)
        init_radius_m = self.cfg.init_pos_range_mm / 1000.0
        r = np.sqrt(np.random.random()) * init_radius_m
        theta = np.random.random() * 2 * np.pi
        init_x = r * np.cos(theta)
        init_y = r * np.sin(theta)

        # Random initial velocity
        init_vel_m_s = self.cfg.init_vel_range_mm_s / 1000.0
        init_vx = np.random.uniform(-init_vel_m_s, init_vel_m_s)
        init_vy = np.random.uniform(-init_vel_m_s, init_vel_m_s)

        # Set ball state
        ball_z = (self.home_z / 1000.0) + self.ball_physics.radius
        self.ball_pos = np.array([[init_x, init_y, ball_z]], dtype=np.float32)
        self.ball_vel = np.array([[init_vx, init_vy, 0.0]], dtype=np.float32)
        self.ball_omega = np.zeros((1, 3), dtype=np.float32)

        # Reset FK warm-start
        self.last_fk_translation = np.array([0.0, 0.0, self.home_z])
        self.last_fk_rotation = np.array([0.0, 0.0, 0.0])

        # Build initial observation frame
        initial_frame = self._get_single_frame()

        # Fill ALL frames with initial observation (same as env_gpu.py)
        for i in range(self.num_frames):
            self.frame_history[i] = initial_frame

        obs = self.frame_history.flatten()

        info = {
            'physics_gt': self.physics_gt.copy(),
            'target_pos_mm': (self.target_x, self.target_y)
        }

        return obs, info

    def _get_single_frame(self) -> np.ndarray:
        """Get single observation frame (7 values)."""
        ball_x_mm = self.ball_pos[0, 0] * 1000.0
        ball_y_mm = self.ball_pos[0, 1] * 1000.0

        # Add camera noise to ball position if enabled
        if self.cfg.use_camera_noise:
            noise_std = self.cfg.position_noise_std_mm
            ball_x_mm += np.random.normal(0, noise_std)
            ball_y_mm += np.random.normal(0, noise_std)

        frame = np.array([
            ball_x_mm / self.platform_radius_mm,           # ball_x normalized
            ball_y_mm / self.platform_radius_mm,           # ball_y normalized
            self.platform_rx / self.max_tilt_deg,          # platform_rx normalized
            self.platform_ry / self.max_tilt_deg,          # platform_ry normalized
            self._current_dt / 0.01,                       # dt normalized (varies with randomization)
            self.target_x / self.platform_radius_mm,       # target_x normalized
            self.target_y / self.platform_radius_mm,       # target_y normalized
        ], dtype=np.float32)

        return frame

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Step the simulation.

        Args:
            action: [rx_target, ry_target] normalized to [-1, 1]

        Returns:
            obs, reward, done, truncated, info
        """
        # Randomize dt if enabled (simulates variable control loop timing)
        if self.cfg.use_dt_randomization:
            # Start with episode's base dt
            step_dt = self._base_dt
            # Add small per-step noise
            step_dt += np.random.normal(0, self.cfg.dt_noise_std)
            # Occasional large jitter
            if np.random.random() < self.cfg.dt_jitter_prob:
                step_dt += self.cfg.dt_jitter_ms / 1000.0
            step_dt = max(0.001, step_dt)  # Ensure positive
        else:
            step_dt = self.dt

        # Scale action to degrees
        rx_target = float(action[0]) * self.max_tilt_deg
        ry_target = float(action[1]) * self.max_tilt_deg

        # Compute IK to get servo angles
        translation = np.array([0.0, 0.0, self.home_z])
        rotation = np.array([rx_target, ry_target, 0.0])

        servo_angles = self.ik.calculate_servo_angles(
            translation, rotation, use_top_surface_offset=True
        )

        if servo_angles is not None:
            # Send commands to all 6 servos
            for i, servo in enumerate(self.servos):
                servo.send_command(servo_angles[i], self.sim_time)

        # Update servo dynamics
        for servo in self.servos:
            servo.update(step_dt, self.sim_time)

        # Get actual servo angles
        actual_angles = np.array([servo.get_angle() for servo in self.servos])

        # Compute forward kinematics to get actual platform pose
        initial_guess = None
        if self.last_fk_translation is not None:
            initial_guess = (self.last_fk_translation, self.last_fk_rotation)

        fk_translation, fk_rotation, success, _ = self.ik.calculate_forward_kinematics(
            actual_angles,
            initial_guess=initial_guess,
            use_top_surface_offset=True
        )

        if success:
            self.last_fk_translation = fk_translation
            self.last_fk_rotation = fk_rotation
            actual_rx = fk_rotation[0]
            actual_ry = fk_rotation[1]
        else:
            # Fallback to target if FK fails
            actual_rx = rx_target
            actual_ry = ry_target

        # Compute platform angular acceleration
        omega_rx = (actual_rx - self.prev_platform_angles['rx']) / step_dt
        omega_ry = (actual_ry - self.prev_platform_angles['ry']) / step_dt
        alpha_rx = (omega_rx - self.platform_angular_vel['rx']) / step_dt
        alpha_ry = (omega_ry - self.platform_angular_vel['ry']) / step_dt

        self.platform_angular_vel['rx'] = omega_rx
        self.platform_angular_vel['ry'] = omega_ry
        self.platform_angular_accel['rx'] = alpha_rx
        self.platform_angular_accel['ry'] = alpha_ry
        self.prev_platform_angles['rx'] = actual_rx
        self.prev_platform_angles['ry'] = actual_ry

        # Update platform state
        self.platform_rx = actual_rx
        self.platform_ry = actual_ry

        # Build platform pose for ball physics (with gravity offset applied)
        # SimpleBallPhysics2D expects [x, y, z, rx, ry, rz] with x,y,z in mm and angles in degrees
        # Gravity offset simulates unlevel surface - adds constant bias to effective tilt
        platform_pose = np.array([[
            0.0,  # x in mm (platform centered)
            0.0,  # y in mm
            self.home_z,  # z in mm
            actual_rx + self.gravity_offset_rx,  # rx + gravity offset
            actual_ry + self.gravity_offset_ry,  # ry + gravity offset
            0.0   # rz in degrees
        ]], dtype=np.float32)

        # Step ball physics
        self.ball_pos, self.ball_vel, self.ball_omega, contact_info = self.ball_physics.step(
            self.ball_pos,
            self.ball_vel,
            self.ball_omega,
            platform_pose,
            step_dt,
            platform_angular_accel=self.platform_angular_accel
        )

        # Check if ball fell off
        done = contact_info.get('fell_off', False)

        # Compute reward
        reward = self._compute_reward(done)

        # Update frame history
        new_frame = self._get_single_frame()
        self.frame_history[:-1] = self.frame_history[1:]  # Shift left
        self.frame_history[-1] = new_frame

        # Update time
        self.sim_time += step_dt
        self.step_count += 1

        # Store current dt for observation
        self._current_dt = step_dt

        # Check truncation
        truncated = self.step_count >= self.cfg.max_steps

        obs = self.frame_history.flatten()

        info = {
            'physics_gt': self.physics_gt.copy(),
            'ball_pos_mm': (self.ball_pos[0, 0] * 1000, self.ball_pos[0, 1] * 1000),
            'target_pos_mm': (self.target_x, self.target_y),
            'platform_angles': (actual_rx, actual_ry),
            'fell_off': done
        }

        return obs, reward, done, truncated, info

    def _compute_reward(self, fell_off: bool) -> float:
        """Compute reward (same formula as env_gpu.py)."""
        if fell_off:
            return float(self.fall_penalty)

        # Ball position relative to target (in mm)
        ball_x_mm = (self.ball_pos[0, 0] - self.target_x / 1000.0) * 1000.0
        ball_y_mm = (self.ball_pos[0, 1] - self.target_y / 1000.0) * 1000.0
        dist_mm = np.sqrt(ball_x_mm**2 + ball_y_mm**2)

        # Ball speed (in mm/s)
        vx_mm_s = self.ball_vel[0, 0] * 1000.0
        vy_mm_s = self.ball_vel[0, 1] * 1000.0
        speed_mm_s = np.sqrt(vx_mm_s**2 + vy_mm_s**2)

        # Base reward: position × speed factor
        dist_factor = 1.0 / (1.0 + dist_mm / self.dist_scale)
        speed_factor = 1.0 / (1.0 + speed_mm_s / self.base_speed_scale)
        base_reward = dist_factor * speed_factor

        # Center bonus: extra reward for being very close to target AND slow
        # Distance component: linear from 0 at radius to max at center
        # Speed component: linear cutoff - full bonus when still, zero bonus above threshold
        if dist_mm < self.center_bonus_radius:
            distance_factor = 1.0 - dist_mm / self.center_bonus_radius
            speed_factor = max(0.0, 1.0 - speed_mm_s / self.center_bonus_speed_scale)
            center_bonus = self.center_bonus_max * distance_factor * speed_factor
        else:
            center_bonus = 0.0

        # Approach reward: bonus for moving towards target at APPROPRIATE speed
        # vel_towards is positive when moving towards target
        approach_reward = 0.0
        if dist_mm > 1e-6 and speed_mm_s > 1e-6:
            vel_towards = -(vx_mm_s * ball_x_mm + vy_mm_s * ball_y_mm) / dist_mm

            # Alignment: how directly is ball moving towards target?
            # cos(0°)=1.0 towards, cos(90°)=0.0 perpendicular, cos(180°)=-1.0 away
            alignment = vel_towards / speed_mm_s

            # Optimal velocity scales with distance (for smooth deceleration to stop)
            # At 30mm, optimal ~60mm/s; at 60mm, optimal ~120mm/s
            optimal_vel = dist_mm * 2.0

            if vel_towards <= 0:
                # Moving away from target: penalty scaled by alignment (-1 at 180°)
                approach_reward = alignment * self.approach_scale
            elif speed_mm_s <= optimal_vel:
                # Total speed within optimal: reward based on towards component and alignment
                approach_reward = (vel_towards / optimal_vel) * alignment * self.approach_scale
            else:
                # Total speed exceeds optimal: penalize excess, scale by alignment
                excess = (speed_mm_s - optimal_vel) / optimal_vel
                approach_reward = alignment * self.approach_scale * np.exp(-excess)

            approach_reward = np.clip(approach_reward, -0.5, 0.5)

        return float(base_reward + center_bonus + approach_reward)


class StewartEnvVec:
    """
    Vectorized wrapper for StewartEnv.

    Creates multiple independent simulation environments.
    """

    def __init__(
        self,
        num_envs: int = 1,
        config: Optional[EnvConfig] = None,
        reward_config: Optional[RewardConfig] = None,
        device: str = "cuda"
    ):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.cfg = config or EnvConfig()

        # Create individual environments
        self.envs = [
            StewartEnv(config=config, reward_config=reward_config, device=device)
            for _ in range(num_envs)
        ]

        # Observation and action dimensions
        self.obs_dim = self.cfg.num_frames * self.cfg.obs_per_frame
        self.action_dim = self.cfg.action_dim

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset all environments."""
        obs_list = []
        physics_gt_list = []

        for env in self.envs:
            obs, info = env.reset()
            obs_list.append(obs)
            physics_gt_list.append(info['physics_gt'])

        obs = np.stack(obs_list, axis=0)
        physics_gt = np.stack(physics_gt_list, axis=0)

        return obs, {'physics_gt': physics_gt}

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Step all environments."""
        obs_list = []
        reward_list = []
        done_list = []
        truncated_list = []
        physics_gt_list = []

        for i, env in enumerate(self.envs):
            obs, reward, done, truncated, info = env.step(actions[i])
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            truncated_list.append(truncated)
            physics_gt_list.append(info['physics_gt'])

            # Auto-reset if done or truncated
            if done or truncated:
                obs, reset_info = env.reset()
                obs_list[-1] = obs
                physics_gt_list[-1] = reset_info['physics_gt']

        obs = np.stack(obs_list, axis=0)
        rewards = np.array(reward_list, dtype=np.float32)
        dones = np.array(done_list, dtype=bool)
        truncateds = np.array(truncated_list, dtype=bool)
        physics_gt = np.stack(physics_gt_list, axis=0)

        return obs, rewards, dones, truncateds, {'physics_gt': physics_gt}


if __name__ == "__main__":
    print("Testing Stewart Platform Environment...")
    print("=" * 60)

    env = StewartEnv()

    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Physics GT: {info['physics_gt']}")

    # Run a few steps
    total_reward = 0
    for step in range(100):
        action = np.random.uniform(-1, 1, 2).astype(np.float32)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        if step % 20 == 0:
            print(f"Step {step}: reward={reward:.3f}, ball_pos={info['ball_pos_mm']}, "
                  f"platform={info['platform_angles']}")

        if done:
            print(f"Ball fell off at step {step}")
            break

    print(f"\nTotal reward: {total_reward:.1f}")
    print("\n[OK] Environment working!")

    # Test vectorized version
    print("\n" + "=" * 60)
    print("Testing Vectorized Environment...")

    vec_env = StewartEnvVec(num_envs=2)
    obs, info = vec_env.reset()
    print(f"Vectorized obs shape: {obs.shape}")

    for step in range(10):
        actions = np.random.uniform(-1, 1, (2, 2)).astype(np.float32)
        obs, rewards, dones, truncateds, info = vec_env.step(actions)
        print(f"Step {step}: rewards={rewards}")

    print("\n[OK] Vectorized environment working!")
