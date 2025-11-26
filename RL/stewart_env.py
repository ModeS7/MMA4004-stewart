"""
Stewart Platform Ball Balancing Environment

Simplified 2D physics for RL training:
- Ball rolls on tilted platform
- Platform tilts in X and Y directions
- Goal: keep ball centered

Uses Numba for fast physics simulation.
"""

import numpy as np
import numba as nb
from rl_config import EnvConfig, RewardConfig


# ============================================================================
# NUMBA-OPTIMIZED PHYSICS
# ============================================================================

@nb.njit(fastmath=True, cache=True)
def clip_value(value, min_val, max_val):
    """Fast clip function."""
    if value < min_val:
        return min_val
    elif value > max_val:
        return max_val
    return value


@nb.njit(fastmath=True, cache=True)
def compute_ball_acceleration(ball_x, ball_y, ball_vx, ball_vy,
                               platform_rx, platform_ry,
                               g=9.81, mass_factor=1.667, mu_roll=0.02):
    """
    Compute ball acceleration on tilted platform.

    Rolling ball: a = g * sin(theta) / mass_factor
    where mass_factor = 1 + I/(m*r^2) = 5/3 for hollow sphere

    Args:
        ball_x, ball_y: Ball position (m)
        ball_vx, ball_vy: Ball velocity (m/s)
        platform_rx, platform_ry: Platform tilt angles (radians)
        g: Gravity (m/s^2)
        mass_factor: 1 + I/(m*r^2), 5/3 for hollow sphere
        mu_roll: Rolling friction coefficient

    Returns:
        ax, ay: Ball acceleration (m/s^2)
    """
    # Gravity components on tilted surface
    # Small angle approximation: sin(theta) ~ theta for |theta| < 15 deg
    sin_rx = np.sin(platform_rx)
    sin_ry = np.sin(platform_ry)
    cos_rx = np.cos(platform_rx)
    cos_ry = np.cos(platform_ry)

    # Acceleration due to gravity on tilted surface
    ax = g * sin_ry * cos_rx / mass_factor
    ay = -g * sin_rx * cos_ry / mass_factor

    # Rolling friction (opposes velocity)
    speed = np.sqrt(ball_vx * ball_vx + ball_vy * ball_vy)
    if speed > 1e-6:
        friction_ax = -mu_roll * g * ball_vx / speed
        friction_ay = -mu_roll * g * ball_vy / speed
        ax += friction_ax
        ay += friction_ay

    return ax, ay


@nb.njit(fastmath=True, cache=True)
def rk4_step(x, y, vx, vy, rx, ry, dt, g, mass_factor, mu_roll):
    """
    RK4 integration step for ball physics.

    Returns:
        new_x, new_y, new_vx, new_vy
    """
    # k1
    ax1, ay1 = compute_ball_acceleration(x, y, vx, vy, rx, ry, g, mass_factor, mu_roll)
    k1_x, k1_y = vx, vy
    k1_vx, k1_vy = ax1, ay1

    # k2
    x2 = x + 0.5 * dt * k1_x
    y2 = y + 0.5 * dt * k1_y
    vx2 = vx + 0.5 * dt * k1_vx
    vy2 = vy + 0.5 * dt * k1_vy
    ax2, ay2 = compute_ball_acceleration(x2, y2, vx2, vy2, rx, ry, g, mass_factor, mu_roll)
    k2_x, k2_y = vx2, vy2
    k2_vx, k2_vy = ax2, ay2

    # k3
    x3 = x + 0.5 * dt * k2_x
    y3 = y + 0.5 * dt * k2_y
    vx3 = vx + 0.5 * dt * k2_vx
    vy3 = vy + 0.5 * dt * k2_vy
    ax3, ay3 = compute_ball_acceleration(x3, y3, vx3, vy3, rx, ry, g, mass_factor, mu_roll)
    k3_x, k3_y = vx3, vy3
    k3_vx, k3_vy = ax3, ay3

    # k4
    x4 = x + dt * k3_x
    y4 = y + dt * k3_y
    vx4 = vx + dt * k3_vx
    vy4 = vy + dt * k3_vy
    ax4, ay4 = compute_ball_acceleration(x4, y4, vx4, vy4, rx, ry, g, mass_factor, mu_roll)
    k4_x, k4_y = vx4, vy4
    k4_vx, k4_vy = ax4, ay4

    # Combine
    new_x = x + (dt / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
    new_y = y + (dt / 6.0) * (k1_y + 2.0 * k2_y + 2.0 * k3_y + k4_y)
    new_vx = vx + (dt / 6.0) * (k1_vx + 2.0 * k2_vx + 2.0 * k3_vx + k4_vx)
    new_vy = vy + (dt / 6.0) * (k1_vy + 2.0 * k2_vy + 2.0 * k3_vy + k4_vy)

    return new_x, new_y, new_vx, new_vy


@nb.njit(fastmath=True, cache=True)
def step_physics_batch(ball_x, ball_y, ball_vx, ball_vy,
                       platform_rx, platform_ry,
                       action_rx, action_ry,
                       dt, max_tilt_rad, platform_radius,
                       g, mass_factor, mu_roll,
                       servo_tau):
    """
    Step physics for a batch of environments.

    Args:
        ball_x, ball_y: Ball positions (N,) in meters
        ball_vx, ball_vy: Ball velocities (N,) in m/s
        platform_rx, platform_ry: Current platform angles (N,) in radians
        action_rx, action_ry: Target angles (N,) in radians
        dt: Time step
        max_tilt_rad: Maximum tilt angle in radians
        platform_radius: Platform radius in meters
        g, mass_factor, mu_roll: Physics constants
        servo_tau: Servo time constant

    Returns:
        new_ball_x, new_ball_y, new_ball_vx, new_ball_vy,
        new_platform_rx, new_platform_ry, fell_off (N,)
    """
    n = ball_x.shape[0]

    new_ball_x = np.empty(n)
    new_ball_y = np.empty(n)
    new_ball_vx = np.empty(n)
    new_ball_vy = np.empty(n)
    new_platform_rx = np.empty(n)
    new_platform_ry = np.empty(n)
    fell_off = np.zeros(n, dtype=nb.boolean)

    for i in range(n):
        # Update platform angles (first-order servo dynamics)
        # y(t+dt) = target + (y(t) - target) * exp(-dt/tau)
        if servo_tau > 1e-6:
            decay = np.exp(-dt / servo_tau)
            new_rx = action_rx[i] + (platform_rx[i] - action_rx[i]) * decay
            new_ry = action_ry[i] + (platform_ry[i] - action_ry[i]) * decay
        else:
            new_rx = action_rx[i]
            new_ry = action_ry[i]

        # Clamp to max tilt
        new_rx = clip_value(new_rx, -max_tilt_rad, max_tilt_rad)
        new_ry = clip_value(new_ry, -max_tilt_rad, max_tilt_rad)

        new_platform_rx[i] = new_rx
        new_platform_ry[i] = new_ry

        # Step ball physics with RK4
        bx, by, bvx, bvy = rk4_step(
            ball_x[i], ball_y[i], ball_vx[i], ball_vy[i],
            new_rx, new_ry, dt, g, mass_factor, mu_roll
        )

        # Check if ball fell off
        dist = np.sqrt(bx * bx + by * by)
        if dist > platform_radius:
            fell_off[i] = True
            # Reset ball to center
            bx, by, bvx, bvy = 0.0, 0.0, 0.0, 0.0

        new_ball_x[i] = bx
        new_ball_y[i] = by
        new_ball_vx[i] = bvx
        new_ball_vy[i] = bvy

    return (new_ball_x, new_ball_y, new_ball_vx, new_ball_vy,
            new_platform_rx, new_platform_ry, fell_off)


# ============================================================================
# ENVIRONMENT CLASS
# ============================================================================

class StewartBallEnv:
    """
    Stewart Platform Ball Balancing Environment.

    State: [ball_x, ball_y, ball_vx, ball_vy, platform_rx, platform_ry]
           Positions in mm, velocities in mm/s, angles in degrees

    Action: [rx_target, ry_target] in [-1, 1], scaled to [-max_tilt, max_tilt]

    Reward: Shaped reward encouraging ball to stay centered with minimal tilt.
    """

    def __init__(self, num_envs=1, config=None, reward_config=None):
        """
        Initialize environment.

        Args:
            num_envs: Number of parallel environments
            config: EnvConfig object (uses defaults if None)
            reward_config: RewardConfig object (uses defaults if None)
        """
        self.num_envs = num_envs
        self.cfg = config or EnvConfig()
        self.reward_cfg = reward_config or RewardConfig()

        # Convert to SI units for physics
        self.platform_radius_m = self.cfg.platform_radius_mm / 1000.0
        self.max_tilt_rad = np.radians(self.cfg.max_tilt_deg)

        # Ball physics (hollow sphere: I = 2/3 * m * r^2)
        self.g = 9.81
        self.mass_factor = 5.0 / 3.0  # 1 + I/(m*r^2) for hollow sphere
        self.mu_roll = 0.02  # Rolling friction

        # Servo dynamics
        self.servo_tau = 0.05  # 50ms time constant

        # State arrays (in SI units internally, converted for observation)
        self.ball_x = np.zeros(num_envs, dtype=np.float32)
        self.ball_y = np.zeros(num_envs, dtype=np.float32)
        self.ball_vx = np.zeros(num_envs, dtype=np.float32)
        self.ball_vy = np.zeros(num_envs, dtype=np.float32)
        self.platform_rx = np.zeros(num_envs, dtype=np.float32)
        self.platform_ry = np.zeros(num_envs, dtype=np.float32)

        # Episode tracking
        self.step_count = np.zeros(num_envs, dtype=np.int32)
        self.episode_reward = np.zeros(num_envs, dtype=np.float32)

        # Pre-compile Numba functions
        self._warmup_numba()

    def _warmup_numba(self):
        """Pre-compile Numba functions with dummy data."""
        dummy = np.zeros(1, dtype=np.float32)
        dummy_bool = np.zeros(1, dtype=np.bool_)
        step_physics_batch(
            dummy, dummy, dummy, dummy,
            dummy, dummy, dummy, dummy,
            self.cfg.dt, self.max_tilt_rad, self.platform_radius_m,
            self.g, self.mass_factor, self.mu_roll, self.servo_tau
        )

    def reset(self, indices=None):
        """
        Reset environments to random initial states.

        Args:
            indices: Optional array of environment indices to reset.
                    If None, resets all environments.

        Returns:
            observations: (num_envs, state_dim) array
        """
        if indices is None:
            indices = np.arange(self.num_envs)

        n = len(indices)

        # Random initial ball position (in meters)
        init_pos_m = self.cfg.init_pos_range_mm / 1000.0
        self.ball_x[indices] = np.random.uniform(-init_pos_m, init_pos_m, n).astype(np.float32)
        self.ball_y[indices] = np.random.uniform(-init_pos_m, init_pos_m, n).astype(np.float32)

        # Random initial velocity (in m/s)
        init_vel_m_s = self.cfg.init_vel_range_mm_s / 1000.0
        self.ball_vx[indices] = np.random.uniform(-init_vel_m_s, init_vel_m_s, n).astype(np.float32)
        self.ball_vy[indices] = np.random.uniform(-init_vel_m_s, init_vel_m_s, n).astype(np.float32)

        # Platform starts level
        self.platform_rx[indices] = 0.0
        self.platform_ry[indices] = 0.0

        # Reset counters
        self.step_count[indices] = 0
        self.episode_reward[indices] = 0.0

        return self._get_observation()

    def step(self, actions):
        """
        Take a step in all environments.

        Args:
            actions: (num_envs, 2) array of [rx_target, ry_target] in [-1, 1]

        Returns:
            observations: (num_envs, state_dim)
            rewards: (num_envs,)
            dones: (num_envs,)
            infos: dict with additional info
        """
        # Scale actions to radians
        action_rx = actions[:, 0].astype(np.float32) * self.max_tilt_rad
        action_ry = actions[:, 1].astype(np.float32) * self.max_tilt_rad

        # Step physics
        (self.ball_x, self.ball_y, self.ball_vx, self.ball_vy,
         self.platform_rx, self.platform_ry, fell_off) = step_physics_batch(
            self.ball_x, self.ball_y, self.ball_vx, self.ball_vy,
            self.platform_rx, self.platform_ry,
            action_rx, action_ry,
            self.cfg.dt, self.max_tilt_rad, self.platform_radius_m,
            self.g, self.mass_factor, self.mu_roll, self.servo_tau
        )

        # Compute rewards
        rewards = self._compute_reward(actions, fell_off)

        # Update step count
        self.step_count += 1
        self.episode_reward += rewards

        # Check done conditions
        dones = (self.step_count >= self.cfg.max_steps) | fell_off

        # Info
        infos = {
            'fell_off': fell_off,
            'episode_reward': self.episode_reward.copy(),
            'step_count': self.step_count.copy()
        }

        return self._get_observation(), rewards, dones, infos

    def _get_observation(self):
        """
        Get observation for all environments.

        Returns:
            obs: (num_envs, 6) array
                 [ball_x_mm, ball_y_mm, ball_vx_mm_s, ball_vy_mm_s, rx_deg, ry_deg]
        """
        obs = np.zeros((self.num_envs, self.cfg.state_dim), dtype=np.float32)

        # Ball position in mm, normalized to [-1, 1] based on platform size
        obs[:, 0] = self.ball_x * 1000.0 / self.cfg.platform_radius_mm
        obs[:, 1] = self.ball_y * 1000.0 / self.cfg.platform_radius_mm

        # Ball velocity in mm/s, normalized (typical max ~500 mm/s)
        obs[:, 2] = self.ball_vx * 1000.0 / 500.0
        obs[:, 3] = self.ball_vy * 1000.0 / 500.0

        # Platform angles normalized to [-1, 1]
        obs[:, 4] = np.degrees(self.platform_rx) / self.cfg.max_tilt_deg
        obs[:, 5] = np.degrees(self.platform_ry) / self.cfg.max_tilt_deg

        return obs

    def _compute_reward(self, actions, fell_off):
        """
        Compute reward for all environments.

        Reward components:
        1. Position penalty: -k * (x^2 + y^2)
        2. Velocity penalty: -k * (vx^2 + vy^2)
        3. Tilt penalty: -k * (rx^2 + ry^2)
        4. Action penalty: -k * (a^2)
        5. Center bonus: +k * exp(-dist/threshold)
        6. Fall penalty: -100 if ball fell off
        """
        cfg = self.reward_cfg

        # Position error (in mm for intuitive scaling)
        pos_x_mm = self.ball_x * 1000.0
        pos_y_mm = self.ball_y * 1000.0
        pos_error_sq = pos_x_mm ** 2 + pos_y_mm ** 2

        # Velocity (in mm/s)
        vel_x_mm_s = self.ball_vx * 1000.0
        vel_y_mm_s = self.ball_vy * 1000.0
        vel_sq = vel_x_mm_s ** 2 + vel_y_mm_s ** 2

        # Tilt (in degrees)
        rx_deg = np.degrees(self.platform_rx)
        ry_deg = np.degrees(self.platform_ry)
        tilt_sq = rx_deg ** 2 + ry_deg ** 2

        # Action magnitude
        action_sq = actions[:, 0] ** 2 + actions[:, 1] ** 2

        # Penalties
        reward = np.zeros(self.num_envs, dtype=np.float32)
        reward -= cfg.k_position * pos_error_sq
        reward -= cfg.k_velocity * vel_sq
        reward -= cfg.k_tilt * tilt_sq
        reward -= cfg.k_action * action_sq

        # Center bonus (exponential decay from center)
        dist_mm = np.sqrt(pos_error_sq)
        center_bonus = cfg.k_center_bonus * np.exp(-dist_mm / cfg.center_threshold_mm)
        reward += center_bonus

        # Fall penalty
        reward[fell_off] = cfg.out_of_bounds_penalty

        return reward

    def get_state_info(self):
        """Get current state for debugging/visualization."""
        return {
            'ball_pos_mm': np.stack([self.ball_x * 1000, self.ball_y * 1000], axis=1),
            'ball_vel_mm_s': np.stack([self.ball_vx * 1000, self.ball_vy * 1000], axis=1),
            'platform_tilt_deg': np.stack([np.degrees(self.platform_rx),
                                           np.degrees(self.platform_ry)], axis=1),
            'step_count': self.step_count,
            'episode_reward': self.episode_reward
        }


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("Testing StewartBallEnv...")

    # Create environment
    env = StewartBallEnv(num_envs=4)
    print(f"Created environment with {env.num_envs} parallel envs")
    print(f"State dim: {env.cfg.state_dim}, Action dim: {env.cfg.action_dim}")

    # Reset
    obs = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Observation range: [{obs.min():.3f}, {obs.max():.3f}]")

    # Run a few steps
    total_reward = np.zeros(4)
    for step in range(100):
        # Random actions
        actions = np.random.uniform(-1, 1, (4, 2)).astype(np.float32)
        obs, rewards, dones, infos = env.step(actions)
        total_reward += rewards

        if step == 0:
            print(f"\nStep {step}:")
            print(f"  Observation: {obs[0]}")
            print(f"  Reward: {rewards[0]:.4f}")
            print(f"  Done: {dones[0]}")

    print(f"\nAfter 100 steps:")
    print(f"  Total rewards: {total_reward}")
    print(f"  Avg reward per step: {total_reward.mean() / 100:.4f}")

    state = env.get_state_info()
    print(f"  Ball positions (mm): {state['ball_pos_mm']}")
    print(f"  Platform tilt (deg): {state['platform_tilt_deg']}")

    print("\n[OK] Environment test passed!")
