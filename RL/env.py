"""
Stewart Platform Environment for Feedforward Agent

Features:
- 12 frames of history with dt timing
- Domain randomization for sim-to-real transfer
- Returns observations and physics ground truth
"""

import numpy as np
import numba as nb

try:
    from .rl_config import EnvConfig, RewardConfig
except ImportError:
    from rl_config import EnvConfig, RewardConfig


# ============================================================================
# NUMBA-OPTIMIZED PHYSICS (same as original)
# ============================================================================

@nb.njit(fastmath=True, cache=True)
def clip_value(value, min_val, max_val):
    if value < min_val:
        return min_val
    elif value > max_val:
        return max_val
    return value


@nb.njit(fastmath=True, cache=True)
def compute_ball_acceleration(ball_x, ball_y, ball_vx, ball_vy,
                               platform_rx, platform_ry,
                               g, mass_factor, mu_roll):
    sin_rx = np.sin(platform_rx)
    sin_ry = np.sin(platform_ry)
    cos_rx = np.cos(platform_rx)
    cos_ry = np.cos(platform_ry)

    ax = g * sin_ry * cos_rx / mass_factor
    ay = -g * sin_rx * cos_ry / mass_factor

    speed = np.sqrt(ball_vx * ball_vx + ball_vy * ball_vy)
    if speed > 1e-6:
        friction_ax = -mu_roll * g * ball_vx / speed
        friction_ay = -mu_roll * g * ball_vy / speed
        ax += friction_ax
        ay += friction_ay

    return ax, ay


@nb.njit(fastmath=True, cache=True)
def rk4_step(x, y, vx, vy, rx, ry, dt, g, mass_factor, mu_roll):
    ax1, ay1 = compute_ball_acceleration(x, y, vx, vy, rx, ry, g, mass_factor, mu_roll)
    k1_x, k1_y = vx, vy
    k1_vx, k1_vy = ax1, ay1

    x2 = x + 0.5 * dt * k1_x
    y2 = y + 0.5 * dt * k1_y
    vx2 = vx + 0.5 * dt * k1_vx
    vy2 = vy + 0.5 * dt * k1_vy
    ax2, ay2 = compute_ball_acceleration(x2, y2, vx2, vy2, rx, ry, g, mass_factor, mu_roll)
    k2_x, k2_y = vx2, vy2
    k2_vx, k2_vy = ax2, ay2

    x3 = x + 0.5 * dt * k2_x
    y3 = y + 0.5 * dt * k2_y
    vx3 = vx + 0.5 * dt * k2_vx
    vy3 = vy + 0.5 * dt * k2_vy
    ax3, ay3 = compute_ball_acceleration(x3, y3, vx3, vy3, rx, ry, g, mass_factor, mu_roll)
    k3_x, k3_y = vx3, vy3
    k3_vx, k3_vy = ax3, ay3

    x4 = x + dt * k3_x
    y4 = y + dt * k3_y
    vx4 = vx + dt * k3_vx
    vy4 = vy + dt * k3_vy
    ax4, ay4 = compute_ball_acceleration(x4, y4, vx4, vy4, rx, ry, g, mass_factor, mu_roll)
    k4_x, k4_y = vx4, vy4
    k4_vx, k4_vy = ax4, ay4

    new_x = x + (dt / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
    new_y = y + (dt / 6.0) * (k1_y + 2.0 * k2_y + 2.0 * k3_y + k4_y)
    new_vx = vx + (dt / 6.0) * (k1_vx + 2.0 * k2_vx + 2.0 * k3_vx + k4_vx)
    new_vy = vy + (dt / 6.0) * (k1_vy + 2.0 * k2_vy + 2.0 * k3_vy + k4_vy)

    return new_x, new_y, new_vx, new_vy


# ============================================================================
# ENVIRONMENT CLASS
# ============================================================================

class StewartEnv:
    """
    Stewart Platform Environment with frame history.

    Observation: [x, y, rx, ry, dt] × num_frames
    Returns physics ground truth for auxiliary loss.
    Supports domain randomization.
    """

    def __init__(self, num_envs=1, config=None, reward_config=None, use_domain_randomization=None):
        self.num_envs = num_envs
        self.cfg = config or EnvConfig()
        self.reward_cfg = reward_config or RewardConfig()

        # Override domain randomization if specified
        if use_domain_randomization is not None:
            self.cfg.use_domain_randomization = use_domain_randomization

        # Convert to SI units
        self.platform_radius_m = self.cfg.platform_radius_mm / 1000.0
        self.max_tilt_rad = np.radians(self.cfg.max_tilt_deg)

        # Default physics (will be randomized per env)
        self.g = 9.81
        self.default_mass_factor = 5.0 / 3.0
        self.default_mu_roll = 0.02
        self.default_servo_tau = 0.05

        # Per-environment physics parameters (for domain randomization)
        self.mass_factor = np.full(num_envs, self.default_mass_factor, dtype=np.float32)
        self.mu_roll = np.full(num_envs, self.default_mu_roll, dtype=np.float32)
        self.servo_tau = np.full(num_envs, self.default_servo_tau, dtype=np.float32)

        # Per-environment timing parameters
        # base_dt: each env has different base timestep (5-20ms)
        # actual_dt: the dt used for current step (with variance and jitter)
        self.base_dt = np.full(num_envs, self.cfg.dt, dtype=np.float32)
        self.actual_dt = np.full(num_envs, self.cfg.dt, dtype=np.float32)

        # Timing randomization settings
        self.dt_range = (0.005, 0.020)  # 5-20ms base dt range
        self.dt_variance = 0.20          # 20% variance per step
        self.jitter_prob = 0.002         # 0.2% chance of jitter
        self.jitter_dt = 0.050           # 50ms jitter when it occurs

        # Platform placement offset (simulates non-level surface)
        # Random 0-2 degree tilt in random direction, set once per env
        self.platform_offset_max_deg = 2.0
        self.platform_offset_rx = np.zeros(num_envs, dtype=np.float32)
        self.platform_offset_ry = np.zeros(num_envs, dtype=np.float32)

        # State arrays
        self.ball_x = np.zeros(num_envs, dtype=np.float32)
        self.ball_y = np.zeros(num_envs, dtype=np.float32)
        self.ball_vx = np.zeros(num_envs, dtype=np.float32)
        self.ball_vy = np.zeros(num_envs, dtype=np.float32)
        self.platform_rx = np.zeros(num_envs, dtype=np.float32)
        self.platform_ry = np.zeros(num_envs, dtype=np.float32)

        # Target position (where ball should go, normalized)
        # For now always 0,0 (center), later can be trajectories
        self.target_x = np.zeros(num_envs, dtype=np.float32)
        self.target_y = np.zeros(num_envs, dtype=np.float32)

        # Episode tracking
        self.step_count = np.zeros(num_envs, dtype=np.int32)
        self.episode_reward = np.zeros(num_envs, dtype=np.float32)
        self.prev_actions = np.zeros((num_envs, 2), dtype=np.float32)

        # Frame history buffer: (num_envs, num_frames, obs_per_frame)
        # obs_per_frame = [ball_x, ball_y, platform_rx, platform_ry, dt]
        self.frame_history = np.zeros(
            (num_envs, self.cfg.num_frames, self.cfg.obs_per_frame),
            dtype=np.float32
        )

        # Last step time for dt calculation (simulated)
        self.last_step_time = np.zeros(num_envs, dtype=np.float32)

        # Warmup numba
        self._warmup_numba()

    def _warmup_numba(self):
        """Pre-compile Numba functions."""
        _ = rk4_step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                     self.cfg.dt, self.g, self.default_mass_factor, self.default_mu_roll)

    def _randomize_platform_offset(self, indices):
        """Randomize platform placement offset (non-level surface simulation)."""
        n = len(indices)

        if self.cfg.use_domain_randomization:
            # Random magnitude 0 to max_deg
            magnitude = np.random.uniform(0, self.platform_offset_max_deg, n)
            # Random direction (angle in radians)
            direction = np.random.uniform(0, 2 * np.pi, n)

            # Convert to rx, ry offsets
            self.platform_offset_rx[indices] = np.radians(magnitude * np.sin(direction)).astype(np.float32)
            self.platform_offset_ry[indices] = np.radians(magnitude * np.cos(direction)).astype(np.float32)
        else:
            self.platform_offset_rx[indices] = 0.0
            self.platform_offset_ry[indices] = 0.0

    def _randomize_physics(self, indices):
        """Randomize physics parameters for specified environments."""
        n = len(indices)

        if self.cfg.use_domain_randomization:
            # Randomize friction
            self.mu_roll[indices] = np.random.uniform(
                self.cfg.friction_range[0],
                self.cfg.friction_range[1],
                n
            ).astype(np.float32)

            # Randomize servo time constant
            self.servo_tau[indices] = np.random.uniform(
                self.cfg.servo_tau_range[0],
                self.cfg.servo_tau_range[1],
                n
            ).astype(np.float32)

            # Randomize mass factor
            self.mass_factor[indices] = np.random.uniform(
                self.cfg.mass_factor_range[0],
                self.cfg.mass_factor_range[1],
                n
            ).astype(np.float32)

            # Randomize base dt (5-20ms per environment)
            self.base_dt[indices] = np.random.uniform(
                self.dt_range[0],
                self.dt_range[1],
                n
            ).astype(np.float32)
        else:
            # Use default values
            self.mu_roll[indices] = self.default_mu_roll
            self.servo_tau[indices] = self.default_servo_tau
            self.mass_factor[indices] = self.default_mass_factor
            self.base_dt[indices] = self.cfg.dt

    def _sample_dt(self):
        """
        Sample actual dt for this step with variance and jitter.

        Each environment has:
        - base_dt: 5-20ms (set at reset)
        - variance: ±20% of base_dt
        - jitter: 0.2% chance of 50ms spike
        """
        # Start with base dt + 20% variance
        variance = 1.0 + np.random.uniform(-self.dt_variance, self.dt_variance, self.num_envs)
        self.actual_dt = (self.base_dt * variance).astype(np.float32)

        # Apply jitter (0.2% chance of 50ms)
        jitter_mask = np.random.random(self.num_envs) < self.jitter_prob
        self.actual_dt[jitter_mask] = self.jitter_dt

        # Clamp to reasonable range (1ms - 60ms)
        self.actual_dt = np.clip(self.actual_dt, 0.001, 0.060)

        return self.actual_dt

    def get_physics_normalized(self):
        """
        Get normalized physics parameters for all environments.

        Returns: (num_envs, 3) array of [friction, servo_tau, mass_factor] in [0, 1]
        """
        physics = np.zeros((self.num_envs, 3), dtype=np.float32)

        # Normalize to [0, 1] based on ranges
        fr = self.cfg.friction_range
        sr = self.cfg.servo_tau_range
        mr = self.cfg.mass_factor_range

        physics[:, 0] = (self.mu_roll - fr[0]) / (fr[1] - fr[0])
        physics[:, 1] = (self.servo_tau - sr[0]) / (sr[1] - sr[0])
        physics[:, 2] = (self.mass_factor - mr[0]) / (mr[1] - mr[0])

        return np.clip(physics, 0, 1)

    def reset(self, indices=None):
        """
        Reset environments.

        Returns:
            obs: (num_envs, obs_dim) observations
            physics_gt: (num_envs, physics_dim) normalized physics ground truth
        """
        if indices is None:
            indices = np.arange(self.num_envs)

        n = len(indices)

        # Randomize physics and platform offset for these environments
        self._randomize_physics(indices)
        self._randomize_platform_offset(indices)

        # Random initial position
        init_pos_m = self.cfg.init_pos_range_mm / 1000.0
        self.ball_x[indices] = np.random.uniform(-init_pos_m, init_pos_m, n).astype(np.float32)
        self.ball_y[indices] = np.random.uniform(-init_pos_m, init_pos_m, n).astype(np.float32)

        # Random initial velocity
        init_vel_m_s = self.cfg.init_vel_range_mm_s / 1000.0
        self.ball_vx[indices] = np.random.uniform(-init_vel_m_s, init_vel_m_s, n).astype(np.float32)
        self.ball_vy[indices] = np.random.uniform(-init_vel_m_s, init_vel_m_s, n).astype(np.float32)

        # Platform starts level
        self.platform_rx[indices] = 0.0
        self.platform_ry[indices] = 0.0

        # Reset counters
        self.step_count[indices] = 0
        self.episode_reward[indices] = 0.0
        self.prev_actions[indices] = 0.0
        self.last_step_time[indices] = 0.0

        # Initialize actual_dt for first frame
        self.actual_dt[indices] = self.base_dt[indices]

        # Initialize frame history with current observation (all frames same)
        initial_obs = self._get_single_frame()
        for i in indices:
            for f in range(self.cfg.num_frames):
                self.frame_history[i, f, :] = initial_obs[i]

        info = {
            'physics_gt': self.get_physics_normalized(),
            'base_dt': self.base_dt.copy(),
        }

        return self._get_observation(), info

    def _get_single_frame(self):
        """
        Get single frame observation for all environments.

        Returns: (num_envs, obs_per_frame) array [x, y, rx, ry, dt, target_x, target_y]
        """
        obs = np.zeros((self.num_envs, self.cfg.obs_per_frame), dtype=np.float32)

        # Ball position with noise
        ball_x_mm = self.ball_x * 1000.0
        ball_y_mm = self.ball_y * 1000.0

        if self.cfg.use_camera_noise:
            noise_x = np.random.normal(0, self.cfg.position_noise_std_mm, self.num_envs)
            noise_y = np.random.normal(0, self.cfg.position_noise_std_mm, self.num_envs)
            dist = np.sqrt(ball_x_mm**2 + ball_y_mm**2)
            depth_scale = 1.0 + dist * self.cfg.noise_depth_scale
            noise_x *= depth_scale
            noise_y *= depth_scale
            ball_x_mm = ball_x_mm + noise_x.astype(np.float32)
            ball_y_mm = ball_y_mm + noise_y.astype(np.float32)

        # Normalize position
        obs[:, 0] = ball_x_mm / self.cfg.platform_radius_mm
        obs[:, 1] = ball_y_mm / self.cfg.platform_radius_mm

        # Normalize platform angles
        obs[:, 2] = np.degrees(self.platform_rx) / self.cfg.max_tilt_deg
        obs[:, 3] = np.degrees(self.platform_ry) / self.cfg.max_tilt_deg

        # dt normalized (nominal dt = 0.01s = 10ms)
        # actual_dt varies per environment and per step
        obs[:, 4] = self.actual_dt / 0.01  # Normalized so 10ms = 1.0

        # Target position (normalized, currently always 0,0)
        obs[:, 5] = self.target_x
        obs[:, 6] = self.target_y

        return obs

    def _get_observation(self):
        """
        Get observation for all environments.

        Returns: (num_envs, obs_dim) array
        """
        return self.frame_history.reshape(self.num_envs, -1)

    def step(self, actions):
        """
        Step all environments.

        Args:
            actions: (num_envs, 2) array [rx_target, ry_target] in [-1, 1]

        Returns:
            obs: (num_envs, obs_dim)
            rewards: (num_envs,)
            dones: (num_envs,)
            infos: dict with physics_gt and other info
        """
        # Sample dt for this step (with variance and jitter)
        self._sample_dt()

        # Scale actions
        action_rx = actions[:, 0].astype(np.float32) * self.max_tilt_rad
        action_ry = actions[:, 1].astype(np.float32) * self.max_tilt_rad

        # Step physics for each environment (with per-env parameters and dt)
        fell_off = np.zeros(self.num_envs, dtype=bool)

        for i in range(self.num_envs):
            dt_i = self.actual_dt[i]

            # Servo dynamics (using per-env dt)
            if self.servo_tau[i] > 1e-6:
                decay = np.exp(-dt_i / self.servo_tau[i])
                new_rx = action_rx[i] + (self.platform_rx[i] - action_rx[i]) * decay
                new_ry = action_ry[i] + (self.platform_ry[i] - action_ry[i]) * decay
            else:
                new_rx = action_rx[i]
                new_ry = action_ry[i]

            new_rx = clip_value(new_rx, -self.max_tilt_rad, self.max_tilt_rad)
            new_ry = clip_value(new_ry, -self.max_tilt_rad, self.max_tilt_rad)

            self.platform_rx[i] = new_rx
            self.platform_ry[i] = new_ry

            # Ball physics (using per-env dt)
            # Add platform offset to simulate non-level surface
            effective_rx = new_rx + self.platform_offset_rx[i]
            effective_ry = new_ry + self.platform_offset_ry[i]

            bx, by, bvx, bvy = rk4_step(
                self.ball_x[i], self.ball_y[i],
                self.ball_vx[i], self.ball_vy[i],
                effective_rx, effective_ry, dt_i,
                self.g, self.mass_factor[i], self.mu_roll[i]
            )

            # Check bounds
            dist = np.sqrt(bx * bx + by * by)
            if dist > self.platform_radius_m:
                fell_off[i] = True
                bx, by, bvx, bvy = 0.0, 0.0, 0.0, 0.0

            self.ball_x[i] = bx
            self.ball_y[i] = by
            self.ball_vx[i] = bvx
            self.ball_vy[i] = bvy

        # Compute rewards
        rewards = self._compute_reward(actions, fell_off)

        # Update tracking
        self.prev_actions = actions.astype(np.float32).copy()
        self.step_count += 1
        self.episode_reward += rewards

        # Update frame history (shift and add new frame)
        new_frame = self._get_single_frame()
        self.frame_history[:, :-1, :] = self.frame_history[:, 1:, :]
        self.frame_history[:, -1, :] = new_frame

        # Check done
        dones = (self.step_count >= self.cfg.max_steps) | fell_off
        truncated = self.step_count >= self.cfg.max_steps

        infos = {
            'fell_off': fell_off,
            'episode_reward': self.episode_reward.copy(),
            'step_count': self.step_count.copy(),
            'physics_gt': self.get_physics_normalized(),
            'actual_dt': self.actual_dt.copy(),
            'base_dt': self.base_dt.copy(),
        }

        return self._get_observation(), rewards, dones, truncated, infos

    def _compute_reward(self, actions, fell_off):
        """Compute reward (same as original)."""
        cfg = self.reward_cfg

        pos_x_mm = self.ball_x * 1000.0
        pos_y_mm = self.ball_y * 1000.0
        pos_error_sq = pos_x_mm ** 2 + pos_y_mm ** 2

        vel_x_mm_s = self.ball_vx * 1000.0
        vel_y_mm_s = self.ball_vy * 1000.0
        vel_sq = vel_x_mm_s ** 2 + vel_y_mm_s ** 2
        speed_mm_s = np.sqrt(vel_sq)

        rx_deg = np.degrees(self.platform_rx)
        ry_deg = np.degrees(self.platform_ry)
        tilt_sq = rx_deg ** 2 + ry_deg ** 2

        action_sq = actions[:, 0] ** 2 + actions[:, 1] ** 2
        action_delta = actions - self.prev_actions
        action_rate_sq = action_delta[:, 0] ** 2 + action_delta[:, 1] ** 2

        reward = np.zeros(self.num_envs, dtype=np.float32)
        reward -= cfg.k_position * pos_error_sq
        reward -= cfg.k_velocity * vel_sq
        reward -= cfg.k_tilt * tilt_sq
        reward -= cfg.k_action * action_sq
        reward -= cfg.k_action_rate * action_rate_sq

        dist_mm = np.sqrt(pos_error_sq)
        center_bonus = cfg.k_center_bonus * np.exp(-dist_mm / cfg.center_threshold_mm)
        reward += center_bonus

        stability_mask = (dist_mm < cfg.center_threshold_mm) & (speed_mm_s < cfg.stability_vel_threshold)
        reward += cfg.k_stability_bonus * stability_mask.astype(np.float32)

        reward[fell_off] = cfg.out_of_bounds_penalty

        return reward


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("Testing StewartEnv...")

    env = StewartEnv(num_envs=4)
    print(f"Created env with {env.num_envs} parallel envs")
    print(f"  Num frames: {env.cfg.num_frames}")
    print(f"  Obs per frame: {env.cfg.obs_per_frame}")
    print(f"  obs dim: {env.cfg.obs_dim}")
    print(f"  Domain randomization: {env.cfg.use_domain_randomization}")

    # Reset
    obs, info = env.reset()
    print(f"\nAfter reset:")
    print(f"  obs shape: {obs.shape}")
    print(f"  physics_gt shape: {info['physics_gt'].shape}")
    print(f"  physics_gt (env 0): {info['physics_gt'][0]}")
    print(f"  base_dt per env (ms): {info['base_dt'] * 1000}")

    # Step and collect dt statistics
    dt_samples = []
    jitter_count = 0
    for step in range(1000):
        actions = np.random.uniform(-1, 1, (4, 2)).astype(np.float32)
        obs, rewards, dones, truncated, infos = env.step(actions)
        dt_samples.extend(infos['actual_dt'].tolist())
        jitter_count += np.sum(infos['actual_dt'] > 0.040)

    dt_samples = np.array(dt_samples)
    print(f"\ndt statistics over 1000 steps × 4 envs:")
    print(f"  Mean dt: {dt_samples.mean()*1000:.2f} ms")
    print(f"  Std dt:  {dt_samples.std()*1000:.2f} ms")
    print(f"  Min dt:  {dt_samples.min()*1000:.2f} ms")
    print(f"  Max dt:  {dt_samples.max()*1000:.2f} ms")
    print(f"  Jitter frames (>40ms): {jitter_count} ({100*jitter_count/len(dt_samples):.2f}%)")

    print("\n[OK] StewartEnv test passed!")
