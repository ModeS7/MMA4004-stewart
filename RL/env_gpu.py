"""
Stewart Platform Environment - GPU Accelerated

Fully vectorized PyTorch implementation for fast parallel simulation.
All physics computed on GPU using tensor operations.
"""

import torch
import numpy as np

try:
    from .rl_config import EnvConfig, RewardConfig
except ImportError:
    from rl_config import EnvConfig, RewardConfig


class StewartEnvGPU:
    """
    GPU-accelerated Stewart Platform Environment.

    All state stored as PyTorch tensors on specified device.
    Physics fully vectorized - no per-environment loops.
    """

    def __init__(self, num_envs=1, config=None, reward_config=None,
                 device="cuda", use_domain_randomization=None):
        self.num_envs = num_envs
        self.cfg = config or EnvConfig()
        self.reward_cfg = reward_config or RewardConfig()
        self.device = torch.device(device)

        # Override domain randomization if specified
        if use_domain_randomization is not None:
            self.cfg.use_domain_randomization = use_domain_randomization

        # Convert to SI units
        self.platform_radius_m = self.cfg.platform_radius_mm / 1000.0
        self.max_tilt_rad = np.radians(self.cfg.max_tilt_deg)

        # Constants as tensors
        self.g = torch.tensor(9.81, device=self.device)

        # Default physics values
        self.default_mass_factor = 5.0 / 3.0
        self.default_mu_roll = 0.02
        self.default_servo_tau = 0.05

        # Per-environment physics parameters
        self.mass_factor = torch.full((num_envs,), self.default_mass_factor,
                                       device=self.device, dtype=torch.float32)
        self.mu_roll = torch.full((num_envs,), self.default_mu_roll,
                                   device=self.device, dtype=torch.float32)
        self.servo_tau = torch.full((num_envs,), self.default_servo_tau,
                                     device=self.device, dtype=torch.float32)

        # Timing parameters
        self.base_dt = torch.full((num_envs,), self.cfg.dt,
                                   device=self.device, dtype=torch.float32)
        self.actual_dt = torch.full((num_envs,), self.cfg.dt,
                                     device=self.device, dtype=torch.float32)

        # Timing randomization settings
        self.dt_range = (0.005, 0.020)
        self.dt_variance = 0.20
        self.jitter_prob = 0.002
        self.jitter_dt = 0.050

        # Platform placement offset
        self.platform_offset_max_deg = 2.0
        self.platform_offset_rx = torch.zeros(num_envs, device=self.device, dtype=torch.float32)
        self.platform_offset_ry = torch.zeros(num_envs, device=self.device, dtype=torch.float32)

        # State tensors
        self.ball_x = torch.zeros(num_envs, device=self.device, dtype=torch.float32)
        self.ball_y = torch.zeros(num_envs, device=self.device, dtype=torch.float32)
        self.ball_vx = torch.zeros(num_envs, device=self.device, dtype=torch.float32)
        self.ball_vy = torch.zeros(num_envs, device=self.device, dtype=torch.float32)
        self.platform_rx = torch.zeros(num_envs, device=self.device, dtype=torch.float32)
        self.platform_ry = torch.zeros(num_envs, device=self.device, dtype=torch.float32)

        # Target position
        self.target_x = torch.zeros(num_envs, device=self.device, dtype=torch.float32)
        self.target_y = torch.zeros(num_envs, device=self.device, dtype=torch.float32)

        # Episode tracking
        self.step_count = torch.zeros(num_envs, device=self.device, dtype=torch.int32)
        self.episode_reward = torch.zeros(num_envs, device=self.device, dtype=torch.float32)
        self.prev_actions = torch.zeros((num_envs, 2), device=self.device, dtype=torch.float32)

        # Frame history buffer
        self.frame_history = torch.zeros(
            (num_envs, self.cfg.num_frames, self.cfg.obs_per_frame),
            device=self.device, dtype=torch.float32
        )

    def _compute_acceleration(self, x, y, vx, vy, rx, ry, mass_factor, mu_roll):
        """
        Compute ball acceleration from platform tilt and friction.
        Fully vectorized - operates on all environments at once.

        All inputs are tensors of shape (num_envs,)
        Returns ax, ay tensors of shape (num_envs,)
        """
        sin_rx = torch.sin(rx)
        sin_ry = torch.sin(ry)
        cos_rx = torch.cos(rx)
        cos_ry = torch.cos(ry)

        # Gravity components
        ax = self.g * sin_ry * cos_rx / mass_factor
        ay = -self.g * sin_rx * cos_ry / mass_factor

        # Rolling friction (opposes velocity)
        speed = torch.sqrt(vx * vx + vy * vy)
        # Avoid division by zero
        speed_safe = torch.clamp(speed, min=1e-6)

        friction_ax = -mu_roll * self.g * vx / speed_safe
        friction_ay = -mu_roll * self.g * vy / speed_safe

        # Only apply friction where speed > threshold
        friction_mask = (speed > 1e-6).float()
        ax = ax + friction_ax * friction_mask
        ay = ay + friction_ay * friction_mask

        return ax, ay

    def _rk4_step(self, x, y, vx, vy, rx, ry, dt, mass_factor, mu_roll):
        """
        Runge-Kutta 4th order integration step.
        Fully vectorized - operates on all environments at once.

        All inputs are tensors of shape (num_envs,)
        Returns new x, y, vx, vy tensors
        """
        # k1
        ax1, ay1 = self._compute_acceleration(x, y, vx, vy, rx, ry, mass_factor, mu_roll)
        k1_x, k1_y = vx, vy
        k1_vx, k1_vy = ax1, ay1

        # k2
        x2 = x + 0.5 * dt * k1_x
        y2 = y + 0.5 * dt * k1_y
        vx2 = vx + 0.5 * dt * k1_vx
        vy2 = vy + 0.5 * dt * k1_vy
        ax2, ay2 = self._compute_acceleration(x2, y2, vx2, vy2, rx, ry, mass_factor, mu_roll)
        k2_x, k2_y = vx2, vy2
        k2_vx, k2_vy = ax2, ay2

        # k3
        x3 = x + 0.5 * dt * k2_x
        y3 = y + 0.5 * dt * k2_y
        vx3 = vx + 0.5 * dt * k2_vx
        vy3 = vy + 0.5 * dt * k2_vy
        ax3, ay3 = self._compute_acceleration(x3, y3, vx3, vy3, rx, ry, mass_factor, mu_roll)
        k3_x, k3_y = vx3, vy3
        k3_vx, k3_vy = ax3, ay3

        # k4
        x4 = x + dt * k3_x
        y4 = y + dt * k3_y
        vx4 = vx + dt * k3_vx
        vy4 = vy + dt * k3_vy
        ax4, ay4 = self._compute_acceleration(x4, y4, vx4, vy4, rx, ry, mass_factor, mu_roll)
        k4_x, k4_y = vx4, vy4
        k4_vx, k4_vy = ax4, ay4

        # Combine
        new_x = x + (dt / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
        new_y = y + (dt / 6.0) * (k1_y + 2.0 * k2_y + 2.0 * k3_y + k4_y)
        new_vx = vx + (dt / 6.0) * (k1_vx + 2.0 * k2_vx + 2.0 * k3_vx + k4_vx)
        new_vy = vy + (dt / 6.0) * (k1_vy + 2.0 * k2_vy + 2.0 * k3_vy + k4_vy)

        return new_x, new_y, new_vx, new_vy

    def _randomize_platform_offset(self, mask):
        """Randomize platform placement offset for environments where mask is True."""
        if not self.cfg.use_domain_randomization:
            self.platform_offset_rx[mask] = 0.0
            self.platform_offset_ry[mask] = 0.0
            return

        n = mask.sum().item()
        if n == 0:
            return

        # Random magnitude 0 to max_deg
        magnitude = torch.rand(n, device=self.device) * self.platform_offset_max_deg
        # Random direction
        direction = torch.rand(n, device=self.device) * 2 * np.pi

        # Convert to rx, ry offsets (in radians)
        self.platform_offset_rx[mask] = torch.deg2rad(magnitude * torch.sin(direction))
        self.platform_offset_ry[mask] = torch.deg2rad(magnitude * torch.cos(direction))

    def _randomize_physics(self, mask):
        """Randomize physics parameters for environments where mask is True."""
        n = mask.sum().item()
        if n == 0:
            return

        if self.cfg.use_domain_randomization:
            # Friction
            self.mu_roll[mask] = torch.rand(n, device=self.device) * \
                (self.cfg.friction_range[1] - self.cfg.friction_range[0]) + self.cfg.friction_range[0]

            # Servo tau
            self.servo_tau[mask] = torch.rand(n, device=self.device) * \
                (self.cfg.servo_tau_range[1] - self.cfg.servo_tau_range[0]) + self.cfg.servo_tau_range[0]

            # Mass factor
            self.mass_factor[mask] = torch.rand(n, device=self.device) * \
                (self.cfg.mass_factor_range[1] - self.cfg.mass_factor_range[0]) + self.cfg.mass_factor_range[0]

            # Base dt
            self.base_dt[mask] = torch.rand(n, device=self.device) * \
                (self.dt_range[1] - self.dt_range[0]) + self.dt_range[0]
        else:
            self.mu_roll[mask] = self.default_mu_roll
            self.servo_tau[mask] = self.default_servo_tau
            self.mass_factor[mask] = self.default_mass_factor
            self.base_dt[mask] = self.cfg.dt

    def _sample_dt(self):
        """Sample actual dt for this step with variance and jitter."""
        # Base dt with variance
        variance = 1.0 + (torch.rand(self.num_envs, device=self.device) * 2 - 1) * self.dt_variance
        self.actual_dt = self.base_dt * variance

        # Jitter (0.2% chance of 50ms)
        jitter_mask = torch.rand(self.num_envs, device=self.device) < self.jitter_prob
        self.actual_dt[jitter_mask] = self.jitter_dt

        # Clamp
        self.actual_dt = torch.clamp(self.actual_dt, 0.001, 0.060)

    def get_physics_normalized(self):
        """Get normalized physics parameters [0, 1] for auxiliary loss."""
        physics = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)

        fr = self.cfg.friction_range
        sr = self.cfg.servo_tau_range
        mr = self.cfg.mass_factor_range

        physics[:, 0] = (self.mu_roll - fr[0]) / (fr[1] - fr[0])
        physics[:, 1] = (self.servo_tau - sr[0]) / (sr[1] - sr[0])
        physics[:, 2] = (self.mass_factor - mr[0]) / (mr[1] - mr[0])

        return torch.clamp(physics, 0, 1)

    def reset(self, indices=None):
        """Reset environments. Returns numpy arrays."""
        if indices is None:
            mask = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        else:
            mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            mask[indices] = True

        # Randomize physics and platform offset
        self._randomize_physics(mask)
        self._randomize_platform_offset(mask)

        # Random initial position
        init_pos_m = self.cfg.init_pos_range_mm / 1000.0
        n = mask.sum().item()
        self.ball_x[mask] = (torch.rand(n, device=self.device) * 2 - 1) * init_pos_m
        self.ball_y[mask] = (torch.rand(n, device=self.device) * 2 - 1) * init_pos_m

        # Random initial velocity
        init_vel_m_s = self.cfg.init_vel_range_mm_s / 1000.0
        self.ball_vx[mask] = (torch.rand(n, device=self.device) * 2 - 1) * init_vel_m_s
        self.ball_vy[mask] = (torch.rand(n, device=self.device) * 2 - 1) * init_vel_m_s

        # Platform starts level
        self.platform_rx[mask] = 0.0
        self.platform_ry[mask] = 0.0

        # Reset counters
        self.step_count[mask] = 0
        self.episode_reward[mask] = 0.0
        self.prev_actions[mask] = 0.0

        # Initialize actual_dt
        self.actual_dt[mask] = self.base_dt[mask]

        # Initialize frame history
        initial_obs = self._get_single_frame()
        for i in range(self.cfg.num_frames):
            self.frame_history[mask, i, :] = initial_obs[mask]

        # Return numpy
        info = {
            'physics_gt': self.get_physics_normalized().cpu().numpy(),
            'base_dt': self.base_dt.cpu().numpy(),
        }
        return self._get_observation().cpu().numpy(), info

    def reset_tensor(self, indices=None):
        """Reset environments. Returns GPU tensors (no CPU transfer)."""
        if indices is None:
            mask = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        else:
            mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            mask[indices] = True

        # Randomize physics and platform offset
        self._randomize_physics(mask)
        self._randomize_platform_offset(mask)

        # Random initial position
        init_pos_m = self.cfg.init_pos_range_mm / 1000.0
        n = mask.sum().item()
        self.ball_x[mask] = (torch.rand(n, device=self.device) * 2 - 1) * init_pos_m
        self.ball_y[mask] = (torch.rand(n, device=self.device) * 2 - 1) * init_pos_m

        # Random initial velocity
        init_vel_m_s = self.cfg.init_vel_range_mm_s / 1000.0
        self.ball_vx[mask] = (torch.rand(n, device=self.device) * 2 - 1) * init_vel_m_s
        self.ball_vy[mask] = (torch.rand(n, device=self.device) * 2 - 1) * init_vel_m_s

        # Platform starts level
        self.platform_rx[mask] = 0.0
        self.platform_ry[mask] = 0.0

        # Reset counters
        self.step_count[mask] = 0
        self.episode_reward[mask] = 0.0
        self.prev_actions[mask] = 0.0

        # Initialize actual_dt
        self.actual_dt[mask] = self.base_dt[mask]

        # Initialize frame history
        initial_obs = self._get_single_frame()
        for i in range(self.cfg.num_frames):
            self.frame_history[mask, i, :] = initial_obs[mask]

        # Return tensors (no CPU transfer)
        return self._get_observation(), self.get_physics_normalized()

    def _get_single_frame(self):
        """Get single frame observation for all environments."""
        obs = torch.zeros((self.num_envs, self.cfg.obs_per_frame),
                          device=self.device, dtype=torch.float32)

        # Ball position with noise
        ball_x_mm = self.ball_x * 1000.0
        ball_y_mm = self.ball_y * 1000.0

        if self.cfg.use_camera_noise:
            noise_x = torch.randn(self.num_envs, device=self.device) * self.cfg.position_noise_std_mm
            noise_y = torch.randn(self.num_envs, device=self.device) * self.cfg.position_noise_std_mm
            dist = torch.sqrt(ball_x_mm**2 + ball_y_mm**2)
            depth_scale = 1.0 + dist * self.cfg.noise_depth_scale
            noise_x = noise_x * depth_scale
            noise_y = noise_y * depth_scale
            ball_x_mm = ball_x_mm + noise_x
            ball_y_mm = ball_y_mm + noise_y

        # Normalize position
        obs[:, 0] = ball_x_mm / self.cfg.platform_radius_mm
        obs[:, 1] = ball_y_mm / self.cfg.platform_radius_mm

        # Normalize platform angles
        obs[:, 2] = torch.rad2deg(self.platform_rx) / self.cfg.max_tilt_deg
        obs[:, 3] = torch.rad2deg(self.platform_ry) / self.cfg.max_tilt_deg

        # dt normalized
        obs[:, 4] = self.actual_dt / 0.01

        # Target position (normalized)
        obs[:, 5] = self.target_x / self.cfg.platform_radius_mm
        obs[:, 6] = self.target_y / self.cfg.platform_radius_mm

        return obs

    def _get_observation(self):
        """Get flattened observation."""
        return self.frame_history.reshape(self.num_envs, -1)

    def step(self, actions):
        """
        Step all environments.

        Args:
            actions: numpy array (num_envs, 2) in [-1, 1]

        Returns:
            obs, rewards, dones, truncated, info (all numpy)
        """
        # Convert to tensor
        actions = torch.from_numpy(actions).to(self.device, dtype=torch.float32)

        # Sample dt
        self._sample_dt()

        # Scale actions
        action_rx = actions[:, 0] * self.max_tilt_rad
        action_ry = actions[:, 1] * self.max_tilt_rad

        # Servo dynamics (vectorized)
        decay = torch.exp(-self.actual_dt / self.servo_tau)
        new_rx = action_rx + (self.platform_rx - action_rx) * decay
        new_ry = action_ry + (self.platform_ry - action_ry) * decay

        # Clamp
        new_rx = torch.clamp(new_rx, -self.max_tilt_rad, self.max_tilt_rad)
        new_ry = torch.clamp(new_ry, -self.max_tilt_rad, self.max_tilt_rad)

        self.platform_rx = new_rx
        self.platform_ry = new_ry

        # Effective tilt (with platform offset)
        effective_rx = new_rx + self.platform_offset_rx
        effective_ry = new_ry + self.platform_offset_ry

        # Ball physics (fully vectorized RK4)
        new_x, new_y, new_vx, new_vy = self._rk4_step(
            self.ball_x, self.ball_y, self.ball_vx, self.ball_vy,
            effective_rx, effective_ry, self.actual_dt,
            self.mass_factor, self.mu_roll
        )

        # Check bounds
        dist = torch.sqrt(new_x**2 + new_y**2)
        fell_off = dist > self.platform_radius_m

        # Reset fallen balls to center (they'll get terminal reward)
        new_x = torch.where(fell_off, torch.zeros_like(new_x), new_x)
        new_y = torch.where(fell_off, torch.zeros_like(new_y), new_y)
        new_vx = torch.where(fell_off, torch.zeros_like(new_vx), new_vx)
        new_vy = torch.where(fell_off, torch.zeros_like(new_vy), new_vy)

        self.ball_x = new_x
        self.ball_y = new_y
        self.ball_vx = new_vx
        self.ball_vy = new_vy

        # Compute rewards
        rewards = self._compute_reward(actions, fell_off)

        # Update tracking
        self.prev_actions = actions.clone()
        self.step_count += 1
        self.episode_reward += rewards

        # Update frame history
        new_frame = self._get_single_frame()
        self.frame_history[:, :-1, :] = self.frame_history[:, 1:, :].clone()
        self.frame_history[:, -1, :] = new_frame

        # Check done
        dones = fell_off | (self.step_count >= self.cfg.max_steps)
        truncated = self.step_count >= self.cfg.max_steps

        # Return numpy
        info = {
            'fell_off': fell_off.cpu().numpy(),
            'episode_reward': self.episode_reward.cpu().numpy(),
            'step_count': self.step_count.cpu().numpy(),
            'physics_gt': self.get_physics_normalized().cpu().numpy(),
            'actual_dt': self.actual_dt.cpu().numpy(),
            'base_dt': self.base_dt.cpu().numpy(),
        }

        return (
            self._get_observation().cpu().numpy(),
            rewards.cpu().numpy(),
            dones.cpu().numpy(),
            truncated.cpu().numpy(),
            info
        )

    def step_tensor(self, actions):
        """
        Step all environments (GPU tensor version - no CPU transfer).

        Args:
            actions: GPU tensor (num_envs, 2) in [-1, 1]

        Returns:
            obs: GPU tensor (num_envs, obs_dim)
            rewards: GPU tensor (num_envs,)
            dones: GPU tensor (num_envs,) bool
            physics_gt: GPU tensor (num_envs, 3)
        """
        # Sample dt
        self._sample_dt()

        # Scale actions
        action_rx = actions[:, 0] * self.max_tilt_rad
        action_ry = actions[:, 1] * self.max_tilt_rad

        # Servo dynamics (vectorized)
        decay = torch.exp(-self.actual_dt / self.servo_tau)
        new_rx = action_rx + (self.platform_rx - action_rx) * decay
        new_ry = action_ry + (self.platform_ry - action_ry) * decay

        # Clamp
        new_rx = torch.clamp(new_rx, -self.max_tilt_rad, self.max_tilt_rad)
        new_ry = torch.clamp(new_ry, -self.max_tilt_rad, self.max_tilt_rad)

        self.platform_rx = new_rx
        self.platform_ry = new_ry

        # Effective tilt (with platform offset)
        effective_rx = new_rx + self.platform_offset_rx
        effective_ry = new_ry + self.platform_offset_ry

        # Ball physics (fully vectorized RK4)
        new_x, new_y, new_vx, new_vy = self._rk4_step(
            self.ball_x, self.ball_y, self.ball_vx, self.ball_vy,
            effective_rx, effective_ry, self.actual_dt,
            self.mass_factor, self.mu_roll
        )

        # Check bounds
        dist = torch.sqrt(new_x**2 + new_y**2)
        fell_off = dist > self.platform_radius_m

        # Reset fallen balls to center (they'll get terminal reward)
        new_x = torch.where(fell_off, torch.zeros_like(new_x), new_x)
        new_y = torch.where(fell_off, torch.zeros_like(new_y), new_y)
        new_vx = torch.where(fell_off, torch.zeros_like(new_vx), new_vx)
        new_vy = torch.where(fell_off, torch.zeros_like(new_vy), new_vy)

        self.ball_x = new_x
        self.ball_y = new_y
        self.ball_vx = new_vx
        self.ball_vy = new_vy

        # Compute rewards
        rewards = self._compute_reward(actions, fell_off)

        # Update tracking
        self.prev_actions = actions.clone()
        self.step_count += 1
        self.episode_reward += rewards

        # Update frame history
        new_frame = self._get_single_frame()
        self.frame_history[:, :-1, :] = self.frame_history[:, 1:, :].clone()
        self.frame_history[:, -1, :] = new_frame

        # Check done
        dones = fell_off | (self.step_count >= self.cfg.max_steps)

        # Return tensors (no CPU transfer)
        return (
            self._get_observation(),
            rewards,
            dones,
            self.get_physics_normalized()
        )

    def _compute_reward(self, actions, fell_off):
        """Compute reward (vectorized)."""
        cfg = self.reward_cfg

        pos_x_mm = self.ball_x * 1000.0
        pos_y_mm = self.ball_y * 1000.0
        pos_error_sq = pos_x_mm**2 + pos_y_mm**2

        vel_x_mm_s = self.ball_vx * 1000.0
        vel_y_mm_s = self.ball_vy * 1000.0
        vel_sq = vel_x_mm_s**2 + vel_y_mm_s**2
        speed_mm_s = torch.sqrt(vel_sq)

        rx_deg = torch.rad2deg(self.platform_rx)
        ry_deg = torch.rad2deg(self.platform_ry)
        tilt_sq = rx_deg**2 + ry_deg**2

        action_sq = actions[:, 0]**2 + actions[:, 1]**2
        action_delta = actions - self.prev_actions
        action_rate_sq = action_delta[:, 0]**2 + action_delta[:, 1]**2

        reward = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        reward -= cfg.k_position * pos_error_sq
        reward -= cfg.k_velocity * vel_sq
        reward -= cfg.k_tilt * tilt_sq
        reward -= cfg.k_action * action_sq
        reward -= cfg.k_action_rate * action_rate_sq

        dist_mm = torch.sqrt(pos_error_sq)
        center_bonus = cfg.k_center_bonus * torch.exp(-dist_mm / cfg.center_threshold_mm)
        reward += center_bonus

        stability_mask = (dist_mm < cfg.center_threshold_mm) & (speed_mm_s < cfg.stability_vel_threshold)
        reward += cfg.k_stability_bonus * stability_mask.float()

        reward = torch.where(fell_off, torch.full_like(reward, cfg.out_of_bounds_penalty), reward)

        return reward


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("Testing StewartEnvGPU...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    env = StewartEnvGPU(num_envs=100, device=device)
    print(f"Created env with {env.num_envs} parallel envs")
    print(f"  Num frames: {env.cfg.num_frames}")
    print(f"  Obs per frame: {env.cfg.obs_per_frame}")
    print(f"  Obs dim: {env.cfg.obs_dim}")

    # Reset
    obs, info = env.reset()
    print(f"\nAfter reset:")
    print(f"  obs shape: {obs.shape}")
    print(f"  physics_gt shape: {info['physics_gt'].shape}")

    # Benchmark
    import time
    n_steps = 1000
    actions = np.random.uniform(-1, 1, (100, 2)).astype(np.float32)

    # Warmup
    for _ in range(10):
        obs, r, d, t, info = env.step(actions)

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_steps):
        obs, r, d, t, info = env.step(actions)
    elapsed = time.perf_counter() - start

    steps_per_sec = (n_steps * 100) / elapsed
    print(f"\nBenchmark ({n_steps} steps × 100 envs):")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Steps/sec: {steps_per_sec:,.0f}")

    print("\n[OK] StewartEnvGPU test passed!")
