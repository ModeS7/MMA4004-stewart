"""
RL Configuration for Stewart Platform Ball Balancing

All hyperparameters for SAC training.
Based on successful Pendulum RL project.
"""

import numpy as np


# ============================================================================
# ENVIRONMENT CONFIGURATION
# ============================================================================

class EnvConfig:
    """Environment settings."""

    # Physics
    dt = 0.01  # Time step (10ms, 100Hz control)
    max_steps = 1000  # Steps per episode - longer for stability learning

    # Platform limits (from core/utils.py)
    platform_radius_mm = 150.0  # Platform radius in mm
    max_tilt_deg = 10.0  # Maximum tilt angle

    # Ball properties (table tennis ball)
    ball_radius_m = 0.02  # 20mm radius
    ball_mass_kg = 0.0027  # 2.7g

    # Frame history settings
    num_frames = 12  # Number of frames in history
    obs_per_frame = 7  # [ball_x, ball_y, platform_rx, platform_ry, dt, target_x, target_y]
    obs_dim = num_frames * obs_per_frame  # 12 * 7 = 84

    # Action space: [rx_target, ry_target] normalized to [-1, 1]
    action_dim = 2
    action_scale = max_tilt_deg  # Maps [-1, 1] to [-max_tilt, max_tilt]

    # Physics estimation output
    physics_dim = 3  # [friction, servo_tau, mass_factor]

    # Initial state randomization
    init_pos_range_mm = 100.0  # Ball starts within [-100, 100] mm (harder)
    init_vel_range_mm_s = 50.0  # Initial velocity range

    # Camera noise (ZED camera model)
    use_camera_noise = False
    position_noise_std_mm = 2.0  # 2mm std noise
    # Noise can also scale with distance from camera (depth-dependent)
    noise_depth_scale = 0.0

    # Domain randomization (for sim-to-real transfer)
    # Only platform tilt offset enabled (simulates gravity vector)
    use_domain_randomization = False  # Physics params fixed
    use_platform_offset = True        # Random tilt offset enabled
    platform_offset_max_deg = 2.0     # Max platform tilt offset (degrees)
    # Physics parameter ranges for randomization (used when enabled)
    # Default rolling_friction = 0.0225 in SimpleBallPhysics2D
    friction_range = (0.015, 0.035)     # Rolling friction coefficient (centered on 0.0225)
    servo_tau_range = (0.03, 0.08)      # Servo time constant (30-80ms)
    mass_factor_range = (1.4, 2.0)      # Ball inertia factor (1.67 nominal)


# ============================================================================
# REWARD CONFIGURATION
# ============================================================================

class RewardConfig:
    """Reward function weights."""

    # Penalties (negative)
    k_position = 0.01  # Position error: -k * (x^2 + y^2) in mm^2 [5x increase to punish off-center]
    k_velocity = 0.005  # Velocity penalty: -k * (vx^2 + vy^2) [increased 50x from original]
    k_tilt = 0.01  # Tilt penalty: -k * (rx^2 + ry^2)
    k_action = 0.001  # Action magnitude: -k * (a^2)
    k_action_rate = 0.2  # Action rate penalty: -k * (da^2) [increased for smoother control]

    # Bonus (positive)
    k_center_bonus = 1.0  # Bonus for being centered
    center_threshold_mm = 10.0  # Distance to get full bonus
    k_stability_bonus = 1.0  # Bonus for being centered AND slow [increased]
    stability_vel_threshold = 10.0  # Velocity threshold for stability bonus (mm/s) [tighter]

    # Termination
    out_of_bounds_penalty = -100.0  # Ball fell off

    # Multiplicative reward scales (used by GPU env)
    dist_scale = 30.0   # ~30mm from center gives 0.5 distance factor
    speed_scale = 50.0  # ~50mm/s gives 0.5 speed factor
    fall_penalty = -10.0  # Penalty when ball falls off
    approach_scale = 0.5  # Approach velocity bonus: reward moving towards target when far


# ============================================================================
# SAC CONFIGURATION
# ============================================================================

class SACConfig:
    """SAC hyperparameters - tuned for parallel GPU training (1000+ envs)."""

    # Network architecture
    hidden_dim = 256  # Hidden layer size
    use_layer_norm = True  # LayerNorm on critic for stability

    # Learning rates (separate for actor/critic per CleanRL/research)
    actor_lr = 3e-4  # Policy learning rate
    critic_lr = 3e-4  # Q-network learning rate (same as actor for stability)

    # SAC parameters (tuned for parallel training with TD3-style stability)
    gamma = 0.99  # Discount factor
    tau = 0.005  # Soft update coefficient (slower for stability)
    alpha = 0.01  # Initial entropy coefficient (lower for parallel - research shows 0.001-0.02)
    automatic_entropy_tuning = True
    policy_delay = 8  # Update actor every N critic updates (TD3-style, prevents following unstable Q)

    # Replay buffer (must be large enough to hold demos + multiple episodes)
    # With 1000 envs × 1000 steps = 1M transitions/episode
    # 10M buffer retains ~10 episodes worth + demos
    buffer_size = 10_000_000  # 10M to retain demos longer
    batch_size = 1024  # Larger batch = more GPU utilization

    # Training (higher update ratio for better sample efficiency)
    warmup_steps = 1000  # Random actions before training (skipped if demos)
    updates_per_step = 32  # Gradient updates per update call
    update_every = 4  # Update every N environment steps (more frequent)


# ============================================================================
# PPO CONFIGURATION
# ============================================================================

class PPOConfig:
    """PPO hyperparameters - tuned for parallel GPU training (1000+ envs)."""

    # Network architecture (same as SAC for consistency)
    hidden_dim = 256

    # PPO parameters
    learning_rate = 3e-4
    gamma = 0.99              # Discount factor
    gae_lambda = 0.95         # GAE lambda
    clip_range = 0.2          # PPO clipping parameter
    clip_range_vf = None      # Value function clipping (None = disabled)

    # Loss coefficients
    ent_coef = 0.01           # Entropy bonus coefficient
    vf_coef = 0.5             # Value function loss coefficient
    max_grad_norm = 0.5       # Gradient clipping

    # Rollout settings (for 1000 parallel envs)
    n_steps = 32              # Steps per env before update (32 * 1000 = 32k samples)
    n_epochs = 4              # Epochs per update
    batch_size = 1024         # Minibatch size


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

class TrainingConfig:
    """Training settings."""

    max_episodes = 1000

    # Logging
    log_interval = 10  # Print every N episodes
    eval_interval = 50  # Evaluate every N episodes
    save_interval = 100  # Save model every N episodes

    # Paths
    save_dir = "RL/checkpoints"
    log_dir = "RL/runs"

    # Hardware
    device = "cuda"  # "cuda" or "cpu"
    seed = 42


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_configs():
    """Return all config objects."""
    return {
        'env': EnvConfig(),
        'reward': RewardConfig(),
        'sac': SACConfig(),
        'ppo': PPOConfig(),
        'training': TrainingConfig()
    }


if __name__ == "__main__":
    print("Stewart Platform RL Configuration")
    print("=" * 50)

    env = EnvConfig()
    reward = RewardConfig()
    sac = SACConfig()
    train = TrainingConfig()

    print(f"\nEnvironment:")
    print(f"  Obs dim: {env.obs_dim} ({env.num_frames} frames × {env.obs_per_frame})")
    print(f"  Action dim: {env.action_dim}")
    print(f"  Max steps: {env.max_steps} ({env.max_steps * env.dt}s)")
    print(f"  Platform radius: {env.platform_radius_mm}mm")
    print(f"  Max tilt: {env.max_tilt_deg} deg")

    print(f"\nSAC:")
    print(f"  Hidden dim: {sac.hidden_dim}")
    print(f"  Learning rate: {sac.lr}")
    print(f"  Batch size: {sac.batch_size}")
    print(f"  Buffer size: {sac.buffer_size}")

    print(f"\nTraining:")
    print(f"  Max episodes: {train.max_episodes}")
    print(f"  Device: {train.device}")
