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
    max_steps = 1000  # Steps per episode (10 seconds) - longer for stability learning

    # Platform limits (from core/utils.py)
    platform_radius_mm = 150.0  # Platform radius in mm
    max_tilt_deg = 10.0  # Maximum tilt angle

    # Ball properties (table tennis ball)
    ball_radius_m = 0.02  # 20mm radius
    ball_mass_kg = 0.0027  # 2.7g

    # Frame history settings
    num_frames = 12  # Number of frames in history
    obs_per_frame = 5  # [ball_x, ball_y, platform_rx, platform_ry, dt]
    obs_dim = num_frames * obs_per_frame  # 12 * 5 = 60

    # Action space: [rx_target, ry_target] normalized to [-1, 1]
    action_dim = 2
    action_scale = max_tilt_deg  # Maps [-1, 1] to [-max_tilt, max_tilt]

    # Physics estimation output
    physics_dim = 3  # [friction, servo_tau, mass_factor]

    # Initial state randomization
    init_pos_range_mm = 50.0  # Ball starts within [-50, 50] mm
    init_vel_range_mm_s = 30.0  # Initial velocity range

    # Camera noise (ZED camera model)
    # ZED typical noise: ~1-2mm at close range
    use_camera_noise = True
    position_noise_std_mm = 3.0  # 3mm std noise
    # Noise can also scale with distance from camera (depth-dependent)
    noise_depth_scale = 0.1  # Additional noise = distance * scale

    # Domain randomization (for sim-to-real transfer)
    use_domain_randomization = True
    # Physics parameter ranges for randomization
    friction_range = (0.01, 0.05)       # Rolling friction coefficient
    servo_tau_range = (0.03, 0.08)      # Servo time constant (30-80ms)
    mass_factor_range = (1.4, 2.0)      # Ball inertia factor (1.67 nominal)


# ============================================================================
# REWARD CONFIGURATION
# ============================================================================

class RewardConfig:
    """Reward function weights."""

    # Penalties (negative)
    k_position = 0.002  # Position error: -k * (x^2 + y^2) in mm^2 [increased]
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


# ============================================================================
# SAC CONFIGURATION
# ============================================================================

class SACConfig:
    """SAC hyperparameters."""

    # Network architecture
    hidden_dim = 256  # Hidden layer size

    # Learning rates
    lr = 3e-4  # Learning rate for all networks

    # SAC parameters
    gamma = 0.99  # Discount factor
    tau = 0.005  # Soft update coefficient
    alpha = 0.2  # Initial entropy coefficient
    automatic_entropy_tuning = True

    # Replay buffer
    buffer_size = 100_000
    batch_size = 256  # Reduced for sequence data

    # Training
    warmup_steps = 1000  # Random actions before training
    updates_per_step = 1


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
