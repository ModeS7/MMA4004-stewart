"""
Unified SAC Training for Stewart Platform Ball Balancing

Single script supporting both MLP and CNN architectures.
"""

import os
import sys
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from RL.env import StewartEnvVec
from RL.rl_config import EnvConfig, RewardConfig
from RL.sac_agent import SACAgent, ReplayBuffer


# ============================================================================
# CONFIGURATION - MODIFY THESE
# ============================================================================

# Architecture: "mlp" (simple) or "cnn" (temporal)
ARCHITECTURE = "cnn"

# Training
NUM_ENVS = 1              # Single environment
MAX_EPISODES = 3000       # Training episodes
MAX_STEPS = 1000          # Steps per episode (30 seconds at 100Hz)
BATCH_SIZE = 512          # Batch size for updates
BUFFER_SIZE = 300_000     # Replay buffer size

# SAC Hyperparameters
HIDDEN_DIM = 256          # Network hidden layer size
LR = 3e-4                 # Learning rate
GAMMA = 0.99              # Discount factor
TAU = 0.005               # Soft update coefficient
ALPHA = 0.2               # Initial entropy coefficient
AUTOMATIC_ENTROPY = True  # Auto-tune entropy

# CNN-specific (only used if ARCHITECTURE="cnn")
USE_PHYSICS_HEAD = False  # Enable physics estimation auxiliary task
PHYSICS_DIM = 3           # Physics estimation dimension
PHYSICS_LOSS_WEIGHT = 0.1 # Weight for physics auxiliary loss

# Domain Randomization (for sim-to-real transfer)
RANDOMIZATION = True              # Enable/disable all randomization
TARGET_RANDOMIZATION = True       # Randomize target position within 100mm radius

# Training Options
WARMUP_STEPS = 1000       # Random actions before training
EVAL_INTERVAL = 5        # Evaluate every N episodes
SAVE_INTERVAL = 10        # Save model every N episodes

# Performance optimizations
USE_COMPILE = True        # torch.compile for faster NN (1.5x speedup after warmup)
USE_AMP = False           # Mixed precision (FP16) - disabled: adds overhead for small networks

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Enable TensorFloat32 for faster matmul on Ampere+ GPUs
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')

# Resume from checkpoint (set to path or None)
CHECKPOINT = "./RL/checkpoints/sac_cnn_20251130_223233/sac_cnn_ep970.pt"


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train():
    """Train the SAC agent on the Stewart platform environment."""
    print("=" * 60)
    print(f"SAC Training - {ARCHITECTURE.upper()} Architecture")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Architecture: {ARCHITECTURE}")
    print(f"Episodes: {MAX_EPISODES}")
    print(f"Steps per episode: {MAX_STEPS}")
    print(f"Buffer size: {BUFFER_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    if ARCHITECTURE == "cnn":
        print(f"Physics head: {USE_PHYSICS_HEAD}")
    print(f"Randomization: {RANDOMIZATION}")
    print()

    # Load configs
    env_cfg = EnvConfig()
    env_cfg.max_steps = MAX_STEPS
    env_cfg.use_domain_randomization = RANDOMIZATION
    env_cfg.use_platform_offset = RANDOMIZATION
    env_cfg.use_camera_noise = RANDOMIZATION
    env_cfg.use_dt_randomization = RANDOMIZATION
    env_cfg.randomize_target = TARGET_RANDOMIZATION
    reward_cfg = RewardConfig()

    # Create simulation environment
    env = StewartEnvVec(
        num_envs=NUM_ENVS,
        config=env_cfg,
        reward_config=reward_cfg,
        device=DEVICE
    )

    state_dim = env_cfg.obs_dim  # 84
    action_dim = env_cfg.action_dim  # 2

    print(f"State dim: {state_dim}")
    print(f"Action dim: {action_dim}")

    # Create agent
    agent = SACAgent(
        architecture=ARCHITECTURE,
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=HIDDEN_DIM,
        num_frames=env_cfg.num_frames,
        obs_per_frame=env_cfg.obs_per_frame,
        use_physics_head=USE_PHYSICS_HEAD,
        physics_dim=PHYSICS_DIM,
        physics_loss_weight=PHYSICS_LOSS_WEIGHT,
        lr=LR,
        gamma=GAMMA,
        tau=TAU,
        alpha=ALPHA,
        automatic_entropy=AUTOMATIC_ENTROPY,
        device=DEVICE,
        compile_model=USE_COMPILE,
        use_amp=USE_AMP
    )
    print(f"Agent created on {agent.device}")
    if USE_COMPILE:
        print("torch.compile enabled (first episodes slower due to compilation)")
    if USE_AMP:
        print("Mixed precision (AMP) enabled for faster training")

    # Load checkpoint if specified
    start_episode = 0
    if CHECKPOINT is not None:
        agent.load(CHECKPOINT)
        print(f"Loaded checkpoint: {CHECKPOINT}")
        try:
            start_episode = int(CHECKPOINT.split('_ep')[-1].split('.')[0])
            print(f"Resuming from episode {start_episode}")
        except:
            pass

    # Create replay buffer
    buffer = ReplayBuffer(BUFFER_SIZE, state_dim, action_dim, device=DEVICE)
    print(f"Replay buffer created (capacity: {BUFFER_SIZE})")

    # Create run directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = f"RL/checkpoints/sac_{ARCHITECTURE}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    # TensorBoard
    writer = SummaryWriter(run_dir)
    print(f"Run directory: {run_dir}")

    # Training metrics
    episode_rewards = []
    episode_lengths = []
    total_steps = 0
    training_start_time = time.time()

    print()
    print("=" * 60)
    print("Starting training...")
    print("=" * 60)

    for episode in range(start_episode, MAX_EPISODES):
        episode_start_time = time.time()

        # Reset environment
        obs_batch, _ = env.reset()

        episode_reward = 0
        episode_length = 0

        for step in range(MAX_STEPS):
            # Select actions for all envs
            if total_steps < WARMUP_STEPS:
                actions = np.random.uniform(-1, 1, (NUM_ENVS, action_dim)).astype(np.float32)
            else:
                actions = np.array([agent.select_action(obs_batch[i], evaluate=False)
                                   for i in range(NUM_ENVS)])

            # Step environment
            next_obs_batch, rewards, dones, truncateds, _ = env.step(actions)

            # Store transitions for all envs
            for i in range(NUM_ENVS):
                buffer.push(obs_batch[i], actions[i], rewards[i], next_obs_batch[i],
                           float(dones[i] or truncateds[i]))

            obs_batch = next_obs_batch
            episode_reward += rewards.sum()
            episode_length += NUM_ENVS
            total_steps += NUM_ENVS

            # Update agent
            if total_steps >= WARMUP_STEPS and len(buffer) >= BATCH_SIZE:
                update_info = agent.update(buffer, BATCH_SIZE)

                # Log to TensorBoard
                if total_steps % 1000 == 0:
                    writer.add_scalar('Loss/Critic', update_info['critic_loss'], total_steps)
                    writer.add_scalar('Loss/Actor', update_info['actor_loss'], total_steps)
                    writer.add_scalar('Params/Alpha', update_info['alpha'], total_steps)
                    if USE_PHYSICS_HEAD and ARCHITECTURE == "cnn":
                        writer.add_scalar('Loss/Physics', update_info['physics_loss'], total_steps)

        # Episode complete
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # Log episode metrics
        writer.add_scalar('Episode/Reward', episode_reward, episode)
        writer.add_scalar('Episode/Length', episode_length, episode)
        writer.add_scalar('Episode/Time', time.time() - episode_start_time, episode)

        # Print progress
        episode_time = time.time() - episode_start_time
        elapsed_time = time.time() - training_start_time
        steps_per_sec = total_steps / elapsed_time if elapsed_time > 0 else 0
        avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)

        # ETA calculation
        episodes_done = episode - start_episode + 1
        episodes_remaining = MAX_EPISODES - episode - 1
        if episodes_done > 0:
            eta_seconds = (elapsed_time / episodes_done) * episodes_remaining
            eta_str = f"{eta_seconds/3600:.1f}h" if eta_seconds > 3600 else f"{eta_seconds/60:.0f}m"
        else:
            eta_str = "?"

        print(f"Ep {episode + 1:4d} | "
              f"R: {episode_reward:7.1f} | "
              f"Avg: {avg_reward:7.1f} | "
              f"α: {agent.alpha:.3f} | "
              f"T: {episode_time:.1f}s | "
              f"SPS: {steps_per_sec:.0f} | "
              f"ETA: {eta_str}")

        # Save checkpoint
        if (episode + 1) % SAVE_INTERVAL == 0:
            checkpoint_path = os.path.join(run_dir, f"sac_{ARCHITECTURE}_ep{episode + 1}.pt")
            agent.save(checkpoint_path)
            print(f"  [SAVE] {checkpoint_path}")

    # Save final model
    final_path = os.path.join(run_dir, f"sac_{ARCHITECTURE}_final.pt")
    agent.save(final_path)
    print(f"\nFinal model saved: {final_path}")

    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(episode_rewards, alpha=0.3)
    if len(episode_rewards) >= 10:
        smoothed = np.convolve(episode_rewards, np.ones(10) / 10, mode='valid')
        axes[0].plot(range(9, len(episode_rewards)), smoothed, 'r-', linewidth=2)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].set_title(f'Episode Rewards ({ARCHITECTURE.upper()})')
    axes[0].grid(True)

    axes[1].plot(episode_lengths)
    axes[1].axhline(y=MAX_STEPS, color='g', linestyle='--', label='Max')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Length')
    axes[1].set_title('Episode Lengths')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'training.png'), dpi=150)
    plt.close()

    writer.close()
    print(f"\nTraining complete!")
    print(f"TensorBoard: tensorboard --logdir {run_dir}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    train()
