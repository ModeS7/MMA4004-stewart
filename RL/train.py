"""
Training script for Feedforward SAC with Physics Estimation.

Uses frame stacking (12 frames) and domain randomization for sim-to-real.
"""

import os
import sys
import time
from datetime import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# Add RL directory to path for standalone execution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rl_config import EnvConfig, SACConfig, RewardConfig, TrainingConfig
from env import StewartEnv
from env_gpu import StewartEnvGPU
from agent import SACAgent, ReplayBuffer


# ============================================================================
# TRAINING CONFIGURATION - MODIFY THESE
# ============================================================================

NUM_ENVS = 1000         # Number of parallel environments
MAX_EPISODES = 1500     # Total training episodes
DEVICE = "cuda"         # "cuda" or "cpu"
USE_GPU_ENV = True      # Use GPU-accelerated environment (faster physics)
CHECKPOINT = None       # Path to checkpoint to resume from, or None

# Logging
LOG_INTERVAL = 10       # Print stats every N episodes
EVAL_INTERVAL = 50      # Evaluate every N episodes
SAVE_INTERVAL = 100     # Save checkpoint every N episodes

# ============================================================================


def evaluate(agent, env_config, reward_config, num_episodes=10):
    """Evaluate agent on fresh environments."""
    if USE_GPU_ENV:
        eval_env = StewartEnvGPU(
            num_envs=num_episodes,
            config=env_config,
            reward_config=reward_config,
            device=DEVICE,
            use_domain_randomization=False  # Fixed physics for evaluation
        )
    else:
        eval_env = StewartEnv(
            num_envs=num_episodes,
            config=env_config,
            reward_config=reward_config,
            use_domain_randomization=False  # Fixed physics for evaluation
        )

    obs, _ = eval_env.reset()
    episode_rewards = np.zeros(num_episodes)
    episode_lengths = np.zeros(num_episodes)
    done_mask = np.zeros(num_episodes, dtype=bool)

    while not done_mask.all():
        actions, _ = agent.select_action_batch(obs, evaluate=True)
        obs, rewards, dones, truncated, info = eval_env.step(actions)

        # Accumulate rewards for non-done episodes
        episode_rewards += rewards * (~done_mask)
        episode_lengths += (~done_mask)

        done_mask |= (dones | truncated)

    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'success_rate': np.mean(episode_lengths >= env_config.max_steps)
    }


def train():
    """Main training loop."""
    print("=" * 60)
    print("SAC Training with Physics Estimation")
    print("=" * 60)

    # Load configs
    env_cfg = EnvConfig()
    sac_cfg = SACConfig()
    reward_cfg = RewardConfig()
    train_cfg = TrainingConfig()

    # Override with script settings
    train_cfg.max_episodes = MAX_EPISODES
    train_cfg.device = DEVICE
    train_cfg.log_interval = LOG_INTERVAL
    train_cfg.eval_interval = EVAL_INTERVAL
    train_cfg.save_interval = SAVE_INTERVAL

    print(f"\nConfiguration:")
    print(f"  Device: {train_cfg.device}")
    print(f"  Num envs: {NUM_ENVS}")
    print(f"  Max episodes: {train_cfg.max_episodes}")
    print(f"  Max steps per episode: {env_cfg.max_steps}")
    print(f"  Num frames: {env_cfg.num_frames}")
    print(f"  Obs per frame: {env_cfg.obs_per_frame}")
    print(f"  obs dim: {env_cfg.obs_dim}")
    print(f"  Physics dim: {env_cfg.physics_dim}")
    print(f"  Domain randomization: {env_cfg.use_domain_randomization}")
    print(f"  Camera noise: {env_cfg.use_camera_noise} (std={env_cfg.position_noise_std_mm}mm)")

    # Create environment
    if USE_GPU_ENV:
        env = StewartEnvGPU(
            num_envs=NUM_ENVS,
            config=env_cfg,
            reward_config=reward_cfg,
            device=DEVICE,
            use_domain_randomization=env_cfg.use_domain_randomization
        )
        print(f"\nGPU Environment created with {NUM_ENVS} parallel envs on {DEVICE}")
    else:
        env = StewartEnv(
            num_envs=NUM_ENVS,
            config=env_cfg,
            reward_config=reward_cfg,
            use_domain_randomization=env_cfg.use_domain_randomization
        )
        print(f"\nCPU Environment created with {NUM_ENVS} parallel envs")

    # Create agent
    agent = SACAgent(
        obs_dim=env_cfg.obs_dim,
        action_dim=env_cfg.action_dim,
        hidden_dim=sac_cfg.hidden_dim,
        physics_dim=env_cfg.physics_dim,
        lr=sac_cfg.lr,
        gamma=sac_cfg.gamma,
        tau=sac_cfg.tau,
        alpha=sac_cfg.alpha,
        physics_loss_weight=0.1,
        automatic_entropy_tuning=sac_cfg.automatic_entropy_tuning,
        device=train_cfg.device
    )
    print(f"Agent created on device: {agent.device}")

    # Load checkpoint if specified
    start_episode = 0
    if CHECKPOINT is not None:
        agent.load(CHECKPOINT)
        # Extract episode number from checkpoint name if possible
        try:
            start_episode = int(CHECKPOINT.split('_ep')[-1].split('.')[0])
        except:
            pass
        print(f"Loaded checkpoint: {CHECKPOINT}")

    # Create replay buffer
    buffer = ReplayBuffer(
        capacity=sac_cfg.buffer_size,
        obs_dim=env_cfg.obs_dim,
        action_dim=env_cfg.action_dim,
        physics_dim=env_cfg.physics_dim
    )
    print(f"Replay buffer created with capacity {sac_cfg.buffer_size}")

    # Training metrics
    episode_rewards = []
    episode_lengths = []
    eval_rewards = []
    losses = {'critic': [], 'actor': [], 'physics': [], 'alpha': []}

    # Create run directory with timestamp
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(train_cfg.save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Create TensorBoard writer
    writer = SummaryWriter(run_dir)
    print(f"Run directory: {run_dir}")

    # Log hyperparameters
    writer.add_text("config/env", f"num_frames={env_cfg.num_frames}, obs_per_frame={env_cfg.obs_per_frame}")
    writer.add_text("config/sac", f"hidden_dim={sac_cfg.hidden_dim}, lr={sac_cfg.lr}, gamma={sac_cfg.gamma}")
    writer.add_text("config/training", f"num_envs={NUM_ENVS}, max_episodes={MAX_EPISODES}")

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    total_steps = 0
    train_start_time = time.time()

    for episode in range(start_episode, train_cfg.max_episodes):
        obs, reset_info = env.reset()
        physics_gt = reset_info['physics_gt']  # Ground truth physics for auxiliary loss

        episode_reward = np.zeros(NUM_ENVS)
        episode_length = np.zeros(NUM_ENVS)
        done_mask = np.zeros(NUM_ENVS, dtype=bool)
        step_in_episode = 0  # Track actual steps for update_every

        episode_start_time = time.time()

        while not done_mask.all():
            # Select action
            if total_steps < sac_cfg.warmup_steps:
                # Random actions during warmup
                actions = np.random.uniform(-1, 1, (NUM_ENVS, env_cfg.action_dim)).astype(np.float32)
            else:
                actions, _ = agent.select_action_batch(obs, evaluate=False)

            # Step environment
            next_obs, rewards, dones, truncated, step_info = env.step(actions)

            # Store transitions (vectorized, only for non-done envs)
            buffer.push_batch(
                obs, actions, rewards, next_obs, dones.astype(np.float32),
                physics_gt, mask=~done_mask
            )

            # Update agent (every N steps to speed up training)
            update_every = getattr(sac_cfg, 'update_every', 1)
            step_in_episode += 1
            if (total_steps >= sac_cfg.warmup_steps and
                len(buffer) >= sac_cfg.batch_size and
                step_in_episode % update_every == 0):
                for _ in range(sac_cfg.updates_per_step):
                    update_info = agent.update(buffer, sac_cfg.batch_size)

                # Track losses (periodically)
                if step_in_episode % 100 == 0:
                    losses['critic'].append(update_info['critic_loss'])
                    losses['actor'].append(update_info['actor_loss'])
                    losses['physics'].append(update_info['physics_loss'])
                    losses['alpha'].append(update_info['alpha'])

                    # TensorBoard logging
                    writer.add_scalar("loss/critic", update_info['critic_loss'], total_steps)
                    writer.add_scalar("loss/actor", update_info['actor_loss'], total_steps)
                    writer.add_scalar("loss/physics", update_info['physics_loss'], total_steps)
                    writer.add_scalar("loss/policy", update_info['policy_loss'], total_steps)
                    writer.add_scalar("params/alpha", update_info['alpha'], total_steps)

            # Accumulate rewards
            episode_reward += rewards * (~done_mask)
            episode_length += (~done_mask)

            # Update state
            obs = next_obs
            physics_gt = step_info['physics_gt']
            done_mask |= (dones | truncated)
            total_steps += NUM_ENVS

        # Episode complete
        mean_reward = np.mean(episode_reward)
        mean_length = np.mean(episode_length)
        episode_rewards.append(mean_reward)
        episode_lengths.append(mean_length)

        episode_time = time.time() - episode_start_time

        # TensorBoard episode logging
        writer.add_scalar("episode/reward", mean_reward, episode)
        writer.add_scalar("episode/length", mean_length, episode)
        writer.add_scalar("episode/time", episode_time, episode)
        writer.add_scalar("buffer/size", len(buffer), episode)

        # Logging
        if (episode + 1) % train_cfg.log_interval == 0:
            recent_rewards = episode_rewards[-train_cfg.log_interval:]
            print(f"Episode {episode + 1:4d} | "
                  f"Reward: {mean_reward:7.1f} (avg: {np.mean(recent_rewards):7.1f}) | "
                  f"Length: {mean_length:4.0f} | "
                  f"Alpha: {agent.alpha:.3f} | "
                  f"Buffer: {len(buffer):6d} | "
                  f"Time: {episode_time:.1f}s")

        # Evaluation
        if (episode + 1) % train_cfg.eval_interval == 0:
            eval_info = evaluate(agent, env_cfg, reward_cfg)
            eval_rewards.append(eval_info['mean_reward'])

            # TensorBoard eval logging
            writer.add_scalar("eval/reward_mean", eval_info['mean_reward'], episode)
            writer.add_scalar("eval/reward_std", eval_info['std_reward'], episode)
            writer.add_scalar("eval/length", eval_info['mean_length'], episode)
            writer.add_scalar("eval/success_rate", eval_info['success_rate'], episode)

            print(f"\n  [EVAL] Mean reward: {eval_info['mean_reward']:.1f} ± {eval_info['std_reward']:.1f} | "
                  f"Success rate: {eval_info['success_rate']*100:.0f}% | "
                  f"Mean length: {eval_info['mean_length']:.0f}\n")

        # Save checkpoint
        if (episode + 1) % train_cfg.save_interval == 0:
            checkpoint_path = os.path.join(run_dir, f"sac_ep{episode + 1}.pt")
            agent.save(checkpoint_path)
            print(f"  [SAVE] Checkpoint saved: {checkpoint_path}")

    # Training complete
    total_time = time.time() - train_start_time
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Total time: {total_time / 60:.1f} minutes")
    print(f"Total steps: {total_steps:,}")
    print(f"Final mean reward: {np.mean(episode_rewards[-10:]):.1f}")

    # Save final model
    final_path = os.path.join(run_dir, "sac_final.pt")
    agent.save(final_path)
    print(f"Final model saved: {final_path}")

    # Plot training curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Episode rewards
    axes[0, 0].plot(episode_rewards, alpha=0.3)
    if len(episode_rewards) >= 10:
        smoothed = np.convolve(episode_rewards, np.ones(10)/10, mode='valid')
        axes[0, 0].plot(range(9, len(episode_rewards)), smoothed, 'r-', linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].grid(True)

    # Episode lengths
    axes[0, 1].plot(episode_lengths)
    axes[0, 1].axhline(y=env_cfg.max_steps, color='g', linestyle='--', label='Max steps')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Length')
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Losses
    if losses['critic']:
        axes[1, 0].plot(losses['critic'], label='Critic', alpha=0.7)
        axes[1, 0].plot(losses['actor'], label='Actor', alpha=0.7)
        axes[1, 0].set_xlabel('Update (x100)')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Losses')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

    # Physics loss and alpha
    if losses['physics']:
        ax2 = axes[1, 1]
        ax2.plot(losses['physics'], 'b-', label='Physics Loss')
        ax2.set_xlabel('Update (x100)')
        ax2.set_ylabel('Physics Loss', color='b')
        ax2.tick_params(axis='y', labelcolor='b')

        ax3 = ax2.twinx()
        ax3.plot(losses['alpha'], 'r-', label='Alpha')
        ax3.set_ylabel('Alpha', color='r')
        ax3.tick_params(axis='y', labelcolor='r')
        ax2.set_title('Physics Loss & Entropy Coefficient')
        ax2.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(run_dir, "training.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Training plot saved: {plot_path}")
    plt.close()

    # Close TensorBoard writer
    writer.close()
    print(f"\nTo view TensorBoard: tensorboard --logdir {train_cfg.save_dir}")


if __name__ == "__main__":
    train()
