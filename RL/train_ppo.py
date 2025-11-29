"""
PPO Training script for Stewart Platform Ball Balancing.

On-policy training loop - more stable than SAC for parallel environments.
Based on Isaac Gym's approach (which uses PPO for BallBalance).
"""

import os
import sys
import time
from datetime import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# Enable TensorFloat32 for faster matmul on Ampere+ GPUs
torch.set_float32_matmul_precision('high')

# Add RL directory to path for standalone execution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rl_config import EnvConfig, PPOConfig, RewardConfig, TrainingConfig
from env_gpu import StewartEnvGPU
from ppo_agent import PPOAgent, RolloutBuffer


# ============================================================================
# TRAINING CONFIGURATION - MODIFY THESE
# ============================================================================

NUM_ENVS = 1000         # Number of parallel environments
TOTAL_TIMESTEPS = 10_000_000  # Total training timesteps
DEVICE = "cuda"         # "cuda" or "cpu"
USE_COMPILE = False     # Use torch.compile (can cause issues with PPO)
CHECKPOINT = None       # Path to checkpoint to resume from, or None

# Logging
LOG_INTERVAL = 1        # Print stats every N updates
EVAL_INTERVAL = 10      # Evaluate every N updates
SAVE_INTERVAL = 50      # Save checkpoint every N updates

# ============================================================================


def evaluate(agent, env_config, reward_config, num_episodes=10, device="cuda"):
    """Evaluate agent on fresh environments."""
    eval_env = StewartEnvGPU(
        num_envs=num_episodes,
        config=env_config,
        reward_config=reward_config,
        device=device,
        use_domain_randomization=False  # Fixed physics for evaluation
    )

    obs, _ = eval_env.reset_tensor()
    episode_rewards = torch.zeros(num_episodes, device=device)
    episode_lengths = torch.zeros(num_episodes, device=device)
    done_mask = torch.zeros(num_episodes, device=device, dtype=torch.bool)

    while not done_mask.all():
        with torch.no_grad():
            actions, _, _ = agent.get_action(obs, deterministic=True)

        obs, rewards, dones, _ = eval_env.step_tensor(actions)

        episode_rewards += rewards * (~done_mask)
        episode_lengths += (~done_mask).float()
        done_mask = done_mask | dones

    return {
        'mean_reward': episode_rewards.mean().item(),
        'std_reward': episode_rewards.std().item(),
        'mean_length': episode_lengths.mean().item(),
        'success_rate': (episode_lengths >= env_config.max_steps).float().mean().item()
    }


def train():
    """Main PPO training loop."""
    print("=" * 60)
    print("PPO Training for Stewart Platform")
    print("=" * 60)

    # Load configs
    env_cfg = EnvConfig()
    ppo_cfg = PPOConfig()
    reward_cfg = RewardConfig()
    train_cfg = TrainingConfig()

    print(f"\nConfiguration:")
    print(f"  Device: {DEVICE}")
    print(f"  Num envs: {NUM_ENVS}")
    print(f"  Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"  N steps: {ppo_cfg.n_steps}")
    print(f"  Batch size: {ppo_cfg.batch_size}")
    print(f"  N epochs: {ppo_cfg.n_epochs}")
    print(f"  Samples per update: {NUM_ENVS * ppo_cfg.n_steps:,}")

    # Create environment
    env = StewartEnvGPU(
        num_envs=NUM_ENVS,
        config=env_cfg,
        reward_config=reward_cfg,
        device=DEVICE,
        use_domain_randomization=env_cfg.use_domain_randomization
    )
    print(f"\nGPU Environment created with {NUM_ENVS} parallel envs on {DEVICE}")

    # Create agent
    agent = PPOAgent(
        obs_dim=env_cfg.obs_dim,
        action_dim=env_cfg.action_dim,
        hidden_dim=ppo_cfg.hidden_dim,
        num_frames=env_cfg.num_frames,
        obs_per_frame=env_cfg.obs_per_frame,
        learning_rate=ppo_cfg.learning_rate,
        gamma=ppo_cfg.gamma,
        gae_lambda=ppo_cfg.gae_lambda,
        clip_range=ppo_cfg.clip_range,
        clip_range_vf=ppo_cfg.clip_range_vf,
        ent_coef=ppo_cfg.ent_coef,
        vf_coef=ppo_cfg.vf_coef,
        max_grad_norm=ppo_cfg.max_grad_norm,
        device=DEVICE,
        compile_model=USE_COMPILE
    )
    print(f"PPO Agent created on device: {agent.device}")

    # Load checkpoint if specified
    if CHECKPOINT is not None:
        agent.load(CHECKPOINT)
        print(f"Loaded checkpoint: {CHECKPOINT}")

    # Create rollout buffer
    rollout_buffer = RolloutBuffer(
        n_steps=ppo_cfg.n_steps,
        num_envs=NUM_ENVS,
        obs_dim=env_cfg.obs_dim,
        action_dim=env_cfg.action_dim,
        device=DEVICE
    )
    print(f"Rollout buffer created: {ppo_cfg.n_steps} steps × {NUM_ENVS} envs")

    # Create run directory
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_ppo"
    run_dir = os.path.join(train_cfg.save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # TensorBoard
    writer = SummaryWriter(run_dir)
    print(f"Run directory: {run_dir}")

    # Training metrics
    all_rewards = []
    all_lengths = []

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    total_timesteps = 0
    num_updates = 0
    train_start_time = time.time()

    # Initial reset
    obs, _ = env.reset_tensor()

    # Track episode stats
    episode_rewards = torch.zeros(NUM_ENVS, device=DEVICE)
    episode_lengths = torch.zeros(NUM_ENVS, device=DEVICE)

    while total_timesteps < TOTAL_TIMESTEPS:
        update_start_time = time.time()

        # Collect rollout
        rollout_buffer.reset()

        for step in range(ppo_cfg.n_steps):
            # Get action
            with torch.no_grad():
                action, log_prob, value = agent.get_action(obs)

            # Step environment
            next_obs, rewards, dones, _ = env.step_tensor(action)

            # Store transition
            rollout_buffer.add(obs, action, log_prob, rewards, dones.float(), value)

            # Track episode stats
            episode_rewards += rewards
            episode_lengths += 1

            # Handle episode ends
            done_indices = dones.nonzero(as_tuple=True)[0]
            if len(done_indices) > 0:
                for idx in done_indices:
                    all_rewards.append(episode_rewards[idx].item())
                    all_lengths.append(episode_lengths[idx].item())

                # Reset episode trackers for done envs
                episode_rewards[done_indices] = 0
                episode_lengths[done_indices] = 0

                # Reset done environments
                next_obs_reset, _ = env.reset_tensor(done_indices)
                next_obs[done_indices] = next_obs_reset[done_indices]

            obs = next_obs
            total_timesteps += NUM_ENVS

        # Compute returns and advantages
        with torch.no_grad():
            last_value = agent.get_value(obs)
        rollout_buffer.compute_returns_and_advantages(last_value, ppo_cfg.gamma, ppo_cfg.gae_lambda)

        # Update policy
        update_info = agent.update(rollout_buffer, ppo_cfg.n_epochs, ppo_cfg.batch_size)
        num_updates += 1

        update_time = time.time() - update_start_time

        # Logging
        if num_updates % LOG_INTERVAL == 0 and len(all_rewards) > 0:
            recent_rewards = all_rewards[-100:] if len(all_rewards) > 100 else all_rewards
            recent_lengths = all_lengths[-100:] if len(all_lengths) > 100 else all_lengths

            mean_reward = np.mean(recent_rewards)
            mean_length = np.mean(recent_lengths)

            print(f"Update {num_updates:4d} | "
                  f"Timesteps: {total_timesteps:,} | "
                  f"Reward: {mean_reward:7.1f} | "
                  f"Length: {mean_length:5.0f} | "
                  f"Policy Loss: {update_info['policy_loss']:.4f} | "
                  f"Value Loss: {update_info['value_loss']:.4f} | "
                  f"Time: {update_time:.1f}s")

            # TensorBoard
            writer.add_scalar("rollout/reward_mean", mean_reward, total_timesteps)
            writer.add_scalar("rollout/length_mean", mean_length, total_timesteps)
            writer.add_scalar("loss/policy", update_info['policy_loss'], total_timesteps)
            writer.add_scalar("loss/value", update_info['value_loss'], total_timesteps)
            writer.add_scalar("loss/entropy", update_info['entropy_loss'], total_timesteps)
            writer.add_scalar("train/clip_fraction", update_info['clip_fraction'], total_timesteps)

        # Evaluation
        if num_updates % EVAL_INTERVAL == 0:
            eval_info = evaluate(agent, env_cfg, reward_cfg, device=DEVICE)

            print(f"\n  [EVAL] Mean reward: {eval_info['mean_reward']:.1f} ± {eval_info['std_reward']:.1f} | "
                  f"Success rate: {eval_info['success_rate']*100:.0f}% | "
                  f"Mean length: {eval_info['mean_length']:.0f}\n")

            writer.add_scalar("eval/reward_mean", eval_info['mean_reward'], total_timesteps)
            writer.add_scalar("eval/success_rate", eval_info['success_rate'], total_timesteps)

        # Save checkpoint
        if num_updates % SAVE_INTERVAL == 0:
            checkpoint_path = os.path.join(run_dir, f"ppo_step{total_timesteps}.pt")
            agent.save(checkpoint_path)
            print(f"  [SAVE] Checkpoint saved: {checkpoint_path}")

    # Training complete
    total_time = time.time() - train_start_time
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Total time: {total_time / 60:.1f} minutes")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Final mean reward: {np.mean(all_rewards[-100:]):.1f}")

    # Save final model
    final_path = os.path.join(run_dir, "ppo_final.pt")
    agent.save(final_path)
    print(f"Final model saved: {final_path}")

    # Plot training curve
    if len(all_rewards) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Rewards
        axes[0].plot(all_rewards, alpha=0.3)
        if len(all_rewards) >= 100:
            smoothed = np.convolve(all_rewards, np.ones(100)/100, mode='valid')
            axes[0].plot(range(99, len(all_rewards)), smoothed, 'r-', linewidth=2)
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Reward')
        axes[0].set_title('Episode Rewards')
        axes[0].grid(True)

        # Lengths
        axes[1].plot(all_lengths)
        axes[1].axhline(y=env_cfg.max_steps, color='g', linestyle='--', label='Max steps')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Length')
        axes[1].set_title('Episode Lengths')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plot_path = os.path.join(run_dir, "training.png")
        plt.savefig(plot_path, dpi=150)
        print(f"Training plot saved: {plot_path}")
        plt.close()

    writer.close()
    print(f"\nTo view TensorBoard: tensorboard --logdir {train_cfg.save_dir}")


if __name__ == "__main__":
    train()
