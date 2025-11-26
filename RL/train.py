"""
Training Script for Stewart Platform RL

Train SAC agent to balance ball on Stewart platform.
"""

import os
import sys
import argparse
from time import time

import numpy as np
import torch
import matplotlib.pyplot as plt

from rl_config import EnvConfig, RewardConfig, SACConfig, TrainingConfig
from stewart_env import StewartBallEnv
from sac_agent import SACAgent, ReplayBuffer


def train(args):
    """Main training function."""

    print("=" * 60)
    print("Stewart Platform Ball Balancing - SAC Training")
    print("=" * 60)

    # Load configs
    env_cfg = EnvConfig()
    reward_cfg = RewardConfig()
    sac_cfg = SACConfig()
    train_cfg = TrainingConfig()

    # Override with command line args
    num_envs = args.num_envs
    max_episodes = args.max_episodes or train_cfg.max_episodes
    device = args.device or train_cfg.device

    print(f"\nConfiguration:")
    print(f"  Parallel environments: {num_envs}")
    print(f"  Max episodes: {max_episodes}")
    print(f"  Device: {device}")
    print(f"  State dim: {env_cfg.state_dim}")
    print(f"  Action dim: {env_cfg.action_dim}")

    # Set seed
    np.random.seed(train_cfg.seed)
    torch.manual_seed(train_cfg.seed)

    # Create environment
    env = StewartBallEnv(num_envs=num_envs, config=env_cfg, reward_config=reward_cfg)
    print(f"\nEnvironment created with {num_envs} parallel envs")

    # Create agent
    agent = SACAgent(
        state_dim=env_cfg.state_dim,
        action_dim=env_cfg.action_dim,
        hidden_dim=sac_cfg.hidden_dim,
        lr=sac_cfg.lr,
        gamma=sac_cfg.gamma,
        tau=sac_cfg.tau,
        alpha=sac_cfg.alpha,
        automatic_entropy_tuning=sac_cfg.automatic_entropy_tuning,
        device=device
    )
    print(f"SAC agent created on {agent.device}")

    # Create replay buffer
    buffer = ReplayBuffer(
        capacity=sac_cfg.buffer_size,
        state_dim=env_cfg.state_dim,
        action_dim=env_cfg.action_dim
    )

    # Load checkpoint if provided
    if args.checkpoint:
        agent.load(args.checkpoint)
        print(f"Loaded checkpoint: {args.checkpoint}")

    # Training metrics
    episode_rewards = []
    episode_lengths = []
    critic_losses = []
    actor_losses = []
    alpha_values = []

    # Training loop
    print(f"\nStarting training...")
    print("-" * 60)

    total_steps = 0
    start_time = time()

    for episode in range(max_episodes):
        # Reset all environments
        obs = env.reset()
        episode_reward = np.zeros(num_envs)
        episode_length = 0

        done_mask = np.zeros(num_envs, dtype=bool)

        # Episode loop
        while not done_mask.all():
            # Select actions
            if total_steps < sac_cfg.warmup_steps:
                # Random exploration
                actions = np.random.uniform(-1, 1, (num_envs, env_cfg.action_dim)).astype(np.float32)
            else:
                actions = agent.select_action_batch(obs)

            # Step environment
            next_obs, rewards, dones, infos = env.step(actions)

            # Store transitions (only for envs that weren't already done)
            for i in range(num_envs):
                if not done_mask[i]:
                    buffer.push(obs[i], actions[i], rewards[i], next_obs[i], float(dones[i]))
                    episode_reward[i] += rewards[i]

            # Update done mask
            done_mask = done_mask | dones
            episode_length += 1
            total_steps += num_envs

            # Update networks
            if len(buffer) > sac_cfg.batch_size and total_steps >= sac_cfg.warmup_steps:
                for _ in range(sac_cfg.updates_per_step):
                    info = agent.update(buffer, sac_cfg.batch_size)
                    critic_losses.append(info['critic_loss'])
                    actor_losses.append(info['actor_loss'])
                    alpha_values.append(info['alpha'])

            obs = next_obs

            # Safety break
            if episode_length >= env_cfg.max_steps:
                break

        # Episode complete
        mean_reward = episode_reward.mean()
        episode_rewards.append(mean_reward)
        episode_lengths.append(episode_length)

        # Logging
        if (episode + 1) % train_cfg.log_interval == 0:
            elapsed = time() - start_time
            avg_reward = np.mean(episode_rewards[-train_cfg.log_interval:])
            avg_length = np.mean(episode_lengths[-train_cfg.log_interval:])

            # Get recent losses
            if critic_losses:
                recent_critic = np.mean(critic_losses[-100:])
                recent_actor = np.mean(actor_losses[-100:])
                recent_alpha = np.mean(alpha_values[-100:])
            else:
                recent_critic = recent_actor = recent_alpha = 0

            steps_per_sec = total_steps / elapsed

            print(f"Ep {episode+1:4d}/{max_episodes} | "
                  f"R: {mean_reward:7.2f} | "
                  f"AvgR: {avg_reward:7.2f} | "
                  f"Len: {avg_length:5.0f} | "
                  f"C: {recent_critic:.3f} | "
                  f"A: {recent_actor:.3f} | "
                  f"a: {recent_alpha:.3f} | "
                  f"{steps_per_sec:.0f} sps")

        # Save checkpoint
        if (episode + 1) % train_cfg.save_interval == 0:
            os.makedirs(train_cfg.save_dir, exist_ok=True)
            save_path = os.path.join(train_cfg.save_dir, f"sac_ep{episode+1}.pt")
            agent.save(save_path)
            print(f"  Saved: {save_path}")

    # Training complete
    elapsed = time() - start_time
    print("-" * 60)
    print(f"Training completed in {elapsed:.1f}s")
    print(f"Total steps: {total_steps:,}")
    print(f"Average speed: {total_steps/elapsed:.0f} steps/sec")

    # Save final model
    os.makedirs(train_cfg.save_dir, exist_ok=True)
    final_path = os.path.join(train_cfg.save_dir, "sac_final.pt")
    agent.save(final_path)
    print(f"Final model saved: {final_path}")

    # Plot training progress
    plot_training(episode_rewards, critic_losses, actor_losses, alpha_values, train_cfg.save_dir)

    return agent


def plot_training(rewards, critic_losses, actor_losses, alpha_values, save_dir):
    """Plot and save training curves."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Episode rewards
    ax = axes[0, 0]
    ax.plot(rewards, alpha=0.3, label='Episode')
    if len(rewards) >= 10:
        window = min(50, len(rewards) // 5)
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(np.arange(window-1, len(rewards)), smoothed, label=f'MA-{window}')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Episode Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Critic loss
    ax = axes[0, 1]
    if critic_losses:
        ax.plot(critic_losses, alpha=0.3)
        if len(critic_losses) >= 100:
            window = 100
            smoothed = np.convolve(critic_losses, np.ones(window)/window, mode='valid')
            ax.plot(np.arange(window-1, len(critic_losses)), smoothed)
    ax.set_xlabel('Update Step')
    ax.set_ylabel('Loss')
    ax.set_title('Critic Loss')
    ax.grid(True, alpha=0.3)

    # Actor loss
    ax = axes[1, 0]
    if actor_losses:
        ax.plot(actor_losses, alpha=0.3)
        if len(actor_losses) >= 100:
            window = 100
            smoothed = np.convolve(actor_losses, np.ones(window)/window, mode='valid')
            ax.plot(np.arange(window-1, len(actor_losses)), smoothed)
    ax.set_xlabel('Update Step')
    ax.set_ylabel('Loss')
    ax.set_title('Actor Loss')
    ax.grid(True, alpha=0.3)

    # Alpha
    ax = axes[1, 1]
    if alpha_values:
        ax.plot(alpha_values)
    ax.set_xlabel('Update Step')
    ax.set_ylabel('Alpha')
    ax.set_title('Entropy Coefficient')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = os.path.join(save_dir, "training_progress.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training plot saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train SAC for Stewart Platform")

    parser.add_argument('--num-envs', type=int, default=10,
                        help='Number of parallel environments (default: 10)')
    parser.add_argument('--max-episodes', type=int, default=None,
                        help='Maximum training episodes')
    parser.add_argument('--device', type=str, default=None,
                        help='Device: cuda or cpu')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
