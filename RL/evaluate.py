"""
Evaluation Script for Stewart Platform RL

Evaluate trained SAC agent and visualize behavior.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation

from rl_config import EnvConfig, RewardConfig
from stewart_env import StewartBallEnv
from sac_agent import SACAgent


def evaluate(args):
    """Evaluate trained agent."""

    print("=" * 60)
    print("Stewart Platform Ball Balancing - Evaluation")
    print("=" * 60)

    # Load config
    env_cfg = EnvConfig()
    reward_cfg = RewardConfig()

    # Create environment (single env for evaluation)
    env = StewartBallEnv(num_envs=1, config=env_cfg, reward_config=reward_cfg)
    print(f"Environment created")

    # Create agent
    agent = SACAgent(
        state_dim=env_cfg.state_dim,
        action_dim=env_cfg.action_dim,
        device=args.device
    )

    # Load model
    if args.model:
        agent.load(args.model)
        print(f"Loaded model: {args.model}")
    else:
        print("No model specified, using random policy")

    # Run evaluation episodes
    num_episodes = args.episodes
    all_rewards = []
    all_lengths = []
    all_trajectories = []

    for ep in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        trajectory = {
            'ball_pos': [],
            'ball_vel': [],
            'platform_tilt': [],
            'actions': [],
            'rewards': []
        }

        for step in range(env_cfg.max_steps):
            # Select action (deterministic for evaluation)
            action = agent.select_action(obs[0], evaluate=True)
            action = action.reshape(1, -1)

            # Step environment
            next_obs, rewards, dones, infos = env.step(action)

            # Record trajectory
            state = env.get_state_info()
            trajectory['ball_pos'].append(state['ball_pos_mm'][0].copy())
            trajectory['ball_vel'].append(state['ball_vel_mm_s'][0].copy())
            trajectory['platform_tilt'].append(state['platform_tilt_deg'][0].copy())
            trajectory['actions'].append(action[0].copy())
            trajectory['rewards'].append(rewards[0])

            episode_reward += rewards[0]
            obs = next_obs

            if dones[0]:
                break

        all_rewards.append(episode_reward)
        all_lengths.append(step + 1)
        all_trajectories.append(trajectory)

        print(f"Episode {ep+1}/{num_episodes}: "
              f"Reward = {episode_reward:.2f}, Length = {step+1}")

    # Summary
    print("\n" + "=" * 60)
    print("Evaluation Summary:")
    print(f"  Episodes: {num_episodes}")
    print(f"  Mean reward: {np.mean(all_rewards):.2f} +/- {np.std(all_rewards):.2f}")
    print(f"  Mean length: {np.mean(all_lengths):.1f}")
    print(f"  Success rate: {sum(l >= env_cfg.max_steps for l in all_lengths) / num_episodes * 100:.1f}%")

    # Plot results
    if args.plot:
        plot_evaluation(all_trajectories, env_cfg, args.save_path)

    # Animate if requested
    if args.animate:
        animate_episode(all_trajectories[0], env_cfg)


def plot_evaluation(trajectories, env_cfg, save_path=None):
    """Plot evaluation results."""

    # Use first trajectory for detailed plot
    traj = trajectories[0]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Ball trajectory (top-down view)
    ax = axes[0, 0]
    ball_pos = np.array(traj['ball_pos'])
    ax.plot(ball_pos[:, 0], ball_pos[:, 1], 'b-', alpha=0.7, linewidth=0.5)
    ax.scatter(ball_pos[0, 0], ball_pos[0, 1], c='green', s=100, zorder=5, label='Start')
    ax.scatter(ball_pos[-1, 0], ball_pos[-1, 1], c='red', s=100, zorder=5, label='End')

    # Draw platform boundary
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(env_cfg.platform_radius_mm * np.cos(theta),
            env_cfg.platform_radius_mm * np.sin(theta),
            'k--', linewidth=2, label='Platform')

    ax.set_xlim(-env_cfg.platform_radius_mm * 1.1, env_cfg.platform_radius_mm * 1.1)
    ax.set_ylim(-env_cfg.platform_radius_mm * 1.1, env_cfg.platform_radius_mm * 1.1)
    ax.set_aspect('equal')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title('Ball Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Ball position over time
    ax = axes[0, 1]
    t = np.arange(len(ball_pos)) * env_cfg.dt
    ax.plot(t, ball_pos[:, 0], label='X')
    ax.plot(t, ball_pos[:, 1], label='Y')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position (mm)')
    ax.set_title('Ball Position')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Ball velocity
    ax = axes[0, 2]
    ball_vel = np.array(traj['ball_vel'])
    ax.plot(t, ball_vel[:, 0], label='Vx')
    ax.plot(t, ball_vel[:, 1], label='Vy')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (mm/s)')
    ax.set_title('Ball Velocity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Platform tilt
    ax = axes[1, 0]
    platform_tilt = np.array(traj['platform_tilt'])
    ax.plot(t, platform_tilt[:, 0], label='Rx')
    ax.plot(t, platform_tilt[:, 1], label='Ry')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.axhline(env_cfg.max_tilt_deg, color='r', linestyle=':', alpha=0.3)
    ax.axhline(-env_cfg.max_tilt_deg, color='r', linestyle=':', alpha=0.3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Tilt (deg)')
    ax.set_title('Platform Tilt')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Actions
    ax = axes[1, 1]
    actions = np.array(traj['actions'])
    ax.plot(t, actions[:, 0], label='Action X')
    ax.plot(t, actions[:, 1], label='Action Y')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Action [-1, 1]')
    ax.set_title('Agent Actions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Rewards
    ax = axes[1, 2]
    rewards = np.array(traj['rewards'])
    ax.plot(t, rewards)
    ax.plot(t, np.cumsum(rewards) / (np.arange(len(rewards)) + 1), '--', label='Running avg')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Reward')
    ax.set_title(f'Rewards (Total: {sum(rewards):.1f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved: {save_path}")
    else:
        plt.savefig('RL/evaluation_results.png', dpi=150)
        print("Plot saved: RL/evaluation_results.png")

    plt.show()


def animate_episode(trajectory, env_cfg):
    """Animate a single episode."""

    ball_pos = np.array(trajectory['ball_pos'])
    platform_tilt = np.array(trajectory['platform_tilt'])

    fig, ax = plt.subplots(figsize=(8, 8))

    # Platform boundary
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(env_cfg.platform_radius_mm * np.cos(theta),
            env_cfg.platform_radius_mm * np.sin(theta),
            'k-', linewidth=2)

    # Ball
    ball = Circle((0, 0), 20, color='orange', zorder=5)
    ax.add_patch(ball)

    # Trail
    trail, = ax.plot([], [], 'b-', alpha=0.3, linewidth=1)

    # Text
    text = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top', fontsize=10)

    ax.set_xlim(-env_cfg.platform_radius_mm * 1.2, env_cfg.platform_radius_mm * 1.2)
    ax.set_ylim(-env_cfg.platform_radius_mm * 1.2, env_cfg.platform_radius_mm * 1.2)
    ax.set_aspect('equal')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title('Ball Balancing Animation')
    ax.grid(True, alpha=0.3)

    def init():
        ball.center = (0, 0)
        trail.set_data([], [])
        text.set_text('')
        return ball, trail, text

    def animate(frame):
        # Ball position
        x, y = ball_pos[frame]
        ball.center = (x, y)

        # Trail (last 50 positions)
        start = max(0, frame - 50)
        trail.set_data(ball_pos[start:frame+1, 0], ball_pos[start:frame+1, 1])

        # Info text
        t = frame * env_cfg.dt
        rx, ry = platform_tilt[frame]
        text.set_text(f't = {t:.2f}s\n'
                      f'Ball: ({x:.1f}, {y:.1f}) mm\n'
                      f'Tilt: ({rx:.1f}, {ry:.1f}) deg')

        return ball, trail, text

    anim = FuncAnimation(fig, animate, init_func=init,
                         frames=len(ball_pos), interval=20, blit=True)

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained SAC agent")

    parser.add_argument('--model', type=str, default=None,
                        help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of evaluation episodes')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu')
    parser.add_argument('--plot', action='store_true',
                        help='Plot evaluation results')
    parser.add_argument('--animate', action='store_true',
                        help='Animate first episode')
    parser.add_argument('--save-path', type=str, default=None,
                        help='Path to save evaluation plot')

    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
