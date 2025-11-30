"""
Simple SAC Training for Stewart Platform Ball Balancing

Clean implementation following Pendulum_RL project structure.
No complex features - just pure SAC with frame stacking.
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from RL.env_gpu import StewartEnvGPU
from RL.rl_config import EnvConfig, RewardConfig


# ============================================================================
# CONFIGURATION
# ============================================================================

NUM_ENVS = 1              # Single environment (like Pendulum_RL)
MAX_EPISODES = 1000       # Training episodes
MAX_STEPS = 1000          # Steps per episode
BATCH_SIZE = 512          # Batch size for updates
BUFFER_SIZE = 100_000     # Replay buffer size
HIDDEN_DIM = 256          # Network hidden layer size
LR = 3e-4                 # Learning rate
GAMMA = 0.99              # Discount factor
TAU = 0.005               # Soft update coefficient
ALPHA = 0.2               # Initial entropy coefficient
AUTOMATIC_ENTROPY = True  # Auto-tune entropy
WARMUP_STEPS = 1000       # Random actions before training
EVAL_INTERVAL = 10        # Evaluate every N episodes
SAVE_INTERVAL = 50        # Save model every N episodes
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# ACTOR NETWORK
# ============================================================================

class Actor(nn.Module):
    """Policy network that outputs action distribution."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Mean and log std for continuous action
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        """Forward pass to get action mean and log std."""
        features = self.network(state)

        # Get mean and constrain it to [-1, 1]
        action_mean = torch.tanh(self.mean(features))

        # Get log standard deviation and clamp it
        action_log_std = self.log_std(features)
        action_log_std = torch.clamp(action_log_std, -20, 2)

        return action_mean, action_log_std

    def sample(self, state):
        """Sample action from the distribution and compute log probability."""
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # Sample from normal distribution with reparameterization trick
        normal = Normal(mean, std)
        x = normal.rsample()

        # Constrain to [-1, 1]
        action = torch.tanh(x)

        # Calculate log probability with squashing correction
        log_prob = normal.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob


# ============================================================================
# CRITIC NETWORK
# ============================================================================

class Critic(nn.Module):
    """Dual Q-network for value estimation."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()

        # Q1 network
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Q2 network (reduces overestimation bias)
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        """Forward pass through both Q networks."""
        x = torch.cat([state, action], 1)
        return self.q1(x), self.q2(x)


# ============================================================================
# REPLAY BUFFER
# ============================================================================

class ReplayBuffer:
    """Simple replay buffer for off-policy learning."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """Add a transition to the buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Randomly sample a batch of transitions."""
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = map(
            np.array, zip(*[self.buffer[i] for i in batch])
        )
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# ============================================================================
# SAC AGENT
# ============================================================================

class SACAgent:
    """Soft Actor-Critic agent for continuous control."""

    def __init__(
            self,
            state_dim,
            action_dim,
            hidden_dim=256,
            lr=3e-4,
            gamma=0.99,
            tau=0.005,
            alpha=0.2,
            automatic_entropy_tuning=True,
            device="cuda"
    ):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.device = torch.device(device)

        # Initialize networks
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(self.device)

        # Copy parameters to target network
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # Optimizers
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=lr)

        # Automatic entropy tuning
        if automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor([action_dim])).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.AdamW([self.log_alpha], lr=lr)

    def select_action(self, state, evaluate=False):
        """Select an action given a state."""
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

        with torch.no_grad():
            if evaluate:
                action, _ = self.actor(state)
            else:
                action, _ = self.actor.sample(state)

            return action.cpu().numpy()[0]

    def update(self, memory, batch_size=512):
        """Update actor and critic parameters."""
        # Sample batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory.sample(batch_size)

        # Convert to tensors
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)

        # === Update Critic ===
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state_batch)
            target_q1, target_q2 = self.critic_target(next_state_batch, next_action)
            target_q = torch.min(target_q1, target_q2)

            if self.automatic_entropy_tuning:
                alpha = self.log_alpha.exp()
            else:
                alpha = self.alpha

            target_q = target_q - alpha * next_log_prob
            target_q = reward_batch + (1 - done_batch) * self.gamma * target_q

        current_q1, current_q2 = self.critic(state_batch, action_batch)
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # === Update Actor ===
        actions, log_probs = self.actor.sample(state_batch)
        q1, q2 = self.critic(state_batch, actions)
        min_q = torch.min(q1, q2)

        if self.automatic_entropy_tuning:
            alpha = self.log_alpha.exp()
        else:
            alpha = self.alpha

        actor_loss = (alpha * log_probs - min_q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # === Update Alpha ===
        alpha_loss = 0.0
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp().item()

        # === Soft Update Target Networks ===
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha if isinstance(self.alpha, float) else self.alpha
        }

    def save(self, path):
        """Save model checkpoint."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, path)

    def load(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train():
    """Train the SAC agent on the Stewart platform environment."""
    print("=" * 60)
    print("Simple SAC Training for Stewart Platform")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Episodes: {MAX_EPISODES}")
    print(f"Steps per episode: {MAX_STEPS}")
    print(f"Buffer size: {BUFFER_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print()

    # Load configs
    env_cfg = EnvConfig()
    reward_cfg = RewardConfig()

    # Create environment (single env for simplicity)
    env = StewartEnvGPU(
        num_envs=NUM_ENVS,
        config=env_cfg,
        reward_config=reward_cfg,
        device=DEVICE,
        use_domain_randomization=False
    )

    state_dim = env_cfg.obs_dim  # 84
    action_dim = env_cfg.action_dim  # 2

    print(f"State dim: {state_dim}")
    print(f"Action dim: {action_dim}")

    # Create agent
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=HIDDEN_DIM,
        lr=LR,
        gamma=GAMMA,
        tau=TAU,
        alpha=ALPHA,
        automatic_entropy_tuning=AUTOMATIC_ENTROPY,
        device=DEVICE
    )
    print(f"Agent created on {DEVICE}")

    # Create replay buffer
    buffer = ReplayBuffer(BUFFER_SIZE)
    print(f"Replay buffer created (capacity: {BUFFER_SIZE})")

    # Create run directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = f"RL/checkpoints/simple_sac_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    # TensorBoard
    writer = SummaryWriter(run_dir)
    print(f"Run directory: {run_dir}")

    # Training metrics
    episode_rewards = []
    episode_lengths = []
    total_steps = 0

    print()
    print("=" * 60)
    print("Starting training...")
    print("=" * 60)

    for episode in range(MAX_EPISODES):
        # Reset environment
        obs, _ = env.reset_tensor()
        obs = obs[0].cpu().numpy()  # Single env, convert to numpy

        episode_reward = 0
        episode_length = 0

        for step in range(MAX_STEPS):
            # Select action
            if total_steps < WARMUP_STEPS:
                action = np.random.uniform(-1, 1, action_dim).astype(np.float32)
            else:
                action = agent.select_action(obs, evaluate=False)

            # Step environment
            action_tensor = torch.tensor([action], device=DEVICE)
            next_obs, reward, done, _ = env.step_tensor(action_tensor)

            next_obs = next_obs[0].cpu().numpy()
            reward = reward[0].item()
            done = done[0].item()

            # Store transition
            buffer.push(obs, action, reward, next_obs, float(done))

            # Update
            if total_steps >= WARMUP_STEPS and len(buffer) >= BATCH_SIZE:
                update_info = agent.update(buffer, BATCH_SIZE)

                # Log to TensorBoard
                if total_steps % 100 == 0:
                    writer.add_scalar('Loss/Critic', update_info['critic_loss'], total_steps)
                    writer.add_scalar('Loss/Actor', update_info['actor_loss'], total_steps)
                    writer.add_scalar('Params/Alpha', update_info['alpha'], total_steps)

            obs = next_obs
            episode_reward += reward
            episode_length += 1
            total_steps += 1

            if done:
                break

        # Episode complete
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # Log episode metrics
        writer.add_scalar('Episode/Reward', episode_reward, episode)
        writer.add_scalar('Episode/Length', episode_length, episode)

        # Print progress
        avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
        print(f"Episode {episode + 1:4d} | "
              f"Reward: {episode_reward:7.1f} | "
              f"Avg: {avg_reward:7.1f} | "
              f"Length: {episode_length:4d} | "
              f"Alpha: {agent.alpha:.3f} | "
              f"Buffer: {len(buffer):6d}")

        # Save checkpoint
        if (episode + 1) % SAVE_INTERVAL == 0:
            checkpoint_path = os.path.join(run_dir, f"sac_ep{episode + 1}.pt")
            agent.save(checkpoint_path)
            print(f"  [SAVE] {checkpoint_path}")

    # Save final model
    final_path = os.path.join(run_dir, "sac_final.pt")
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
    axes[0].set_title('Episode Rewards')
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


if __name__ == "__main__":
    train()
