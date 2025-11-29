"""
PPO Agent for Stewart Platform Ball Balancing

On-policy algorithm that's more stable than SAC for parallel training.
Based on CleanRL's PPO implementation, adapted for GPU environments.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


# ============================================================================
# ACTOR-CRITIC NETWORK
# ============================================================================

class ActorCritic(nn.Module):
    """
    Actor-Critic network with 1D CNN backbone for temporal processing.

    Same architecture as SAC Actor, but with separate value head.
    """

    def __init__(self, obs_dim, action_dim, hidden_dim=256,
                 num_frames=12, obs_per_frame=7):
        super().__init__()

        self.num_frames = num_frames
        self.obs_per_frame = obs_per_frame

        # Shared 1D CNN backbone
        self.conv = nn.Sequential(
            nn.Conv1d(obs_per_frame, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
        )

        # CNN output: 64 × 5 = 320
        cnn_out_dim = 64 * 5

        # Shared MLP
        self.shared_fc = nn.Sequential(
            nn.Linear(cnn_out_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor head (policy)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))

        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with orthogonal initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Actor output layer with smaller scale
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.zeros_(self.actor_mean.bias)

    def _get_features(self, obs):
        """Extract features from observations."""
        batch_size = obs.shape[0]

        # Reshape: (batch, 84) -> (batch, 12, 7) -> (batch, 7, 12)
        x = obs.view(batch_size, self.num_frames, self.obs_per_frame)
        x = x.permute(0, 2, 1)

        # CNN
        x = self.conv(x)
        x = x.flatten(1)

        # Shared MLP
        features = self.shared_fc(x)
        return features

    def get_value(self, obs):
        """Get value estimate for observations."""
        features = self._get_features(obs)
        return self.critic(features)

    def get_action_and_value(self, obs, action=None):
        """
        Get action, log probability, entropy, and value.

        If action is provided, compute log_prob for that action.
        Otherwise, sample a new action.
        """
        features = self._get_features(obs)

        # Actor
        action_mean = self.actor_mean(features)
        action_std = self.actor_log_std.exp().expand_as(action_mean)

        dist = Normal(action_mean, action_std)

        if action is None:
            action = dist.sample()

        # Squash action to [-1, 1]
        action_tanh = torch.tanh(action)

        # Log probability with tanh correction
        log_prob = dist.log_prob(action)
        log_prob = log_prob - torch.log(1 - action_tanh.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)

        # Entropy
        entropy = dist.entropy().sum(dim=-1)

        # Value
        value = self.critic(features)

        return action_tanh, log_prob, entropy, value


# ============================================================================
# ROLLOUT BUFFER
# ============================================================================

class RolloutBuffer:
    """
    Buffer to store rollout data for PPO.

    Stores n_steps of experience from num_envs parallel environments.
    """

    def __init__(self, n_steps, num_envs, obs_dim, action_dim, device):
        self.n_steps = n_steps
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

        # Pre-allocate tensors on GPU
        self.observations = torch.zeros((n_steps, num_envs, obs_dim),
                                         dtype=torch.float32, device=device)
        self.actions = torch.zeros((n_steps, num_envs, action_dim),
                                   dtype=torch.float32, device=device)
        self.log_probs = torch.zeros((n_steps, num_envs),
                                     dtype=torch.float32, device=device)
        self.rewards = torch.zeros((n_steps, num_envs),
                                   dtype=torch.float32, device=device)
        self.dones = torch.zeros((n_steps, num_envs),
                                 dtype=torch.float32, device=device)
        self.values = torch.zeros((n_steps, num_envs),
                                  dtype=torch.float32, device=device)

        # Computed after rollout
        self.advantages = torch.zeros((n_steps, num_envs),
                                      dtype=torch.float32, device=device)
        self.returns = torch.zeros((n_steps, num_envs),
                                   dtype=torch.float32, device=device)

        self.ptr = 0

    def add(self, obs, action, log_prob, reward, done, value):
        """Add a step of experience."""
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value.squeeze(-1)
        self.ptr += 1

    def compute_returns_and_advantages(self, last_value, gamma, gae_lambda):
        """
        Compute returns and GAE advantages.

        Args:
            last_value: Value estimate for the state after the last step
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        last_gae = 0

        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_value = last_value.squeeze(-1)
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]

            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            self.advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae

        self.returns = self.advantages + self.values

    def get_batches(self, batch_size):
        """
        Generate random minibatches for training.

        Yields:
            Tuples of (obs, actions, old_log_probs, advantages, returns)
        """
        # Flatten time and env dimensions
        total_size = self.n_steps * self.num_envs
        indices = torch.randperm(total_size, device=self.device)

        # Flatten all tensors
        flat_obs = self.observations.view(-1, self.obs_dim)
        flat_actions = self.actions.view(-1, self.action_dim)
        flat_log_probs = self.log_probs.view(-1)
        flat_advantages = self.advantages.view(-1)
        flat_returns = self.returns.view(-1)

        # Normalize advantages
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

        for start in range(0, total_size, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]

            yield (
                flat_obs[batch_indices],
                flat_actions[batch_indices],
                flat_log_probs[batch_indices],
                flat_advantages[batch_indices],
                flat_returns[batch_indices]
            )

    def reset(self):
        """Reset buffer pointer."""
        self.ptr = 0


# ============================================================================
# PPO AGENT
# ============================================================================

class PPOAgent:
    """
    PPO Agent for parallel GPU environments.

    Features:
        - Clipped surrogate objective
        - GAE for advantage estimation
        - Entropy bonus for exploration
        - Value function clipping (optional)
    """

    def __init__(
            self,
            obs_dim,
            action_dim=2,
            hidden_dim=256,
            num_frames=12,
            obs_per_frame=7,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,  # Value function clipping (None = disabled)
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            device="cpu",
            compile_model=False
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

        # Network
        self.network = ActorCritic(
            obs_dim, action_dim, hidden_dim,
            num_frames, obs_per_frame
        ).to(self.device)

        # Optional: compile for speedup
        if compile_model and hasattr(torch, 'compile'):
            self.network = torch.compile(self.network)

        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate, eps=1e-5)

    def get_action(self, obs, deterministic=False):
        """
        Get action for a batch of observations.

        Args:
            obs: Tensor of observations (num_envs, obs_dim)
            deterministic: If True, return mean action (for evaluation)

        Returns:
            action, log_prob, value
        """
        with torch.no_grad():
            if deterministic:
                features = self.network._get_features(obs)
                action = torch.tanh(self.network.actor_mean(features))
                value = self.network.critic(features)
                return action, None, value
            else:
                action, log_prob, _, value = self.network.get_action_and_value(obs)
                return action, log_prob, value

    def get_value(self, obs):
        """Get value estimate for observations."""
        with torch.no_grad():
            return self.network.get_value(obs)

    def update(self, rollout_buffer, n_epochs, batch_size):
        """
        Update policy and value function using collected rollouts.

        Args:
            rollout_buffer: Buffer containing rollout data
            n_epochs: Number of epochs to train on the data
            batch_size: Minibatch size

        Returns:
            dict with training metrics
        """
        # Track metrics
        pg_losses = []
        value_losses = []
        entropy_losses = []
        clip_fractions = []

        for epoch in range(n_epochs):
            for batch in rollout_buffer.get_batches(batch_size):
                obs, actions, old_log_probs, advantages, returns = batch

                # Get current policy outputs
                _, new_log_probs, entropy, new_values = self.network.get_action_and_value(obs, actions)
                new_values = new_values.squeeze(-1)

                # Policy loss (clipped surrogate)
                log_ratio = new_log_probs - old_log_probs
                ratio = log_ratio.exp()

                pg_loss1 = -advantages * ratio
                pg_loss2 = -advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                if self.clip_range_vf is not None:
                    # Clipped value loss
                    values_pred = rollout_buffer.values.view(-1)[batch]
                    value_clipped = values_pred + torch.clamp(
                        new_values - values_pred, -self.clip_range_vf, self.clip_range_vf
                    )
                    value_loss1 = (new_values - returns).pow(2)
                    value_loss2 = (value_clipped - returns).pow(2)
                    value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
                else:
                    value_loss = 0.5 * (new_values - returns).pow(2).mean()

                # Entropy loss
                entropy_loss = -entropy.mean()

                # Total loss
                loss = pg_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Track metrics
                pg_losses.append(pg_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())

                with torch.no_grad():
                    clip_fraction = ((ratio - 1).abs() > self.clip_range).float().mean().item()
                    clip_fractions.append(clip_fraction)

        return {
            'policy_loss': np.mean(pg_losses),
            'value_loss': np.mean(value_losses),
            'entropy_loss': np.mean(entropy_losses),
            'clip_fraction': np.mean(clip_fractions),
        }

    def save(self, path):
        """Save model."""
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("Testing PPO Agent...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create agent
    agent = PPOAgent(
        obs_dim=84,
        action_dim=2,
        hidden_dim=256,
        num_frames=12,
        obs_per_frame=7,
        device=device
    )

    # Test forward pass
    obs = torch.randn(100, 84, device=device)
    action, log_prob, value = agent.get_action(obs)

    print(f"Action shape: {action.shape}")
    print(f"Log prob shape: {log_prob.shape}")
    print(f"Value shape: {value.shape}")

    # Test rollout buffer
    buffer = RolloutBuffer(
        n_steps=16,
        num_envs=100,
        obs_dim=84,
        action_dim=2,
        device=device
    )

    # Fill buffer
    for _ in range(16):
        obs = torch.randn(100, 84, device=device)
        action, log_prob, value = agent.get_action(obs)
        reward = torch.randn(100, device=device)
        done = torch.zeros(100, device=device)
        buffer.add(obs, action, log_prob, reward, done, value)

    # Compute returns
    last_value = agent.get_value(obs)
    buffer.compute_returns_and_advantages(last_value, gamma=0.99, gae_lambda=0.95)

    # Test update
    metrics = agent.update(buffer, n_epochs=4, batch_size=256)
    print(f"Update metrics: {metrics}")

    print("\n[OK] PPO Agent test passed!")
