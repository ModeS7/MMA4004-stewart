"""
Soft Actor-Critic (SAC) Agent

Based on successful Pendulum RL implementation.
Adapted for Stewart Platform ball balancing.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


# ============================================================================
# ACTOR NETWORK
# ============================================================================

class Actor(nn.Module):
    """
    Policy network that outputs a Gaussian action distribution.

    Uses tanh squashing for bounded actions.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        """Get action mean and log std."""
        features = self.network(state)
        action_mean = torch.tanh(self.mean(features))
        action_log_std = torch.clamp(self.log_std(features), -20, 2)
        return action_mean, action_log_std

    def sample(self, state):
        """Sample action with reparameterization trick."""
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # Sample from Gaussian
        normal = Normal(mean, std)
        x = normal.rsample()

        # Squash to [-1, 1]
        action = torch.tanh(x)

        # Log probability with squashing correction
        log_prob = normal.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob


# ============================================================================
# CRITIC NETWORK
# ============================================================================

class Critic(nn.Module):
    """
    Twin Q-networks for reducing overestimation bias.
    """

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

        # Q2 network
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        """Forward pass through both Q networks."""
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)


# ============================================================================
# REPLAY BUFFER
# ============================================================================

class ReplayBuffer:
    """Experience replay buffer for off-policy learning."""

    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.position = 0
        self.size = 0

        # Pre-allocate arrays
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        """Add a single transition."""
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def push_batch(self, states, actions, rewards, next_states, dones):
        """Add a batch of transitions."""
        batch_size = states.shape[0]

        for i in range(batch_size):
            self.push(states[i], actions[i], rewards[i], next_states[i], dones[i])

    def sample(self, batch_size):
        """Sample a batch of transitions."""
        indices = np.random.randint(0, self.size, size=batch_size)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )

    def __len__(self):
        return self.size


# ============================================================================
# SAC AGENT
# ============================================================================

class SACAgent:
    """
    Soft Actor-Critic agent for continuous control.

    Features:
    - Automatic entropy tuning
    - Twin Q-networks
    - Soft target updates
    """

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

        # Device
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Networks
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(self.device)

        # Copy parameters to target
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # Optimizers
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=lr)

        # Automatic entropy tuning
        if automatic_entropy_tuning:
            self.target_entropy = -action_dim  # Heuristic: -dim(A)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.AdamW([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item()

    def select_action(self, state, evaluate=False):
        """
        Select action for a single state.

        Args:
            state: (state_dim,) numpy array
            evaluate: If True, use mean action (no exploration)

        Returns:
            action: (action_dim,) numpy array
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if evaluate:
                action, _ = self.actor(state)
            else:
                action, _ = self.actor.sample(state)

        return action.cpu().numpy()[0]

    def select_action_batch(self, states, evaluate=False):
        """
        Select actions for a batch of states.

        Args:
            states: (batch, state_dim) numpy array
            evaluate: If True, use mean action (no exploration)

        Returns:
            actions: (batch, action_dim) numpy array
        """
        states = torch.FloatTensor(states).to(self.device)

        with torch.no_grad():
            if evaluate:
                actions, _ = self.actor(states)
            else:
                actions, _ = self.actor.sample(states)

        return actions.cpu().numpy()

    def update(self, replay_buffer, batch_size=512):
        """
        Update actor and critic networks.

        Returns:
            dict with critic_loss, actor_loss, alpha
        """
        # Sample batch
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # ===== Update Critic =====
        with torch.no_grad():
            # Sample next actions
            next_actions, next_log_probs = self.actor.sample(next_states)

            # Target Q values
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)

            # Entropy-regularized target
            if self.automatic_entropy_tuning:
                alpha = self.log_alpha.exp()
            else:
                alpha = self.alpha

            target_q = target_q - alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * target_q

        # Current Q estimates
        current_q1, current_q2 = self.critic(states, actions)

        # Critic loss
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ===== Update Actor =====
        new_actions, log_probs = self.actor.sample(states)
        q1, q2 = self.critic(states, new_actions)
        min_q = torch.min(q1, q2)

        if self.automatic_entropy_tuning:
            alpha = self.log_alpha.exp()
        else:
            alpha = self.alpha

        # Actor loss (maximize Q - alpha * log_prob)
        actor_loss = (alpha * log_probs - min_q).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ===== Update Alpha =====
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp().item()

        # ===== Soft Update Target =====
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha
        }

    def save(self, path):
        """Save actor and critic models."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'log_alpha': self.log_alpha if self.automatic_entropy_tuning else None
        }, path)

    def load(self, path):
        """Load actor and critic models."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        if self.automatic_entropy_tuning and checkpoint['log_alpha'] is not None:
            self.log_alpha = checkpoint['log_alpha'].to(self.device)

    def load_actor(self, path):
        """Load only actor model (for evaluation)."""
        checkpoint = torch.load(path, map_location=self.device)
        if 'actor' in checkpoint:
            self.actor.load_state_dict(checkpoint['actor'])
        else:
            # Assume it's just the actor state dict
            self.actor.load_state_dict(checkpoint)


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("Testing SAC Agent...")

    # Create agent
    state_dim = 6
    action_dim = 2
    agent = SACAgent(state_dim, action_dim)
    print(f"Created agent on device: {agent.device}")

    # Test action selection
    state = np.random.randn(state_dim).astype(np.float32)
    action = agent.select_action(state)
    print(f"Single action: {action}, shape: {action.shape}")

    # Test batch action selection
    states = np.random.randn(10, state_dim).astype(np.float32)
    actions = agent.select_action_batch(states)
    print(f"Batch actions: shape {actions.shape}")

    # Test replay buffer
    buffer = ReplayBuffer(10000, state_dim, action_dim)
    for _ in range(1000):
        s = np.random.randn(state_dim).astype(np.float32)
        a = np.random.randn(action_dim).astype(np.float32)
        r = np.random.randn()
        s2 = np.random.randn(state_dim).astype(np.float32)
        d = np.random.choice([0, 1])
        buffer.push(s, a, r, s2, d)

    print(f"Buffer size: {len(buffer)}")

    # Test update
    info = agent.update(buffer, batch_size=256)
    print(f"Update info: {info}")

    print("\n[OK] SAC Agent test passed!")
