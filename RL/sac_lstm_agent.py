"""
LSTM-based Soft Actor-Critic (SAC) Agent

Uses LSTM to process observation sequences and infer velocity/acceleration
from position history. Designed for noisy camera observations.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


# ============================================================================
# LSTM ACTOR NETWORK
# ============================================================================

class LSTMActor(nn.Module):
    """
    LSTM-based policy network.

    Processes observation sequence with LSTM to infer dynamics,
    then outputs Gaussian action distribution with tanh squashing.
    """

    def __init__(self, obs_dim, action_dim, seq_length,
                 lstm_hidden_dim=128, lstm_layers=1, hidden_dim=256):
        super(LSTMActor, self).__init__()

        self.obs_dim = obs_dim
        self.seq_length = seq_length
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers

        # LSTM to process observation sequence
        self.lstm = nn.LSTM(
            input_size=obs_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
        )

        # MLP head after LSTM
        self.network = nn.Sequential(
            nn.Linear(lstm_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs_seq, hidden=None):
        """
        Forward pass.

        Args:
            obs_seq: (batch, seq_length, obs_dim) observation sequence
            hidden: Optional LSTM hidden state

        Returns:
            action_mean, action_log_std, new_hidden
        """
        # LSTM forward
        lstm_out, hidden = self.lstm(obs_seq, hidden)

        # Use last timestep output
        features = lstm_out[:, -1, :]  # (batch, lstm_hidden_dim)

        # MLP head
        features = self.network(features)

        action_mean = torch.tanh(self.mean(features))
        action_log_std = torch.clamp(self.log_std(features), -20, 2)

        return action_mean, action_log_std, hidden

    def sample(self, obs_seq, hidden=None):
        """Sample action with reparameterization trick."""
        mean, log_std, hidden = self.forward(obs_seq, hidden)
        std = log_std.exp()

        # Sample from Gaussian
        normal = Normal(mean, std)
        x = normal.rsample()

        # Squash to [-1, 1]
        action = torch.tanh(x)

        # Log probability with squashing correction
        log_prob = normal.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob, hidden


# ============================================================================
# LSTM CRITIC NETWORK
# ============================================================================

class LSTMCritic(nn.Module):
    """
    LSTM-based twin Q-networks.

    Processes observation sequence with LSTM, concatenates with action,
    then outputs Q-value.
    """

    def __init__(self, obs_dim, action_dim, seq_length,
                 lstm_hidden_dim=128, lstm_layers=1, hidden_dim=256):
        super(LSTMCritic, self).__init__()

        self.obs_dim = obs_dim
        self.seq_length = seq_length
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers

        # Shared LSTM for observation encoding (used by both Q networks)
        self.lstm = nn.LSTM(
            input_size=obs_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
        )

        # Q1 network
        self.q1 = nn.Sequential(
            nn.Linear(lstm_hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Q2 network
        self.q2 = nn.Sequential(
            nn.Linear(lstm_hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs_seq, action, hidden=None):
        """
        Forward pass through both Q networks.

        Args:
            obs_seq: (batch, seq_length, obs_dim) observation sequence
            action: (batch, action_dim) actions
            hidden: Optional LSTM hidden state

        Returns:
            q1, q2: Q-value estimates
        """
        # LSTM forward
        lstm_out, hidden = self.lstm(obs_seq, hidden)

        # Use last timestep output
        features = lstm_out[:, -1, :]  # (batch, lstm_hidden_dim)

        # Concatenate with action
        x = torch.cat([features, action], dim=-1)

        return self.q1(x), self.q2(x)


# ============================================================================
# SEQUENCE REPLAY BUFFER
# ============================================================================

class SequenceReplayBuffer:
    """
    Experience replay buffer for sequence observations.

    Stores full observation sequences for LSTM training.
    """

    def __init__(self, capacity, seq_length, obs_dim, action_dim):
        self.capacity = capacity
        self.position = 0
        self.size = 0

        self.seq_length = seq_length
        self.obs_dim = obs_dim

        # Pre-allocate arrays for sequences
        self.states = np.zeros((capacity, seq_length, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, seq_length, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def push(self, state_seq, action, reward, next_state_seq, done):
        """Add a single transition with observation sequences."""
        self.states[self.position] = state_seq
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state_seq
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
# LSTM SAC AGENT
# ============================================================================

class LSTMSACAgent:
    """
    LSTM-based Soft Actor-Critic agent.

    Uses LSTM networks to process observation sequences and infer
    velocity/acceleration from position history.
    """

    def __init__(
            self,
            obs_dim,
            action_dim,
            seq_length,
            lstm_hidden_dim=128,
            lstm_layers=1,
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
        self.seq_length = seq_length
        self.obs_dim = obs_dim

        # Device
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Networks
        self.actor = LSTMActor(
            obs_dim, action_dim, seq_length,
            lstm_hidden_dim, lstm_layers, hidden_dim
        ).to(self.device)

        self.critic = LSTMCritic(
            obs_dim, action_dim, seq_length,
            lstm_hidden_dim, lstm_layers, hidden_dim
        ).to(self.device)

        self.critic_target = LSTMCritic(
            obs_dim, action_dim, seq_length,
            lstm_hidden_dim, lstm_layers, hidden_dim
        ).to(self.device)

        # Copy parameters to target
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # Optimizers
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=lr)

        # Automatic entropy tuning
        if automatic_entropy_tuning:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.AdamW([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item()

    def select_action(self, obs_seq, evaluate=False):
        """
        Select action for a single observation sequence.

        Args:
            obs_seq: (seq_length, obs_dim) numpy array
            evaluate: If True, use mean action (no exploration)

        Returns:
            action: (action_dim,) numpy array
        """
        obs_seq = torch.FloatTensor(obs_seq).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if evaluate:
                action, _, _ = self.actor(obs_seq)
            else:
                action, _, _ = self.actor.sample(obs_seq)

        return action.cpu().numpy()[0]

    def select_action_batch(self, obs_seqs, evaluate=False):
        """
        Select actions for a batch of observation sequences.

        Args:
            obs_seqs: (batch, seq_length, obs_dim) numpy array
            evaluate: If True, use mean action (no exploration)

        Returns:
            actions: (batch, action_dim) numpy array
        """
        obs_seqs = torch.FloatTensor(obs_seqs).to(self.device)

        with torch.no_grad():
            if evaluate:
                actions, _, _ = self.actor(obs_seqs)
            else:
                actions, _, _ = self.actor.sample(obs_seqs)

        return actions.cpu().numpy()

    def update(self, replay_buffer, batch_size=256):
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
            next_actions, next_log_probs, _ = self.actor.sample(next_states)

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
        new_actions, log_probs, _ = self.actor.sample(states)
        q1, q2 = self.critic(states, new_actions)
        min_q = torch.min(q1, q2)

        if self.automatic_entropy_tuning:
            alpha = self.log_alpha.exp()
        else:
            alpha = self.alpha

        # Actor loss
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


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("Testing LSTM SAC Agent...")

    # Config
    obs_dim = 4  # [ball_x, ball_y, platform_rx, platform_ry]
    action_dim = 2
    seq_length = 10

    # Create agent
    agent = LSTMSACAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        seq_length=seq_length,
        lstm_hidden_dim=128,
        lstm_layers=1,
        hidden_dim=256
    )
    print(f"Created LSTM agent on device: {agent.device}")

    # Test action selection
    obs_seq = np.random.randn(seq_length, obs_dim).astype(np.float32)
    action = agent.select_action(obs_seq)
    print(f"Single action: {action}, shape: {action.shape}")

    # Test batch action selection
    obs_seqs = np.random.randn(10, seq_length, obs_dim).astype(np.float32)
    actions = agent.select_action_batch(obs_seqs)
    print(f"Batch actions: shape {actions.shape}")

    # Test replay buffer
    buffer = SequenceReplayBuffer(10000, seq_length, obs_dim, action_dim)
    for _ in range(1000):
        s = np.random.randn(seq_length, obs_dim).astype(np.float32)
        a = np.random.randn(action_dim).astype(np.float32)
        r = np.random.randn()
        s2 = np.random.randn(seq_length, obs_dim).astype(np.float32)
        d = np.random.choice([0, 1])
        buffer.push(s, a, r, s2, d)

    print(f"Buffer size: {len(buffer)}")

    # Test update
    info = agent.update(buffer, batch_size=256)
    print(f"Update info: {info}")

    print("\n[OK] LSTM SAC Agent test passed!")
