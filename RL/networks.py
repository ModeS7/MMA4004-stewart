"""
Neural Network Architectures for SAC

Contains both MLP and CNN variants for Actor and Critic networks.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal


# ============================================================================
# MLP NETWORKS (Simple feedforward)
# ============================================================================

class ActorMLP(nn.Module):
    """Simple MLP actor for SAC."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorMLP, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        """Forward pass to get action mean and log std."""
        features = self.network(state)
        action_mean = torch.tanh(self.mean(features))
        action_log_std = torch.clamp(self.log_std(features), -20, 2)
        return action_mean, action_log_std

    def sample(self, state):
        """Sample action with reparameterization trick."""
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        x = normal.rsample()
        action = torch.tanh(x)

        # Log probability with squashing correction
        log_prob = normal.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def get_deterministic(self, state):
        """Get deterministic action (for evaluation)."""
        mean, _ = self.forward(state)
        return mean


class CriticMLP(nn.Module):
    """Twin Q-networks with MLP architecture."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(CriticMLP, self).__init__()

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
# CNN NETWORKS (Temporal 1D convolutions)
# ============================================================================

class ActorCNN(nn.Module):
    """
    1D CNN actor that processes frame history temporally.

    Optionally includes physics estimation head for sim-to-real.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256,
                 num_frames=12, obs_per_frame=7, physics_dim=3,
                 use_physics_head=False):
        super(ActorCNN, self).__init__()

        self.num_frames = num_frames
        self.obs_per_frame = obs_per_frame
        self.use_physics_head = use_physics_head

        # 1D CNN backbone: (batch, obs_per_frame, num_frames)
        self.conv = nn.Sequential(
            nn.Conv1d(obs_per_frame, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
        )

        # Calculate CNN output size: 64 channels × 5 time steps = 320
        cnn_out_dim = 64 * 5

        # MLP after CNN
        self.fc = nn.Sequential(
            nn.Linear(cnn_out_dim, hidden_dim),
            nn.ReLU(),
        )

        # Action head
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

        # Optional physics estimation head
        if use_physics_head:
            self.physics_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, physics_dim),
                nn.Sigmoid()  # Output in [0, 1]
            )

    def forward(self, obs):
        """Forward pass."""
        batch_size = obs.shape[0]

        # Reshape: (batch, 84) -> (batch, 12, 7) -> (batch, 7, 12)
        x = obs.view(batch_size, self.num_frames, self.obs_per_frame)
        x = x.permute(0, 2, 1)

        # CNN feature extraction
        x = self.conv(x)
        x = x.flatten(1)

        # MLP
        features = self.fc(x)

        # Action output
        action_mean = torch.tanh(self.mean(features))
        action_log_std = torch.clamp(self.log_std(features), -20, 2)

        if self.use_physics_head:
            physics_est = self.physics_head(features)
            return action_mean, action_log_std, physics_est

        return action_mean, action_log_std

    def sample(self, obs):
        """Sample action with reparameterization trick."""
        if self.use_physics_head:
            mean, log_std, physics_est = self.forward(obs)
        else:
            mean, log_std = self.forward(obs)
            physics_est = None

        std = log_std.exp()
        normal = Normal(mean, std)
        x = normal.rsample()
        action = torch.tanh(x)

        log_prob = normal.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        if self.use_physics_head:
            return action, log_prob, physics_est
        return action, log_prob

    def get_deterministic(self, obs):
        """Get deterministic action (for evaluation)."""
        if self.use_physics_head:
            mean, _, physics_est = self.forward(obs)
            return mean, physics_est
        mean, _ = self.forward(obs)
        return mean


class CriticCNN(nn.Module):
    """Twin Q-networks with 1D CNN for temporal processing."""

    def __init__(self, state_dim, action_dim, hidden_dim=256,
                 num_frames=12, obs_per_frame=7):
        super(CriticCNN, self).__init__()

        self.num_frames = num_frames
        self.obs_per_frame = obs_per_frame

        # Shared CNN for observation encoding
        self.conv = nn.Sequential(
            nn.Conv1d(obs_per_frame, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
        )

        cnn_out_dim = 64 * 5

        # Q1 network with LayerNorm for stability
        self.q1 = nn.Sequential(
            nn.Linear(cnn_out_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Q2 network
        self.q2 = nn.Sequential(
            nn.Linear(cnn_out_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, action):
        """Forward pass through both Q networks."""
        batch_size = obs.shape[0]

        # Reshape: (batch, 84) -> (batch, 7, 12)
        x = obs.view(batch_size, self.num_frames, self.obs_per_frame)
        x = x.permute(0, 2, 1)

        # CNN feature extraction
        x = self.conv(x)
        x = x.flatten(1)

        # Concatenate with action
        x = torch.cat([x, action], dim=-1)

        return self.q1(x), self.q2(x)
