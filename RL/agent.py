"""
Feedforward SAC Agent with Physics Estimation

Uses frame stacking (12 frames) with timing info for sim-to-real transfer.
Auxiliary physics estimation head helps learn robust representations.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


# ============================================================================
# ACTOR NETWORK WITH PHYSICS ESTIMATION
# ============================================================================

class Actor(nn.Module):
    """
    Feedforward actor that takes observation history.

    Outputs:
        - Action (2D): platform tilt targets
        - Physics estimate (3D): [friction, servo_tau, mass_factor]
    """

    def __init__(self, input_dim, action_dim, hidden_dim=256, physics_dim=3):
        super(Actor, self).__init__()

        self.input_dim = input_dim
        self.physics_dim = physics_dim

        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Action head (Gaussian policy)
        self.action_mean = nn.Linear(hidden_dim, action_dim)
        self.action_log_std = nn.Linear(hidden_dim, action_dim)

        # Physics estimation head (auxiliary task)
        self.physics_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, physics_dim),
            nn.Sigmoid()  # Output in [0, 1], scale to actual ranges later
        )

    def forward(self, obs):
        """
        Forward pass.

        Args:
            obs: (batch, input_dim) observation history

        Returns:
            action_mean, action_log_std, physics_estimate
        """
        features = self.backbone(obs)

        # Action output
        action_mean = torch.tanh(self.action_mean(features))
        action_log_std = torch.clamp(self.action_log_std(features), -20, 2)

        # Physics estimation
        physics_est = self.physics_head(features)

        return action_mean, action_log_std, physics_est

    def sample(self, obs):
        """Sample action with reparameterization trick."""
        mean, log_std, physics_est = self.forward(obs)
        std = log_std.exp()

        # Sample from Gaussian
        normal = Normal(mean, std)
        x = normal.rsample()

        # Squash to [-1, 1]
        action = torch.tanh(x)

        # Log probability with squashing correction
        log_prob = normal.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob, physics_est

    def get_action_deterministic(self, obs):
        """Get deterministic action (for evaluation)."""
        mean, _, physics_est = self.forward(obs)
        return mean, physics_est


# ============================================================================
# CRITIC NETWORK
# ============================================================================

class Critic(nn.Module):
    """
    Twin Q-networks for observations.
    """

    def __init__(self, input_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()

        # Q1 network
        self.q1 = nn.Sequential(
            nn.Linear(input_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Q2 network
        self.q2 = nn.Sequential(
            nn.Linear(input_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, action):
        """Forward pass through both Q networks."""
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)


# ============================================================================
# REPLAY BUFFER
# ============================================================================

class ReplayBuffer:
    """
    Replay buffer for observations.

    Stores physics ground truth for auxiliary loss.
    """

    def __init__(self, capacity, obs_dim, action_dim, physics_dim=3):
        self.capacity = capacity
        self.position = 0
        self.size = 0

        # Pre-allocate arrays
        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.physics_gt = np.zeros((capacity, physics_dim), dtype=np.float32)

    def push(self, state, action, reward, next_state, done, physics_gt=None):
        """Add a single transition."""
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        if physics_gt is not None:
            self.physics_gt[self.position] = physics_gt

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """Sample a batch of transitions."""
        indices = np.random.randint(0, self.size, size=batch_size)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            self.physics_gt[indices]
        )

    def __len__(self):
        return self.size


# ============================================================================
# FLAT SAC AGENT
# ============================================================================

class SACAgent:
    """
    SAC agent with frame stacking and physics estimation.

    Features:
        - 12-frame history input
        - Variable dt support for real-time deployment
        - Auxiliary physics estimation for sim-to-real
        - Standard SAC with automatic entropy tuning
    """

    def __init__(
            self,
            obs_dim,          # Total observation dim (e.g., 12 * 5 = 60)
            action_dim=2,
            hidden_dim=256,
            physics_dim=3,    # [friction, servo_tau, mass_factor]
            lr=3e-4,
            gamma=0.99,
            tau=0.005,
            alpha=0.2,
            physics_loss_weight=0.1,  # Weight for auxiliary physics loss
            automatic_entropy_tuning=True,
            device="cpu"
    ):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.physics_loss_weight = physics_loss_weight
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.obs_dim = obs_dim
        self.physics_dim = physics_dim

        # Device
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

        # Networks
        self.actor = Actor(obs_dim, action_dim, hidden_dim, physics_dim).to(self.device)
        self.critic = Critic(obs_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = Critic(obs_dim, action_dim, hidden_dim).to(self.device)

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

    def select_action(self, obs, evaluate=False):
        """
        Select action for a single observation.

        Args:
            obs: (obs_dim,) numpy array
            evaluate: If True, use deterministic action

        Returns:
            action: (action_dim,) numpy array
            physics_est: (physics_dim,) numpy array
        """
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if evaluate:
                action, physics_est = self.actor.get_action_deterministic(obs)
            else:
                action, _, physics_est = self.actor.sample(obs)

        return action.cpu().numpy()[0], physics_est.cpu().numpy()[0]

    def select_action_batch(self, obs_batch, evaluate=False):
        """
        Select actions for a batch of observations.

        Args:
            obs_batch: (batch, obs_dim) numpy array
            evaluate: If True, use deterministic action

        Returns:
            actions: (batch, action_dim) numpy array
            physics_est: (batch, physics_dim) numpy array
        """
        obs_batch = torch.FloatTensor(obs_batch).to(self.device)

        with torch.no_grad():
            if evaluate:
                actions, physics_est = self.actor.get_action_deterministic(obs_batch)
            else:
                actions, _, physics_est = self.actor.sample(obs_batch)

        return actions.cpu().numpy(), physics_est.cpu().numpy()

    def update(self, replay_buffer, batch_size=256):
        """
        Update actor and critic networks.

        Returns:
            dict with losses and metrics
        """
        # Sample batch
        states, actions, rewards, next_states, dones, physics_gt = replay_buffer.sample(batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        physics_gt = torch.FloatTensor(physics_gt).to(self.device)

        # ===== Update Critic =====
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)

            if self.automatic_entropy_tuning:
                alpha = self.log_alpha.exp()
            else:
                alpha = self.alpha

            target_q = target_q - alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * target_q

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ===== Update Actor =====
        new_actions, log_probs, physics_est = self.actor.sample(states)
        q1, q2 = self.critic(states, new_actions)
        min_q = torch.min(q1, q2)

        if self.automatic_entropy_tuning:
            alpha = self.log_alpha.exp()
        else:
            alpha = self.alpha

        # Actor loss = policy loss + physics auxiliary loss
        policy_loss = (alpha * log_probs - min_q).mean()
        physics_loss = nn.MSELoss()(physics_est, physics_gt)
        actor_loss = policy_loss + self.physics_loss_weight * physics_loss

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
            'policy_loss': policy_loss.item(),
            'physics_loss': physics_loss.item(),
            'alpha': self.alpha
        }

    def save(self, path):
        """Save model."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'log_alpha': self.log_alpha if self.automatic_entropy_tuning else None
        }, path)

    def load(self, path):
        """Load model."""
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
    print("Testing SAC Agent with Physics Estimation...")

    # Config: 12 frames × 5 values (x, y, rx, ry, dt) = 60
    num_frames = 12
    obs_per_frame = 5
    obs_dim = num_frames * obs_per_frame
    action_dim = 2
    physics_dim = 3

    print(f"  Frames: {num_frames}")
    print(f"  Obs per frame: {obs_per_frame}")
    print(f"  Total input dim: {obs_dim}")

    # Create agent
    agent = SACAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        physics_dim=physics_dim,
        device="cpu"
    )
    print(f"Created agent on device: {agent.device}")

    # Test action selection
    obs = np.random.randn(obs_dim).astype(np.float32)
    action, physics_est = agent.select_action(obs)
    print(f"Action: {action}, shape: {action.shape}")
    print(f"Physics estimate: {physics_est}")

    # Test batch action selection
    obs_batch = np.random.randn(10, obs_dim).astype(np.float32)
    actions, physics_batch = agent.select_action_batch(obs_batch)
    print(f"Batch actions shape: {actions.shape}")

    # Test replay buffer
    buffer = ReplayBuffer(10000, obs_dim, action_dim, physics_dim)
    for _ in range(1000):
        s = np.random.randn(obs_dim).astype(np.float32)
        a = np.random.randn(action_dim).astype(np.float32)
        r = np.random.randn()
        s2 = np.random.randn(obs_dim).astype(np.float32)
        d = np.random.choice([0, 1])
        phys = np.random.rand(physics_dim).astype(np.float32)
        buffer.push(s, a, r, s2, d, phys)

    print(f"Buffer size: {len(buffer)}")

    # Test update
    info = agent.update(buffer, batch_size=256)
    print(f"Update info: {info}")

    print("\n[OK] SAC Agent test passed!")
