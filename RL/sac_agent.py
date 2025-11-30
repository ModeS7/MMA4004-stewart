"""
Unified SAC Agent

Single agent class that works with both MLP and CNN architectures.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from RL.networks import ActorMLP, ActorCNN, CriticMLP, CriticCNN


# ============================================================================
# REPLAY BUFFER
# ============================================================================

class ReplayBuffer:
    """Simple replay buffer for off-policy learning."""

    def __init__(self, capacity, obs_dim, action_dim, device="cpu"):
        self.capacity = capacity
        self.position = 0
        self.size = 0
        self.device = torch.device(device)

        # Pre-allocate numpy arrays (CPU storage, transfer on sample)
        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
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

    def sample(self, batch_size):
        """Sample a batch and return as tensors on device."""
        indices = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.states[indices]).to(self.device),
            torch.FloatTensor(self.actions[indices]).to(self.device),
            torch.FloatTensor(self.rewards[indices]).to(self.device),
            torch.FloatTensor(self.next_states[indices]).to(self.device),
            torch.FloatTensor(self.dones[indices]).to(self.device),
        )

    def __len__(self):
        return self.size


# ============================================================================
# UNIFIED SAC AGENT
# ============================================================================

class SACAgent:
    """
    Unified SAC agent supporting both MLP and CNN architectures.

    Args:
        architecture: "mlp" or "cnn"
        state_dim: Observation dimension (e.g., 84 = 12 frames × 7 features)
        action_dim: Action dimension (default 2 for platform tilt)
        hidden_dim: Hidden layer size
        num_frames: Number of frames for CNN (default 12)
        obs_per_frame: Features per frame for CNN (default 7)
        use_physics_head: Enable physics estimation (CNN only)
        physics_dim: Physics estimation dimension
        lr: Learning rate
        gamma: Discount factor
        tau: Soft update coefficient
        alpha: Initial entropy coefficient
        automatic_entropy: Auto-tune entropy
        device: "cuda" or "cpu"
    """

    def __init__(
            self,
            architecture="mlp",
            state_dim=84,
            action_dim=2,
            hidden_dim=256,
            num_frames=12,
            obs_per_frame=7,
            use_physics_head=False,
            physics_dim=3,
            physics_loss_weight=0.1,
            lr=3e-4,
            gamma=0.99,
            tau=0.005,
            alpha=0.2,
            automatic_entropy=True,
            device="cuda"
    ):
        self.architecture = architecture
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.automatic_entropy = automatic_entropy
        self.use_physics_head = use_physics_head and architecture == "cnn"
        self.physics_loss_weight = physics_loss_weight
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Create networks based on architecture
        if architecture == "mlp":
            self.actor = ActorMLP(state_dim, action_dim, hidden_dim).to(self.device)
            self.critic = CriticMLP(state_dim, action_dim, hidden_dim).to(self.device)
            self.critic_target = CriticMLP(state_dim, action_dim, hidden_dim).to(self.device)
        elif architecture == "cnn":
            self.actor = ActorCNN(
                state_dim, action_dim, hidden_dim,
                num_frames, obs_per_frame, physics_dim, self.use_physics_head
            ).to(self.device)
            self.critic = CriticCNN(
                state_dim, action_dim, hidden_dim, num_frames, obs_per_frame
            ).to(self.device)
            self.critic_target = CriticCNN(
                state_dim, action_dim, hidden_dim, num_frames, obs_per_frame
            ).to(self.device)
        else:
            raise ValueError(f"Unknown architecture: {architecture}. Use 'mlp' or 'cnn'.")

        # Copy parameters to target
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # Optimizers
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=lr)

        # Automatic entropy tuning
        if automatic_entropy:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.AdamW([self.log_alpha], lr=lr)

    def select_action(self, obs, evaluate=False):
        """
        Select action for a single observation.

        Args:
            obs: (state_dim,) numpy array
            evaluate: If True, use deterministic action

        Returns:
            action: (action_dim,) numpy array
        """
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if evaluate:
                if self.use_physics_head:
                    action, _ = self.actor.get_deterministic(obs)
                else:
                    action = self.actor.get_deterministic(obs)
            else:
                if self.use_physics_head:
                    action, _, _ = self.actor.sample(obs)
                else:
                    action, _ = self.actor.sample(obs)

        return action.cpu().numpy()[0]

    def update(self, replay_buffer, batch_size=512, physics_gt=None):
        """
        Update actor and critic networks.

        Args:
            replay_buffer: ReplayBuffer instance
            batch_size: Batch size for update
            physics_gt: Optional physics ground truth for CNN with physics head

        Returns:
            dict with losses and metrics
        """
        # Sample batch
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        # ===== Update Critic =====
        with torch.no_grad():
            if self.use_physics_head:
                next_actions, next_log_probs, _ = self.actor.sample(next_states)
            else:
                next_actions, next_log_probs = self.actor.sample(next_states)

            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)

            if self.automatic_entropy:
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
        if self.use_physics_head:
            new_actions, log_probs, physics_est = self.actor.sample(states)
        else:
            new_actions, log_probs = self.actor.sample(states)

        q1, q2 = self.critic(states, new_actions)
        min_q = torch.min(q1, q2)

        if self.automatic_entropy:
            alpha = self.log_alpha.exp()
        else:
            alpha = self.alpha

        actor_loss = (alpha * log_probs - min_q).mean()

        # Add physics loss if enabled and ground truth provided
        physics_loss_val = 0.0
        if self.use_physics_head and physics_gt is not None:
            physics_gt_tensor = torch.FloatTensor(physics_gt).to(self.device)
            physics_loss = nn.MSELoss()(physics_est, physics_gt_tensor)
            actor_loss = actor_loss + self.physics_loss_weight * physics_loss
            physics_loss_val = physics_loss.item()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ===== Update Alpha =====
        alpha_loss_val = 0.0
        if self.automatic_entropy:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp().item()
            alpha_loss_val = alpha_loss.item()

        # ===== Soft Update Target =====
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'physics_loss': physics_loss_val,
            'alpha': self.alpha if isinstance(self.alpha, float) else self.alpha
        }

    def save(self, path):
        """Save model checkpoint."""
        torch.save({
            'architecture': self.architecture,
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha.detach() if self.automatic_entropy else None,
            'alpha': self.alpha,
        }, path)

    def load(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])

        if 'actor_optimizer' in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        if 'critic_optimizer' in checkpoint:
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        if self.automatic_entropy and checkpoint.get('log_alpha') is not None:
            self.log_alpha.data.copy_(checkpoint['log_alpha'])
            self.alpha = self.log_alpha.exp().item()
        elif 'alpha' in checkpoint:
            self.alpha = checkpoint['alpha']


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_agent(architecture, state_dim, action_dim, **kwargs):
    """
    Factory function to create SAC agent.

    Args:
        architecture: "mlp" or "cnn"
        state_dim: Observation dimension
        action_dim: Action dimension
        **kwargs: Additional arguments for SACAgent

    Returns:
        SACAgent instance
    """
    return SACAgent(
        architecture=architecture,
        state_dim=state_dim,
        action_dim=action_dim,
        **kwargs
    )
