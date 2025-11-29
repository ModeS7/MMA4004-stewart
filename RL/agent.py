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
    1D CNN actor that processes frame history temporally.

    Outputs:
        - Action (2D): platform tilt targets
        - Physics estimate (3D): [friction, servo_tau, mass_factor]
    """

    def __init__(self, input_dim, action_dim, hidden_dim=256, physics_dim=3,
                 num_frames=12, obs_per_frame=7):
        super(Actor, self).__init__()

        self.input_dim = input_dim
        self.physics_dim = physics_dim
        self.num_frames = num_frames
        self.obs_per_frame = obs_per_frame

        # 1D CNN backbone: (batch, 7, 12) -> (batch, features)
        # Kernel spans all features, slides across time
        self.conv = nn.Sequential(
            nn.Conv1d(obs_per_frame, 32, kernel_size=3, padding=1),  # (batch, 32, 12)
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),             # (batch, 64, 12)
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=2),              # (batch, 64, 5)
            nn.ReLU(),
        )

        # Calculate CNN output size: 64 channels × 5 time steps = 320
        cnn_out_dim = 64 * 5

        # MLP after CNN
        self.fc = nn.Sequential(
            nn.Linear(cnn_out_dim, hidden_dim),
            nn.ReLU(),
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
            obs: (batch, input_dim) observation history (84 = 12 frames × 7 features)

        Returns:
            action_mean, action_log_std, physics_estimate
        """
        batch_size = obs.shape[0]

        # Reshape: (batch, 84) -> (batch, 12, 7) -> (batch, 7, 12)
        # Conv1d expects (batch, channels, length)
        x = obs.view(batch_size, self.num_frames, self.obs_per_frame)
        x = x.permute(0, 2, 1)  # (batch, 7, 12)

        # CNN feature extraction
        x = self.conv(x)
        x = x.flatten(1)  # (batch, 320)

        # MLP
        features = self.fc(x)

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
    Twin Q-networks with 1D CNN for temporal observation processing.
    Uses LayerNorm for training stability in parallel environments.
    """

    def __init__(self, input_dim, action_dim, hidden_dim=256,
                 num_frames=12, obs_per_frame=7, use_layer_norm=True):
        super(Critic, self).__init__()

        self.num_frames = num_frames
        self.obs_per_frame = obs_per_frame

        # Shared CNN for observation encoding
        self.conv = nn.Sequential(
            nn.Conv1d(obs_per_frame, 32, kernel_size=3, padding=1),  # (batch, 32, 12)
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),             # (batch, 64, 12)
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=2),              # (batch, 64, 5)
            nn.ReLU(),
        )

        # CNN output: 64 × 5 = 320, plus action_dim
        cnn_out_dim = 64 * 5

        # Q1 network with LayerNorm for stability
        if use_layer_norm:
            self.q1 = nn.Sequential(
                nn.Linear(cnn_out_dim + action_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            self.q2 = nn.Sequential(
                nn.Linear(cnn_out_dim + action_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        else:
            self.q1 = nn.Sequential(
                nn.Linear(cnn_out_dim + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            self.q2 = nn.Sequential(
                nn.Linear(cnn_out_dim + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )

    def forward(self, obs, action):
        """Forward pass through both Q networks."""
        batch_size = obs.shape[0]

        # Reshape: (batch, 84) -> (batch, 7, 12)
        x = obs.view(batch_size, self.num_frames, self.obs_per_frame)
        x = x.permute(0, 2, 1)  # (batch, 7, 12)

        # CNN feature extraction
        x = self.conv(x)
        x = x.flatten(1)  # (batch, 320)

        # Concatenate with action
        x = torch.cat([x, action], dim=-1)

        return self.q1(x), self.q2(x)


# ============================================================================
# REPLAY BUFFER
# ============================================================================

class ReplayBuffer:
    """
    Replay buffer for observations.

    Stores physics ground truth for auxiliary loss.
    """

    def __init__(self, capacity, obs_dim, action_dim, physics_dim=3, device="cpu"):
        self.capacity = capacity
        self.position = 0
        self.size = 0
        self.device = torch.device(device)
        self.on_gpu = device != "cpu"

        if self.on_gpu:
            # GPU tensors
            self.states = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=self.device)
            self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=self.device)
            self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=self.device)
            self.next_states = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=self.device)
            self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device=self.device)
            self.physics_gt = torch.zeros((capacity, physics_dim), dtype=torch.float32, device=self.device)
        else:
            # CPU numpy arrays
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

    def push_batch(self, states, actions, rewards, next_states, dones, physics_gt=None, mask=None):
        """Add multiple transitions at once (vectorized)."""
        if mask is not None:
            # Filter by mask
            indices = np.where(mask)[0]
            states = states[indices]
            actions = actions[indices]
            rewards = rewards[indices]
            next_states = next_states[indices]
            dones = dones[indices]
            if physics_gt is not None:
                physics_gt = physics_gt[indices]

        n = len(states)
        if n == 0:
            return

        # Convert to tensors if on GPU
        if self.on_gpu:
            states = torch.from_numpy(states).to(self.device)
            actions = torch.from_numpy(actions).to(self.device)
            rewards = torch.from_numpy(rewards).to(self.device).reshape(-1, 1)
            next_states = torch.from_numpy(next_states).to(self.device)
            dones = torch.from_numpy(dones).to(self.device).reshape(-1, 1)
            if physics_gt is not None:
                physics_gt = torch.from_numpy(physics_gt).to(self.device)
        else:
            rewards = rewards.reshape(-1, 1)
            dones = dones.reshape(-1, 1)

        # Handle wrap-around
        end_pos = self.position + n
        if end_pos <= self.capacity:
            # No wrap
            self.states[self.position:end_pos] = states
            self.actions[self.position:end_pos] = actions
            self.rewards[self.position:end_pos] = rewards
            self.next_states[self.position:end_pos] = next_states
            self.dones[self.position:end_pos] = dones
            if physics_gt is not None:
                self.physics_gt[self.position:end_pos] = physics_gt
        else:
            # Wrap around
            first_part = self.capacity - self.position
            second_part = n - first_part

            self.states[self.position:] = states[:first_part]
            self.states[:second_part] = states[first_part:]

            self.actions[self.position:] = actions[:first_part]
            self.actions[:second_part] = actions[first_part:]

            self.rewards[self.position:] = rewards[:first_part]
            self.rewards[:second_part] = rewards[first_part:]

            self.next_states[self.position:] = next_states[:first_part]
            self.next_states[:second_part] = next_states[first_part:]

            self.dones[self.position:] = dones[:first_part]
            self.dones[:second_part] = dones[first_part:]

            if physics_gt is not None:
                self.physics_gt[self.position:] = physics_gt[:first_part]
                self.physics_gt[:second_part] = physics_gt[first_part:]

        self.position = end_pos % self.capacity
        self.size = min(self.size + n, self.capacity)

    def push_batch_tensor(self, states, actions, rewards, next_states, dones, physics_gt, mask=None):
        """Add all transitions from GPU tensors (no CPU transfer, no filtering).

        Note: mask is ignored to avoid CUDA syncs. All transitions are stored.
        """
        n = states.shape[0]  # Known at compile time, no sync
        rewards = rewards.unsqueeze(-1)
        dones = dones.float().unsqueeze(-1)

        # Handle wrap-around (n is fixed NUM_ENVS, so end_pos is predictable)
        end_pos = self.position + n
        if end_pos <= self.capacity:
            # No wrap
            self.states[self.position:end_pos] = states
            self.actions[self.position:end_pos] = actions
            self.rewards[self.position:end_pos] = rewards
            self.next_states[self.position:end_pos] = next_states
            self.dones[self.position:end_pos] = dones
            self.physics_gt[self.position:end_pos] = physics_gt
        else:
            # Wrap around
            first_part = self.capacity - self.position
            second_part = n - first_part

            self.states[self.position:] = states[:first_part]
            self.states[:second_part] = states[first_part:]

            self.actions[self.position:] = actions[:first_part]
            self.actions[:second_part] = actions[first_part:]

            self.rewards[self.position:] = rewards[:first_part]
            self.rewards[:second_part] = rewards[first_part:]

            self.next_states[self.position:] = next_states[:first_part]
            self.next_states[:second_part] = next_states[first_part:]

            self.dones[self.position:] = dones[:first_part]
            self.dones[:second_part] = dones[first_part:]

            self.physics_gt[self.position:] = physics_gt[:first_part]
            self.physics_gt[:second_part] = physics_gt[first_part:]

        self.position = end_pos % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size):
        """Sample a batch of transitions. Returns tensors if on GPU, numpy if on CPU."""
        indices = torch.randint(0, self.size, (batch_size,), device=self.device) if self.on_gpu else \
                  np.random.randint(0, self.size, size=batch_size)

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
            obs_dim,          # Total observation dim (e.g., 12 * 7 = 84)
            action_dim=2,
            hidden_dim=256,
            physics_dim=3,    # [friction, servo_tau, mass_factor]
            num_frames=12,    # Number of frames in history
            obs_per_frame=7,  # Features per frame
            actor_lr=3e-4,    # Policy learning rate
            critic_lr=3e-4,   # Q-network learning rate
            gamma=0.99,       # Discount factor
            tau=0.005,        # Soft update coefficient (slower for stability)
            alpha=0.01,       # Initial entropy coefficient (lower for parallel)
            alpha_lr=1e-4,    # Slower learning rate for alpha (prevents collapse)
            alpha_min=0.005,  # Minimum alpha floor
            policy_delay=8,   # Update actor every N critic updates (TD3-style stability)
            physics_loss_weight=0.1,  # Weight for auxiliary physics loss
            automatic_entropy_tuning=True,
            use_layer_norm=True,  # LayerNorm on critic for stability
            device="cpu",
            compile_model=False,  # Use torch.compile for speedup (PyTorch 2.0+)
            use_amp=False  # Use automatic mixed precision (bfloat16)
    ):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.alpha_min = alpha_min
        self.policy_delay = policy_delay
        self.update_counter = 0  # Track critic updates for policy delay
        self.physics_loss_weight = physics_loss_weight
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.obs_dim = obs_dim
        self.physics_dim = physics_dim
        self.use_amp = use_amp and device != "cpu"

        # Device
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

        # AMP dtype (bfloat16 is more stable than float16 for RL)
        self.amp_dtype = torch.bfloat16 if self.use_amp else torch.float32

        # Networks (1D CNN for temporal processing)
        self.actor = Actor(obs_dim, action_dim, hidden_dim, physics_dim,
                          num_frames, obs_per_frame).to(self.device)
        self.critic = Critic(obs_dim, action_dim, hidden_dim,
                            num_frames, obs_per_frame, use_layer_norm).to(self.device)
        self.critic_target = Critic(obs_dim, action_dim, hidden_dim,
                                   num_frames, obs_per_frame, use_layer_norm).to(self.device)

        # Optional: compile networks for speedup (PyTorch 2.0+)
        if compile_model and hasattr(torch, 'compile'):
            # Use 'default' mode - 'reduce-overhead' uses CUDA graphs which conflict with SAC
            self.actor = torch.compile(self.actor)
            self.critic = torch.compile(self.critic)
            self.critic_target = torch.compile(self.critic_target)

        # Copy parameters to target
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # Optimizers (separate learning rates for actor and critic)
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=critic_lr)

        # Automatic entropy tuning with slower learning rate
        if automatic_entropy_tuning:
            self.target_entropy = -action_dim
            # Initialize log_alpha from configured alpha (not zeros which gives alpha=1.0)
            self.log_alpha = torch.tensor([np.log(alpha)], requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.AdamW([self.log_alpha], lr=alpha_lr)  # Slower LR
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

    def select_action_batch_tensor(self, obs_tensor, evaluate=False):
        """
        Select actions for a batch of observations (GPU tensor version).

        Args:
            obs_tensor: (batch, obs_dim) GPU tensor
            evaluate: If True, use deterministic action

        Returns:
            actions: (batch, action_dim) GPU tensor
        """
        with torch.no_grad():
            if evaluate:
                actions, _ = self.actor.get_action_deterministic(obs_tensor)
            else:
                actions, _, _ = self.actor.sample(obs_tensor)
        return actions

    def update(self, replay_buffer, batch_size=256):
        """
        Update actor and critic networks with TD3-style policy delay.

        Critic is updated every call, but actor only every policy_delay calls.
        This prevents the actor from following unstable Q-value estimates.

        Returns:
            dict with losses and metrics
        """
        # Increment update counter
        self.update_counter += 1

        # Sample batch
        states, actions, rewards, next_states, dones, physics_gt = replay_buffer.sample(batch_size)

        # Convert to tensors (skip if already tensors from GPU buffer)
        if not isinstance(states, torch.Tensor):
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.FloatTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.FloatTensor(dones).to(self.device)
            physics_gt = torch.FloatTensor(physics_gt).to(self.device)

        # ===== Update Critic (every step) =====
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                next_actions, next_log_probs, _ = self.actor.sample(next_states)
                target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)

            if self.automatic_entropy_tuning:
                alpha = self.log_alpha.exp()
            else:
                alpha = self.alpha

            target_q = target_q - alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * target_q

        with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
            current_q1, current_q2 = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q1.float(), target_q.float()) + nn.MSELoss()(current_q2.float(), target_q.float())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Initialize return values (in case actor not updated this step)
        actor_loss_val = 0.0
        policy_loss_val = 0.0
        physics_loss_val = 0.0

        # ===== Update Actor (every policy_delay steps) =====
        if self.update_counter % self.policy_delay == 0:
            with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                new_actions, log_probs, physics_est = self.actor.sample(states)
                q1, q2 = self.critic(states, new_actions)
            min_q = torch.min(q1, q2)

            if self.automatic_entropy_tuning:
                alpha = self.log_alpha.exp()
            else:
                alpha = self.alpha

            # Actor loss = policy loss + physics auxiliary loss (in float32 for stability)
            policy_loss = (alpha * log_probs.float() - min_q.float()).mean()
            physics_loss = nn.MSELoss()(physics_est.float(), physics_gt)
            actor_loss = policy_loss + self.physics_loss_weight * physics_loss

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            actor_loss_val = actor_loss.item()
            policy_loss_val = policy_loss.item()
            physics_loss_val = physics_loss.item()

            # ===== Update Alpha (only when actor updates) =====
            if self.automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_probs.float() + self.target_entropy).detach()).mean()

                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

                # Apply minimum alpha floor to prevent entropy collapse
                self.alpha = max(self.log_alpha.exp().item(), self.alpha_min)

            # ===== Soft Update Target (only when actor updates) =====
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss_val,
            'policy_loss': policy_loss_val,
            'physics_loss': physics_loss_val,
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
