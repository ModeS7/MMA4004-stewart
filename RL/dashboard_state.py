"""
Shared State for Interactive Training Dashboard

Thread-safe state object shared between training loop and Gradio UI.
"""

from dataclasses import dataclass, field
from typing import List
import threading


@dataclass
class TrainingState:
    """
    Shared state between training loop and dashboard.

    Training loop updates metrics, dashboard updates controls.
    All access should be through the provided methods for thread safety.
    """

    # ===== Metrics (updated by training loop) =====
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[float] = field(default_factory=list)
    eval_rewards: List[float] = field(default_factory=list)

    current_episode: int = 0
    current_step: int = 0
    current_alpha: float = 0.0
    current_critic_loss: float = 0.0
    current_actor_loss: float = 0.0

    # ===== Environment Controls (updated by dashboard) =====
    use_domain_randomization: bool = False
    use_platform_offset: bool = True
    use_camera_noise: bool = False

    platform_offset_max_deg: float = 2.0
    camera_noise_std_mm: float = 2.0

    # ===== Reward Controls (updated by dashboard) =====
    dist_scale: float = 30.0      # Distance scale for reward
    speed_scale: float = 50.0     # Speed scale for reward
    fall_penalty: float = -10.0   # Penalty for falling off

    # ===== Training Controls =====
    paused: bool = False
    save_requested: bool = False
    stop_requested: bool = False

    # ===== Internal =====
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_episode(self, reward: float, length: float):
        """Thread-safe add episode metrics."""
        with self._lock:
            self.episode_rewards.append(reward)
            self.episode_lengths.append(length)
            self.current_episode = len(self.episode_rewards)

    def add_eval(self, reward: float):
        """Thread-safe add eval reward."""
        with self._lock:
            self.eval_rewards.append(reward)

    def update_losses(self, critic: float, actor: float, alpha: float):
        """Thread-safe update current losses."""
        with self._lock:
            self.current_critic_loss = critic
            self.current_actor_loss = actor
            self.current_alpha = alpha

    def get_metrics(self):
        """Thread-safe get all metrics."""
        with self._lock:
            return {
                'episode_rewards': self.episode_rewards.copy(),
                'episode_lengths': self.episode_lengths.copy(),
                'eval_rewards': self.eval_rewards.copy(),
                'current_episode': self.current_episode,
                'current_alpha': self.current_alpha,
                'critic_loss': self.current_critic_loss,
                'actor_loss': self.current_actor_loss,
            }

    def get_env_settings(self):
        """Thread-safe get environment settings."""
        with self._lock:
            return {
                'use_domain_randomization': self.use_domain_randomization,
                'use_platform_offset': self.use_platform_offset,
                'use_camera_noise': self.use_camera_noise,
                'platform_offset_max_deg': self.platform_offset_max_deg,
                'camera_noise_std_mm': self.camera_noise_std_mm,
            }

    def get_reward_settings(self):
        """Thread-safe get reward settings."""
        with self._lock:
            return {
                'dist_scale': self.dist_scale,
                'speed_scale': self.speed_scale,
                'fall_penalty': self.fall_penalty,
            }

    def request_save(self):
        """Request a checkpoint save."""
        with self._lock:
            self.save_requested = True

    def clear_save_request(self):
        """Clear save request (called by training loop after saving)."""
        with self._lock:
            self.save_requested = False

    def toggle_pause(self):
        """Toggle pause state."""
        with self._lock:
            self.paused = not self.paused
            return self.paused

    def request_stop(self):
        """Request training stop."""
        with self._lock:
            self.stop_requested = True


# Global state instance
_global_state: TrainingState = None


def get_state() -> TrainingState:
    """Get or create the global training state."""
    global _global_state
    if _global_state is None:
        _global_state = TrainingState()
    return _global_state


def reset_state():
    """Reset the global state (for new training runs)."""
    global _global_state
    _global_state = TrainingState()
    return _global_state
