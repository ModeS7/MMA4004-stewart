"""
Profile training to identify performance bottlenecks.

Usage:
    python RL/profile_training.py

View results:
    tensorboard --logdir RL/profiler_logs
"""

import os
import sys
import numpy as np
import torch
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler, schedule

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from RL.env import StewartEnvVec
from RL.rl_config import EnvConfig, RewardConfig
from RL.sac_agent import SACAgent, ReplayBuffer

# Configuration
NUM_STEPS = 500           # Steps to profile
WARMUP_STEPS = 100        # Random actions before training
BATCH_SIZE = 256
DEVICE = "cuda"
USE_COMPILE = True        # Enable torch.compile for kernel fusion
USE_AMP = True           # Mixed precision (FP16) - disabled: adds overhead for small networks


def main():
    print("=" * 60)
    print("PyTorch Profiler - Stewart Platform Training")
    print("=" * 60)

    # Setup
    env_cfg = EnvConfig()
    reward_cfg = RewardConfig()
    env = StewartEnvVec(num_envs=1, config=env_cfg, reward_config=reward_cfg, device=DEVICE)

    agent = SACAgent(
        architecture="cnn",
        state_dim=env_cfg.obs_dim,
        action_dim=env_cfg.action_dim,
        hidden_dim=256,
        num_frames=env_cfg.num_frames,
        obs_per_frame=env_cfg.obs_per_frame,
        device=DEVICE,
        compile_model=USE_COMPILE,
        use_amp=USE_AMP
    )

    if USE_COMPILE:
        print("torch.compile enabled (first few steps will be slower due to compilation)")
    if USE_AMP:
        print("Mixed precision (AMP) enabled for faster training")

    buffer = ReplayBuffer(50000, env_cfg.obs_dim, env_cfg.action_dim, device=DEVICE)

    # Create log directory
    log_dir = "RL/profiler_logs"
    os.makedirs(log_dir, exist_ok=True)
    print(f"Profiler logs: {log_dir}")
    print(f"View with: tensorboard --logdir {log_dir}")
    print()

    # Profile schedule: wait, warmup, active, repeat
    profiler_schedule = schedule(
        wait=50,      # Skip first 50 steps (buffer filling)
        warmup=50,    # Warmup for 50 steps (no recording)
        active=200,   # Record 200 steps
        repeat=1      # Only one cycle
    )

    obs, _ = env.reset()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=profiler_schedule,
        on_trace_ready=tensorboard_trace_handler(log_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True
    ) as prof:

        for step in range(NUM_STEPS):
            # Mark regions for better visualization
            with torch.profiler.record_function("ACTION_SELECT"):
                if step < WARMUP_STEPS:
                    action = np.random.uniform(-1, 1, (1, 2)).astype(np.float32)
                else:
                    action = agent.select_action(obs[0], evaluate=False)
                    action = action.reshape(1, -1)

            with torch.profiler.record_function("ENV_STEP"):
                next_obs, rewards, dones, truncateds, _ = env.step(action)

            with torch.profiler.record_function("BUFFER_PUSH"):
                buffer.push(obs[0], action[0], rewards[0], next_obs[0], float(dones[0] or truncateds[0]))

            obs = next_obs

            with torch.profiler.record_function("AGENT_UPDATE"):
                if step >= WARMUP_STEPS and len(buffer) >= BATCH_SIZE:
                    agent.update(buffer, BATCH_SIZE)

            # Auto-reset
            if dones[0] or truncateds[0]:
                obs, _ = env.reset()

            # Signal profiler
            prof.step()

            if (step + 1) % 100 == 0:
                print(f"Step {step + 1}/{NUM_STEPS}")

    print()
    print("=" * 60)
    print("Profiling complete!")
    print("=" * 60)
    print()
    print("Key Averages (CPU time):")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))
    print()
    print("Key Averages (CUDA time):")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    print()
    print(f"View detailed trace: tensorboard --logdir {log_dir}")


if __name__ == "__main__":
    main()
