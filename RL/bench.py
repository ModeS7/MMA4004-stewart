"""
Benchmark inference speed for SAC model.

Tests the feedforward model with physics estimation.
"""

import time
import numpy as np
import torch

try:
    from .rl_config import EnvConfig
    from .agent import SACAgent
except ImportError:
    from rl_config import EnvConfig
    from agent import SACAgent


# ============================================================================
# BENCHMARK CONFIG
# ============================================================================

NUM_ITERATIONS = 1000    # Number of inference calls to average
WARMUP_ITERATIONS = 100  # Warmup iterations (not counted)
DEVICE = "cpu"           # "cpu" for laptop, "cuda" for desktop GPU

# ============================================================================


def benchmark():
    """Benchmark SAC agent with frame stacking."""
    print("\n--- SAC (12 frames + physics estimation) ---")

    env_cfg = EnvConfig()

    # Create agent
    agent = SACAgent(
        obs_dim=env_cfg.obs_dim,  # 12 * 5 = 60
        action_dim=env_cfg.action_dim,
        hidden_dim=256,
        physics_dim=env_cfg.physics_dim,
        device=DEVICE
    )
    agent.actor.eval()

    print(f"  Input dim: {env_cfg.obs_dim} ({env_cfg.num_frames} frames × {env_cfg.obs_per_frame} values)")
    print(f"  Action dim: {env_cfg.action_dim}")
    print(f"  Physics dim: {env_cfg.physics_dim}")

    # Count parameters
    total_params = sum(p.numel() for p in agent.actor.parameters())
    print(f"  Actor parameters: {total_params:,}")

    # Create dummy input (12 frames × 5 values per frame)
    obs = np.random.randn(env_cfg.obs_dim).astype(np.float32)

    # Warmup
    print(f"\nWarming up ({WARMUP_ITERATIONS} iterations)...")
    for _ in range(WARMUP_ITERATIONS):
        _ = agent.select_action(obs, evaluate=True)

    # Synchronize CUDA
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    # Benchmark single inference
    print(f"Benchmarking single inference ({NUM_ITERATIONS} iterations)...")
    start = time.perf_counter()

    for _ in range(NUM_ITERATIONS):
        _ = agent.select_action(obs, evaluate=True)

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    avg_time_ms = (elapsed / NUM_ITERATIONS) * 1000
    fps = NUM_ITERATIONS / elapsed

    return avg_time_ms, fps


def benchmark_batch(batch_size=10):
    """Benchmark batched inference."""
    print(f"\n--- Batched Inference (batch_size={batch_size}) ---")

    env_cfg = EnvConfig()

    agent = SACAgent(
        obs_dim=env_cfg.obs_dim,
        action_dim=env_cfg.action_dim,
        hidden_dim=256,
        physics_dim=env_cfg.physics_dim,
        device=DEVICE
    )
    agent.actor.eval()

    # Create batch input
    obs_batch = np.random.randn(batch_size, env_cfg.obs_dim).astype(np.float32)

    # Warmup
    for _ in range(WARMUP_ITERATIONS):
        _ = agent.select_action_batch(obs_batch, evaluate=True)

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()

    for _ in range(NUM_ITERATIONS):
        _ = agent.select_action_batch(obs_batch, evaluate=True)

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    avg_time_ms = (elapsed / NUM_ITERATIONS) * 1000
    per_sample_ms = avg_time_ms / batch_size
    fps = (NUM_ITERATIONS * batch_size) / elapsed

    return avg_time_ms, per_sample_ms, fps


def main():
    print("=" * 60)
    print("SAC Inference Benchmark")
    print("=" * 60)
    print(f"\nConfig:")
    print(f"  Device: {DEVICE}")
    print(f"  Iterations: {NUM_ITERATIONS}")

    if DEVICE == "cuda" and torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    elif DEVICE == "cuda":
        print("  WARNING: CUDA requested but not available, using CPU")

    # Run benchmarks
    single_time, single_fps = benchmark()
    batch_time, per_sample_time, batch_fps = benchmark_batch(batch_size=10)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nSingle inference:")
    print(f"  Time: {single_time:.3f} ms")
    print(f"  Throughput: {single_fps:.1f} Hz")

    print(f"\nBatched inference (10 envs):")
    print(f"  Total time: {batch_time:.3f} ms")
    print(f"  Per sample: {per_sample_time:.3f} ms")
    print(f"  Throughput: {batch_fps:.1f} Hz")

    # Check if fast enough for control
    print(f"\n100Hz control loop budget: 10.0 ms")
    print(f"  Single: {'OK' if single_time < 10 else 'TOO SLOW'} ({single_time:.2f} ms)")

    print(f"\n50Hz control loop budget: 20.0 ms")
    print(f"  Single: {'OK' if single_time < 20 else 'TOO SLOW'} ({single_time:.2f} ms)")


if __name__ == "__main__":
    main()
