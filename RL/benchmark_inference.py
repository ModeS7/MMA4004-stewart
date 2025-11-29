"""
Benchmark inference speed for RL models.

Compares feedforward vs LSTM inference times.
"""

import time
import numpy as np
import torch

from rl_config import EnvConfig, SACConfig
from sac_agent import SACAgent
from sac_lstm_agent import LSTMSACAgent


# ============================================================================
# BENCHMARK CONFIG
# ============================================================================

NUM_ITERATIONS = 1000   # Number of inference calls to average
WARMUP_ITERATIONS = 100  # Warmup iterations (not counted)
BATCH_SIZE = 1          # Batch size for inference (1 = single env)
DEVICE = "cuda"         # "cuda" or "cpu"

# ============================================================================


def benchmark_feedforward():
    """Benchmark feedforward SAC agent."""
    print("\n--- Feedforward SAC ---")

    # Create agent
    agent = SACAgent(
        state_dim=6,  # Original: [x, y, vx, vy, rx, ry]
        action_dim=2,
        hidden_dim=256,
        device=DEVICE
    )
    agent.actor.eval()

    # Create dummy input
    state = np.random.randn(6).astype(np.float32)

    # Warmup
    print(f"Warming up ({WARMUP_ITERATIONS} iterations)...")
    for _ in range(WARMUP_ITERATIONS):
        _ = agent.select_action(state, evaluate=True)

    # Synchronize CUDA
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    print(f"Benchmarking ({NUM_ITERATIONS} iterations)...")
    start = time.perf_counter()

    for _ in range(NUM_ITERATIONS):
        _ = agent.select_action(state, evaluate=True)

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    avg_time_ms = (elapsed / NUM_ITERATIONS) * 1000
    fps = NUM_ITERATIONS / elapsed

    print(f"  Average inference time: {avg_time_ms:.3f} ms")
    print(f"  Throughput: {fps:.1f} Hz")

    return avg_time_ms, fps


def benchmark_lstm():
    """Benchmark LSTM SAC agent."""
    print("\n--- LSTM SAC ---")

    env_cfg = EnvConfig()
    sac_cfg = SACConfig()

    # Create agent
    agent = LSTMSACAgent(
        obs_dim=env_cfg.obs_per_step,
        action_dim=env_cfg.action_dim,
        seq_length=env_cfg.seq_length,
        lstm_hidden_dim=sac_cfg.lstm_hidden_dim,
        lstm_layers=sac_cfg.lstm_layers,
        hidden_dim=sac_cfg.hidden_dim,
        device=DEVICE
    )
    agent.actor.eval()

    # Create dummy input sequence
    obs_seq = np.random.randn(env_cfg.seq_length, env_cfg.obs_per_step).astype(np.float32)

    # Warmup
    print(f"Warming up ({WARMUP_ITERATIONS} iterations)...")
    for _ in range(WARMUP_ITERATIONS):
        _ = agent.select_action(obs_seq, evaluate=True)

    # Synchronize CUDA
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    print(f"Benchmarking ({NUM_ITERATIONS} iterations)...")
    start = time.perf_counter()

    for _ in range(NUM_ITERATIONS):
        _ = agent.select_action(obs_seq, evaluate=True)

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    avg_time_ms = (elapsed / NUM_ITERATIONS) * 1000
    fps = NUM_ITERATIONS / elapsed

    print(f"  Average inference time: {avg_time_ms:.3f} ms")
    print(f"  Throughput: {fps:.1f} Hz")
    print(f"  Sequence length: {env_cfg.seq_length}")
    print(f"  LSTM hidden dim: {sac_cfg.lstm_hidden_dim}")

    return avg_time_ms, fps


def benchmark_lstm_compiled():
    """Benchmark LSTM SAC agent with torch.compile()."""
    print("\n--- LSTM SAC (torch.compile) ---")

    if not hasattr(torch, 'compile'):
        print("  torch.compile not available (requires PyTorch 2.0+)")
        return None, None

    env_cfg = EnvConfig()
    sac_cfg = SACConfig()

    # Create agent
    agent = LSTMSACAgent(
        obs_dim=env_cfg.obs_per_step,
        action_dim=env_cfg.action_dim,
        seq_length=env_cfg.seq_length,
        lstm_hidden_dim=sac_cfg.lstm_hidden_dim,
        lstm_layers=sac_cfg.lstm_layers,
        hidden_dim=sac_cfg.hidden_dim,
        device=DEVICE
    )
    agent.actor.eval()

    # Compile the actor
    print("  Compiling model...")
    try:
        agent.actor = torch.compile(agent.actor, mode="reduce-overhead")
    except Exception as e:
        print(f"  Compilation failed: {e}")
        return None, None

    # Create dummy input sequence
    obs_seq = np.random.randn(env_cfg.seq_length, env_cfg.obs_per_step).astype(np.float32)

    # Warmup (longer for compiled model)
    print(f"Warming up ({WARMUP_ITERATIONS * 2} iterations)...")
    for _ in range(WARMUP_ITERATIONS * 2):
        _ = agent.select_action(obs_seq, evaluate=True)

    # Synchronize CUDA
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    print(f"Benchmarking ({NUM_ITERATIONS} iterations)...")
    start = time.perf_counter()

    for _ in range(NUM_ITERATIONS):
        _ = agent.select_action(obs_seq, evaluate=True)

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    avg_time_ms = (elapsed / NUM_ITERATIONS) * 1000
    fps = NUM_ITERATIONS / elapsed

    print(f"  Average inference time: {avg_time_ms:.3f} ms")
    print(f"  Throughput: {fps:.1f} Hz")

    return avg_time_ms, fps


def main():
    print("=" * 60)
    print("RL Model Inference Benchmark")
    print("=" * 60)
    print(f"\nConfig:")
    print(f"  Device: {DEVICE}")
    print(f"  Iterations: {NUM_ITERATIONS}")
    print(f"  Batch size: {BATCH_SIZE}")

    if DEVICE == "cuda" and torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Run benchmarks
    ff_time, ff_fps = benchmark_feedforward()
    lstm_time, lstm_fps = benchmark_lstm()
    lstm_comp_time, lstm_comp_fps = benchmark_lstm_compiled()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':<25} {'Time (ms)':<12} {'Throughput (Hz)':<15}")
    print("-" * 52)
    print(f"{'Feedforward':<25} {ff_time:<12.3f} {ff_fps:<15.1f}")
    print(f"{'LSTM':<25} {lstm_time:<12.3f} {lstm_fps:<15.1f}")
    if lstm_comp_time:
        print(f"{'LSTM (compiled)':<25} {lstm_comp_time:<12.3f} {lstm_comp_fps:<15.1f}")

    print(f"\nLSTM slowdown vs Feedforward: {lstm_time/ff_time:.1f}x")

    # Check if fast enough for 100Hz control
    print(f"\n100Hz control loop budget: 10.0 ms")
    print(f"  Feedforward: {'OK' if ff_time < 10 else 'TOO SLOW'} ({ff_time:.2f} ms)")
    print(f"  LSTM: {'OK' if lstm_time < 10 else 'TOO SLOW'} ({lstm_time:.2f} ms)")


if __name__ == "__main__":
    main()
