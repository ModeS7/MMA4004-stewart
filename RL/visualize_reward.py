"""
Visualize the RL Reward Function

Shows the multiplicative reward structure:
    reward = 1/(1 + dist/dist_scale) * 1/(1 + speed/speed_scale)

Where:
    - dist_scale = 30mm (30mm from center gives 0.5 distance factor)
    - speed_scale = 50mm/s (50mm/s gives 0.5 speed factor)
    - Fall penalty = -10
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def compute_reward(dist_mm, speed_mm_s, dist_scale=30.0, speed_scale=50.0):
    """Compute reward given distance and speed."""
    dist_factor = 1.0 / (1.0 + dist_mm / dist_scale)
    speed_factor = 1.0 / (1.0 + speed_mm_s / speed_scale)
    return dist_factor * speed_factor


def main():
    # Parameters (from env_gpu.py)
    dist_scale = 30.0   # mm
    speed_scale = 50.0  # mm/s
    platform_radius = 150.0  # mm
    fall_penalty = -10.0

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('RL Reward Function Visualization', fontsize=14, fontweight='bold')

    # =========================================================================
    # 1. 3D Surface Plot
    # =========================================================================
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')

    dist = np.linspace(0, platform_radius, 100)
    speed = np.linspace(0, 300, 100)  # mm/s
    D, S = np.meshgrid(dist, speed)
    R = compute_reward(D, S, dist_scale, speed_scale)

    surf = ax1.plot_surface(D, S, R, cmap='viridis', alpha=0.8, edgecolor='none')
    ax1.set_xlabel('Distance from center (mm)')
    ax1.set_ylabel('Speed (mm/s)')
    ax1.set_zlabel('Reward')
    ax1.set_title('3D Reward Surface')
    ax1.view_init(elev=25, azim=-60)
    fig.colorbar(surf, ax=ax1, shrink=0.5, label='Reward')

    # =========================================================================
    # 2. Contour Plot
    # =========================================================================
    ax2 = fig.add_subplot(2, 2, 2)

    contour = ax2.contourf(D, S, R, levels=20, cmap='viridis')
    ax2.contour(D, S, R, levels=[0.1, 0.25, 0.5, 0.75, 0.9], colors='white', linewidths=0.5)
    ax2.set_xlabel('Distance from center (mm)')
    ax2.set_ylabel('Speed (mm/s)')
    ax2.set_title('Reward Contours')
    fig.colorbar(contour, ax=ax2, label='Reward')

    # Mark key points
    ax2.axvline(x=dist_scale, color='red', linestyle='--', alpha=0.7, label=f'dist_scale={dist_scale}mm')
    ax2.axhline(y=speed_scale, color='orange', linestyle='--', alpha=0.7, label=f'speed_scale={speed_scale}mm/s')
    ax2.legend(loc='upper right')

    # =========================================================================
    # 3. Distance Factor (speed=0)
    # =========================================================================
    ax3 = fig.add_subplot(2, 2, 3)

    dist_factor = 1.0 / (1.0 + dist / dist_scale)
    ax3.plot(dist, dist_factor, 'b-', linewidth=2, label='Distance Factor')
    ax3.axvline(x=dist_scale, color='red', linestyle='--', alpha=0.7, label=f'dist_scale={dist_scale}mm (factor=0.5)')
    ax3.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)

    # Mark platform edge
    ax3.axvline(x=platform_radius, color='black', linestyle='-', alpha=0.5, label=f'Platform edge ({platform_radius}mm)')

    ax3.set_xlabel('Distance from center (mm)')
    ax3.set_ylabel('Distance Factor')
    ax3.set_title('Distance Factor: 1/(1 + dist/30)')
    ax3.set_xlim(0, platform_radius)
    ax3.set_ylim(0, 1.05)
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Add annotations
    for d in [0, 15, 30, 60, 90, 150]:
        f = 1.0 / (1.0 + d / dist_scale)
        ax3.plot(d, f, 'ro', markersize=6)
        ax3.annotate(f'{f:.2f}', (d, f), textcoords='offset points', xytext=(5, 5), fontsize=8)

    # =========================================================================
    # 4. Speed Factor (dist=0)
    # =========================================================================
    ax4 = fig.add_subplot(2, 2, 4)

    speed_factor = 1.0 / (1.0 + speed / speed_scale)
    ax4.plot(speed, speed_factor, 'g-', linewidth=2, label='Speed Factor')
    ax4.axvline(x=speed_scale, color='orange', linestyle='--', alpha=0.7, label=f'speed_scale={speed_scale}mm/s (factor=0.5)')
    ax4.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)

    ax4.set_xlabel('Speed (mm/s)')
    ax4.set_ylabel('Speed Factor')
    ax4.set_title('Speed Factor: 1/(1 + speed/50)')
    ax4.set_xlim(0, 300)
    ax4.set_ylim(0, 1.05)
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    # Add annotations
    for s in [0, 25, 50, 100, 150, 200]:
        f = 1.0 / (1.0 + s / speed_scale)
        ax4.plot(s, f, 'go', markersize=6)
        ax4.annotate(f'{f:.2f}', (s, f), textcoords='offset points', xytext=(5, 5), fontsize=8)

    plt.tight_layout()

    # =========================================================================
    # Second figure: Reward examples and fall penalty
    # =========================================================================
    fig2, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig2.suptitle('Reward Examples & Episode Returns', fontsize=14, fontweight='bold')

    # Example scenarios
    ax5 = axes[0]
    scenarios = [
        ('Centered, still', 0, 0),
        ('Centered, slow', 0, 25),
        ('Centered, fast', 0, 100),
        ('Near center, still', 15, 0),
        ('Near center, slow', 15, 25),
        ('Mid-range, moving', 50, 50),
        ('Edge, still', 100, 0),
        ('Edge, fast', 100, 150),
    ]

    names = [s[0] for s in scenarios]
    rewards = [compute_reward(s[1], s[2]) for s in scenarios]
    colors = plt.cm.viridis(np.array(rewards))

    bars = ax5.barh(names, rewards, color=colors)
    ax5.set_xlabel('Reward per step')
    ax5.set_title('Example Scenarios')
    ax5.set_xlim(0, 1)
    ax5.grid(True, alpha=0.3, axis='x')

    for bar, r in zip(bars, rewards):
        ax5.text(r + 0.02, bar.get_y() + bar.get_height()/2, f'{r:.3f}', va='center', fontsize=9)

    # Episode return simulation
    ax6 = axes[1]
    steps = np.arange(0, 1000)

    # Simulate different behaviors
    # Perfect: stays at center
    perfect_return = np.cumsum(np.ones(1000) * 1.0)

    # Good: average reward ~0.7
    good_return = np.cumsum(np.ones(1000) * 0.7)

    # Medium: average reward ~0.4
    medium_return = np.cumsum(np.ones(1000) * 0.4)

    # Poor: falls at step 200
    poor_rewards = np.ones(200) * 0.3
    poor_rewards = np.append(poor_rewards, [-10])  # Fall
    poor_return = np.cumsum(poor_rewards)
    poor_steps = np.arange(len(poor_return))

    ax6.plot(steps, perfect_return, 'g-', label='Perfect (r=1.0)', linewidth=2)
    ax6.plot(steps, good_return, 'b-', label='Good (r=0.7)', linewidth=2)
    ax6.plot(steps, medium_return, 'orange', label='Medium (r=0.4)', linewidth=2)
    ax6.plot(poor_steps, poor_return, 'r-', label='Falls at step 200', linewidth=2)

    ax6.set_xlabel('Step')
    ax6.set_ylabel('Cumulative Return')
    ax6.set_title('Episode Returns (1000 steps max)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Fall penalty visualization
    ax7 = axes[2]
    x = np.linspace(-1, 1, 100)
    normal_rewards = np.linspace(0, 1, 100)
    ax7.fill_between(x[:50], 0, 1, alpha=0.3, color='green', label='Normal reward range [0, 1]')
    ax7.axhline(y=fall_penalty, color='red', linewidth=3, label=f'Fall penalty = {fall_penalty}')
    ax7.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax7.axhline(y=1, color='gray', linestyle='--', alpha=0.5)

    ax7.set_ylim(-12, 2)
    ax7.set_xlim(-1, 1)
    ax7.set_ylabel('Reward')
    ax7.set_title('Reward Range')
    ax7.legend()
    ax7.set_xticks([])
    ax7.grid(True, alpha=0.3, axis='y')

    # Add text annotations
    ax7.text(0, 0.5, 'Normal\nreward\n[0, 1]', ha='center', va='center', fontsize=12, fontweight='bold')
    ax7.text(0, fall_penalty, f'FALL\n{fall_penalty}', ha='center', va='top', fontsize=12, fontweight='bold', color='red')

    plt.tight_layout()

    # =========================================================================
    # Print summary
    # =========================================================================
    print("=" * 60)
    print("REWARD FUNCTION SUMMARY")
    print("=" * 60)
    print()
    print("Formula:")
    print("  reward = dist_factor * speed_factor")
    print("  dist_factor = 1 / (1 + dist_mm / 30)")
    print("  speed_factor = 1 / (1 + speed_mm_s / 50)")
    print()
    print("Parameters:")
    print(f"  dist_scale = {dist_scale} mm")
    print(f"  speed_scale = {speed_scale} mm/s")
    print(f"  fall_penalty = {fall_penalty}")
    print()
    print("Reward range: [0, 1] (normal), -10 (fall)")
    print()
    print("Example rewards:")
    for name, d, s in scenarios:
        r = compute_reward(d, s)
        print(f"  {name:25s}: dist={d:3.0f}mm, speed={s:3.0f}mm/s -> reward={r:.3f}")
    print()
    print("=" * 60)

    plt.show()


if __name__ == "__main__":
    main()
