"""
Visualize the RL Reward Function

Matches the actual reward computation in env.py:
    base = 1 / (1 + dist_mm / dist_scale)
    alignment = vel_towards / speed  (cos of angle to target)
    optimal_vel = dist_mm * 2.0

    if vel_towards <= 0:
        approach = alignment * approach_scale  (penalty for moving away)
    elif speed <= optimal_vel:
        approach = (vel_towards / optimal_vel) * alignment * approach_scale
    else:
        approach = alignment * approach_scale * exp(-(speed-optimal)/optimal)

    reward = base + clip(approach, -0.5, 0.5)

Key features:
    - Optimal velocity = 2 * distance (decelerate as you approach)
    - Alignment factor penalizes perpendicular/orbiting motion
    - Moving away is penalized proportionally to how directly away
"""

import numpy as np
import matplotlib.pyplot as plt


def compute_base_reward(dist_mm, speed_mm_s=0.0, dist_scale=30.0, speed_scale=100.0):
    """Compute base reward based on distance and speed."""
    dist_factor = 1.0 / (1.0 + dist_mm / dist_scale)
    speed_factor = 1.0 / (1.0 + speed_mm_s / speed_scale)
    return dist_factor * speed_factor


def compute_center_bonus(dist_mm, speed_mm_s=0.0, radius=5.0, max_bonus=1.0, speed_scale=40.0):
    """
    Compute center bonus (extra reward for being very close to target AND slow).

    Distance factor: linear from 0 at radius to 1 at center
    Speed factor: linear cutoff - full bonus when still, zero above threshold
    """
    if isinstance(dist_mm, np.ndarray):
        bonus = np.zeros_like(dist_mm)
        mask = dist_mm < radius
        distance_factor = 1.0 - dist_mm[mask] / radius
        if isinstance(speed_mm_s, np.ndarray):
            speed_factor = np.maximum(0.0, 1.0 - speed_mm_s[mask] / speed_scale)
        else:
            speed_factor = max(0.0, 1.0 - speed_mm_s / speed_scale)
        bonus[mask] = max_bonus * distance_factor * speed_factor
        return bonus
    else:
        if dist_mm < radius:
            distance_factor = 1.0 - dist_mm / radius
            speed_factor = max(0.0, 1.0 - speed_mm_s / speed_scale)
            return max_bonus * distance_factor * speed_factor
        return 0.0


def compute_approach_reward(dist_mm, vel_towards, speed, approach_scale=0.5):
    """
    Approach reward with cosine alignment factor.

    vel_towards: component of velocity towards target (can be negative)
    speed: total speed (magnitude of velocity)

    alignment = cos(angle) = vel_towards / speed
    0° → 1.0, 90° → 0.0, 180° → -1.0
    """
    if dist_mm < 1e-6 or speed < 1e-6:
        return 0.0

    alignment = vel_towards / speed  # This IS cosine

    optimal_vel = dist_mm * 2.0

    if vel_towards <= 0:
        approach = alignment * approach_scale
    elif speed <= optimal_vel:
        approach = (vel_towards / optimal_vel) * alignment * approach_scale
    else:
        excess = (speed - optimal_vel) / optimal_vel
        approach = alignment * approach_scale * np.exp(-excess)

    return np.clip(approach, -0.5, 0.5)


def compute_total_reward(dist_mm, vel_towards, speed, dist_scale=30.0, base_speed_scale=100.0,
                         approach_scale=0.5, center_radius=5.0, center_bonus_max=1.0,
                         center_speed_scale=40.0):
    """Compute total reward."""
    base = compute_base_reward(dist_mm, speed, dist_scale, base_speed_scale)
    center = compute_center_bonus(dist_mm, speed, center_radius, center_bonus_max, center_speed_scale)
    approach = compute_approach_reward(dist_mm, vel_towards, speed, approach_scale)
    return base + center + approach


def main():
    # Parameters
    dist_scale = 30.0
    base_speed_scale = 100.0  # Base reward speed factor: 0.5 at 100mm/s
    center_radius = 5.0
    center_bonus_max = 1.0
    center_speed_scale = 40.0  # Linear cutoff: full bonus at 0mm/s, zero at 40mm/s
    approach_scale = 0.5
    platform_radius = 150.0
    fall_penalty = -10.0

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('RL Reward Function (Alignment-Based Approach)', fontsize=14, fontweight='bold')

    # =========================================================================
    # 1. Center Bonus vs Distance (at different speeds)
    # =========================================================================
    ax1 = axes[0, 0]

    dist = np.linspace(0, center_radius, 200)
    speeds = [0, 10, 20, 30, 40]  # Linear cutoff at 40mm/s
    colors = ['green', 'blue', 'orange', 'red', 'purple']

    for speed, color in zip(speeds, colors):
        bonus = compute_center_bonus(dist, speed, center_radius, center_bonus_max, center_speed_scale)
        ax1.plot(dist, bonus, color=color, linewidth=2, label=f'speed={speed}mm/s')

    ax1.axvline(x=center_radius, color='gray', linestyle='--', alpha=0.5)
    ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)

    ax1.set_xlabel('Distance from center (mm)')
    ax1.set_ylabel('Center Bonus')
    ax1.set_title(f'Center Bonus (distance × speed factor)')
    ax1.set_xlim(0, center_radius)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    # =========================================================================
    # 2. Alignment Factor: Linear vs Smoothed
    # =========================================================================
    ax2 = axes[0, 1]

    angles = np.linspace(0, 180, 200)

    # Linear alignment
    alignment_linear = np.cos(np.radians(angles))
    # Smoothed alignment
    alignment_smooth = alignment_linear * (1.0 + alignment_linear) / 2.0

    ax2.plot(angles, alignment_linear, 'b-', linewidth=2, label='Linear: cos(θ)', alpha=0.7)
    ax2.plot(angles, alignment_smooth, 'g-', linewidth=2.5, label='Smoothed: cos(θ)×(1+cos(θ))/2')
    ax2.axvline(x=90, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    ax2.set_xlabel('Angle from target direction (degrees)')
    ax2.set_ylabel('Alignment Factor')
    ax2.set_title('Alignment: Linear vs Smoothed')
    ax2.set_xlim(0, 180)
    ax2.set_ylim(-0.6, 1.1)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Annotate key points on smoothed curve
    for angle in [0, 45, 90, 135, 180]:
        align = np.cos(np.radians(angle)) * (1.0 + np.cos(np.radians(angle))) / 2.0
        ax2.plot(angle, align, 'ro', markersize=6)
        ax2.annotate(f'{align:+.2f}', (angle, align),
                    textcoords='offset points', xytext=(5, 5), fontsize=8)

    # =========================================================================
    # 3. Approach Reward vs Speed (direct approach, angle=0)
    # =========================================================================
    ax3 = axes[0, 2]

    speeds = np.linspace(0, 300, 200)
    distances = [30, 60, 90]
    colors = ['blue', 'green', 'orange']

    for d, c in zip(distances, colors):
        optimal = d * 2.0
        approaches = [compute_approach_reward(d, s, s, approach_scale) for s in speeds]
        ax3.plot(speeds, approaches, color=c, linewidth=2, label=f'd={d}mm (opt={optimal:.0f})')
        ax3.axvline(x=optimal, color=c, linestyle=':', alpha=0.5)

    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.axhline(y=0.5, color='black', linestyle=':', alpha=0.3)

    ax3.set_xlabel('Speed directly towards target (mm/s)')
    ax3.set_ylabel('Approach Reward')
    ax3.set_title('Approach vs Speed (moving directly, angle=0°)')
    ax3.set_xlim(0, 300)
    ax3.set_ylim(-0.1, 0.6)
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # =========================================================================
    # 4. Total Reward Heatmap (direct approach)
    # =========================================================================
    ax4 = axes[1, 0]

    dist_grid = np.linspace(1, platform_radius, 100)
    speed_grid = np.linspace(0, 300, 100)
    D, S = np.meshgrid(dist_grid, speed_grid)

    R = np.zeros_like(D)
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            # Direct approach (angle=0, so vel_towards = speed)
            R[i, j] = compute_total_reward(D[i, j], S[i, j], S[i, j], dist_scale, base_speed_scale, approach_scale)

    im = ax4.imshow(R, extent=[1, platform_radius, 0, 300], origin='lower',
                    aspect='auto', cmap='RdYlGn', vmin=0, vmax=1.5)
    ax4.contour(D, S, R, levels=[0.5, 0.75, 1.0, 1.25], colors='black', linewidths=0.5)

    # Plot optimal velocity line
    ax4.plot(dist_grid, dist_grid * 2.0, 'b--', linewidth=2, label='Optimal velocity')

    ax4.set_xlabel('Distance from center (mm)')
    ax4.set_ylabel('Speed towards target (mm/s)')
    ax4.set_title('Total Reward (direct approach, angle=0°)')
    ax4.legend(loc='upper left')
    fig.colorbar(im, ax=ax4, label='Reward')

    # =========================================================================
    # 5. Total Reward Heatmap (angled approach, 45°)
    # =========================================================================
    ax5 = axes[1, 1]

    angle = 45
    R = np.zeros_like(D)
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            vel_towards = S[i, j] * np.cos(np.radians(angle))
            R[i, j] = compute_total_reward(D[i, j], vel_towards, S[i, j], dist_scale, base_speed_scale, approach_scale)

    im = ax5.imshow(R, extent=[1, platform_radius, 0, 300], origin='lower',
                    aspect='auto', cmap='RdYlGn', vmin=0, vmax=1.5)
    ax5.contour(D, S, R, levels=[0.5, 0.75, 1.0], colors='black', linewidths=0.5)

    ax5.plot(dist_grid, dist_grid * 2.0, 'b--', linewidth=2, label='Optimal total speed')

    ax5.set_xlabel('Distance from center (mm)')
    ax5.set_ylabel('Total speed (mm/s)')
    ax5.set_title(f'Total Reward (angled approach, {angle}°)')
    ax5.legend(loc='upper left')
    fig.colorbar(im, ax=ax5, label='Reward')

    # =========================================================================
    # 6. Example Scenarios
    # =========================================================================
    ax6 = axes[1, 2]

    # (name, dist, vel_towards, speed)
    scenarios = [
        ('Centered, still', 0, 0, 0),
        ('Centered, 50mm/s', 0, 0, 50),
        ('2mm, still', 2, 0, 0),
        ('2mm, 20mm/s', 2, 0, 20),
        ('5mm, still', 5, 0, 0),
        ('10mm, direct 20mm/s', 10, 20, 20),
        ('30mm, still', 30, 0, 0),
        ('30mm, direct 60mm/s', 30, 60, 60),
    ]

    names = [s[0] for s in scenarios]
    base_rewards = [compute_base_reward(s[1], s[3], dist_scale, base_speed_scale) for s in scenarios]
    center_bonuses = [compute_center_bonus(s[1], s[3], center_radius, center_bonus_max, center_speed_scale) for s in scenarios]
    approach_rewards = [compute_approach_reward(s[1], s[2], s[3], approach_scale) for s in scenarios]
    total_rewards = [b + c + a for b, c, a in zip(base_rewards, center_bonuses, approach_rewards)]

    y_pos = np.arange(len(names))

    # Stack: base, then center bonus, then approach
    bars_base = ax6.barh(y_pos, base_rewards, color='steelblue', label='Base')
    bars_center = ax6.barh(y_pos, center_bonuses, left=base_rewards,
                           color='gold', alpha=0.8, label='Center')
    approach_left = [b + c for b, c in zip(base_rewards, center_bonuses)]
    approach_colors = ['green' if a >= 0 else 'red' for a in approach_rewards]
    bars_approach = ax6.barh(y_pos, approach_rewards, left=approach_left,
                             color=approach_colors, alpha=0.7, label='Approach')

    ax6.set_yticks(y_pos)
    ax6.set_yticklabels(names, fontsize=8)
    ax6.set_xlabel('Total Reward')
    ax6.set_title('Example Scenarios')
    ax6.axvline(x=0, color='black', linewidth=0.5)
    ax6.set_xlim(-0.5, 2.6)
    ax6.grid(True, alpha=0.3, axis='x')

    for i, total in enumerate(total_rewards):
        ax6.text(max(total, 0.05) + 0.05, i, f'{total:.2f}', va='center', fontsize=8)

    ax6.legend(loc='lower right', fontsize=8)

    plt.tight_layout()

    # =========================================================================
    # Print summary
    # =========================================================================
    print("=" * 75)
    print("REWARD FUNCTION SUMMARY (env.py)")
    print("=" * 75)
    print()
    print("Formula:")
    print(f"  base = 1/(1 + dist/{dist_scale}) × 1/(1 + speed/{base_speed_scale})")
    print(f"  center_bonus = {center_bonus_max} × (1-dist/{center_radius}) × max(0, 1-speed/{center_speed_scale})")
    print(f"                 if dist < {center_radius}mm (linear cutoff at {center_speed_scale}mm/s)")
    print(f"  alignment = vel_towards / speed  (cosine: 0°→1, 90°→0, 180°→-1)")
    print(f"  optimal_vel = dist_mm × 2.0")
    print(f"  approach:")
    print(f"    if vel_towards <= 0:    alignment × {approach_scale}")
    print(f"    if speed <= optimal:    (vel_towards/optimal) × alignment × {approach_scale}")
    print(f"    if speed > optimal:     alignment × {approach_scale} × exp(-(speed-opt)/opt)")
    print(f"  reward = base + center_bonus + clip(approach, -0.5, +0.5)")
    print()
    print("Key features:")
    print(f"  - Base reward: dist × speed factors (soft decay, never zero)")
    print(f"  - Center bonus: ZERO when speed >= {center_speed_scale}mm/s (hard cutoff)")
    print("  - Optimal velocity = 2 × distance (decelerate as you approach)")
    print("  - Alignment = cosine: 0°→+1, 90°→0, 180°→-1 (full penalty for retreat)")
    print()
    print("Example rewards:")
    print(f"  {'Scenario':<28s} {'Base':>6s} {'Center':>7s} {'Appr':>6s} {'Total':>7s}")
    print("-" * 62)
    for i, (name, d, vt, s) in enumerate(scenarios):
        base = base_rewards[i]
        center = center_bonuses[i]
        appr = approach_rewards[i]
        total = total_rewards[i]
        print(f"  {name:<28s} {base:>6.2f} {center:>7.2f} {appr:>+6.2f} {total:>7.2f}")
    print()
    print("=" * 75)

    plt.show()


if __name__ == "__main__":
    main()
