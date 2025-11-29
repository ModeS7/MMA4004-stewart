"""
Interactive Training Dashboard using Gradio

Provides real-time control and monitoring of RL training.
"""

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
import threading

from dashboard_state import TrainingState, get_state


def create_reward_plot(state: TrainingState):
    """Create episode rewards plot."""
    metrics = state.get_metrics()
    rewards = metrics['episode_rewards']

    fig, ax = plt.subplots(figsize=(8, 4))

    if len(rewards) > 0:
        ax.plot(rewards, alpha=0.3, color='blue', label='Raw')

        # Smoothed line (rolling average)
        if len(rewards) >= 10:
            window = min(50, len(rewards) // 2)
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(rewards)), smoothed, color='blue', linewidth=2, label=f'Smoothed ({window})')

        # Eval rewards
        eval_rewards = metrics['eval_rewards']
        if len(eval_rewards) > 0:
            eval_episodes = np.linspace(0, len(rewards)-1, len(eval_rewards))
            ax.scatter(eval_episodes, eval_rewards, color='red', s=50, zorder=5, label='Eval')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title(f'Episode Rewards (Episode {metrics["current_episode"]})')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_length_plot(state: TrainingState):
    """Create episode lengths plot."""
    metrics = state.get_metrics()
    lengths = metrics['episode_lengths']

    fig, ax = plt.subplots(figsize=(8, 4))

    if len(lengths) > 0:
        ax.plot(lengths, color='green', alpha=0.7)
        ax.axhline(y=1000, color='red', linestyle='--', alpha=0.5, label='Max (1000)')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Length')
    ax.set_title('Episode Lengths')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def get_status_text(state: TrainingState):
    """Get current training status."""
    metrics = state.get_metrics()
    env_settings = state.get_env_settings()
    reward_settings = state.get_reward_settings()

    status = f"""Episode: {metrics['current_episode']}
Alpha: {metrics['current_alpha']:.4f}
Critic Loss: {metrics['critic_loss']:.4f}
Actor Loss: {metrics['actor_loss']:.4f}

Environment:
  Domain Rand: {'ON' if env_settings['use_domain_randomization'] else 'OFF'}
  Platform Tilt: {'ON' if env_settings['use_platform_offset'] else 'OFF'} (max {env_settings['platform_offset_max_deg']:.1f}°)
  Camera Noise: {'ON' if env_settings['use_camera_noise'] else 'OFF'}

Reward:
  Dist Scale: {reward_settings['dist_scale']:.0f}mm
  Speed Scale: {reward_settings['speed_scale']:.0f}mm/s
  Fall Penalty: {reward_settings['fall_penalty']:.0f}

Status: {'PAUSED' if state.paused else 'RUNNING'}"""

    return status


def create_dashboard(state: Optional[TrainingState] = None):
    """Create the Gradio dashboard interface."""
    if state is None:
        state = get_state()

    with gr.Blocks(title="RL Training Dashboard", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎮 Stewart Platform RL Training Dashboard")

        with gr.Row():
            # ===== Left Column: Controls =====
            with gr.Column(scale=1):
                gr.Markdown("## 🌍 Environment")

                domain_rand = gr.Checkbox(
                    label="Domain Randomization (physics)",
                    value=state.use_domain_randomization
                )
                platform_tilt = gr.Checkbox(
                    label="Platform Tilt Offset (gravity)",
                    value=state.use_platform_offset
                )
                camera_noise = gr.Checkbox(
                    label="Camera Noise",
                    value=state.use_camera_noise
                )

                offset_slider = gr.Slider(
                    minimum=0, maximum=10, value=state.platform_offset_max_deg,
                    step=0.5, label="Max Tilt Offset (degrees)"
                )

                gr.Markdown("## 🎯 Reward Function")

                dist_scale = gr.Slider(
                    minimum=10, maximum=100, value=state.dist_scale,
                    step=5, label="Distance Scale (mm) - lower = stricter"
                )
                speed_scale = gr.Slider(
                    minimum=10, maximum=200, value=state.speed_scale,
                    step=10, label="Speed Scale (mm/s) - lower = stricter"
                )
                fall_penalty = gr.Slider(
                    minimum=-100, maximum=-1, value=state.fall_penalty,
                    step=1, label="Fall Penalty"
                )

                gr.Markdown("## 🎛️ Training Control")

                with gr.Row():
                    pause_btn = gr.Button("⏸️ Pause/Resume", variant="secondary")
                    save_btn = gr.Button("💾 Save", variant="primary")

                stop_btn = gr.Button("🛑 Stop Training", variant="stop")

            # ===== Right Column: Plots =====
            with gr.Column(scale=2):
                reward_plot = gr.Plot(label="Episode Rewards")
                length_plot = gr.Plot(label="Episode Lengths")
                status_text = gr.Textbox(
                    label="Status",
                    value=get_status_text(state),
                    lines=15,
                    interactive=False
                )

        # ===== Event Handlers =====

        # Environment toggles
        def update_domain_rand(value):
            state.use_domain_randomization = value
            return f"Domain Randomization: {'ON' if value else 'OFF'}"

        def update_platform_tilt(value):
            state.use_platform_offset = value
            return f"Platform Tilt: {'ON' if value else 'OFF'}"

        def update_camera_noise(value):
            state.use_camera_noise = value
            return f"Camera Noise: {'ON' if value else 'OFF'}"

        def update_offset_max(value):
            state.platform_offset_max_deg = value

        # Reward sliders
        def update_dist_scale(value):
            state.dist_scale = value

        def update_speed_scale(value):
            state.speed_scale = value

        def update_fall_penalty(value):
            state.fall_penalty = value

        # Training control
        def toggle_pause():
            paused = state.toggle_pause()
            return "⏸️ PAUSED" if paused else "▶️ RUNNING"

        def request_save():
            state.request_save()
            return "💾 Save requested..."

        def request_stop():
            state.request_stop()
            return "🛑 Stopping after current episode..."

        # Connect handlers
        domain_rand.change(update_domain_rand, inputs=domain_rand)
        platform_tilt.change(update_platform_tilt, inputs=platform_tilt)
        camera_noise.change(update_camera_noise, inputs=camera_noise)
        offset_slider.change(update_offset_max, inputs=offset_slider)
        dist_scale.change(update_dist_scale, inputs=dist_scale)
        speed_scale.change(update_speed_scale, inputs=speed_scale)
        fall_penalty.change(update_fall_penalty, inputs=fall_penalty)

        pause_btn.click(toggle_pause, outputs=pause_btn)
        save_btn.click(request_save, outputs=save_btn)
        stop_btn.click(request_stop, outputs=stop_btn)

        # ===== Auto-refresh Timer =====
        def refresh_plots():
            return (
                create_reward_plot(state),
                create_length_plot(state),
                get_status_text(state)
            )

        timer = gr.Timer(2)  # Refresh every 2 seconds
        timer.tick(refresh_plots, outputs=[reward_plot, length_plot, status_text])

    return demo


def launch_dashboard(state: Optional[TrainingState] = None, port: int = 7860):
    """Launch the dashboard in a background thread."""
    if state is None:
        state = get_state()

    def run():
        demo = create_dashboard(state)
        demo.launch(
            server_port=port,
            share=False,
            prevent_thread_lock=True,
            show_error=True,
            quiet=True
        )

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    print(f"\n🎮 Dashboard running at http://localhost:{port}\n")
    return thread


# Test
if __name__ == "__main__":
    import time

    state = get_state()

    # Simulate some training data
    for i in range(50):
        state.add_episode(
            reward=100 + i * 2 + np.random.randn() * 20,
            length=500 + i * 5 + np.random.randn() * 50
        )
        if i % 10 == 0:
            state.add_eval(150 + i * 3)

    state.update_losses(critic=0.5, actor=0.3, alpha=0.15)

    # Launch dashboard
    demo = create_dashboard(state)
    demo.launch(server_port=7860)
