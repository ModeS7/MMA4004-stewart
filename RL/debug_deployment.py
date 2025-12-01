"""
Debug script to compare training vs deployment network behavior.
Run this to identify mismatches between training and deployment.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

# Load training network
from RL.networks import ActorCNN

# Load deployment network (extract the class)
def get_deployment_actor():
    """Recreate the deployment Actor class from control_core.py (with double tanh fix)"""
    import torch.nn as nn

    class Actor(nn.Module):
        def __init__(self, input_dim=84, action_dim=2, hidden_dim=256, physics_dim=3,
                     num_frames=12, obs_per_frame=7):
            super().__init__()
            self.num_frames = num_frames
            self.obs_per_frame = obs_per_frame

            self.conv = nn.Sequential(
                nn.Conv1d(obs_per_frame, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(64, 64, kernel_size=3, stride=2),
                nn.ReLU(),
            )

            cnn_out_dim = 64 * 5

            self.fc = nn.Sequential(
                nn.Linear(cnn_out_dim, hidden_dim),
                nn.ReLU(),
            )

            self.mean = nn.Linear(hidden_dim, action_dim)
            self.log_std = nn.Linear(hidden_dim, action_dim)

            self.physics_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, physics_dim),
                nn.Sigmoid()
            )

        def forward(self, obs):
            batch_size = obs.shape[0]
            x = obs.view(batch_size, self.num_frames, self.obs_per_frame)
            x = x.permute(0, 2, 1)
            x = self.conv(x)
            x = x.flatten(1)
            features = self.fc(x)
            # Single tanh - matches get_deterministic() in training
            action_mean = torch.tanh(self.mean(features))
            return action_mean

    return Actor


def main():
    # Find checkpoint
    checkpoint_path = 'RL/checkpoints/sac_final.pt'
    if not os.path.exists(checkpoint_path):
        import glob
        checkpoints = glob.glob('RL/checkpoints/*/sac_cnn_*.pt')
        if checkpoints:
            checkpoint_path = max(checkpoints, key=os.path.getmtime)

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Print checkpoint info
    print("\n=== CHECKPOINT INFO ===")
    print(f"Keys: {list(checkpoint.keys())}")
    if 'architecture' in checkpoint:
        print(f"Architecture: {checkpoint['architecture']}")

    actor_state = checkpoint['actor']
    print(f"\nActor state_dict keys:")
    for key, value in actor_state.items():
        print(f"  {key}: {value.shape}")

    # Check if checkpoint has physics_head
    has_physics_head = 'physics_head.0.weight' in actor_state
    print(f"\nCheckpoint has physics_head: {has_physics_head}")

    # Create training network
    print("\n=== TRAINING NETWORK (ActorCNN) ===")
    training_actor = ActorCNN(
        state_dim=84, action_dim=2, hidden_dim=256,
        num_frames=12, obs_per_frame=7, physics_dim=3,
        use_physics_head=has_physics_head
    )

    print("Training network keys:")
    for key, value in training_actor.state_dict().items():
        print(f"  {key}: {value.shape}")

    # Check for key mismatches
    training_keys = set(training_actor.state_dict().keys())
    checkpoint_keys = set(actor_state.keys())

    missing_in_checkpoint = training_keys - checkpoint_keys
    extra_in_checkpoint = checkpoint_keys - training_keys

    if missing_in_checkpoint:
        print(f"\n⚠️  Keys in training network but NOT in checkpoint:")
        for k in missing_in_checkpoint:
            print(f"    {k}")

    if extra_in_checkpoint:
        print(f"\n⚠️  Keys in checkpoint but NOT in training network:")
        for k in extra_in_checkpoint:
            print(f"    {k}")

    # Load weights into training network
    training_actor.load_state_dict(actor_state, strict=False)
    training_actor.eval()

    # Create deployment network
    print("\n=== DEPLOYMENT NETWORK ===")
    DeploymentActor = get_deployment_actor()
    deployment_actor = DeploymentActor(input_dim=84, action_dim=2, hidden_dim=256)

    print("Deployment network keys:")
    for key, value in deployment_actor.state_dict().items():
        print(f"  {key}: {value.shape}")

    # Load weights into deployment network
    deployment_actor.load_state_dict(actor_state, strict=False)
    deployment_actor.eval()

    # Compare weights
    print("\n=== WEIGHT COMPARISON ===")
    all_match = True
    for key in actor_state.keys():
        if key in training_actor.state_dict() and key in deployment_actor.state_dict():
            train_w = training_actor.state_dict()[key]
            deploy_w = deployment_actor.state_dict()[key]
            if torch.allclose(train_w, deploy_w):
                print(f"  ✓ {key}: MATCH")
            else:
                print(f"  ✗ {key}: MISMATCH!")
                all_match = False
        else:
            print(f"  ? {key}: key missing in one network")

    # Test with sample input
    print("\n=== OUTPUT COMPARISON ===")
    # Create a realistic observation (12 frames x 7 features)
    # [ball_x, ball_y, platform_rx, platform_ry, dt, target_x, target_y]
    np.random.seed(42)
    obs = np.zeros((12, 7), dtype=np.float32)
    for i in range(12):
        obs[i] = [
            0.3,   # ball_x normalized (45mm / 150mm)
            0.2,   # ball_y normalized (30mm / 150mm)
            0.1 * i / 11,  # platform_rx gradually increasing
            0.05 * i / 11, # platform_ry gradually increasing
            1.0,   # dt normalized (10ms)
            0.0,   # target_x (center)
            0.0,   # target_y (center)
        ]

    obs_flat = obs.flatten()
    obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0)

    print(f"Input observation shape: {obs_tensor.shape}")
    print(f"Input sample (first frame): {obs[0]}")
    print(f"Input sample (last frame): {obs[11]}")

    with torch.no_grad():
        # Training network output (may return 2 or 3 values depending on physics_head)
        train_out = training_actor.forward(obs_tensor)
        if isinstance(train_out, tuple):
            train_action = train_out[0]  # action_mean (already has tanh)
        else:
            train_action = train_out

        # Deployment network output
        deploy_action = deployment_actor.forward(obs_tensor)

    print(f"\nTraining network output: {train_action.numpy()[0]}")
    print(f"Deployment network output: {deploy_action.numpy()[0]}")
    print(f"Difference: {(train_action - deploy_action).abs().numpy()[0]}")

    if torch.allclose(train_action, deploy_action, atol=1e-5):
        print("\n✓ Outputs MATCH - networks are equivalent")
    else:
        print("\n✗ Outputs DIFFER - there's a mismatch!")

    # Test with extreme inputs (what deployment might see)
    print("\n=== EDGE CASE TESTS ===")
    test_cases = [
        ("All zeros", np.zeros((12, 7), dtype=np.float32)),
        ("Ball at edge", np.tile([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], (12, 1)).astype(np.float32)),
        ("Platform tilted max", np.tile([0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0], (12, 1)).astype(np.float32)),
        ("High velocity (large pos change)", None),  # Will create dynamically
    ]

    # High velocity case
    high_vel_obs = np.zeros((12, 7), dtype=np.float32)
    for i in range(12):
        high_vel_obs[i] = [
            -0.5 + i * 0.1,  # ball moving from -0.5 to +0.6
            0.0,
            0.0, 0.0,
            1.0,
            0.0, 0.0
        ]
    test_cases[3] = ("High velocity (large pos change)", high_vel_obs)

    for name, obs in test_cases:
        obs_tensor = torch.FloatTensor(obs.flatten()).unsqueeze(0)
        with torch.no_grad():
            deploy_out = deployment_actor(obs_tensor).numpy()[0]
        print(f"  {name}: output = {deploy_out}")
        if abs(deploy_out[0]) > 0.95 or abs(deploy_out[1]) > 0.95:
            print(f"    ⚠️  Near saturation!")


if __name__ == "__main__":
    main()
