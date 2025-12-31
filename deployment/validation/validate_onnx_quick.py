#!/usr/bin/env python3
"""Quick ONNX validation without running full Isaac Lab simulation.

This script validates the ONNX export by:
1. Loading the PyTorch checkpoint
2. Generating test observations using training statistics
3. Running both PyTorch and ONNX inference
4. Comparing outputs

This is much faster than running a full simulation (seconds vs 8+ minutes).

Usage:
    python validate_onnx_quick.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Constants
OBS_DIM = 50
ACTION_DIM = 12
CHECKPOINT_PATH = Path(__file__).parent.parent.parent / "logs/skrl/harold_direct/2025-12-30_07-30-54_ppo_torch/checkpoints/best_agent.pt"
ONNX_PATH = Path(__file__).parent.parent / "policy/harold_policy.onnx"


def load_pytorch_model(checkpoint_path: Path):
    """Load PyTorch model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract running stats
    running_mean = checkpoint["state_preprocessor"]["running_mean"].numpy()
    running_var = checkpoint["state_preprocessor"]["running_variance"].numpy()

    # Extract policy weights
    policy_state = checkpoint["policy"]

    print(f"Loaded checkpoint: {checkpoint_path.name}")
    print(f"  Running mean shape: {running_mean.shape}")
    print(f"  Running var shape: {running_var.shape}")
    print(f"  Policy keys: {list(policy_state.keys())}")

    return checkpoint, running_mean, running_var


def create_pytorch_policy(checkpoint):
    """Recreate the policy network from checkpoint.

    skrl uses this structure:
    - net_container.0/2/4: Hidden layers (512 -> 256 -> 128)
    - policy_layer: Output layer (128 -> 12)
    - log_std_parameter: Log std for stochastic actions (not used for mean)
    """
    import torch.nn as nn

    # Match skrl's GaussianMixin policy structure
    class PolicyNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            # Hidden layers: net_container is a Sequential
            self.net_container = nn.Sequential(
                nn.Linear(50, 512),
                nn.ELU(),
                nn.Linear(512, 256),
                nn.ELU(),
                nn.Linear(256, 128),
                nn.ELU(),
            )
            # Output layer
            self.policy_layer = nn.Linear(128, 12)

        def forward(self, x):
            features = self.net_container(x)
            return self.policy_layer(features)

    # Create model
    model = PolicyNetwork()

    # Extract only the policy weights (not log_std or value)
    policy_state = checkpoint["policy"]
    model_state = {}

    for k, v in policy_state.items():
        if k.startswith("net_container.") or k.startswith("policy_layer."):
            model_state[k] = v

    # Load weights
    try:
        model.load_state_dict(model_state, strict=True)
        print("  PyTorch model weights loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load weights: {e}")
        print(f"  Checkpoint keys: {list(policy_state.keys())}")
        print(f"  Model keys: {list(model.state_dict().keys())}")
        model.load_state_dict(model_state, strict=False)

    model.eval()
    return model


def normalize_observation(obs: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Apply running stats normalization."""
    eps = 1e-8
    normalized = (obs - mean) / np.sqrt(var + eps)
    return np.clip(normalized, -5.0, 5.0)


def generate_test_observations(mean: np.ndarray, var: np.ndarray, n_samples: int = 10) -> np.ndarray:
    """Generate test observations around the training distribution."""
    np.random.seed(42)  # For reproducibility

    observations = []

    # 1. Use the training mean directly (normalized = 0)
    observations.append(mean.copy())

    # 2. Small perturbations from mean
    std = np.sqrt(var + 1e-8)
    for _ in range(n_samples - 1):
        # Random perturbation within 2 std
        perturbation = np.random.randn(OBS_DIM) * std * 0.5
        obs = mean + perturbation
        observations.append(obs)

    return np.array(observations, dtype=np.float32)


def main():
    print("=" * 70)
    print("ONNX Quick Validation")
    print("=" * 70)

    # Check files exist
    if not CHECKPOINT_PATH.exists():
        print(f"ERROR: Checkpoint not found: {CHECKPOINT_PATH}")
        sys.exit(1)

    if not ONNX_PATH.exists():
        print(f"ERROR: ONNX model not found: {ONNX_PATH}")
        sys.exit(1)

    # Load PyTorch model
    print("\n1. Loading PyTorch checkpoint...")
    checkpoint, running_mean, running_var = load_pytorch_model(CHECKPOINT_PATH)

    # Try to create and run PyTorch model
    print("\n2. Creating PyTorch policy...")
    try:
        pytorch_model = create_pytorch_policy(checkpoint)
        print("  PyTorch model created successfully")
    except Exception as e:
        print(f"  WARNING: Could not create PyTorch model: {e}")
        print("  Will skip PyTorch comparison, only validate ONNX sanity")
        pytorch_model = None

    # Load ONNX model
    print("\n3. Loading ONNX model...")
    try:
        import onnxruntime as ort
        onnx_session = ort.InferenceSession(str(ONNX_PATH), providers=['CPUExecutionProvider'])
        print(f"  ONNX model loaded: {ONNX_PATH.name}")
        print(f"  Input: {onnx_session.get_inputs()[0].name} {onnx_session.get_inputs()[0].shape}")
        print(f"  Output: {onnx_session.get_outputs()[0].name} {onnx_session.get_outputs()[0].shape}")
    except ImportError:
        print("  ERROR: onnxruntime not installed")
        sys.exit(1)

    # Generate test observations
    print("\n4. Generating test observations...")
    raw_observations = generate_test_observations(running_mean, running_var, n_samples=20)
    print(f"  Generated {len(raw_observations)} test observations")

    # Normalize observations
    normalized_observations = np.array([
        normalize_observation(obs, running_mean, running_var)
        for obs in raw_observations
    ])

    # IMPORTANT: ONNX model includes normalization internally!
    # It expects RAW observations, not normalized.
    # See policy/export_policy.py: NormalizedPolicy wrapper applies normalization.
    print("\n5. Running ONNX inference (with RAW observations)...")
    onnx_actions = []
    for obs_raw in raw_observations:
        # ONNX expects raw observations - it normalizes internally
        obs_f32 = obs_raw.astype(np.float32).reshape(1, -1)
        outputs = onnx_session.run(['mean'], {'obs': obs_f32})
        onnx_actions.append(outputs[0][0])
    onnx_actions = np.array(onnx_actions)

    print(f"  ONNX output shape: {onnx_actions.shape}")
    print(f"  ONNX output range: [{onnx_actions.min():.4f}, {onnx_actions.max():.4f}]")

    # Run PyTorch inference if available
    # PyTorch model expects NORMALIZED observations (no normalization wrapper)
    if pytorch_model is not None:
        print("\n6. Running PyTorch inference (with NORMALIZED observations)...")
        with torch.no_grad():
            obs_tensor = torch.from_numpy(normalized_observations.astype(np.float32))
            pytorch_actions = pytorch_model(obs_tensor).numpy()

        print(f"  PyTorch output shape: {pytorch_actions.shape}")
        print(f"  PyTorch output range: [{pytorch_actions.min():.4f}, {pytorch_actions.max():.4f}]")

        # Compare
        print("\n7. Comparing outputs...")
        diff = np.abs(onnx_actions - pytorch_actions)
        print(f"  Max difference: {diff.max():.6f}")
        print(f"  Mean difference: {diff.mean():.6f}")

        if diff.max() < 1e-4:
            print("\n✓ PASS: ONNX outputs match PyTorch outputs!")
        elif diff.max() < 1e-2:
            print("\n⚠ WARNING: Small differences detected (may be floating point)")
        else:
            print("\n✗ FAIL: Significant differences detected!")
            print("\nPer-sample comparison:")
            for i in range(min(5, len(onnx_actions))):
                print(f"  Sample {i}: max_diff={np.abs(onnx_actions[i] - pytorch_actions[i]).max():.6f}")

    # Sanity check ONNX outputs
    print("\n" + "=" * 70)
    print("ONNX Sanity Check")
    print("=" * 70)

    print("\nSample ONNX outputs (first 5 samples):")
    for i in range(min(5, len(onnx_actions))):
        print(f"  Sample {i}: {onnx_actions[i][:4]}... (first 4 joints)")

    # Check for NaN/Inf
    if np.any(np.isnan(onnx_actions)):
        print("\n✗ ERROR: ONNX outputs contain NaN!")
    elif np.any(np.isinf(onnx_actions)):
        print("\n✗ ERROR: ONNX outputs contain Inf!")
    elif np.abs(onnx_actions).max() > 10:
        print(f"\n⚠ WARNING: Large ONNX outputs detected (max={np.abs(onnx_actions).max():.2f})")
    else:
        print("\n✓ ONNX outputs look reasonable")

    # Export test data for RPi validation
    print("\n" + "=" * 70)
    print("Exporting Test Data for RPi")
    print("=" * 70)

    import json

    test_data = {
        "metadata": {
            "checkpoint": str(CHECKPOINT_PATH),
            "num_samples": len(raw_observations),
            "obs_dim": OBS_DIM,
            "action_dim": ACTION_DIM,
            "running_mean": running_mean.tolist(),
            "running_var": running_var.tolist(),
        },
        "timesteps": [
            {
                "step": i,
                "obs_raw": raw_observations[i].tolist(),
                "obs_normalized": normalized_observations[i].tolist(),
                "action": onnx_actions[i].tolist(),
            }
            for i in range(len(raw_observations))
        ]
    }

    output_path = Path(__file__).parent / "sim_episode.json"
    with open(output_path, "w") as f:
        json.dump(test_data, f, indent=2)

    print(f"Saved test data to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")

    print("\nTo validate on RPi:")
    print(f"  scp {output_path} pi@harold.local:~/Desktop/harold/deployment/validation/")
    print("  python validation/validate_onnx_vs_sim.py")


if __name__ == "__main__":
    main()
