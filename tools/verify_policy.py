# verify_policy.py
import numpy as np
from onnxruntime import InferenceSession
import pickle
import os

def verify_onnx_policy():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    policy_dir = os.path.join(os.path.dirname(script_dir), "policy")
    ONNX_PATH = os.path.join(policy_dir, "harold_policy.onnx")
    CONFIG_PATH = os.path.join(policy_dir, "action_config.pt")
    
    print("\n=== Verifying ONNX Policy ===")
    # Create inference session
    ort_session = InferenceSession(ONNX_PATH)
    
    # Create dummy input (50 dimensions)
    dummy_input = np.random.randn(1, 50).astype(np.float32)
    
    # Run inference
    outputs = ort_session.run(None, {'obs': dummy_input})
    
    print("Input shape:", dummy_input.shape)
    print("Output shape:", outputs[0].shape)
    print("Sample output:", outputs[0][0])
    print("=== ONNX Policy Verification Successful ===")
    
    # Load and verify action config
    print("\n=== Verifying Action Config ===")
    try:
        with open(CONFIG_PATH, 'rb') as f:
            config = pickle.load(f)
        print("\nAction configuration loaded successfully!")
        print(f"Action scale: {config['action_scale']}")
        print("\nDefault joint positions:")
        for joint, pos in config['default_joint_pos'].items():
            print(f"{joint}: {pos:.4f}")

        print("\n=== Action Config Verification Successful ===")
    except Exception as e:
        print(f"Error loading action config: {e}")

if __name__ == "__main__":
    verify_onnx_policy()
