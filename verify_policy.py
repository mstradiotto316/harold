# verify_policy.py
import numpy as np
from onnxruntime import InferenceSession
import pickle
import os

def verify_onnx_policy():
    # Paths
    ONNX_PATH = "/home/matteo/Desktop/Harold_V5/harold_policy.onnx"
    CONFIG_PATH = os.path.join(os.path.dirname(ONNX_PATH), "action_config.pt")
    
    print("\n=== Verifying ONNX Policy ===")
    # Create inference session
    ort_session = InferenceSession(ONNX_PATH)
    
    # Create dummy input (38 dimensions)
    dummy_input = np.random.randn(1, 38).astype(np.float32)
    
    # Run inference
    outputs = ort_session.run(None, {'obs': dummy_input})
    
    print("Input shape:", dummy_input.shape)
    print("Output shape:", outputs[0].shape)
    print("Sample output:", outputs[0][0][:3])
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
