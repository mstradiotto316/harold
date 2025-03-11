import torch
import sys

# Load the current action_config.pt file
config_path = "../policy/action_config.pt"
action_config = torch.load(config_path)

# Display current joint positions
print("Current default joint positions:")
for joint, pos in action_config['default_joint_pos'].items():
    print(f"{joint}: {pos}")

# Modify joint positions based on command line arguments
if len(sys.argv) >= 3:
    joint_name = sys.argv[1]
    new_value = float(sys.argv[2])
    
    if joint_name in action_config['default_joint_pos']:
        action_config['default_joint_pos'][joint_name] = new_value
        print(f"\nUpdated {joint_name} to {new_value}")
        
        # Save the modified config
        torch.save(action_config, config_path)
        print(f"Saved updated configuration to {config_path}")
    else:
        print(f"Error: Joint name '{joint_name}' not found")
        print("Available joints:", list(action_config['default_joint_pos'].keys()))
else:
    print("\nUsage: python edit_action_config.py <joint_name> <new_value>")
    print("Example: python edit_action_config.py fl_shoulder_joint 0.5")
    print("Available joints:", list(action_config['default_joint_pos'].keys()))