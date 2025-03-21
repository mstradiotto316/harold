#!/usr/bin/env python3
import serial
import time
import numpy as np
from onnxruntime import InferenceSession
import signal
import pickle
import os
import sys

# ===========================#
#   USER CONFIGURATIONS     #
# ===========================#

# Serial port configuration for Arduino
SERIAL_PORT = '/dev/ttyACM0'  # Default serial port, can be overridden via command line
BAUD_RATE = 115200
HANDSHAKE_MSG = "ARDUINO_READY"

# ONNX Policy Path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ONNX_PATH = os.path.join(SCRIPT_DIR, "harold_policy.onnx")
ACTION_CONFIG_PATH = os.path.join(SCRIPT_DIR, "action_config.pt")

# Action Reduction Factor (for safety - set to 1.0 for full policy output)
ACTION_REDUCTION_FACTOR = 1.0

# Control Loop Target Frequency (MATCH Isaac Lab Simulation - 20Hz)
CONTROL_FREQUENCY = 20.0
CONTROL_PERIOD = 1.0 / CONTROL_FREQUENCY

# ===========================#
#   GLOBAL OBJECTS/FLAGS    #
# ===========================#
ser = None
policy_session = None
action_scale = 2
default_positions = None
safe_positions_str = None  # Global safe command string

# ===========================#
#   FUNCTIONS & CALLBACKS   #
# ===========================#

def send_to_arduino(ser_obj, positions_line):
    """Send joint position commands to Arduino"""
    message_to_send = positions_line + '#'
    ser_obj.write(message_to_send.encode())
    ser_obj.flush()
    time.sleep(0.005)

def read_from_arduino(ser_obj):
    """Read messages from Arduino"""
    messages = []
    while ser_obj.in_waiting:
        message = ser_obj.readline().decode().strip()
        if message:
            messages.append(message)
    return messages

def wait_for_arduino_handshake(ser_obj):
    """Wait for Arduino to respond with handshake message"""
    startup_flag = False
    start_time = time.time()
    ser_obj.reset_input_buffer()
    ser_obj.write(b'READY?#')
    while not startup_flag and time.time() - start_time < 5:
        if b'ARDUINO_READY' in ser_obj.read_all():
            startup_flag = True
            return True
        time.sleep(0.1)
    return False

def process_policy_action(raw_actions):
    """Process policy actions into joint position commands"""
    global action_scale, default_positions
    scaled_actions = raw_actions * action_scale  # Apply scaling

    # Robot is wired to match simulation joint order directly
    reordered_positions = []
    for i in range(12):
        reordered_positions.append(default_positions[i] + scaled_actions[i])

    # Clip final positions to safe bounds ([-1.5, 1.5])
    clipped_positions = np.clip(reordered_positions, -1.5, 1.5)
    positions_line = ','.join(f'{pos:.4f}' for pos in clipped_positions)
    return positions_line

def load_policy_and_config():
    """Load the ONNX policy model and configuration file"""
    global policy_session, action_scale, default_positions, safe_positions_str
    
    print("Loading policy model from:", ONNX_PATH)
    policy_session = InferenceSession(ONNX_PATH)
    
    print("Loading action configuration from:", ACTION_CONFIG_PATH)
    with open(ACTION_CONFIG_PATH, 'rb') as f:
        config = pickle.load(f)
    
    action_scale = config['action_scale']
    
    # Get default joint positions from config
    default_positions = np.array([
        config['default_joint_pos'][name]
        for name in [
            'fl_shoulder_joint', 'fr_shoulder_joint', 'bl_shoulder_joint', 'br_shoulder_joint',
            'fl_thigh_joint', 'fr_thigh_joint', 'bl_thigh_joint', 'br_thigh_joint',
            'fl_knee_joint', 'fr_knee_joint', 'bl_knee_joint', 'br_knee_joint',
        ]
    ], dtype=np.float32)
    
    print("Policy and Action Config loaded successfully")
    
    # Compute the safe default command: zero raw action means default positions
    safe_actions = np.zeros(12, dtype=np.float32)
    safe_positions_str = process_policy_action(safe_actions)
    print("Safe default joint positions computed:")
    print(safe_positions_str)

def emergency_shutdown(sig, frame):
    """Handle emergency shutdown (Ctrl+C)"""
    global ser, safe_positions_str
    print("\nEmergency Shutdown Initiated!")
    if ser is not None and ser.is_open:
        print("Sending safe default positions and closing serial...")
        send_to_arduino(ser, safe_positions_str)
        time.sleep(0.1)
        ser.close()
    print("Exiting...")
    exit(0)

def create_observation():
    """Create an observation from IMU data"""
    try:
        # Import IMU reader
        sys.path.append(os.path.join(os.path.dirname(SCRIPT_DIR), "sensors"))
        from imu_reader import IMUReader
        
        # Initialize IMU reader if not already done
        if not hasattr(create_observation, 'imu_reader'):
            create_observation.imu_reader = IMUReader()
            print("IMU reader initialized")
        
        # Get IMU data
        imu_data = create_observation.imu_reader.get_filtered_data()
        
        # Format into observation vector
        # NOTE: This needs to be configured based on your specific policy's observation format
        obs = np.zeros(38, dtype=np.float32)
        
        # Add accelerometer data
        obs[0:3] = imu_data['accel']
        
        # Add gyroscope data
        obs[3:6] = imu_data['gyro']
        
        # Add orientation data (if available)
        if 'orientation' in imu_data:
            obs[6:8] = imu_data['orientation']
        
        # Reshape to match policy input format
        return obs.reshape(1, -1)
        
    except Exception as e:
        print(f"Warning: Error reading from IMU: {e}")
        print("Using zero observation vector as fallback")
        if not hasattr(create_observation, 'warned'):
            print("TIP: If testing without IMU, consider using robot_controller_observations_playback.py instead")
            create_observation.warned = True
        
        # Return zero observation as fallback
        return np.zeros(38, dtype=np.float32).reshape(1, -1)

def run_policy_step(observation):
    """Run a single step of the policy"""
    global policy_session, ACTION_REDUCTION_FACTOR
    
    # Get raw action from policy
    action = policy_session.run(None, {'obs': observation})[0][0]
    
    # Clip raw actions to [-1, 1]
    clipped_raw_actions = np.clip(action, -1.0, 1.0)
    
    # Apply action reduction factor
    reduced_actions = clipped_raw_actions * ACTION_REDUCTION_FACTOR
    
    # Process actions into joint positions
    positions_command_str = process_policy_action(reduced_actions)
    
    return positions_command_str

def main():
    """Main function"""
    global ser, policy_session, SERIAL_PORT
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Harold Robot Controller")
    parser.add_argument('--serial_port', type=str, help='Serial port for Arduino connection')
    args = parser.parse_args()
    
    # Override serial port if provided
    if args.serial_port:
        SERIAL_PORT = args.serial_port
    
    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, emergency_shutdown)
    
    print("=== Starting Harold Robot Controller ===")
    print(f"Serial port: {SERIAL_PORT}")
    print(f"Action reduction factor: {ACTION_REDUCTION_FACTOR}")
    
    # Load policy and configuration
    print("\nLoading policy and configuration...")
    load_policy_and_config()
    
    # Connect to Arduino
    print("\nOpening serial connection to Arduino...")
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print("Serial connection established")
    except Exception as e:
        print(f"Error opening serial port: {e}")
        exit(1)
    
    # Wait for Arduino handshake
    print("Waiting for Arduino handshake...")
    if not wait_for_arduino_handshake(ser):
        print("Error: Arduino handshake failed. Exiting.")
        if ser is not None and ser.is_open:
            ser.close()
        exit(1)
    print("Arduino handshake successful!")
    
    # Wait for servos to initialize
    print("Waiting for servo initialization...")
    time.sleep(4.0)
    
    # Main control loop
    print("\n=== Robot Control Started ===")
    try:
        while True:
            # Start timing
            start_time = time.time()
            
            # Get observation from sensors
            observation = create_observation()
            
            # Run policy to get joint positions
            positions_command_str = run_policy_step(observation)
            
            # Send command to Arduino
            send_to_arduino(ser, positions_command_str)
            
            # Maintain control frequency
            elapsed_time = time.time() - start_time
            control_rate_delay = max(0.0, CONTROL_PERIOD - elapsed_time)
            time.sleep(control_rate_delay)
            
    except Exception as e:
        print(f"\n=== ERROR in Main Loop: {e} ===")
        emergency_shutdown(None, None)
    finally:
        print("\n=== Robot Control Finished ===")
        if ser is not None and ser.is_open:
            ser.close()

if __name__ == "__main__":
    main()