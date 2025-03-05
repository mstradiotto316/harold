#!/usr/bin/env python3
import serial
import time
import numpy as np
from onnxruntime import InferenceSession
import signal
import pickle
import argparse
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
import os
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
bus = None
slow_mode = False
obs_log_file_path = None
safe_positions_str = None  # Global safe command string

# ===========================#
#   FUNCTIONS & CALLBACKS   #
# ===========================#

def send_to_arduino(ser_obj, positions_line):
    message_to_send = positions_line + '#'
    ser_obj.write(message_to_send.encode())
    ser_obj.flush()
    time.sleep(0.005)

def read_from_arduino(ser_obj):
    messages = []
    while ser_obj.in_waiting:
        message = ser_obj.readline().decode().strip()
        if message:
            messages.append(message)
    return messages

def wait_for_arduino_handshake(ser_obj):
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
    global action_scale, default_positions
    scaled_actions = raw_actions * action_scale  # Apply scaling

    # Get default positions from the config
    joint_position_dict = dict(zip([
        'fl_shoulder_joint', 'fr_shoulder_joint', 'bl_shoulder_joint', 'br_shoulder_joint',
        'fl_thigh_joint', 'fr_thigh_joint', 'bl_thigh_joint', 'br_thigh_joint',
        'fl_knee_joint', 'fr_knee_joint', 'bl_knee_joint', 'br_knee_joint',
    ], default_positions))

    # Joint mapping with multipliers to account for mirror symmetry.
    joint_mapping = {
        0: ('fl_shoulder_joint', 1),
        1: ('fr_shoulder_joint', 1),
        2: ('bl_shoulder_joint', 1),
        3: ('br_shoulder_joint', 1),
        4: ('fl_thigh_joint', 1),
        5: ('fr_thigh_joint', 1),
        6: ('bl_thigh_joint', 1),
        7: ('br_thigh_joint', 1),
        8: ('fl_knee_joint', 1),
        9: ('fr_knee_joint', 1),
        10: ('bl_knee_joint', 1),
        11: ('br_knee_joint', 1),
}

    reordered_positions = []
    for i in range(12):
        joint_name, multiplier = joint_mapping[i]
        reordered_positions.append(
            (joint_position_dict[joint_name] + scaled_actions[i]) * multiplier)

    # Clip final positions to safe bounds ([-1.5, 1.5]) ? adjust if needed.
    clipped_positions = np.clip(reordered_positions, -1.5, 1.5)
    positions_line = ','.join(f'{pos:.4f}' for pos in clipped_positions)
    return positions_line

def load_policy_and_config():
    global policy_session, action_scale, default_positions, safe_positions_str
    policy_session = InferenceSession(ONNX_PATH)
    with open(ACTION_CONFIG_PATH, 'rb') as f:
        config = pickle.load(f)
    action_scale = config['action_scale']
    default_positions = np.array([
        config['default_joint_pos'][name]
        for name in [
            'fl_shoulder_joint', 'fr_shoulder_joint', 'bl_shoulder_joint', 'br_shoulder_joint',
            'fl_thigh_joint', 'fr_thigh_joint', 'bl_thigh_joint', 'br_thigh_joint',
            'fl_knee_joint', 'fr_knee_joint', 'bl_knee_joint', 'br_knee_joint',
        ]
    ], dtype=np.float32)
    print("Policy and Action Config Loaded (using pickle).")
    # Compute the safe default command: zero raw action means default positions.
    safe_actions = np.zeros(12, dtype=np.float32)
    safe_positions_str = process_policy_action(safe_actions)
    print("Safe command (default joint positions) computed as:")
    print(safe_positions_str)

def emergency_shutdown(sig, frame):
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
    # Placeholder observation function - replace with actual IMU reading
    # You can uncomment the IMU reader code when ready to use real sensor data
    """
    # Import IMU reader
    sys.path.append(os.path.join(os.path.dirname(SCRIPT_DIR), "sensors"))
    from imu_reader import IMUReader
    
    # Initialize IMU reader if not already done
    global imu_reader
    if not hasattr(create_observation, 'imu_reader'):
        create_observation.imu_reader = IMUReader()
        
    # Get IMU data
    imu_data = create_observation.imu_reader.get_filtered_data()
    # Format into observation vector (this would need to be adjusted based on your policy's expected format)
    # ...
    """
    # For now, just return zeros
    return np.zeros(38, dtype=np.float32).reshape(1, -1)

def run_policy_step(observation):
    global policy_session, ACTION_REDUCTION_FACTOR
    action = policy_session.run(None, {'obs': observation})[0][0]  # Get raw action from policy
    print("Raw Actions:")
    print(action)
    clipped_raw_actions = np.clip(action, -1.0, 1.0)  # Clip raw actions to [-1, 1]
    print("Clipped Actions:")
    print(clipped_raw_actions)
    reduced_actions = clipped_raw_actions * ACTION_REDUCTION_FACTOR  # Apply action reduction factor (now 1.0)
    print("Reduced and Clipped Actions:")
    print(reduced_actions)
    positions_command_str = process_policy_action(reduced_actions)  # Process actions
    return positions_command_str

def main():
    global ser, policy_session, slow_mode, obs_log_file_path, SERIAL_PORT
    parser = argparse.ArgumentParser(description="Harold Deployment Script")
    parser.add_argument('--slow_mode', action='store_true', help='Enable slower control loop for safer testing')
    parser.add_argument('--obs_log_file', type=str, help='Path to observation log file')
    parser.add_argument('--serial_port', type=str, help='Serial port for Arduino connection')
    args = parser.parse_args()
    slow_mode = args.slow_mode
    obs_log_file_path = args.obs_log_file
    
    # Override serial port if provided
    if args.serial_port:
        SERIAL_PORT = args.serial_port
    signal.signal(signal.SIGINT, emergency_shutdown)
    print("Starting Harold Deployment Script...")
    if slow_mode:
        print("--- SLOW MODE ENABLED ---")
    print(f"--- Action Reduction Factor: {ACTION_REDUCTION_FACTOR} ---")
    if obs_log_file_path:
        print(f"--- Observation Log Playback from: {obs_log_file_path} ---")
    print("Loading Policy and Configuration...")
    load_policy_and_config()
    print("Opening Serial Port...")
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print("Serial Port Opened.")
    print("Waiting for Arduino Handshake...")
    if not wait_for_arduino_handshake(ser):
        print("Error: Arduino Handshake Failed. Exiting.")
        if ser is not None and ser.is_open:
            ser.close()
        exit(1)
    print("Arduino Handshake Successful!\n")
    print("Waiting 4 seconds for Arduino servo startup...")
    time.sleep(4.0)
    print("\n--- Deployment STARTED ---")
    try:
        if obs_log_file_path:  # Log playback mode
            with open(obs_log_file_path, 'r') as log_file:
                for line in log_file:
                    start_time = time.time()
                    obs_list_str = line.strip().strip('[]').split(',')
                    obs_list = [float(x) for x in obs_list_str]
                    observation = np.array(obs_list, dtype=np.float32).reshape(1, -1)
                    positions_command_str = run_policy_step(observation)
                    send_to_arduino(ser, positions_command_str)
                    elapsed_time = time.time() - start_time
                    control_rate_delay = max(0.0, CONTROL_PERIOD - elapsed_time)
                    time.sleep(control_rate_delay)
                    if slow_mode:
                        time.sleep(0.1)
        else:  # Normal mode (IMU reading)
            while True:
                start_time = time.time()
                observation = create_observation()
                positions_command_str = run_policy_step(observation)
                send_to_arduino(ser, positions_command_str)
                elapsed_time = time.time() - start_time
                control_rate_delay = max(0.0, CONTROL_PERIOD - elapsed_time)
                time.sleep(control_rate_delay)
                if slow_mode:
                    time.sleep(0.1)
    except Exception as e:
        print(f"\n--- ERROR in Main Loop: {e} ---")
        emergency_shutdown(None, None)
    finally:
        print("\n--- Deployment FINISHED ---")
        if ser is not None and ser.is_open:
            ser.close()

if __name__ == "__main__":
    main()

