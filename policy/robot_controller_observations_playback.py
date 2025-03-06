#!/usr/bin/env python3
import serial
import time
import numpy as np
from onnxruntime import InferenceSession
import signal
import pickle
import os
import sys
import argparse

# ===========================#
#   USER CONFIGURATIONS     #
# ===========================#

# Serial port configuration for Arduino
SERIAL_PORT = '/dev/ttyACM0'  # Default serial port, can be overridden via command line
BAUD_RATE = 115200
HANDSHAKE_MSG = "ARDUINO_READY"

# Path configurations
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
ONNX_PATH = os.path.join(SCRIPT_DIR, "harold_policy.onnx")
ACTION_CONFIG_PATH = os.path.join(SCRIPT_DIR, "action_config.pt")
SIMULATION_LOGS_DIR = os.path.join(ROOT_DIR, "simulation_logs")
DEFAULT_OBS_LOG_PATH = os.path.join(SIMULATION_LOGS_DIR, "observations.log")

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
last_raw_actions = None  # Store raw policy outputs
last_clipped_raw_actions = None  # Store clipped raw actions for printing
last_scaled_actions = None  # Store scaled actions for printing
last_final_positions = None  # Store final joint positions

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
    global action_scale, default_positions, last_final_positions
    scaled_actions = raw_actions * action_scale  # Apply scaling

    # Robot is wired to match simulation joint order directly
    reordered_positions = []
    for i in range(12):
        reordered_positions.append(default_positions[i] + scaled_actions[i])

    # Store final positions for printing
    last_final_positions = np.array(reordered_positions)
    
    # No additional clipping - allow full range of motion to match simulation
    positions_line = ','.join(f'{pos:.4f}' for pos in reordered_positions)
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

def run_policy_step(observation):
    """Run a single step of the policy"""
    global policy_session, ACTION_REDUCTION_FACTOR, last_raw_actions, last_clipped_raw_actions, last_scaled_actions, action_scale
    
    # Get raw action from policy
    action = policy_session.run(None, {'obs': observation})[0][0]
    
    # Store raw policy actions for printing
    last_raw_actions = action.copy()
    
    # Clip raw actions to [-1, 1]
    clipped_raw_actions = np.clip(action, -1.0, 1.0)
    
    # Store clipped but unscaled actions for printing
    last_clipped_raw_actions = clipped_raw_actions.copy()
    
    # Apply action reduction factor
    reduced_actions = clipped_raw_actions * ACTION_REDUCTION_FACTOR
    
    # Store the scaled actions for printing 
    last_scaled_actions = reduced_actions * action_scale  # Simple scaling with action_scale
    
    # Process actions into joint positions
    positions_command_str = process_policy_action(reduced_actions)
    
    return positions_command_str

def main():
    """Main function for observation log playback"""
    global ser, policy_session, SERIAL_PORT
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Harold Robot Observation Playback")
    parser.add_argument('--obs_log_file', type=str, 
                        help=f'Path to observation log file (default: {DEFAULT_OBS_LOG_PATH})')
    parser.add_argument('--serial_port', type=str, 
                        help='Serial port for Arduino connection')
    parser.add_argument('--loop', action='store_true', 
                        help='Loop the observation log continuously')
    args = parser.parse_args()
    
    # Use provided log file or default
    obs_log_file_path = args.obs_log_file or DEFAULT_OBS_LOG_PATH
    loop_playback = args.loop
    
    # Check if observation log file exists
    if not os.path.exists(obs_log_file_path):
        print(f"Error: Observation log file not found: {obs_log_file_path}")
        print(f"Please place an observations.log file in the simulation_logs directory")
        print(f"or specify a file with --obs_log_file")
        exit(1)
    
    # Override serial port if provided
    if args.serial_port:
        SERIAL_PORT = args.serial_port
    
    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, emergency_shutdown)
    
    print("=== Starting Harold Robot Observation Playback ===")
    print(f"Serial port: {SERIAL_PORT}")
    print(f"Observation log file: {obs_log_file_path}")
    print(f"Action reduction factor: {ACTION_REDUCTION_FACTOR}")
    print(f"Loop playback: {'Yes' if loop_playback else 'No'}")
    
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
    
    # Main playback loop
    print("\n=== Playback Started ===")
    try:
        while True:  # Outer loop for repeated playback
            print(f"Playing observation log: {obs_log_file_path}")
            line_count = 0
            
            with open(obs_log_file_path, 'r') as log_file:
                for line in log_file:
                    line_count += 1
                    
                    # Start timing
                    start_time = time.time()
                    
                    # Parse observation from log line
                    obs_list_str = line.strip().strip('[]').split(',')
                    obs_list = [float(x) for x in obs_list_str]
                    observation = np.array(obs_list, dtype=np.float32).reshape(1, -1)
                    
                    # Run policy to get joint positions
                    positions_command_str = run_policy_step(observation)
                    
                    # Print current observations and all action outputs for comparison
                    print("\n--- STEP DATA ---")
                    print(f"Observations: {observation[0]}")
                    print(f"Raw policy outputs: {last_raw_actions}")
                    print(f"Clipped actions [-1,1]: {last_clipped_raw_actions}")
                    print(f"Clipped + scaled actions: {last_scaled_actions}")
                    print(f"Final joint positions: {last_final_positions}")
                    print(f"Command string: {positions_command_str}")
                    print("----------------")
                    
                    # Send command to Arduino
                    send_to_arduino(ser, positions_command_str)
                    
                    # Print progress every 10 lines
                    if line_count % 10 == 0:
                        print(f"Processed {line_count} observations")
                    
                    # Maintain control frequency
                    elapsed_time = time.time() - start_time
                    control_rate_delay = max(0.0, CONTROL_PERIOD - elapsed_time)
                    time.sleep(control_rate_delay)
            
            print(f"Playback complete. Processed {line_count} observations.")
            
            # If not looping, exit after one playback
            if not loop_playback:
                break
            
            print("Restarting playback...")
    
    except Exception as e:
        print(f"\n=== ERROR in Playback: {e} ===")
        emergency_shutdown(None, None)
    finally:
        print("\n=== Playback Finished ===")
        # Send safe positions before closing
        if ser is not None and ser.is_open:
            print("Sending safe positions...")
            send_to_arduino(ser, safe_positions_str)
            time.sleep(0.5)
            ser.close()

if __name__ == "__main__":
    main()