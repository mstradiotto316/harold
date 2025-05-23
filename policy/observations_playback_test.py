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
SERIAL_PORT = '/dev/ttyACM0'  # Default serial port
BAUD_RATE = 115200
HANDSHAKE_MSG = "ARDUINO_READY"

# Path configurations
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

ONNX_PATH = os.path.join(SCRIPT_DIR, "harold_policy.onnx")
ACTION_CONFIG_PATH = os.path.join(SCRIPT_DIR, "action_config.pt")

SIMULATION_LOGS_DIR = os.path.join(ROOT_DIR, "simulation_logs/observations.log")
ACTION_LOGS_DIR = os.path.join(ROOT_DIR, "simulation_logs/actions.log")
SCALED_ACTION_LOGS_DIR = os.path.join(ROOT_DIR, "simulation_logs/processed_actions.log")

# Control Loop Target Frequency (MATCH Isaac Lab Simulation - 20Hz)
CONTROL_FREQUENCY = 20.0
CONTROL_PERIOD = 1.0 / CONTROL_FREQUENCY

# ===========================#
#   GLOBAL OBJECTS/FLAGS    #
# ===========================#
ser = None
policy_session = None
action_scale = 1
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
    
    # Read any responses after command
    responses = []
    start_time = time.time()
    while time.time() - start_time < 0.1:  # Wait up to 100ms for response
        if ser_obj.in_waiting:
            try:
                message = ser_obj.readline().decode().strip()
                if message:
                    responses.append(message)
                    print(f"Arduino response: {message}")
            except Exception as e:
                print(f"Error reading response: {e}")
        else:
            break
    
    return responses

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

    # Define joint angle limits (matching simulation)
    JOINT_ANGLE_MIN = np.array([-0.3491, -0.3491, -0.3491, -0.3491, 
                               -0.7853, -0.7853, -0.7853, -0.7853,
                               -0.7853, -0.7853, -0.7853, -0.7853], dtype=np.float32)
    JOINT_ANGLE_MAX = np.array([0.3491, 0.3491, 0.3491, 0.3491,
                               0.7853, 0.7853, 0.7853, 0.7853,
                               0.7853, 0.7853, 0.7853, 0.7853], dtype=np.float32)

    # Apply scaling
    scaled_actions = raw_actions * action_scale
    print(f"Scaled actions: {np.array2string(scaled_actions, precision=4, suppress_small=True)}")
    
    # Clamp to joint angle limits (matching simulation)
    processed_actions = np.clip(scaled_actions, JOINT_ANGLE_MIN, JOINT_ANGLE_MAX)
    print(f"Processed actions: {np.array2string(processed_actions, precision=4, suppress_small=True)}")
    
    positions_line = ','.join(f'{pos:.4f}' for pos in processed_actions)
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
    print(f"Action scale: {action_scale:.4f}")
    
    # Get default joint positions from config
    default_positions = np.array([
        config['default_joint_pos'][name]
        for name in [
            'fl_shoulder_joint', 'fr_shoulder_joint', 'bl_shoulder_joint', 'br_shoulder_joint',
            'fl_thigh_joint', 'fr_thigh_joint', 'bl_thigh_joint', 'br_thigh_joint',
            'fl_knee_joint', 'fr_knee_joint', 'bl_knee_joint', 'br_knee_joint',
        ]
    ], dtype=np.float32)
    print(f"Default positions: {np.array2string(default_positions, precision=4, suppress_small=True)}")
    print("Policy and Action Config loaded successfully!")
    
    # Compute the safe default command: zero raw action means default positions
    safe_actions = np.zeros(12, dtype=np.float32)
    safe_positions_str = process_policy_action(safe_actions)
    print("Safe default joint positions computed:")
    print(safe_positions_str)

def emergency_shutdown(sig, frame):
    """Handle emergency shutdown (Ctrl+C)"""
    global ser, safe_positions_str
    print("\nEmergency Shutdown Initiated!")
    try:
        if ser is not None and ser.is_open:
            print("Sending safe positions before closing...")
            # Send STOP command to immediately halt servos
            ser.write(b'STOP#')
            ser.flush()
            time.sleep(0.1)
            
            # Then send safe positions multiple times to ensure they're received
            if safe_positions_str:
                for i in range(3):  # Send 3 times to ensure receipt
                    send_to_arduino(ser, safe_positions_str)
                    time.sleep(0.1)
            
            # Wait for servos to reach safe position
            time.sleep(0.5)
            
            print("Closing serial connection...")
            ser.close()
    except Exception as e:
        print(f"Error during shutdown: {e}")
    print("Exiting...")
    # Use sys.exit instead of exit for better compatibility
    sys.exit(0)

def run_policy_step(observation):
    """Run a single step of the policy"""
    global policy_session, action_scale
    
    try:
        # Get raw action from policy
        action = policy_session.run(None, {'obs': observation})[0][0]
        print(f"Raw action: {np.array2string(action, precision=4, suppress_small=True)}")
        
        # Clip raw actions to [-1, 1]
        clipped_actions = np.clip(action, -1.0, 1.0)
        print(f"Clipped action: {np.array2string(clipped_actions, precision=4, suppress_small=True)}")
        
        # Process actions into joint positions
        positions_command_str = process_policy_action(clipped_actions)
        
        return positions_command_str
    except Exception as e:
        print(f"Error in run_policy_step: {e}")
        # Return safe positions as fallback
        return safe_positions_str if safe_positions_str else "0,0,0,0,0,0,0,0,0,0,0,0"

def main():
    """Main function for observation log playback"""
    global ser, policy_session, SERIAL_PORT
  
    # Check if observation log file exists
    if not os.path.exists(SIMULATION_LOGS_DIR):
        print(f"Error: Observation log file not found: {SIMULATION_LOGS_DIR}")
        print(f"Please place an observations.log file in the simulation_logs directory")
        exit(1)
    elif not os.path.exists(ACTION_LOGS_DIR):
        print(f"Error: Action log file not found: {ACTION_LOGS_DIR}")
        print(f"Please place an actions.log file in the simulation_logs directory")
        exit(1)
    elif not os.path.exists(SCALED_ACTION_LOGS_DIR):
        print(f"Error: Scaled action log file not found: {SCALED_ACTION_LOGS_DIR}")
        print(f"Please place an scaled_actions.log file in the simulation_logs directory")
        exit(1)
    
    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, emergency_shutdown)
    
    print("=== Starting Harold Robot Observation Playback ===")
    print(f"Serial port: {SERIAL_PORT}")
    print(f"Simulated observations log file: {SIMULATION_LOGS_DIR}")
    print(f"Simulated actions log file: {ACTION_LOGS_DIR}")
    print(f"Simulated scaled actions log file: {SCALED_ACTION_LOGS_DIR}")

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

        line_count = 0
        total_lines = 0
        
        # Count total lines for progress tracking
        with open(SIMULATION_LOGS_DIR, 'r') as f:
            total_lines = len(f.readlines())
        print(f"Total observations to process: {total_lines}")
        
        with open(SIMULATION_LOGS_DIR, 'r') as obs_log_file, \
             open(ACTION_LOGS_DIR, 'r') as action_log_file, \
             open(SCALED_ACTION_LOGS_DIR, 'r') as scaled_action_log_file:
            
            print("Starting playback loop...")
            for obs_line, action_line, scaled_action_line in zip(obs_log_file, action_log_file, scaled_action_log_file):
                line_count += 1
                
                # Start timing
                start_time = time.time()
                print(f"Processing line {line_count}/{total_lines} of {total_lines}")
               
                # Parse observation from log line
                try:
                    obs_list_str = obs_line.strip().strip('[]').split(',')
                    obs_list = [float(x) for x in obs_list_str]
                    observation = np.array(obs_list, dtype=np.float32).reshape(1, -1)
                except Exception as e:
                    print(f"Error parsing observation line {line_count}: {e}")
                    print(f"Observation line content: {obs_line[:100]}...")
                    # Use an empty observation as fallback
                    observation = np.zeros((1, 50), dtype=np.float32)

                # Parse action and scaled action (optional, for logging/comparison)
                try:
                    action_list_str = action_line.strip().strip('[]').split(',')
                    action_list = [float(x) for x in action_list_str]
                except Exception as e:
                    print(f"Error parsing action line {line_count}: {e}")
                    print(f"Action line content: {action_line[:100]}...")
                    action_list = []
                
                try:
                    scaled_action_list_str = scaled_action_line.strip().strip('[]').split(',')
                    scaled_action_list = [float(x) for x in scaled_action_list_str]
                except Exception as e:
                    print(f"Error parsing scaled action line {line_count}: {e}")
                    print(f"Scaled action line content: {scaled_action_line[:100]}...")
                    scaled_action_list = []

                print("\n--- STEP DATA ---")
                print(f"Simulated Observations: {np.array2string(observation[0], precision=4, suppress_small=True)}")
                print(f"Simulated Actions: {', '.join([f'{x:.4f}' for x in action_list])}")
                print(f"Simulated Scaled Actions: {', '.join([f'{x:.4f}' for x in scaled_action_list])}")
                
                # Run policy to get joint positions
                positions_command_str = run_policy_step(observation)
                
                print(f"Command string: {positions_command_str}")
                print("----------------")
                
                # Send command to Arduino and get responses
                responses = send_to_arduino(ser, positions_command_str)
                
                # Print progress every 10 lines
                if line_count % 10 == 0:
                    print(f"Processed {line_count} observations")
                
                # Maintain control frequency with timeout
                elapsed_time = time.time() - start_time
                control_rate_delay = max(0.0, CONTROL_PERIOD - elapsed_time)
                
                # Limit maximum delay to 0.25 seconds to avoid getting stuck
                if control_rate_delay > 0.25:
                    print(f"Warning: Long delay detected ({control_rate_delay:.2f}s), limiting to 0.25s")
                    control_rate_delay = 0.25
                    
                time.sleep(control_rate_delay)
        
        print(f"Playback complete. Processed {line_count} observations.")
    
    except Exception as e:
        print(f"\n=== ERROR in Playback: {e} ===")
        emergency_shutdown(None, None)
    finally:
        print("\n=== Playback Finished ===")
        print(f"Processed {line_count} observations out of {total_lines} total observations")
        # Send STOP command and safe positions before closing
        if ser is not None and ser.is_open:
            print("Sending STOP command and safe positions...")
            # Send STOP command to immediately halt all motion
            ser.write(b'STOP#')
            ser.flush()
            time.sleep(0.2)
            
            # Then send safe positions multiple times to ensure proper position
            for i in range(3):
                send_to_arduino(ser, safe_positions_str)
                time.sleep(0.1)
                
            # Wait for servos to reach safe position
            time.sleep(0.7)
            ser.close()
        
        print("Program completed successfully")
        sys.exit(0)

if __name__ == "__main__":
    main()
