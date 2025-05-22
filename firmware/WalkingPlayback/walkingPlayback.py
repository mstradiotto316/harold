#!/usr/bin/env python3
import serial
import struct
import time
import math
import json

# Serial port configuration
SERIAL_PORT = '/dev/ttyUSB0'  # Change this to match your system
BAUD_RATE = 115200

# Movement parameters
STEP_DELAY = 0.01  # 10ms between steps

def send_angles(ser, angles):
    """Send all joint angles to the Arduino."""
    # Pack the angles as 32-bit floats
    data = struct.pack('f' * len(angles), *angles)
    ser.write(data)
    
    # Read and discard debug output until we get the acknowledgment
    while True:
        if ser.in_waiting > 0:
            byte = ser.read()
            if byte == b'A':
                break
            # Optionally print debug output
            print(byte.decode('utf-8', errors='ignore'), end='')
        time.sleep(0.001)

def main():
    # Read the recorded actions
    with open('simulation_logs/processed_actions.log', 'r') as f:
        actions = [json.loads(line) for line in f if line.strip()]
    
    # Open serial port
    with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
        print("Connected to Arduino. Starting playback...")
        time.sleep(2)  # Wait for Arduino to initialize
        
        # Play back each recorded action
        for action in actions:
            # Convert from radians to degrees
            angles_deg = [math.degrees(angle) for angle in action[0]]
            send_angles(ser, angles_deg)
            time.sleep(STEP_DELAY)
        
        print("\nDone - holding final pose.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    except serial.SerialException as e:
        print(f"Serial port error: {e}")
    except Exception as e:
        print(f"Error: {e}")