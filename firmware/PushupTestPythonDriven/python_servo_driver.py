#!/usr/bin/env python3
import serial
import struct
import time
import math

# Serial port configuration
SERIAL_PORT = '/dev/ttyUSB0'  # Change this to match your system
BAUD_RATE = 115200

# Movement parameters
THIGH_START = 0.0
THIGH_END = -45.0
CALF_START = 0.0
CALF_END = 90.0
STEPS = 80
STEP_DELAY = 0.01  # 50ms between steps

def send_angles(ser, thigh_angle, calf_angle):
    """Send thigh and calf angles to the Arduino."""
    # Pack the angles as 32-bit floats
    data = struct.pack('ff', thigh_angle, calf_angle)
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
    # Open serial port
    with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
        print("Connected to Arduino. Starting pushup sequence...")
        time.sleep(2)  # Wait for Arduino to initialize
        
        # Perform 5 pushup repetitions
        for rep in range(5):
            print(f"\nRep {rep + 1} DOWN...")
            
            # Down phase
            for step in range(1, STEPS + 1):
                t = step / STEPS
                thigh_angle = THIGH_START + (THIGH_END - THIGH_START) * t
                calf_angle = CALF_START + (CALF_END - CALF_START) * t
                
                send_angles(ser, thigh_angle, calf_angle)
                time.sleep(STEP_DELAY)
            
            time.sleep(0.3)  # Bottom pause
            
            print("...UP")
            # Up phase
            for step in range(STEPS, -1, -1):
                t = step / STEPS
                thigh_angle = THIGH_START + (THIGH_END - THIGH_START) * t
                calf_angle = CALF_START + (CALF_END - CALF_START) * t
                
                send_angles(ser, thigh_angle, calf_angle)
                time.sleep(STEP_DELAY)
            
            time.sleep(0.3)  # Top pause
        
        print("\nDone - holding straight pose.")
        send_angles(ser, 0, 0)  # Return to straight position

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    except serial.SerialException as e:
        print(f"Serial port error: {e}")
    except Exception as e:
        print(f"Error: {e}")
