import serial
import time

def angle_to_position(angle):
    """Convert angle in degrees to servo position value (0-4095)"""
    # Dynamixel position range is 0-4095 for 0-360 degrees
    return int((angle % 360) * (4095 / 360))

def position_to_angle(position):
    """Convert servo position value (0-4095) to angle in degrees"""
    return (position * 360) / 4095

def read_servo_register(servo_id, address, length, port='/dev/ttyACM0', baudrate=1000000):
    """Read data from servo register"""
    packet = [
        0xFF, 0xFF,     # Header
        servo_id,       # ID
        4,              # Length of remaining bytes
        0x02,           # Instruction: READ
        address,        # Starting address
        length          # Length of data to read
    ]
    
    checksum = (~sum(packet[2:])) & 0xFF
    packet.append(checksum)
    
    with serial.Serial(port, baudrate, timeout=0.1) as ser:
        # Clear any existing data
        ser.reset_input_buffer()
        
        # Send the packet
        ser.write(bytearray(packet))
        
        # Small delay to ensure servo has time to respond
        time.sleep(0.01)
        
        # Read response
        # First, wait for header bytes
        header = ser.read(2)
        if len(header) != 2 or header != b'\xff\xff':
            raise TimeoutError(f"Invalid header from servo {servo_id}")
            
        # Read ID byte
        servo_response_id = ser.read(1)
        if len(servo_response_id) != 1 or servo_response_id[0] != servo_id:
            raise TimeoutError(f"Invalid servo ID in response")
            
        # Read length byte
        length_byte = ser.read(1)
        if len(length_byte) != 1:
            raise TimeoutError(f"No length byte received")
        response_length = length_byte[0]
        
        # Read remaining bytes (error byte + parameters + checksum)
        remaining = ser.read(response_length)
        if len(remaining) != response_length:
            raise TimeoutError(f"Incomplete response from servo {servo_id}")
            
        # Check error byte
        error_byte = remaining[0]
        if error_byte != 0:
            error_codes = []
            if error_byte & 0x01: error_codes.append("Input Voltage Error")
            if error_byte & 0x02: error_codes.append("Angle Limit Error")
            if error_byte & 0x04: error_codes.append("Overheating Error")
            if error_byte & 0x08: error_codes.append("Range Error")
            if error_byte & 0x10: error_codes.append("Checksum Error")
            if error_byte & 0x20: error_codes.append("Overload Error")
            if error_byte & 0x40: error_codes.append("Instruction Error")
            raise RuntimeError(f"Servo {servo_id} reported errors: {', '.join(error_codes)}")
            
        # Return parameter bytes (excluding error byte and checksum)
        return remaining[1:-1]

def get_servo_position(servo_id, port='/dev/ttyACM0', baudrate=1000000):
    """Read current position of servo and return in degrees"""
    try:
        # Position is at address 0x24, 2 bytes
        position_bytes = read_servo_register(servo_id, 0x24, 2, port, baudrate)
        position = position_bytes[0] + (position_bytes[1] << 8)
        angle = position_to_angle(position)
        return angle
    except Exception as e:
        print(f"Error reading position from servo {servo_id}: {e}")
        return None

def get_servo_load(servo_id, port='/dev/ttyACM0', baudrate=1000000):
    """Read current load (torque) of servo"""
    try:
        # Load is at address 0x28, 2 bytes
        load_bytes = read_servo_register(servo_id, 0x28, 2, port, baudrate)
        load = load_bytes[0] + (load_bytes[1] << 8)
        # Convert to percentage and direction
        direction = "CCW" if load >= 1024 else "CW"
        percentage = (load & 1023) / 1023 * 100
        return direction, percentage
    except Exception as e:
        print(f"Error reading load from servo {servo_id}: {e}")
        return None

def get_servo_temperature(servo_id, port='/dev/ttyACM0', baudrate=1000000):
    """Read current temperature of servo in degrees Celsius"""
    try:
        # Temperature is at address 0x2B, 1 byte
        temp_byte = read_servo_register(servo_id, 0x2B, 1, port, baudrate)
        return temp_byte[0]
    except Exception as e:
        print(f"Error reading temperature from servo {servo_id}: {e}")
        return None

def monitor_servos(servo_ids=[1, 2, 3], port='/dev/ttyACM0', baudrate=1000000):
    """Monitor and display current status of all servos"""
    print("\nServo Status:")
    print("-" * 50)
    
    # Open serial port once for all operations
    with serial.Serial(port, baudrate, timeout=0.1) as ser:
        for servo_id in servo_ids:
            try:
                # Get all readings with a single serial connection
                position = get_servo_position(servo_id, port, baudrate)
                time.sleep(0.01)  # Small delay between reads
                load_info = get_servo_load(servo_id, port, baudrate)
                time.sleep(0.01)  # Small delay between reads
                temp = get_servo_temperature(servo_id, port, baudrate)
                
                print(f"Servo {servo_id}:")
                if position is not None:
                    print(f"  Position: {position:.1f}°")
                if load_info is not None:
                    direction, percentage = load_info
                    print(f"  Load: {percentage:.1f}% ({direction})")
                if temp is not None:
                    print(f"  Temperature: {temp}°C")
            except Exception as e:
                print(f"Servo {servo_id}:")
                print(f"  Error: {str(e)}")
            print("-" * 50)

def set_servos_angles(angles, servo_ids=[1, 2, 3], port='/dev/ttyACM0', baudrate=1000000):
    """Set multiple servos to specific angles in degrees simultaneously"""
    if len(angles) != len(servo_ids):
        raise ValueError("Number of angles must match number of servo IDs")

    packets = []
    # Prepare packets for all servos
    for servo_id, angle in zip(servo_ids, angles):
        goal_pos = angle_to_position(angle)
        position_l = goal_pos & 0xFF
        position_h = (goal_pos >> 8) & 0xFF

        packet = [
            0xFF, 0xFF,             # Header
            servo_id,               # ID
            5,                      # Length of remaining bytes
            0x03,                   # Instruction: WRITE
            0x2A,                   # Address: Goal Position
            position_l, position_h  # Data: 2 bytes (little-endian)
        ]
        checksum = (~sum(packet[2:])) & 0xFF
        packet.append(checksum)
        packets.append(packet)

    # Send all packets in quick succession
    with serial.Serial(port, baudrate, timeout=0.1) as ser:
        for packet, servo_id, angle in zip(packets, servo_ids, angles):
            ser.write(bytearray(packet))
            print(f"Set servo ID {servo_id} to {angle}°")

def sweep_servos(start_angles, end_angles, servo_ids=[1, 2, 3], steps=10, delay=0.1, port='/dev/ttyACM0', baudrate=1000000):
    """Sweep multiple servos between their respective start and end angles"""
    if len(start_angles) != len(end_angles) or len(start_angles) != len(servo_ids):
        raise ValueError("Number of start angles, end angles, and servo IDs must match")
    
    if start_angles == end_angles:
        print("Start and end angles are the same, no sweep needed")
        return
    
    # Calculate step sizes for each servo
    angle_steps = [(end - start) / steps for start, end in zip(start_angles, end_angles)]
    
    # Perform the sweep
    for step in range(steps + 1):
        current_angles = [
            start + (step * step_size)
            for start, step_size in zip(start_angles, angle_steps)
        ]
        set_servos_angles(current_angles, servo_ids, port, baudrate)
        time.sleep(delay)
        
    print(f"Completed synchronized sweep for servos {servo_ids}")

if __name__ == "__main__":
    print("INITIATING SEQUENCE...")
    #time.sleep(3)
    
    # Example usage showing position monitoring
    set_servos_angles([45, 45, 45])
    time.sleep(1)
    
    # Monitor current status
    monitor_servos()
    
    # Example sweep with monitoring
    print("\nPerforming sweep with monitoring...")
    sweep_servos(
        start_angles=[0, 0, 0],
        end_angles=[90, 45, 180],
        steps=20,
        delay=0.05
    )
    
    # Monitor final positions
    time.sleep(0.5)  # Give servos time to settle
    monitor_servos()
    
    print("\nRESETTING SERVO POSITIONS...")
    set_servos_angles([0, 0, 0])
    
    print("COMPLETE!")
