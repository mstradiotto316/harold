#!/usr/bin/env python3
import smbus
import time
import numpy as np

# ===========================#
#   USER CONFIGURATIONS     #
# ===========================#

# MPU6050 Configuration (I2C address, etc. - adjust if needed)
MPU6050_ADDR = 0x68
PWR_MGMT_1   = 0x6B
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H  = 0x43


# ===========================#
#   FUNCTIONS & CALLBACKS   #
# ===========================#

bus = None # I2C bus for MPU6050

def initialize_mpu6050():
    """Initializes the MPU6050 IMU."""
    global bus
    bus = smbus.SMBus(1)  # "1" for I2C bus 1 on Jetson Nano
    bus.write_byte_data(MPU6050_ADDR, PWR_MGMT_1, 0)  # Wake up IMU
    time.sleep(0.1) # Give time to wake up
    print("MPU6050 IMU Initialized.")


def read_imu_data():
    """Reads and processes data from MPU6050 (Accel and Gyro)."""
    accel_x = read_word_2c(ACCEL_XOUT_H)
    accel_y = read_word_2c(ACCEL_XOUT_H+2)
    accel_z = read_word_2c(ACCEL_XOUT_H+4)
    gyro_x  = read_word_2c(GYRO_XOUT_H)
    gyro_y  = read_word_2c(GYRO_XOUT_H+2) # Corrected register address
    gyro_z  = read_word_2c(GYRO_XOUT_H+4) # Corrected register address

    # Scale and return (adjust scaling factors if needed)
    accel_scale = 16384.0 # Sensitivity scale factor for +/- 2g range
    gyro_scale  = 131.0   # Sensitivity scale factor for +/- 250 deg/sec range

    return np.array([
        accel_x / accel_scale,
        accel_y / accel_scale,
        accel_z / accel_scale,
        gyro_x / gyro_scale,
        gyro_y / gyro_scale,
        gyro_z / gyro_scale
    ], dtype=np.float32)


def read_word_2c(reg):
    """Helper function to read signed 16-bit word from MPU6050."""
    high = bus.read_byte_data(MPU6050_ADDR, reg)
    low  = bus.read_byte_data(MPU6050_ADDR, reg+1)
    val = (high << 8) | low
    if val >= 0x8000:
        return -((65535 - val) + 1)
    else:
        return val


def create_observation():
    """Creates the observation vector for the policy with labeled components."""
    imu_data = read_imu_data() # Get IMU readings

    # Placeholder for other observation components (replace with actual if you have encoders etc.)
    root_lin_vel   = np.zeros(3, dtype=np.float32)
    root_ang_vel   = imu_data[3:6] # Use gyro data as angular velocity (x,y,z)
    gravity_vec    = np.array([0, 0, 1.345], dtype=np.float32) # Compensating for Z-axis offset - Using corrected value
    command_input  = np.array([0.2, 0.0, 0.0], dtype=np.float32) # Example command (forward, lateral, yaw)
    joint_positions_normalized = np.zeros(12, dtype=np.float32) # Placeholder
    joint_velocities         = np.zeros(12, dtype=np.float32) # Placeholder
    time_signals             = np.array([np.sin(time.time()), np.cos(time.time())], dtype=np.float32) # Time-based signals

    observation = np.concatenate([
        root_lin_vel,       # 0-2:  Root Linear Velocity (placeholder)
        root_ang_vel,       # 3-5:  Root Angular Velocity (Gyro data)
        gravity_vec,        # 6-8:  Gravity Vector (corrected Z)
        command_input,      # 9-11: Command Input (placeholder)
        joint_positions_normalized, # 12-23: Joint Positions Normalized (placeholder)
        joint_velocities,   # 24-35: Joint Velocities (placeholder)
        time_signals       # 36-37: Time Signals (sin, cos)
    ])

    component_names = [
        "root_lin_vel_x", "root_lin_vel_y", "root_lin_vel_z",
        "root_ang_vel_x", "root_ang_vel_y", "root_ang_vel_z",
        "gravity_vec_x",  "gravity_vec_y",  "gravity_vec_z",
        "command_input_x", "command_input_y", "command_input_z",
        "joint_pos_norm_0", "joint_pos_norm_1", "joint_pos_norm_2", "joint_pos_norm_3",
        "joint_pos_norm_4", "joint_pos_norm_5", "joint_pos_norm_6", "joint_pos_norm_7",
        "joint_pos_norm_8", "joint_pos_norm_9", "joint_pos_norm_10", "joint_pos_norm_11",
        "joint_vel_0", "joint_vel_1", "joint_vel_2", "joint_vel_3",
        "joint_vel_4", "joint_vel_5", "joint_vel_6", "joint_vel_7",
        "joint_vel_8", "joint_vel_9", "joint_vel_10", "joint_vel_11",
        "time_signal_sin", "time_signal_cos"
    ]

    labeled_obs = {}
    for i, name in enumerate(component_names):
        labeled_obs[name] = observation[i]

    return labeled_obs


def main():
    """Main function to test MPU6050 IMU readings and observation creation."""
    print("Starting Harold Observation Test Script...")

    print("Initializing MPU6050 IMU...")
    initialize_mpu6050() # Initialize IMU

    print("\n--- Observation Output ---")
    try:
        while True: # Main loop to read and print observations
            observation = create_observation()

            print("\n--- Observation Vector Components ---")
            for name, value in observation.items():
                print(f"{name}: {value:.6f}") # Print each component with label and formatting

            time.sleep(0.5) # Update rate of 2Hz for observation (adjust as needed)

    except KeyboardInterrupt:
        print("\n--- Observation Test FINISHED ---")
        print("Exiting.")


if __name__ == "__main__":
    main()
