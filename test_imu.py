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


def create_imu_observation():
    """Creates the IMU part of the observation vector."""
    imu_data = read_imu_data() # Get IMU readings
    root_ang_vel   = imu_data[3:6] # Use gyro data as angular velocity (x,y,z)
    return imu_data[0:3], root_ang_vel # Return accel and gyro separately


def main():
    """Main function to test MPU6050 IMU readings."""
    print("Starting MPU6050 IMU Test...")

    print("Initializing MPU6050 IMU...")
    initialize_mpu6050() # Initialize IMU

    print("\n--- IMU Readings ---")
    try:
        while True: # Main loop to read IMU data continuously
            accel_data, gyro_data = create_imu_observation()

            print(f"Accel (g): X={accel_data[0]:.3f}, Y={accel_data[1]:.3f}, Z={accel_data[2]:.3f}  |  Gyro (deg/s): X={gyro_data[0]:.3f}, Y={gyro_data[1]:.3f}, Z={gyro_data[2]:.3f}")
            time.sleep(0.1) # Update rate of 10Hz
    except KeyboardInterrupt:
        print("\n--- IMU Test FINISHED ---")
        print("Exiting.")


if __name__ == "__main__":
    main()
