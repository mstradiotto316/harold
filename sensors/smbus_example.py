import smbus
import time

# MPU6050 Registers
MPU6050_ADDR = 0x68
PWR_MGMT_1   = 0x6B
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H  = 0x43

bus = smbus.SMBus(1)  # "1" for I2C bus 1 on Jetson Nano

def read_word(reg):
    high = bus.read_byte_data(MPU6050_ADDR, reg)
    low  = bus.read_byte_data(MPU6050_ADDR, reg+1)
    return (high << 8) | low

def read_word_2c(reg):
    val = read_word(reg)
    if val >= 0x8000:
        return -((65535 - val) + 1)
    else:
        return val

# Initialize the MPU-6050
bus.write_byte_data(MPU6050_ADDR, PWR_MGMT_1, 0)  # wake up the IMU

while True:
    # Read accelerometer
    accel_x = read_word_2c(ACCEL_XOUT_H)
    accel_y = read_word_2c(ACCEL_XOUT_H+2)
    accel_z = read_word_2c(ACCEL_XOUT_H+4)
    
    # Convert to 'g's
    accel_x_g = accel_x / 16384.0
    accel_y_g = accel_y / 16384.0
    accel_z_g = accel_z / 16384.0
    
    # Read gyroscope
    gyro_x = read_word_2c(GYRO_XOUT_H)
    gyro_y = read_word_2c(GYRO_XOUT_H+2)
    gyro_z = read_word_2c(GYRO_XOUT_H+4)
    
    # Convert to deg/sec
    gyro_x_dps = gyro_x / 131.0
    gyro_y_dps = gyro_y / 131.0
    gyro_z_dps = gyro_z / 131.0

    print(f"Accel (g): {accel_x_g:.2f}, {accel_y_g:.2f}, {accel_z_g:.2f},  Gyro (dps): {gyro_x_dps:.2f}, {gyro_y_dps:.2f}, {gyro_z_dps:.2f}")
    time.sleep(0.1)

