import smbus
import time
import math
import numpy as np

class IMUReader:
    def __init__(self):
        self.bus = smbus.SMBus(1)
        self.MPU6050_ADDR = 0x68
        self._init_imu()
        
        # Filter parameters
        self.alpha = 0.98
        self.dt = 0.01
        self.roll = 0.0
        self.pitch = 0.0
        self.offsets = np.zeros(6)
        
    def _init_imu(self):
        self.bus.write_byte_data(self.MPU6050_ADDR, 0x6B, 0)
        time.sleep(0.1)

    def _read_word(self, reg):
        high = self.bus.read_byte_data(self.MPU6050_ADDR, reg)
        low = self.bus.read_byte_data(self.MPU6050_ADDR, reg+1)
        return (high << 8) | low

    def _read_word_2c(self, reg):
        val = self._read_word(reg)
        return val - 0x10000 if val >= 0x8000 else val

    def get_raw_data(self):
        """Return raw accelerometer and gyroscope measurements."""
        accel_x = self._read_word_2c(0x3B) / 16384.0
        accel_y = self._read_word_2c(0x3D) / 16384.0
        accel_z = self._read_word_2c(0x3F) / 16384.0

        gyro_x = self._read_word_2c(0x43) / 131.0
        gyro_y = self._read_word_2c(0x45) / 131.0
        gyro_z = self._read_word_2c(0x47) / 131.0

        return {
            'accel': [accel_x, accel_y, accel_z],
            'gyro': [gyro_x, gyro_y, gyro_z]
        }

    def get_filtered_data(self):
        """Return accelerometer, gyro and orientation using a complementary filter."""
        data = self.get_raw_data()
        accel_x, accel_y, accel_z = data['accel']
        gyro_x, gyro_y, gyro_z = data['gyro']

        if hasattr(self, 'offsets'):
            accel_x -= self.offsets[0]
            accel_y -= self.offsets[1]
            accel_z -= self.offsets[2]
            gyro_x -= self.offsets[3]
            gyro_y -= self.offsets[4]
            gyro_z -= self.offsets[5]

        # Complementary filter
        acc_pitch = math.atan2(accel_y, math.sqrt(accel_x**2 + accel_z**2)) * 180/math.pi
        acc_roll = math.atan2(-accel_x, math.sqrt(accel_y**2 + accel_z**2)) * 180/math.pi
        
        self.pitch = self.alpha*(self.pitch + gyro_y*self.dt) + (1-self.alpha)*acc_pitch
        self.roll = self.alpha*(self.roll + gyro_x*self.dt) + (1-self.alpha)*acc_roll
        
        return {
            'accel': [accel_x, accel_y, accel_z],
            'gyro': [gyro_x, gyro_y, gyro_z],
            'orientation': [self.roll, self.pitch]
        }


    def calibrate(self, duration=5):
        """Calibrate accelerometer and gyroscope offsets."""
        print("Calibrating IMU...")
        offsets = np.zeros(6)
        samples = 0
        start = time.time()
        while time.time() - start < duration:
            data = self.get_raw_data()
            offsets += np.array([
                data['accel'][0], data['accel'][1], data['accel'][2],
                data['gyro'][0], data['gyro'][1], data['gyro'][2]
            ])
            samples += 1
            time.sleep(self.dt)

        if samples:
            self.offsets = offsets / samples
        else:
            self.offsets = np.zeros(6)

        print("Calibration complete! Offsets:", self.offsets)
