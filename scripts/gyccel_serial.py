#!/usr/bin/env python

import time
import serial
import numpy as np
from time import sleep
from typing import Optional, Union

class GyccelSerial():

	READ_CMD = 'r'
	ACCEL_SCALE = 16384.0  # LSB/g (±2g range)
	GYRO_SCALE = 131.0     # LSB/(deg/s) (±250°/s range)

	def __init__(self, 
							port: Optional[str]="/dev/ttyACM0",
							baud: Optional[int]=38400,
							wait: int=2.0,
							) -> None:
			
		self.wait = wait
		self.ser = serial.Serial(port, baud, timeout=10.0)
		sleep(wait) 
		self.read()
		self.flush()

		self.q_init = np.array([ -0.7068252, 0, 0, 0.7073883 ]) 
		self.t_last = time.time()

	def read(self) -> str:
		msg = self.ser.readline().decode()
		return msg
	
	def readQuats(self) -> np.ndarray:
		self.write(self.READ_CMD)
		sleep(0.1)
		msg = self.ser.readline().decode()
		while not 'quat' in msg:
			msg = self.ser.readline().decode()
			sleep(0.1)
		msg = msg.replace("quat", "").replace("\r\n", "").strip()
		msg = msg.split("\t")
		x = float(msg[1])
		y = float(msg[2])
		z = float(msg[3])
		w = float(msg[0])
		quats = np.array([x, y, z, w], dtype=np.float32)
		return quats

	def multQuats(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
		x1, y1, z1, w1 = q1
		x2, y2, z2, w2 = q2
		w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
		x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
		y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
		z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
		return np.array([x, y, z, w])

	def normQuats(self, q: np.ndarray) -> np.ndarray:
		"""Normalize a quaternion to unit length."""
		norm = np.linalg.norm(q)
		if norm == 0:
			return q
		return q / norm

	def gyro2Quat(self, gyro: np.ndarray, dt: float, q_prev: np.ndarray) -> np.ndarray:
		gyro_rad = np.array(gyro) / self.GYRO_SCALE * np.pi / 180.0
		angle = np.linalg.norm(gyro_rad) * dt
		if angle > 0:
			axis = gyro_rad / np.linalg.norm(gyro_rad)
			q_rot = np.array([np.sin(angle/2) * axis[0],
							np.sin(angle/2) * axis[1],
							np.sin(angle/2) * axis[2],
							np.cos(angle/2)])
			q_new = self.multQuats(q_rot, q_prev)
			return self.normQuats(q_new)
		return q_prev

	def accel2Quat(self, accel: np.ndarray) -> np.ndarray:
		accel_g = np.array(accel) / self.ACCEL_SCALE
		accel_g = accel_g / np.linalg.norm(accel_g)
		gravity = np.array([0, 0, -1])
		axis = np.cross(accel_g, gravity)
		angle = np.arccos(np.dot(accel_g, gravity))
		if np.linalg.norm(axis) > 0:
			axis = axis / np.linalg.norm(axis)
			q_acc = np.array([np.sin(angle/2) * axis[0],
							np.sin(angle/2) * axis[1],
							np.sin(angle/2) * axis[2],
							np.cos(angle/2)])
			return self.normQuats(q_acc)
		return np.array([0, 0, 0, 1])

	def fuseQuats(self, q_gyro: np.ndarray, q_acc: np.ndarray, alpha: float=0.02) -> np.ndarray:
		q_fused = (1 - alpha) * q_gyro + alpha * q_acc
		return self.normQuats(q_fused)

	def read_raw_data(ser):
		"""Read raw accel/gyro data from serial."""
		line = ser.readline().decode('utf-8').strip()
		if line.startswith('a/g:'):
			values = line.split('\t')[1:]
			if len(values) == 6:
				ax, ay, az, gx, gy, gz = map(int, values)
				return [ax, ay, az], [gx, gy, gz]
		return None, None

	def readOrientation(self) -> np.ndarray:
		raw = self.readRaw()
		if raw is None:
			return self.q_init
		accel = raw[: 3]
		gyro = raw[3 :]

		current_time = time.time()
		dt = current_time - self.t_last
		self.t_last = current_time

		q_gyro = self.gyro2Quat(gyro, dt, self.q_init)
		q_acc = self.accel2Quat(accel)
		q = self.fuseQuats(q_gyro, q_acc, alpha=0.02)

		return q
	
	def readRaw(self) -> Union[None, np.ndarray]:
		"""       
			ax_rot
			ay_rot
			az_rot
			gx_rot
			gy_rot
			gz_rot
		"""
		self.write(self.READ_CMD)
		sleep(0.1)

		msg = self.ser.readline().decode()
		while not 'g' in msg:
			print(msg)
			msg = self.ser.readline().decode()
			print(msg)
			sleep(0.1)

		msg = msg.replace("a/g:", "").replace("\r\n", "").strip()
		msg = msg.split("\t")
		try:
			ax_rot = float(msg[0])
			ay_rot = float(msg[1])
			az_rot = float(msg[2])
			gx_rot = float(msg[3])
			gy_rot = float(msg[4])
			gz_rot = float(msg[5])
			return np.array([ax_rot, ay_rot, az_rot, gx_rot, gy_rot, gz_rot], dtype=np.float32)
		except:
			return None

	def flush(self) -> None:
		self.ser.reset_input_buffer()

	def reset(self) -> None:
		print("Resetting imu ... ", end="")
		self.ser.close()
		self.ser.open()
		sleep(self.wait) 
		self.read()
		self.flush()
		print("done")

	def close(self) -> None:
		self.ser.close()

	def write(self, msg: str) -> None:
		 self.ser.write(msg.encode("utf-8"))

if __name__ == "__main__":
	gs = GyccelSerial()
	while 1:
		print(gs.readOrientation())
		sleep(0.1)
