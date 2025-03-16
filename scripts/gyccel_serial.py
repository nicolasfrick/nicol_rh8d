#!/usr/bin/env python

import serial
import numpy as np
from time import sleep
from typing import Optional

class GyccelSerial():

	READ_CMD = 'r'

	def __init__(self, 
							port: Optional[str]="/dev/ttyACM0",
							baud: Optional[int]=38400,
							wait: int=2.0,
							) -> None:
			
		self.wait = wait
		self.ser = serial.Serial(port, baud, timeout=10.0)
		sleep(wait)  # delay to wait for arduino to reset
		self.read()
		self.flush()

	def read(self) -> str:
		msg = self.ser.readline().decode()
		return msg
	
	def readQuats(self) -> np.ndarray:
		self.write('r')
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
	
	def readRaw(self) -> np.ndarray:
		self.write('r')
		sleep(0.1)
		msg = self.ser.readline().decode()
		while not 'g' in msg:
			msg = self.ser.readline().decode()
			print(msg)
			sleep(0.1)
		msg = msg.replace("g/a:", "").replace("\r\n", "").strip()
		msg = msg.split("\t")
		# x = float(msg[1])
		# y = float(msg[2])
		# z = float(msg[3])
		# w = float(msg[0])
		# quats = np.array([x, y, z, w], dtype=np.float32)
		return msg

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
		print(gs.readRaw())
		sleep(0.1)
