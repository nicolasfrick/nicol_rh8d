import serial
import numpy as np
import pandas as pd
from typing import Tuple, Union, Optional

class QdecSerial():

	START_BYTE = 0xAA 
	END_BYTE = 0xBB 
	ESCAPE_BYTE = 0x7D
	QUADRATURE_PPR = 2048
	QUADRATURE_COUNTS = QUADRATURE_PPR * 4
	INIT_COUNTS = QUADRATURE_PPR * 2
	COLS = ["prox", "med", "dist"]

	def __init__(self,
				 port: Optional[str]='/dev/ttyUSB0',
				 baud: Optional[int]=19200,
				 tout: Optional[int]=1,
				 retry: Optional[int]=100, 
				 filter_iters: Optional[int]=100,
				 ) -> None:
		
		self.retry = retry
		self.filter_iters = filter_iters
		self.ser = serial.Serial(port, baud, timeout=tout)
		self.qdecReset()
		self.ser.flush()

	def close(self) -> None:
		self.ser.close()

	def qdecReset(self) -> None:
		"""Reset counter values to INIT_COUNTS.
		"""
		try:
			return self.ser.write(bytes([self.END_BYTE]))
		except Exception as e:
			print(e)

	def readMedianAnglesRad(self) -> Tuple[int, int, int]:
		filter = self.readAngles() 
		median = filter.median()
		return self._toAngle(median.prox, False), self._toAngle(median.med, False), self._toAngle(median.dist, False)
	
	def readMedianAnglesDeg(self) -> Tuple[int, int, int]:
		filter = self.readAngles() 
		median = filter.median()
		return self._toAngle(median.prox), self._toAngle(median.med), self._toAngle(median.dist)

	def readAngles(self) -> pd.DataFrame:
		""" Read num filter_iters counter values and
			return the angle computed from their medians.
		"""
		filter = pd.DataFrame(columns=self.COLS)
		for idx in range(self.filter_iters):
			tpl = self._txRxData()
			if tpl: 
				filter.loc[idx] = tpl
		return filter
	
	def _toAngle(self, cnt_val: int, deg: bool=True) -> float:
		base = 360.0 if deg else 2*np.pi
		# start with zero
		angle_cnt = cnt_val - self.INIT_COUNTS
		# restrict to positive angles
		if angle_cnt <= 0:
			return 0.0
		return (base / self.QUADRATURE_COUNTS) * angle_cnt

	def _txRxData(self) -> Union[Tuple[int, int, int], bool]:
		""" Read bytes for 3 uint16 values.
			Return the [proximal, medial, distal] counter values.
		"""
		data_buffer = []
		escape = False
		cnt = self.retry
		self.ser.flush()
		
		try:
			# trigger data tx
			self.ser.write(bytes([self.START_BYTE]))

			while cnt > 0:
				# read 1 byte
				data = self.ser.read(1)
				# timeout
				if len(data) == 0:
					cnt -= 1
					continue

				byte = ord(data)
				if byte == self.START_BYTE:
					# reset buffer for new data packet
					data_buffer = [] 
					escape = False
					continue

				# end byte rec
				if byte == self.END_BYTE:
					# we expect 3 uint16 numbers (6 bytes)
					if len(data_buffer) >= 6:  
						num1 = (data_buffer[0] << 8) | data_buffer[1]
						num2 = (data_buffer[2] << 8) | data_buffer[3]
						num3 = (data_buffer[4] << 8) | data_buffer[5]
						return num1, num2, num3
					else:
						# invalid length
						print("Invalid rx packet length")
						return False

				# handle escape byte
				if byte == self.ESCAPE_BYTE:
					escape = True
					continue

				# if escape flag is set, 
				# XOR the byte with 0x20
				if escape:
					byte ^= 0x20
					escape = False

				# add byte to data buffer
				data_buffer.append(byte)

		except Exception as e:
			print(e)

		print("Max number of rx retries exceeded")
		return False
	
if __name__ == '__main__':
	import time
	s = QdecSerial()
	s.qdecReset()
	while 1:
		deg = s.readMedianAnglesDeg()
		print(deg)
		time.sleep(0.2)
