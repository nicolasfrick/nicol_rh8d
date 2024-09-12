#!/usr/bin/env python3

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from scipy.spatial.transform import Rotation as R

class PoseFilter():
	"""
		Kalman Filter implementation for 6D pose object tracking as 
		depicted in https://campar.in.tum.de/Chair/KalmanFilter and 
		https://docs.opencv.org/3.3.0/dc/d2c/tutorial_real_time_pose.html

		@param f_ctrl Loop frequency
		@type float
		@param state_init Initialize the states of the filter (x,y,z,r,p,y)
		@type Tuple[float,float,float,float,float,float]
		@param num_states Number of filter states (visible + hidden)
		@type int
		@param num_measurements Number of input parameters
		@type int
		@param process_noise Noise model initialization for process
		@type float
		@param measurement_noise Noise model initialization for measurement
		@type float
	"""

	def __init__(self,
			  				f_ctrl: float,
							state_init: Optional[Tuple[float,float,float,float,float,float]]=(0.0,0.0,0.0,0.0,0.0,0.0),
							num_states: Optional[int]=18,
							num_measurements: Optional[int]=6,
							process_noise: Optional[float]=1e-5,
							measurement_noise: Optional[float]=1e-2) -> None:
		
		self.dt = 1/f_ctrl
		self.state_init = state_init
		self.num_states = num_states
		self.num_measurements = num_measurements
		self.process_noise = process_noise
		self.measurement_noise = measurement_noise
		self.translation_estimated = np.zeros(3, dtype=np.float32)
		self.rotation_estimated = np.zeros(3, dtype=np.float32)
		self.filter = cv2.KalmanFilter(self.num_states, self.num_measurements)
		self.initFilter()

	@property
	def rotation_as_euler(self) -> np.ndarray:
		return self.rotation_estimated
	@property
	def rotation_as_matrix(self) -> np.ndarray:
		return  R.from_euler('xyz', self.rotation_estimated) 
	@property
	def translation(self) -> np.ndarray:
		return self.translation_estimated

	def initFilter(self) -> None:
		"""        
			Initialize the filter matrices.

			DYNAMIC MODEL 
			[1 0 0 dt  0  0 dt2   0   0 0 0 0  0  0  0   0   0   0]
			[0 1 0  0 dt  0   0 dt2   0 0 0 0  0  0  0   0   0   0]
			[0 0 1  0  0 dt   0   0 dt2 0 0 0  0  0  0   0   0   0]
			[0 0 0  1  0  0  dt   0   0 0 0 0  0  0  0   0   0   0]
			[0 0 0  0  1  0   0  dt   0 0 0 0  0  0  0   0   0   0]
			[0 0 0  0  0  1   0   0  dt 0 0 0  0  0  0   0   0   0]
			[0 0 0  0  0  0   1   0   0 0 0 0  0  0  0   0   0   0]
			[0 0 0  0  0  0   0   1   0 0 0 0  0  0  0   0   0   0]
			[0 0 0  0  0  0   0   0   1 0 0 0  0  0  0   0   0   0]
			[0 0 0  0  0  0   0   0   0 1 0 0 dt  0  0 dt2   0   0]
			[0 0 0  0  0  0   0   0   0 0 1 0  0 dt  0   0 dt2   0]
			[0 0 0  0  0  0   0   0   0 0 0 1  0  0 dt   0   0 dt2]
			[0 0 0  0  0  0   0   0   0 0 0 0  1  0  0  dt   0   0]
			[0 0 0  0  0  0   0   0   0 0 0 0  0  1  0   0  dt   0]
			[0 0 0  0  0  0   0   0   0 0 0 0  0  0  1   0   0  dt]
			[0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   1   0   0]
			[0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   0   1   0]
			[0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   0   0   1]

			MEASUREMENT MODEL 
			[1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
			[0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
			[0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
			[0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0]
			[0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0]
			[0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0]
		"""

		# covariance matrices
		self.filter.errorCovPost = cv2.setIdentity(self.filter.errorCovPost, 1)
		self.filter.processNoiseCov = cv2.setIdentity(self.filter.processNoiseCov, self.process_noise)
		self.filter.measurementNoiseCov = cv2.setIdentity(self.filter.measurementNoiseCov, self.measurement_noise)

		# transition matrix
		transition_matrix = np.eye(self.num_states, dtype=np.float32)
		# position
		transition_matrix[0, 3] = self.dt
		transition_matrix[1, 4] = self.dt
		transition_matrix[2, 5] = self.dt
		transition_matrix[3, 6] = self.dt
		transition_matrix[4, 7] = self.dt
		transition_matrix[5, 8] = self.dt
		transition_matrix[0, 6] = 0.5*pow(self.dt, 2)
		transition_matrix[1, 7] = 0.5*pow(self.dt, 2)
		transition_matrix[2, 8] = 0.5*pow(self.dt, 2)
		# orientation
		transition_matrix[9, 12] = self.dt
		transition_matrix[10, 13] = self.dt
		transition_matrix[11, 14] = self.dt
		transition_matrix[12, 15] = self.dt
		transition_matrix[13, 16] = self.dt
		transition_matrix[14, 17] = self.dt
		transition_matrix[9, 15] = 0.5*pow(self.dt, 2)
		transition_matrix[10, 16] = 0.5*pow(self.dt, 2)
		transition_matrix[11, 17] = 0.5*pow(self.dt, 2)
		self.filter.transitionMatrix = transition_matrix

		# measurement matrix
		measurement_matrix = np.zeros((self.num_measurements, self.num_states), dtype=np.float32)
		measurement_matrix[0, 0] = 1  # x
		measurement_matrix[1, 1] = 1  # y
		measurement_matrix[2, 2] = 1  # z
		measurement_matrix[3, 9] = 1  # roll
		measurement_matrix[4, 10] = 1 # pitch
		measurement_matrix[5, 11] = 1 # yaw
		self.filter.measurementMatrix = measurement_matrix

		# states
		init_state = np.zeros((self.num_states, 1), dtype=np.float32)
		init_state[0][0] = self.state_init[0]
		init_state[1][0] = self.state_init[1]
		init_state[2][0] = self.state_init[2]
		init_state[9][0] = self.state_init[3]
		init_state[10][0] = self.state_init[4]
		init_state[11][0] = self.state_init[5]
		self.filter.statePre = init_state
		self.filter.statePost = init_state

	def updateFilter(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
		# predict, to update the internal statePre variable
		prediction = self.filter.predict()
		prediction = prediction.flatten()
		# correct phase uses predicted value and measurement
		estimated = self.filter.correct(measurement)
		estimated = estimated.flatten()

		self.translation_estimated = estimated[:3]
		self.rotation_estimated = estimated[9:12]
		return self.rotation_estimated, self.translation_estimated

	def toMeasurement(self, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
		# convert rotation vector to euler angles
		roudriges, _ = cv2.Rodrigues(np.array(rvec))
		mat = R.from_matrix(roudriges)
		measured_eulers = mat.as_euler('xyz')

		measurement = np.zeros(self.num_measurements, dtype=np.float32)
		measurement[:3] = tvec[:] 
		measurement[3:] = measured_eulers[:]
		return measurement

if __name__=="__main__":
	fq, size = 30, 100
	init = (0,0,0,0.1,0,0)
	proc_noise = 0.1
	meas_noise = 0.8
	print(meas_noise)
	f = PoseFilter(fq, init, process_noise=proc_noise, measurement_noise=meas_noise)

	var = 3
	error_2 = 0.01
	est, inp = np.zeros((size, 6), dtype=np.float32), np.zeros((size, 6), dtype=np.float32)
	for i in range(size):
		meas = np.random.uniform(init[var]-error_2, init[var]+error_2, 6).astype(np.float32)
		(rot, trans) = f.updateFilter(f.toMeasurement(meas[:3], meas[3:]))
		inp[i] = meas
		est[i,:3] = rot
		est[i,3:] = trans

	xs = [x/fq for x in range(size)]
	mean = sum(inp[:, var])/len(inp[:, var])
	print("Last estimation: ", est[-1,var], ", mean: ", mean, ", true value: ", init[var])
	plt.plot(xs, inp[:, var], label='measurement')
	plt.plot(xs, [init[var] for _ in range(size)], label='init')
	plt.plot(xs, [mean for _ in range(size)], label='mean')
	plt.plot(xs, est[:, var], label='estimate', linewidth=2)
	plt.legend()
	plt.show()
