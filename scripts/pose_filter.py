#!/usr/bin/env python3

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from scipy.spatial.transform import Rotation as R

class PoseFilter():
	"""
		Kalman Filter implementation for 6D pose tracking as 
		depicted in https://campar.in.tum.de/Chair/KalmanFilter and 
		https://docs.opencv.org/3.3.0/dc/d2c/tutorial_real_time_pose.html
		and https://pieriantraining.com/kalman-filter-opencv-python-example/
		the latter is using a linear motion model that is suitable for slow movements.

		@param f_ctrl Loop frequency
		@type float
		@param state_init Initialize the states of the filter (x,y,z,r,p,y)
		@type Tuple[float,float,float,float,float,float]
		@param process_noise Noise model initialization for process
		@type float
		@param measurement_noise Noise model initialization for measurement
		@type float
		@param error_post Post state covariance error
		@type float
		@param pos_only Filter only xyz position
		@type bool
	"""

	num_states = 18
	num_measurements = 6

	def __init__(self,
			  				f_ctrl: float,
							state_init: Optional[Tuple[float,float,float,float,float,float]]=(0.0,0.0,0.0,0.0,0.0,0.0),
							process_noise: Optional[float]=1e-5,
							measurement_noise: Optional[float]=1e-2,
							error_post: Optional[float]=1.0,
							pos_only: Optional[bool]=False) -> None:
		
		self.dt = 1/f_ctrl
		self.state_init = state_init
		self.process_noise = process_noise
		self.measurement_noise = measurement_noise
		self.error_post = error_post
		self.pos_only = pos_only
		self.translation_estimated = np.zeros(3, dtype=np.float32)
		self.rotation_estimated = None if pos_only else np.zeros(3, dtype=np.float32) 
		self.translation_predicted = np.zeros(3, dtype=np.float32)
		self.rotation_predicted = None if pos_only else np.zeros(3, dtype=np.float32)
		if pos_only:
			self.num_states = 6
			self.num_measurements = 3

		self.filter = cv2.KalmanFilter(self.num_states, self.num_measurements)
		self.initFilter()

	@property
	def est_rotation_as_euler(self) -> np.ndarray:
		return None if self.pos_only else self.rotation_estimated
	@property
	def est_rotation_as_matrix(self) -> np.ndarray:
		return  None if self.pos_only else R.from_euler('xyz', self.rotation_estimated) 
	@property
	def est_translation(self) -> np.ndarray:
		return self.translation_estimated
	@property
	def pred_rotation_as_euler(self) -> np.ndarray:
		return None if self.pos_only else self.rotation_predicted
	@property
	def pred_rotation_as_matrix(self) -> np.ndarray:
		return  None if self.pos_only else R.from_euler('xyz', self.rotation_predicted) 
	@property
	def pred_translation(self) -> np.ndarray:
		return self.translation_predicted

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
		self.filter.errorCovPost = cv2.setIdentity(self.filter.errorCovPost, self.error_post)
		self.filter.processNoiseCov = cv2.setIdentity(self.filter.processNoiseCov, self.process_noise)
		self.filter.measurementNoiseCov = cv2.setIdentity(self.filter.measurementNoiseCov, self.measurement_noise)

		# transition matrix
		if self.pos_only:
			self.filter.transitionMatrix = np.array([[1, 0, 0, 1, 0, 0],
																					 [0, 1, 0, 0, 1, 0],
																				  	 [0, 0, 1, 0, 0, 1],
																					 [0, 0, 0, 1, 0, 0],
																					 [0, 0, 0, 0, 1, 0],
																					 [0, 0, 0, 0, 0, 1]], np.float32)
			self.filter.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0],
																							  [0, 1, 0, 0, 0, 0],
																							  [0, 0, 1, 0, 0, 0]], np.float32)
		else:
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
		if not self.pos_only:
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

		self.translation_predicted= prediction[:3]
		self.translation_estimated = estimated[:3]
		if not self.pos_only:
			self.rotation_predicted = prediction[9:12]
			self.rotation_estimated = estimated[9:12]
		return self.rotation_estimated, self.translation_estimated

	def poseToMeasurement(self,  tvec: np.ndarray, rvec: np.ndarray=None) -> np.ndarray:
		measurement = np.zeros(self.num_measurements, dtype=np.float32)
		measurement[:3] = tvec[:] 

		if not self.pos_only:
			# convert rotation vector to euler angles
			roudriges, _ = cv2.Rodrigues(np.array(rvec))
			mat = R.from_matrix(roudriges)
			measured_eulers = mat.as_euler('xyz')
			measurement[3:] = measured_eulers[:]
		return measurement
	
def test6DFilter():
	fq, size = 30, 500
	init = (0,1,-2,21,0,0)
	proc_noise = 1e-5
	meas_noise = 1e-2
	error = 1
	f = PoseFilter(fq, init, process_noise=proc_noise, measurement_noise=meas_noise, error_post=error)

	for var in range(6):
		error_2 = 0.01
		est, inp = np.zeros((size, 6), dtype=np.float32), np.zeros((size, 6), dtype=np.float32)
		for i in range(size):
			meas = np.random.uniform(init[var]-error_2, init[var]+error_2, 6).astype(np.float32)
			(rot, trans) = f.updateFilter(f.poseToMeasurement(meas[:3], meas[3:]))
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

def test3DFilter():
	fq, size = 30, 50
	init = (-3,5,0)
	proc_noise = 1e-10
	meas_noise = 1e-10
	error = 0
	f = PoseFilter(fq, (0,0,0), process_noise=proc_noise, measurement_noise=meas_noise, error_post=error, pos_only=True)

	for var in range(3):
		error_2 = 0.01
		est, inp = np.zeros((size, 3), dtype=np.float32), np.zeros((size, 3), dtype=np.float32)
		for i in range(size):
			meas = np.random.uniform(init[var]-error_2, init[var]+error_2, 3).astype(np.float32)
			(rot, trans) = f.updateFilter(f.poseToMeasurement(meas))
			inp[i] = meas
			est[i] = trans

		xs = [x/fq for x in range(size)]
		mean = sum(inp[:, var])/len(inp[:, var])
		print("Last estimation: ", est[-1,var], ", mean: ", mean, ", true value: ", init[var])
		plt.plot(xs, inp[:, var], label='measurement')
		plt.plot(xs, [init[var] for _ in range(size)], label='init')
		plt.plot(xs, [mean for _ in range(size)], label='mean')
		plt.plot(xs, est[:, var], label='estimate', linewidth=2)
		plt.legend()
		plt.show()

if __name__=="__main__":
	test6DFilter()
