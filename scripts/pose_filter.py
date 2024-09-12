#!/usr/bin/env python3

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

class PoseFilter():
	"""
		Kalman Filter implementation for 6D pose object tracking as 
		depicted in https://campar.in.tum.de/Chair/KalmanFilter and 
		https://docs.opencv.org/3.3.0/dc/d2c/tutorial_real_time_pose.html

		@param f_ctrl Loop frequency
		@type float
		@param num_states Number of filter states (visible + hidden)
		@type int
		@param num_measurements Number of input parameters
		@type int
	"""

	def __init__(self,
			  				f_ctrl: float,
							num_states: int=18,
							num_measurements: int=6,
							process_noise: float=1e-5,
							measurement_noise: float=1e-2) -> None:
		
		self.dt = 1/f_ctrl
		self.num_states = num_states
		self.num_measurements = num_measurements
		self.process_noise = process_noise
		self.measurement_noise = measurement_noise

		self.filter = cv2.KalmanFilter(self.num_states, self.num_measurements)
		self.measurements = np.zeros(self.num_measurements, dtype=np.float32)

		self.initFilter()

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
		# position
		self.filter.transitionMatrix[0, 3] = self.dt
		self.filter.transitionMatrix[1, 4] = self.dt
		self.filter.transitionMatrix[2, 5] = self.dt
		self.filter.transitionMatrix[3, 6] = self.dt
		self.filter.transitionMatrix[4, 7] = self.dt
		self.filter.transitionMatrix[5, 8] = self.dt
		self.filter.transitionMatrix[0, 6] = 0.5*pow(self.dt, 2)
		self.filter.transitionMatrix[1, 7] = 0.5*pow(self.dt, 2)
		self.filter.transitionMatrix[2, 8] = 0.5*pow(self.dt, 2)
		# orientation
		self.filter.transitionMatrix[9, 12] = self.dt
		self.filter.transitionMatrix[10, 13] = self.dt
		self.filter.transitionMatrix[11, 14] = self.dt
		self.filter.transitionMatrix[12, 15] = self.dt
		self.filter.transitionMatrix[13, 16] = self.dt
		self.filter.transitionMatrix[14, 17] = self.dt
		self.filter.transitionMatrix[9, 15] = 0.5*pow(self.dt, 2)
		self.filter.transitionMatrix[10, 16] = 0.5*pow(self.dt, 2)
		self.filter.transitionMatrix[11, 17] = 0.5*pow(self.dt, 2)

		# measurement matrix
		self.filter.measurementMatrix[0, 0] = 1  # x
		self.filter.measurementMatrix[1, 1] = 1  # y
		self.filter.measurementMatrix[2, 2] = 1  # z
		self.filter.measurementMatrix[3, 9] = 1  # roll
		self.filter.measurementMatrix[4, 10] = 1 # pitch
		self.filter.measurementMatrix[5, 11] = 1 # yaw

	def updateFilter(self, measurement):
		# predict, to update the internal statePre variable
		prediction = self.filter.predict()
		# correct phase uses predicted value and measurement
		estimated = self.filter.correct(measurement)

		# translation estimate
		translation_estimated = np.zeros(3, dtype=np.float32)
		translation_estimated[0] = estimated[0]
		translation_estimated[1] = estimated[1]
		translation_estimated[2] = estimated[2]
		# euler angle estimate
		euler_estimated = np.zeros(3, dtype=np.float32)
		euler_estimated[0] = estimated[9]
		euler_estimated[1] = estimated[10]
		euler_estimated[2] = estimated[11]
		# convert to rotation matrix
		rotation_estimated = R.from_euler('xyz', euler_estimated)

		return translation_estimated, rotation_estimated

	def addMeasurement(self, rvec, tvec): #Mat &measurements,const Mat &translation_measured, const Mat &rotation_measured)
		# convert rotation vector to euler angles
		roudriges, _ = cv2.Rodrigues(np.array(rvec))
		mat = R.from_matrix(roudriges)
		measured_eulers = mat.as_euler('xyz')

		# set measurement to predict
		self.measurements[0] = tvec[0] # x
		self.measurements[1] = tvec[1] # y
		self.measurements[2] = tvec[2] # z
		self.measurements[3] = measured_eulers[0] # roll
		self.measurements[4] = measured_eulers[1] # pitch
		self.measurements[5] = measured_eulers[2] # yaw

	def  measure(self, inliers_idx):
		if inliers_idx.rows >= self.minInliersKalman:
				# Get the measured translation
				# Mat translation_measured = pnp_detection.get_t_matrix();

				# // Get the measured rotation
				# Mat rotation_measured = pnp_detection.get_R_matrix();

				# // fill the measurements vector
				# self.fillMeasurements(measurements, translation_measured, rotation_measured)
				self.good_measurement = True

		# update the Kalman filter with good measurements, otherwise with previous valid measurements
		# Mat translation_estimated(3, 1, CV_64FC1);
		# Mat rotation_estimated(3, 3, CV_64FC1);
		# updateKalmanFilter( KF, measurements,
		#                     translation_estimated, rotation_estimated);
		# -- Step 6: Set estimated projection matrix
		# pnp_detection_est.set_P_matrix(rotation_estimated, translation_estimated);

if __name__=="__main__":
	 f = PoseFilter()
