#!/usr/bin/env python3

import os
import yaml
import cv2
import cv2.aruco as aru
import dataclasses
import numpy as np
from threading import Lock
from typing import Sequence, Optional, Tuple, Union, Any

DATA_PTH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  "datasets/aruco")

@dataclasses.dataclass(eq=False)
class ReconfParams(aru.DetectorParameters):

	# denoising
	denoise: bool = False
	h: float=10.0
	templateWindowSize: int=7
	searchWindowSize: int=21

	def __init__(self):
		# detector params
		super().__init__()

	# adaptiveThreshWinSizeMin: int = 3
	# adaptiveThreshWinSizeMax: int = 23
	# adaptiveThreshWinSizeStep: int = 10
	# adaptiveThreshConstant: float = 7.0
	# minMarkerPerimeterRate: float = 0.03
	# maxMarkerPerimeterRate: float = 4.0  
	# polygonalApproxAccuracyRate: float = 0.03
	# minCornerDistanceRate: float = 0.05
	# minDistanceToBorder: int = 3 
	# minMarkerDistanceRate: float =  0.125
	# minGroupDistance: float = 0.21
	# cornerRefinementMethod: int = 3
	# cornerRefinementWinSize: int = 5
	# relativeCornerRefinmentWinSize: float = 0.3
	# cornerRefinementMaxIterations: int = 30
	# cornerRefinementMinAccuracy: float = 0.1
	# markerBorderBits: int = 1
	# perspectiveRemovePixelPerCell: int = 4
	# perspectiveRemoveIgnoredMarginPerCell: float = 0.13
	# maxErroneousBitsInBorderRate: float = 0.35
	# minOtsuStdDev: float = 5.0
	# errorCorrectionRate: float = 0.6
	# aprilTagQuadDecimate: float = 0.0
	# aprilTagQuadSigma: float = 0.0
	# aprilTagMinClusterPixels: int = 5
	# aprilTagMaxNmaxima: int = 10
	# aprilTagCriticalRad: float = 0.1745329201221466
	# aprilTagMaxLineFitMse: float = 10.0
	# aprilTagMinWhiteBlackDiff: int = 5
	# aprilTagDeglitch: int = 0  
	# detectInvertedMarker: bool = False
	# useAruco3Detection: bool = False
	# minSideLengthCanonicalImg: int = 32
	# minMarkerLengthRatioOriginalImg: float = 0.0 
	# # denoise params
	# denoise: bool = False
	# h: float = 10.0
	# templateWindowSize: int = 7
	# searchWindowSize: int = 21

class ArucoDetector():
	"""Detect Aruco marker from an image.

		@param marker_length:  Physical dimension (sidelength) of the marker in mm to m.
														  Unit determines the unit of the output
		@type  float
		@param K: camera intrinsics
		@type  tuple
		@param D: camera distorsion coefficients (plumb bob)
		@type  tuple
		@param bbox: bounding box ((x_min, x_max),(y_min,y_max)), if provided, image will be cropped
		@type  tuple
		@param denoise: Whether to apply denoising to the cropped image
		@type  bool
		@param dict_type:  type of the aruco dictionary out of ARUCO_DICT, loaded if no yaml path provided
		@type  str
		@param dict_yaml:  path to a dictionary yaml file in 'datasets/aruco/' directory used in favour of dict_type
		@type  str
		@param print_stats Print detection statistics
		@type bool
		@param 	estimate_pattern Enum to set Aruco estimate pattern. One of [ARUCO_CCW_CENTER, ARUCO_CW_TOP_LEFT_CORNER]
		@type int
		@param solve_pnp_method Enum to set Aruco solver method. One of [SOLVEPNP_ITERATIVE, SOLVEPNP_EPNP, SOLVEPNP_P3P, SOLVEPNP_DLS, SOLVEPNP_UPNP, SOLVEPNP_AP3P, SOLVEPNP_IPPE, SOLVEPNP_IPPE_SQUARE, SOLVEPNP_SQPNP, SOLVEPNP_MAX_COUNT]
		@type int

		 Generate markers with opencv (custom build in opencv_build directory):
		.opencv_build/opencv/build/bin/example_cpp_aruco_dict_utils /
		 nicol_rh8d/datasets/aruco/custom_matrix_4x4_20.yml /
		-nMarkers=20 /
		-markerSize=4 /
		-r 
	"""

	# ArUco dictionaries built into the OpenCV library
	ARUCO_DICT = {
								"DICT_4X4_50": aru.DICT_4X4_50,
								"DICT_4X4_100": aru.DICT_4X4_100,
								"DICT_4X4_250": aru.DICT_4X4_250,
								"DICT_4X4_1000": aru.DICT_4X4_1000,
								"DICT_5X5_50": aru.DICT_5X5_50,
								"DICT_5X5_100": aru.DICT_5X5_100,
								"DICT_5X5_250": aru.DICT_5X5_250,
								"DICT_5X5_1000": aru.DICT_5X5_1000,
								"DICT_6X6_50": aru.DICT_6X6_50,
								"DICT_6X6_100": aru.DICT_6X6_100,
								"DICT_6X6_250": aru.DICT_6X6_250,
								"DICT_6X6_1000": aru.DICT_6X6_1000,
								"DICT_7X7_50": aru.DICT_7X7_50,
								"DICT_7X7_100": aru.DICT_7X7_100,
								"DICT_7X7_250": aru.DICT_7X7_250,
								"DICT_7X7_1000": aru.DICT_7X7_1000,
								"DICT_ARUCO_ORIGINAL": aru.DICT_ARUCO_ORIGINAL
	}
	RED = (0,0,255)
	GREEN = (0,255,0)
	BLUE = (255,0,0)
	AXIS_LENGTH = 1.5 # axis drawing
	AXIS_THICKNESS = 2 # axis drawing
	CIRCLE_SIZE = 3
	CIRCLE_CLR = BLUE

	def __init__(self,
							marker_length: float,
							K: Tuple,
							D: Tuple,
							bbox: Optional[Tuple]=None,
							denoise: Optional[bool]=False,
							dict_type: Optional[str]="DICT_4X4_50",
							dict_yaml: Optional[str]="",
							print_stats: Optional[bool]=True,
							estimate_pattern: Optional[int]=aru.ARUCO_CCW_CENTER,
							solve_pnp_method: Optional[int]=cv2.SOLVEPNP_IPPE_SQUARE) -> None:
		
		self.aruco_dict = aru.getPredefinedDictionary(self.ARUCO_DICT[dict_type]) if dict_yaml == "" else self.loadArucoYaml(dict_yaml)
		self.estimate_params = aru.EstimateParameters()
		assert(estimate_pattern >= aru.ARUCO_CCW_CENTER and estimate_pattern <= aru.ARUCO_CW_TOP_LEFT_CORNER)
		self.estimate_params.pattern = estimate_pattern
		assert(solve_pnp_method >= cv2.SOLVEPNP_ITERATIVE and solve_pnp_method <= cv2.SOLVEPNP_MAX_COUNT)
		self.estimate_params.solvePnPMethod = solve_pnp_method
		self.params_lock = Lock()
		self.params = ReconfParams()

		self.denoise = denoise
		self.print_stats = print_stats
		self.marker_length = marker_length
		self.cmx = np.asanyarray(K).reshape(3,3)
		self.dist =  np.asanyarray(D)
		self.bbox = bbox
		self.t_total = 0
		self.it_total = 0
		self.printSettings()

	def printSettings(self) -> None:
		txt = f"Running Aruco Detector\n \
					denoise: {self.denoise},\n\
					print_stats: {self.print_stats},\n\
					marker_length: {self.marker_length},\n \
					estimate_pattern: {self.estimate_params.pattern},\n \
				    solvePnPMethod: {self.estimate_params.solvePnPMethod}"
		print()
		print(txt)
		print()

	def setDetectorParams(self, config, level):
		with self.params_lock:
			for k,v in config.groups.groups.image_denoising.parameters.items():
				self.params.__setattr__(k, v)
			for k,v in config.groups.groups.marker_detection.parameters.items():
				self.params.__setattr__(k, v)

		# with self.den_params_lock:
		# 	self.denoise = self.denoise_arg or config.denoise
		# 	self.den_params.h = config.h 
		# 	self.den_params.templateWindowSize = config.templateWindowSize 
		# 	self.den_params.searchWindowSize = config.searchWindowSize
		# with self.est_params_lock:
		# 	self.estimate_params.pattern = config.estimate_pattern
		# 	self.estimate_params.solvePnPMethod = config.solvePnPMethod
		# with self.det_params_lock:
		# 	self.det_params.adaptiveThreshWinSizeMin = config.adaptiveThreshWinSizeMin 
		# 	self.det_params.adaptiveThreshWinSizeMax = config.adaptiveThreshWinSizeMax 
		# 	self.det_params.adaptiveThreshWinSizeStep = config.adaptiveThreshWinSizeStep 
		# 	self.det_params.adaptiveThreshConstant = config.adaptiveThreshConstant 
		# 	self.det_params.minMarkerPerimeterRate = config.minMarkerPerimeterRate 
		# 	self.det_params.maxMarkerPerimeterRate = config.maxMarkerPerimeterRate 
		# 	self.det_params.polygonalApproxAccuracyRate = config.polygonalApproxAccuracyRate 
		# 	self.det_params.minCornerDistanceRate = config.minCornerDistanceRate 
		# 	self.det_params.minDistanceToBorder = config.minDistanceToBorder 
		# 	self.det_params.minMarkerDistanceRate = config.minMarkerDistanceRate 
		# 	self.det_params.minGroupDistance = config.minGroupDistance 
		# 	self.det_params.cornerRefinementMethod = config.cornerRefinementMethod 
		# 	self.det_params.cornerRefinementWinSize = config.cornerRefinementWinSize 
		# 	self.det_params.relativeCornerRefinmentWinSize = config.relativeCornerRefinmentWinSize 
		# 	self.det_params.cornerRefinementMaxIterations = config.cornerRefinementMaxIterations 
		# 	self.det_params.cornerRefinementMinAccuracy = config.cornerRefinementMinAccuracy 
		# 	self.det_params.markerBorderBits = config.markerBorderBits 
		# 	self.det_params.perspectiveRemovePixelPerCell = config.perspectiveRemovePixelPerCell 
		# 	self.det_params.perspectiveRemoveIgnoredMarginPerCell = config.perspectiveRemoveIgnoredMarginPerCell 
		# 	self.det_params.maxErroneousBitsInBorderRate = config.maxErroneousBitsInBorderRate 
		# 	self.det_params.minOtsuStdDev = config.minOtsuStdDev 
		# 	self.det_params.errorCorrectionRate = config.errorCorrectionRate 
		# 	self.det_params.aprilTagQuadDecimate = config.aprilTagQuadDecimate 
		# 	self.det_params.aprilTagQuadSigma = config.aprilTagQuadSigma 
		# 	self.det_params.aprilTagMinClusterPixels = config.aprilTagMinClusterPixels 
		# 	self.det_params.aprilTagMaxNmaxima = config.aprilTagMaxNmaxima 
		# 	self.det_params.aprilTagCriticalRad = config.aprilTagCriticalRad 
		# 	self.det_params.aprilTagMaxLineFitMse = config.aprilTagMaxLineFitMse 
		# 	self.det_params.aprilTagMinWhiteBlackDiff = config.aprilTagMinWhiteBlackDiff 
		# 	self.det_params.aprilTagDeglitch = config.aprilTagDeglitch 
		# 	self.det_params.detectInvertedMarker = config.detectInvertedMarker 
		# 	self.det_params.useAruco3Detection = config.useAruco3Detection 
		# 	self.det_params.minSideLengthCanonicalImg = config.minSideLengthCanonicalImg 
		# 	self.det_params.minMarkerLengthRatioOriginalImg = config.minMarkerLengthRatioOriginalImg 
		return config
		
	def setBBox(self, bbox: Tuple) -> None:
		self.bbox = bbox

	@classmethod
	def loadArucoYaml(self, filename: str) -> aru.Dictionary:
		"""Load yaml dict from 'datasets/aruco/' directory"""

		fl = os.path.join(DATA_PTH, filename)
		dct = aru.Dictionary()

		with open(fl, 'r') as fr:
			txt = yaml.safe_load(fr)
			num_markers = txt.pop('nmarkers')
			dct.markerSize = txt.pop('markersize')
			dct.maxCorrectionBits = txt.pop('maxCorrectionBits')
			# num rows as used in aruco_dict_utils.cpp
			nbytes =  int((dct.markerSize * dct.markerSize + 8 - 1) / 8)
			dct.bytesList = np.empty(shape=(num_markers, nbytes , 4), dtype = np.uint8)

			for key, val in txt.items():
				# convert str to int array
				bit_arr = np.array(list(map(int, val)), dtype=np.uint8)
				# size  
				bit_arr = bit_arr.reshape(dct.markerSize, dct.markerSize) # TODO: check if this works with n cols > 8
				compressed = aru.Dictionary.getByteListFromBits(bit_arr)
				# append to index
				idx = int(key.replace('marker_', ""))
				dct.bytesList[idx] = compressed

		return dct

	def _genSquarePoints(self, length: float) -> None:
		"""
			https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html
			cv::SOLVEPNP_IPPE_SQUARE Special case suitable for marker pose estimation. 
			Number of input points must be 4. Object points must be defined in the following order:
				point 0: [-squareLength / 2, squareLength / 2, 0]
				point 1: [ squareLength / 2, squareLength / 2, 0]
				point 2: [ squareLength / 2, -squareLength / 2, 0]
				point 3: [-squareLength / 2, -squareLength / 2, 0]
		"""

		self.obj_points = np.zeros((4, 3), dtype=np.float32)
		self.obj_points[0,:] = np.array([-length/2, length/2, 0])
		self.obj_points[1,:] = np.array([length/2, length/2, 0])
		self.obj_points[2,:] = np.array([length/2, -length/2, 0])
		self.obj_points[3,:] = np.array([-length/2, -length/2, 0])
		
	def projPoints(self, img, obj_points, rvec, tvec):
		"""Test the solvePnP by projecting the 3D Points to camera"""
		proj, _ = cv2.projectPoints(obj_points, rvec, tvec, self.cmx, self.dist)
		for p in proj:
			cv2.circle(img, (int(p[0][0]), int(p[0][1])), self.CIRCLE_SIZE, self.CIRCLE_CLR, -1)
		return img

	def _cropImage(self, img: np.ndarray) -> np.ndarray:
		return img[self.bbox[0][0]: self.bbox[0][1], self.bbox[1][0]: self.bbox[1][1]]
			
	def _printStats(self, tick: int) -> None:
		t_current = (cv2.getTickCount() - tick) / cv2.getTickFrequency()
		self.t_total += t_current
		self.it_total += 1
		if self.it_total % 100 == 0:
			print("Detection Time = {} ms (Mean = {} ms)".format(t_current * 1000, 1000 * self.t_total / self.it_total))

	def drawCamCS(self, img):
		thckns = 2
		arw_len = 100
		img_center =(int(img.shape[1]/2), int(img.shape[0]/2))
		cv2.arrowedLine(img, img_center, (img_center[0]+arw_len, img_center[1]), self.RED, thckns, cv2.LINE_AA)
		cv2.arrowedLine(img, img_center, (img_center[0], img_center[1]+arw_len), self.GREEN, thckns, cv2.LINE_AA)
		cv2.putText(img, 'X', (img_center[0]-10, img_center[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 1, self.BLUE, thckns, cv2.LINE_AA)
		cv2.circle(img, img_center, 5, self.CIRCLE_CLR, -1)

	def detMarkerPoses(self, img: np.ndarray) -> Tuple[dict, np.ndarray, np.ndarray]:	
		"""Detect Aruco marker from bgr image.
			@param img Input image with 'bgr' encoding
			@type np.ndarray
			@return Detected marker poses, marker detection image, processed image
		"""
		with self.params_lock:
			tick = cv2.getTickCount()
			# grasycale image
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			#  image denoising using Non-local Means Denoising algorithm
			if self.denoise or self.params.denoise:
				gray = cv2.fastNlMeansDenoising(gray, None, self.params.h, self.params.templateWindowSize, self.params.searchWindowSize)
			
			# detection
			marker_poses = {}
			(corners, ids, rejected) = aru.detectMarkers(gray, self.aruco_dict, parameters=self.params)
			if len(corners) > 0:
				ids = ids.flatten()
				zipped = zip(ids, corners)
				for id, corner in sorted(zipped):
					# estimate camera pose relative to the marker (image center T marker center) using the unit provided by obj_points
					(rvec, tvec, obj_points) = aru.estimatePoseSingleMarkers(corner, self.marker_length, self.cmx, self.dist, estimateParameters=self.estimate_params)
					marker_poses.update({id: {'rvec': rvec.flatten(), 'tvec': tvec.flatten(), 'points': obj_points, 'corners': corner}})
				if self.print_stats:
					self._printStats(tick)
			else:
				print("No marker found")
			
			# draw detection
			out_img = aru.drawDetectedMarkers(img, corners, ids)
			self.drawCamCS(out_img)
			for id, pose in marker_poses.items():
				out_img = cv2.drawFrameAxes(out_img, self.cmx, self.dist, pose['rvec'], pose['tvec'], self.marker_length*self.AXIS_LENGTH, self.AXIS_THICKNESS)
				out_img = self.projPoints(out_img, pose['points'], pose['rvec'], pose['tvec'])
			return marker_poses, out_img, gray
	
def saveArucoImgMatrix(aruco_dict: aru.Dictionary, show: bool=False, filename: str="aruco_matrix.png", bbits: int=1, num: int=0, scale: float=5, save_indiv: bool=False) -> None:
	"""Aligns the markers in a mxn matrix where m >= n .
		Saves markers with border as image in  'datasets/aruco/' directory

		@param aruco_dict:  Aruco dictionary.
		@type  aru.Dictionary
		@param show:  show image in cv2 window
		@type  bool
		@param filename:  image filename, saved inside 'datasets/aruco/' directory
		@type  str
		@param bbits:  border bits for the marker, adds to marker size
		@type  int
		@param num:  number of markers added to the image
		@type  int
		@param scale:  scaling applied to the marker image
		@type  float
		@param save_indiv Save markers to one file per marker
		@type bool
"""

	num_markers = aruco_dict.bytesList.shape[0] if num == 0 else num
	size = aruco_dict.markerSize + 2*bbits

	# matrix
	n = int(np.sqrt(num_markers))  # cols
	residual = num_markers - np.square(n)
	m = n + residual//n + (1 if residual%n > 0 else 0) # rows

	# entries
	v_border = np.ones((size, size))*255 # white vertical spacing = 1 *marker size
	v_border[:, 1] = v_border[:, -2] = [size*30] # add grey border lines
	h_border =  np.ones((size, 3* size))*255 if save_indiv else np.ones((size, (2*n*size) + size))*255 # white horizontal spacing = n * (marker size + v_border size) + v_border size
	h_border[1, :] = h_border[-2, :] = [size*30] # add horizontal grey border lines
	h_border[:, 1::2*size] = h_border[:, size-2::2*size] = [size*30]  # add vertical grey border lines
	rows = v_border.copy()
	matrix = h_border.copy()

	# mkdir
	if save_indiv:
		file_pth = filename.replace(".png", "")
		file_pth = os.path.join(DATA_PTH, file_pth)
		if not os.path.isdir(file_pth):
			os.mkdir(file_pth)

	# draw
	idx = 1
	print("Order of ", num_markers, " Aruco markers:")
	print("-"*num_markers)
	# cols
	for _ in range(m):
			# rows
			for _ in range(n):
				if idx >= num_markers+1 and save_indiv:
					print()
					break
				print(idx, " ", end="") if idx < num_markers+1 else print("pad ", end="")
				aruco_img = aru.generateImageMarker(aruco_dict, idx, size) if idx < num_markers+1 else v_border
				# append border
				rows = np.hstack((rows, aruco_img, v_border))
				if save_indiv:
					matrix = np.vstack((matrix, rows, h_border))
					matrix = cv2.resize(matrix, None, fx=scale, fy=scale, interpolation= cv2.INTER_AREA)
					cv2.imwrite(os.path.join(file_pth,  str(idx) + ".png"),  matrix)
					# reset
					rows = v_border.copy()
					matrix = h_border.copy()
				idx += 1
			if save_indiv:
				continue
			# stack horiz border
			matrix = np.vstack((matrix, rows, h_border))
			rows = v_border.copy()
			print()
	print("-"*num_markers)
	print("Scaling by factor ", scale)

	# resize and save
	if not save_indiv:
		matrix = cv2.resize(matrix, None, fx=scale, fy=scale, interpolation= cv2.INTER_AREA)
		cv2.imwrite(os.path.join(DATA_PTH, filename),  matrix)
	if show:
		cv2.imshow("aruco_img", matrix)
		if cv2.waitKey(0) == ord("q"):
			cv2.destroyAllWindows()

if __name__ == "__main__":
	# adict = ArucoDetector.loadArucoYaml("custom_matrix_4x4_32_consider_flipped.yml")
	# adict = aru.getPredefinedDictionary(aru.DICT_4X4_50)
	# saveArucoImgMatrix(adict, False, "matrix_4x4_32.png", num=32, save_indiv=False)
	a = ArucoDetector(0, np.eye(3), [])
