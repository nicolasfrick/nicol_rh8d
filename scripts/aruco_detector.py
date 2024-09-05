#!/usr/bin/env python3

import os
import yaml
import numpy as np
import cv2
import cv2.aruco as aru
from typing import Sequence, Optional, Tuple, Union, Any

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
		@param det_marker_center Location of the marker center (centered, top left corner)
		@type bool
		@param solvepnp_square Whether to use ippe square method for solvePnp method, else default value (iterative)
		@type bool
		@param print_stats Print detection statistics
		@type bool

		 Generate markers with opencv (custom build in opencv_build directory):
		.opencv_build/opencv/build/bin/example_cpp_aruco_dict_utils /
		 nicol_rh8d/datasets/aruco/custom_matrix_4x4_20.yml /
		-nMarkers=20 /
		-markerSize=4 /
		-r 
	"""

	# ArUco dictionaries built into the OpenCV library
	ARUCO_DICT = {
								"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
								"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
								"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
								"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
								"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
								"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
								"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
								"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
								"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
								"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
								"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
								"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
								"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
								"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
								"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
								"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
								"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL
	}

	AXIS_LENGTH = 1.5 # axis drawing
	AXIS_THICKNESS = 2 # axis drawing

	def __init__(self,
							marker_length: float,
							K: Tuple,
							D: Tuple,
							bbox: Optional[Tuple]=None,
							denoise: Optional[bool]=False,
							dict_type: Optional[str]="DICT_4X4_50",
							dict_yaml: Optional[str]="",
							det_marker_center: Optional[bool]=True,
							solvepnp_square: Optional[bool]=True,
							print_stats: Optional[bool]=True) -> None:
		
		self.aruco_dict = aru.getPredefinedDictionary(self.ARUCO_DICT[dict_type]) if dict_yaml == "" else self.loadArucoYaml(dict_yaml)
		self.det_params = aru.DetectorParameters()

		self.estimate_params = aru.EstimateParameters()
		self.estimate_params.pattern = aru.ARUCO_CCW_CENTER if det_marker_center else aru.ARUCO_CW_TOP_LEFT_CORNER
		if solvepnp_square:
			self.estimate_params.solvePnPMethod = cv2.SOLVEPNP_IPPE_SQUARE
		self.estimate_params.useExtrinsicGuess = False # not implemented here

		self.marker_length = marker_length
		self.genSquarePoints(marker_length)
		self.cmx = np.asanyarray(K).reshape(3,3)
		self.dist =  np.asanyarray(D)
		self.bbox = bbox
		self.denoise = denoise
		self.t_total = 0
		self.it_total = 0
		self.print_stats = print_stats
		
	def setBBox(self, bbox: Tuple) -> None:
		self.bbox = bbox

	@classmethod
	def loadArucoYaml(self, filename: str) -> aru.Dictionary:
		"""Load yaml dict from 'datasets/aruco/' directory"""

		fl = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets/aruco/" + filename)
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

	def genSquarePoints(self, length: float) -> None:
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
		
	# def drawPoints(self):
	# 	#Test the solvePnP by projecting the 3D Points to camera
	# 	projPoints = cv2.projectPoints(points_3D, rvecs, tvecs, K, distCoeffs)[0]
	# 	for p in points_2D:
	# 	cv2.circle(im, (int(p[0]), int(p[1])), 3, (0,255,0), -1)
	# 	for p in projPoints:
	# 	cv2.circle(im, (int(p[0][0]), int(p[0][1])), 3, (255,0,0), -1)

	def cropImage(self, img: np.ndarray) -> np.ndarray:
		return img[self.bbox[0][0]: self.bbox[0][1], self.bbox[1][0]: self.bbox[1][1]]
	
	def denoiseImage(self, img: np.ndarray) -> np.ndarray:
		return cv2.fastNlMeansDenoisingColored(img, None, 10,10,7,21)
	
	def printStats(self, tick: int) -> None:
		t_current = (cv2.getTickCount() - tick) / cv2.getTickFrequency()
		self.t_total += t_current
		self.it_total += 1
		if self.it_total % 100 == 0:
			print("Detection Time = {} ms (Mean = {} ms)".format(t_current * 1000, 1000 * self.t_total / self.it_total))

	def detMarkerPoses(self, img: np.ndarray) -> dict:
		tick = cv2.getTickCount()

		marker_poses = {}
		(corners, ids, rejected) = aru.detectMarkers(img, self.aruco_dict, parameters=self.det_params)
		if len(corners) > 0:
			ids = ids.flatten()
			zipped = zip(ids, corners)
			for id, corner in sorted(zipped):
				# estimate camera pose relative to the marker using the unit provided by obj_points
				(rvec, tvec, obj_points) = aru.estimatePoseSingleMarkers(corner, self.marker_length, self.cmx, self.dist, estimateParameters=self.estimate_params)
				marker_poses.update({id: {'rvec': rvec.flatten(), 'tvec': tvec.flatten(), 'points': obj_points}})

			if self.print_stats:
				self.printStats(tick)
		else:
			print("No marker found")
		
		# draw detection
		out_img = aru.drawDetectedMarkers(img, corners, ids)
		if marker_poses:
			for id, pose in marker_poses.items():
				out_img = cv2.drawFrameAxes(out_img, self.cmx, self.dist, pose['rvec'], pose['tvec'], self.marker_length*self.AXIS_LENGTH, self.AXIS_THICKNESS)
		else:
			print("No pose detected")
					
		return marker_poses, out_img
	
def saveArucoImgMatrix(aruco_dict: aru.Dictionary, show: bool=False, filename: str="aruco_matrix.png", bbits: int=1, num: int=0, scale: float=5) -> None:
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
	h_border = np.ones((size, (2*n*size) + size))*255 # white horizontal spacing = n * (marker size + v_border size) + v_border size
	h_border[1, :] = h_border[-2, :] = [size*30] # add horizontal grey border lines
	h_border[:, 1::2*size] = h_border[:, size-2::2*size] = [size*30]  # add vertical grey border lines
	rows = v_border.copy()
	matrix = h_border.copy()

	# draw
	idx = 1
	print("Order of ", num_markers, " Aruco markers:")
	print("-"*num_markers)
	# cols
	for _ in range(m):
			# rows
			for _ in range(n):
				print(idx, " ", end="") if idx < num_markers+1 else print("pad ", end="")
				aruco_img = aru.generateImageMarker(aruco_dict, idx, size) if idx < num_markers+1 else v_border
				rows = np.hstack((rows, aruco_img, v_border))
				idx += 1
			matrix = np.vstack((matrix, rows, h_border))
			rows = v_border.copy()
			print()
	print("-"*num_markers)
	print("Scaling by factor ", scale)

	# resize and save
	matrix = cv2.resize(matrix, None, fx=scale, fy=scale, interpolation= cv2.INTER_AREA)
	cv2.imwrite(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets/aruco/" + filename),  matrix)
	if show:
		cv2.imshow("aruco_img", matrix)
		if cv2.waitKey(0) == ord("q"):
			cv2.destroyAllWindows()

if __name__ == "__main__":
	# adict = ArucoDetector.loadArucoYaml("custom_matrix_4x4_32_consider_flipped.yml")
	# saveArucoImgMatrix(adict, False, "custom_matrix_4x4_32_consider_flipped.png", num=5)
	# ad = ArucoDetector(0.0, np.eye(3), [])
	rot = np.array([0,0,0], dtype=np.float32)
	trans = np.array([-1,2,3])
	r , _ = cv2.Rodrigues(rot)
	r = np.matrix(r).T
	print(r)
	print(r.shape)
	d = np.dot(-r, trans)
	print(d)
	print(d.shape)

# /camera_info
# header:
#   seq: 45
#   stamp:
#     secs: 1724064207
#     nsecs: 449997425
#   frame_id: "marker_realsense_color_optical_frame"
# height: 1080
# width: 1920
# distortion_model: "plumb_bob"

# The distortion parameters, size depending on the distortion model.
# For "plumb_bob", the 5 parameters are: (k1, k2, t1, t2, k3).
# D: [0.0, 0.0, 0.0, 0.0, 0.0]

# Intrinsic camera matrix for the raw (distorted) images.
#        [fx  0 cx]
# K = [ 0 fy cy]
#        [ 0  0  1]
# Projects 3D points in the camera coordinate frame to 2D pixel
# coordinates using the focal lengths (fx, fy) and principal point (cx, cy).
# K: [1396.5938720703125, 0.0, 944.5514526367188, 0.0, 1395.5264892578125, 547.0949096679688, 0.0, 0.0, 1.0]

# Rectification matrix (stereo cameras only)
# R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

# Projection/camera matrix
#        [fx'  0  cx' Tx]
# P = [ 0  fy' cy' Ty]
#        [ 0   0   1   0]
# By convention, this matrix specifies the intrinsic (camera) matrix
#  of the processed (rectified) image. That is, the left 3x3 portion
#  is the normal camera intrinsic matrix for the rectified image.
# It projects 3D points in the camera coordinate frame to 2D pixel
#  coordinates using the focal lengths (fx', fy') and principal point
#  (cx', cy') - these may differ from the values in K.
# For monocular cameras, Tx = Ty = 0. Normally, monocular cameras will
#  also have R = the identity and P[1:3,1:3] = K.
# Given a 3D point [X Y Z]', the projection (x, y) of the point onto
#  the rectified image is given by:
#  [u v w]' = P * [X Y Z 1]'
#         x = u / w
#         y = v / w
#  This holds for both images of a stereo pair.
# P: [1396.5938720703125, 0.0, 944.5514526367188, 0.0, 0.0, 1395.5264892578125, 547.0949096679688, 0.0, 0.0, 0.0, 1.0, 0.0]

# Binning refers here to any camera setting which combines rectangular
#  neighborhoods of pixels into larger "super-pixels." It reduces the
#  resolution of the output image to
#  (width / binning_x) x (height / binning_y).
# The default values binning_x = binning_y = 0 is considered the same
#  as binning_x = binning_y = 1 (no subsampling).
# binning_x: 0
# binning_y: 0

# Region of interest (subwindow of full camera resolution), given in
#  full resolution (unbinned) image coordinates. A particular ROI
#  always denotes the same window of pixels on the camera sensor,
#  regardless of binning settings.
# The default setting of roi (all values 0) is considered the same as
#  full resolution (roi.width = width, roi.height = height).
# roi:
#   x_offset: 0
#   y_offset: 0
#   height: 0
#   width: 0
#   do_rectify: False
# ---

# rs-enumerate-devices -c
# Device info:
# Name                          :     Intel RealSense D415
# Serial Number                 :     822512060411
# ...
#   Width:        1920
#   Height:       1080
#   PPX:          944.551452636719
#   PPY:          547.094909667969
#   Fx:           1396.59387207031
#   Fy:           1395.52648925781
#   Distortion:   Inverse Brown Conrady
#   Coeffs:       0       0       0       0       0
#   FOV (deg):    69 x 42.31
