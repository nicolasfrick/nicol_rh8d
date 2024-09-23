#!/usr/bin/env python3

import os
import cv2
import yaml
import dataclasses
import numpy as np
import cv2.aruco as aru
import dt_apriltags as apl
from threading import Lock
from typing import Optional, Tuple, Any, Callable
from pose_filter import *

DATA_PTH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  "datasets/aruco")

@dataclasses.dataclass(eq=False)
class ReconfParams():
	aruco_params = aru.DetectorParameters()
	estimate_params = aru.EstimateParameters()

class MarkerDetectorBase():
	"""Base clas for fiducial marker detection.

		@param marker_length:  Physical dimension (sidelength) of the marker in meters!
		@type  float
		@param K: camera intrinsics
		@type  tuple
		@param D: camera distorsion coefficients (plumb bob)
		@type  tuple
		@param f_ctrl Control loop frequency
		@type int
		@param bbox: bounding box ((x_min, x_max),(y_min,y_max)), if provided, image will be cropped
		@type  tuple
		@param print_stats Print detection statistics
		@type bool
		@param filter_type Determine whether the detections are filtered and the type of filter
		@type FilterTypes
	"""

	RED = (0,0,255)
	GREEN = (0,255,0)
	BLUE = (255,0,0)
	AXIS_LENGTH = 1.5
	AXIS_THICKNESS = 2
	CIRCLE_SIZE = 3
	CIRCLE_CLR = BLUE
	FONT_THCKNS = 2
	FONT_SCALE = 0.5

	def __init__(self,					
			  				K: Tuple,
							D: Tuple,
							marker_length: float,
							f_ctrl: Optional[int]=30,
							bbox: Optional[Tuple]=None,
							print_stats: Optional[bool]=True,
							filter_type: Optional[Union[FilterTypes, str]]=FilterTypes.NONE,
							) -> None:
		
		# params
		self.params_lock = Lock()
		self.params = ReconfParams()		
		# initialize params
		fl = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cfg/detector_params.yaml")
		with open(fl, 'r') as fr:
			self.params_config = yaml.safe_load(fr)
		for _, v in self.params_config.items():
			self.setDetectorParams(v, 0)

		self.print_stats = print_stats
		self.marker_length = marker_length
		self.filter_type = filter_type 
		self.f_ctrl = f_ctrl
		self.bbox = bbox
		self.filters = {}
		self.cmx = np.asanyarray(K).reshape(3,3)
		self.dist =  np.asanyarray(D)
		self.t_total = 0
		self.it_total = 0

	def setDetectorParams(self, config: Any, level: int) -> Any:
		with self.params_lock:
			for k,v in config.items():
				if k in self.params_config['pose_estimation']:
					self.params.estimate_params.__setattr__(k, v)
				elif k in self.params_config['aruco_detection']:
					self.params.aruco_params.__setattr__(k, v)
				else:
					self.params.__setattr__(k, v)
			self.params_change = True
			return config
	
	def setBBox(self, bbox: Tuple[float, float, float, float]) -> None:
		self.bbox = bbox

	def getFilteredTranslationById(self, id: int) -> Union[np.ndarray, None]:
		f = self.filters.get(id)
		if f is not None:
			return f.est_translation
		return None
	
	def getFilteredRotationEulerById(self, id: int) -> Union[np.ndarray, None]:
		f = self.filters.get(id)
		if f is not None:
			return f.est_rotation_as_euler
		return None
	
	def _genSquarePoints(self, length: float) -> np.ndarray:
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

	def _projPoints(self, img: cv2.typing.MatLike, obj_points: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> cv2.typing.MatLike:
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

	def _drawCamCS(self, img: cv2.typing.MatLike) -> None:
		thckns = 2
		arw_len = 100
		img_center =(int(img.shape[1]/2), int(img.shape[0]/2))
		cv2.arrowedLine(img, img_center, (img_center[0]+arw_len, img_center[1]), self.RED, thckns, cv2.LINE_AA)
		cv2.arrowedLine(img, img_center, (img_center[0], img_center[1]+arw_len), self.GREEN, thckns, cv2.LINE_AA)
		cv2.putText(img, 'X', (img_center[0]-10, img_center[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 1, self.BLUE, thckns, cv2.LINE_AA)
		cv2.circle(img, img_center, self.CIRCLE_SIZE, self.CIRCLE_CLR, -1)

	def _printSettings(self) -> None:
		raise NotImplementedError
	
	def _detectionRoutine(self):
		raise NotImplementedError
		
	def detMarkerPoses(self, img: cv2.typing.MatLike, subroutine: Callable[[cv2.typing.MatLike, cv2.typing.MatLike], Tuple[dict, cv2.typing.MatLike, cv2.typing.MatLike]]) -> Tuple[dict, cv2.typing.MatLike, cv2.typing.MatLike]:	
		"""Generic marker detection method."""
		with self.params_lock:
			tick = cv2.getTickCount()
			# grasycale image
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			# improve contrast
			if self.params.hist_equalization:
				gray = cv2.equalizeHist(gray)
			#  image denoising using Non-local Means Denoising algorithm
			if self.params.denoise:
				gray = cv2.fastNlMeansDenoising(gray, None, self.params.h, self.params.templateWindowSize, self.params.searchWindowSize)

			(marker_poses, out_img, gray) = subroutine(img, gray)

			self._drawCamCS(out_img)
			for _, pose in marker_poses.items():
				out_img = cv2.drawFrameAxes(out_img, self.cmx, self.dist, pose['rvec'], pose['tvec'], self.marker_length*self.AXIS_LENGTH, self.AXIS_THICKNESS)
				out_img = self._projPoints(out_img, pose['points'], pose['rvec'], pose['tvec'])

			if self.print_stats:
				self._printStats(tick)

			return marker_poses, img, gray
	
class AprilDetector(MarkerDetectorBase):
	"""
		Apriltag marker detector.
		
		@param marker_family Type of Apriltag marker
		@type str
													
		Max detection distance in meters = t /(2 * tan( (b* f * p) / (2 * r ) ) )
		t = size of your tag in meters
		b = the number of bits that span the width of the tag (excluding the white border for Apriltag 2). ex: 36h11 = 8, 25h9 = 6, standard41h12 = 9
		f = horizontal FOV of your camera
		r = horizontal resolution of you camera
		p = the number of pixels required to detect a bit. This is an adjustable constant. We recommend 5. Lowest number we recommend is 2 which is the Nyquist Frequency. 
				We recommend 5 to avoid some of the detection pitfalls mentioned above.
	"""
	def __init__(self,
			  				K: Tuple,
							D: Tuple,
							marker_length: float,
							f_ctrl: Optional[int]=30,
							bbox: Optional[Tuple]=None,
							print_stats: Optional[bool]=True,
							filter_type: Optional[Union[FilterTypes, str]]=FilterTypes.NONE,
							marker_family: Optional[str]='tag16h5',
							) -> None:
		
		super().__init__(K=K, 
				   						D=D, 
										marker_length=marker_length, 
										f_ctrl=f_ctrl, 
										bbox=bbox, 
										print_stats=print_stats, 
										filter_type=filter_type,
										)
		self.det = apl.Detector(families=marker_family, 
													nthreads=self.params.nthreads,
													quad_decimate=self.params.quad_decimate,   
													quad_sigma=self.params.quad_sigma,
													refine_edges=self.params.refine_edges, 
													decode_sharpening=self.params.decode_sharpening,
													debug=False,
													)
		self.params_change = False
		self.camera_params = (K[0], K[4], K[2], K[5])
		self._genSquarePoints(marker_length)
		self._printSettings()

	def _printSettings(self) -> None:
		txt =   f"Running Apriltag Detector with settings:\n"
		txt += f"Camera params fx: {self.camera_params[0]}, fy: {self.camera_params[1]}, cx: {self.camera_params[2]}, cy: {self.camera_params[3]}\n"
		txt += f"Distorsion: {self.dist}\n"
		txt += f"print_stats: {self.print_stats},\n"
		txt += f"marker_length: {self.marker_length},\n"
		txt += f"control frequency: {self.f_ctrl},\n"
		txt += f"filter type: {self.filter_type},\n"
		for attr in dir(self.params):
			if not attr.startswith('__'):
				txt += f"{attr}: {self.params.__getattribute__(attr)},\n"
		print(txt)
		print()

	def validateDetection(self, detection: apl.Detection) -> bool:
		corners = detection.corners.astype(int)
		marker_width = np.linalg.norm(corners[0] - corners[1])
		marker_height = np.linalg.norm(corners[1] - corners[2])
		return detection.decision_margin > self.params.decision_margin \
						and detection.hamming < self.params.max_hamming \
							and marker_width > self.params.min_marker_width \
								and marker_height > self.params.min_marker_height

	def drawMarkers(self, detection: apl.Detection, img: cv2.typing.MatLike) -> None:
		corners = detection.corners.astype(int)
		cv2.putText(img, str(detection.tag_id), tuple(corners[0]), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.RED, self.FONT_THCKNS)
		for i in range(4):
			cv2.line(img, tuple(corners[i]), tuple(corners[(i + 1) % 4]), self.GREEN, self.AXIS_THICKNESS)

	def _adaptParams(self):
		self.det.tag_detector_ptr.contents.nthreads = int(self.params.nthreads)
		self.det.tag_detector_ptr.contents.quad_decimate = float(self.params.quad_decimate)
		self.det.tag_detector_ptr.contents.quad_sigma = float(self.params.quad_sigma)
		self.det.tag_detector_ptr.contents.refine_edges = int(self.params.refine_edges)
		self.det.tag_detector_ptr.contents.decode_sharpening = int(self.params.decode_sharpening)
		self.params_change = False
		
	def _detectionRoutine(self, img: cv2.typing.MatLike, gray: cv2.typing.MatLike)  -> Tuple[dict, cv2.typing.MatLike, cv2.typing.MatLike]:	
		marker_poses = {}
		if self.params_change:
			self._adaptParams()
		detections = self.det.detect(gray, estimate_tag_pose=True, camera_params=self.camera_params, tag_size=self.marker_length)
		if len(detections) > 0:
			for detection in detections:
				# proc detections
				if self.validateDetection(detection):
					id = detection.tag_id
					tvec = detection.pose_t.flatten()
					# filtering
					if id in self.filters.keys():
						self.filters[id].updateFilter(PoseFilterBase.poseToMeasurement(tvec=tvec, rot_mat=detection.pose_R))
					else:
						self.filters.update( {id: createFilter(self.filter_type, PoseFilterBase.poseToMeasurement(tvec=tvec, rot_mat=detection.pose_R), self.f_ctrl)} )
					# result
					marker_poses.update({id: {'rvec': cv2.Rodrigues(detection.pose_R)[0], 
							   											'rot_mat': detection.pose_R,
							   											'tvec': tvec, 
																		'points': self.obj_points, 
																		'corners': detection.corners, 
																		'ftrans': self.getFilteredTranslationById(id), 
																		'frot': self.getFilteredRotationEulerById(id),
																		'homography': detection.homography,
																		'center': detection.center,
																		'pose_err': detection.pose_err}})
					self.drawMarkers(detection, img)
				else:
					self.drawMarkers(detection, gray)

		return marker_poses, img, gray

	def detMarkerPoses(self, img: np.ndarray) -> Tuple[dict, np.ndarray]:
		"""Detect Apriltag marker in bgr image.
			@param img Input image with 'bgr' encoding
			@type np.ndarray
			@return Detected marker poses, marker detection image, processed image
		"""
		return super().detMarkerPoses(img, self._detectionRoutine)

class ArucoDetector(MarkerDetectorBase):
	"""Detect Aruco marker from an image.

		@param dict_type:  type of the aruco dictionary out of ARUCO_DICT, loaded if no yaml path provided
		@type  str
		@param dict_yaml:  path to a dictionary yaml file in 'datasets/aruco/' directory used in favour of dict_type
		@type  str
		@param 	estimate_pattern Enum to set Aruco estimate pattern. One of [ARUCO_CCW_CENTER, ARUCO_CW_TOP_LEFT_CORNER]
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

	def __init__(self,
			  				K: Tuple,
							D: Tuple,
							marker_length: float,
							f_ctrl: Optional[int]=30,
							bbox: Optional[Tuple]=None,
							print_stats: Optional[bool]=True,
							filter_type: Optional[Union[FilterTypes, str]]=FilterTypes.NONE,
							dict_yaml: Optional[str]="",
							dict_type: Optional[str]="DICT_4X4_50",
							) -> None:
		
		super().__init__(K=K, 
				   						D=D, 
										marker_length=marker_length, 
										f_ctrl=f_ctrl, 
										bbox=bbox, 
										print_stats=print_stats, 
										filter_type=filter_type,
										)
		self.aruco_dict = aru.getPredefinedDictionary(self.ARUCO_DICT[dict_type]) if dict_yaml == "" else self.loadArucoYaml(dict_yaml)
		self._printSettings()

	def _printSettings(self) -> None:
		txt =   f"Running Aruco Detector with settings:\n"
		txt += f"Camera matrix: {self.cmx}\n"
		txt += f"Camera distorsion: {self.dist}\n"
		txt += f"print_stats: {self.print_stats},\n"
		txt += f"marker_length: {self.marker_length},\n"
		txt += f"hist_equalization: {self.params.hist_equalization}\n"
		txt += f"denoise: {self.params.denoise}\n"
		txt += f"h: {self.params.h}\n"
		txt += f"templateWindowSize: {self.params.templateWindowSize}\n"
		txt += f"searchWindowSize: {self.params.searchWindowSize}\n"
		for attr in dir(self.params.aruco_params):
			if not attr.startswith('__'):
				txt += f"{attr}: {self.params.aruco_params.__getattribute__(attr)},\n"
		for attr in dir(self.params.estimate_params):
			if not attr.startswith('__'):
				txt += f"{attr}: {self.params.estimate_params.__getattribute__(attr)},\n"
		print(txt)
		print()

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
	
	def _detectionRoutine(self,  img: cv2.typing.MatLike, gray: cv2.typing.MatLike)  -> Tuple[dict, cv2.typing.MatLike, cv2.typing.MatLike]:	
		# detection
		marker_poses = {}
		(corners, ids, rejected) = aru.detectMarkers(gray, self.aruco_dict, parameters=self.params.aruco_params)
		if len(corners) > 0:
			ids = ids.flatten()
			zipped = zip(ids, corners)
			for id, corner in sorted(zipped):
				# estimate camera pose relative to the marker (image center T marker center) using the unit provided by obj_points
				(rvec, tvec, obj_points) = aru.estimatePoseSingleMarkers(corner, self.marker_length, self.cmx, self.dist, estimateParameters=self.params.estimate_params)
				tvec = tvec.flatten()
				rvec = rvec.flatten()
				# filtering
				if id in self.filters.keys():
					self.filters[id].updateFilter(PoseFilterBase.poseToMeasurement(tvec=tvec, rvec=rvec))
				else:
					self.filters.update( {id: createFilter(self.filter_type, PoseFilterBase.poseToMeasurement(tvec=tvec, rvec=rvec), self.f_ctrl)} )
				# results
				marker_poses.update({id: {'rvec': rvec, 
							   										'rot_mat': cv2.Rodrigues(rvec)[0],
							   										'tvec': tvec, 
																	'points': obj_points, 
																	'corners': corner.reshape((4,2)), 
																	'ftrans': self.getFilteredTranslationById(id), 
																	'frot': self.getFilteredRotationEulerById(id),
																	'homography': None,
																	'center': None,
																	'pose_err': None}})

		# draw detection
		out_img = aru.drawDetectedMarkers(img, corners, ids)
		gray = aru.drawDetectedMarkers(gray, rejected)
		return marker_poses, out_img, gray

	def detMarkerPoses(self, img: np.ndarray) -> Tuple[dict, np.ndarray, np.ndarray]:	
		"""Detect Aruco marker in bgr image.
			@param img Input image with 'bgr' encoding
			@type np.ndarray
			@return Detected marker poses, marker detection image, processed image
		"""
		return super().detMarkerPoses(img, self._detectionRoutine)
	
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
	adict = ArucoDetector.loadArucoYaml("custom_matrix_4x4_32_consider_flipped.yml")
	saveArucoImgMatrix(adict, False, "matrix_4x4_32.png", num=32, save_indiv=False)
