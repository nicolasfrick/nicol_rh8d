#!/usr/bin/env python3

import os
import sys
import cv2
import yaml
import time
import rospy
import tf2_ros
import cv_bridge
import threading
import numpy as np
import sensor_msgs.msg
from collections import deque
from datetime import datetime
import dynamic_reconfigure.server
from sensor_msgs.msg import Image
from typing import Optional, Any, Tuple
from scipy.optimize import least_squares
from sensor_msgs.msg import JointState
from realsense2_camera.msg import Extrinsics
from scipy.spatial.transform import Rotation as R
from message_filters import Subscriber, ApproximateTimeSynchronizer
from util import *
from move import *
from plot_record import *
from pose_filter import *
from qdec_serial import *
from rh8d_serial import *
from nicol_rh8d.cfg import ArucoDetectorConfig
from marker_detector import ArucoDetector, AprilDetector
np.set_printoptions(threshold=sys.maxsize, suppress=True)

dt_now = datetime.now()
dt_now = '' # dt_now.strftime("%H_%M_%S")

# data records
DATA_PTH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets/detection')
QDEC_DET_PTH = os.path.join(DATA_PTH, 'qdec/detection_' + dt_now + '.json')
KEYPT_DET_PTH = os.path.join(DATA_PTH, 'keypoint/detection_' + dt_now + '.json')
# image records
JPG_QUALITY = 60
REC_DIR = os.path.join(os.path.expanduser('~'), 'rh8d_dataset')
if not os.path.exists(REC_DIR):
	os.mkdir(REC_DIR)
print("Writing images to", REC_DIR)

QDEC_REC_DIR = os.path.join(REC_DIR, 'qdec')
QDEC_ORIG_REC_DIR = os.path.join(QDEC_REC_DIR, 'orig')
QDEC_DET_REC_DIR = os.path.join(QDEC_REC_DIR, 'det')
if not os.path.exists(QDEC_REC_DIR):
	os.mkdir(QDEC_REC_DIR)
if not os.path.exists(QDEC_ORIG_REC_DIR):
	os.mkdir(QDEC_ORIG_REC_DIR)
if not os.path.exists(QDEC_DET_REC_DIR):
	os.mkdir(QDEC_DET_REC_DIR)

KEYPT_REC_DIR = os.path.join(REC_DIR, 'keypoint_'+ dt_now)
KEYPT_ORIG_REC_DIR = os.path.join(KEYPT_REC_DIR, 'orig')
KEYPT_DET_REC_DIR = os.path.join(KEYPT_REC_DIR, 'det')
KEYPT_R_EYE_REC_DIR = os.path.join(KEYPT_REC_DIR, 'right_eye')
KEYPT_L_EYE_REC_DIR = os.path.join(KEYPT_REC_DIR, 'left_eye')
KEYPT_TOP_CAM_REC_DIR = os.path.join(KEYPT_REC_DIR, 'top_cam')
KEYPT_HEAD_CAM_REC_DIR = os.path.join(KEYPT_REC_DIR, 'head_cam')
if not os.path.exists(KEYPT_REC_DIR):
	os.mkdir(KEYPT_REC_DIR)
if not os.path.exists(KEYPT_ORIG_REC_DIR):
	os.mkdir(KEYPT_ORIG_REC_DIR)
if not os.path.exists(KEYPT_DET_REC_DIR):
	os.mkdir(KEYPT_DET_REC_DIR)
if not os.path.exists(KEYPT_R_EYE_REC_DIR):
	os.mkdir(KEYPT_R_EYE_REC_DIR)
if not os.path.exists(KEYPT_L_EYE_REC_DIR):
	os.mkdir(KEYPT_L_EYE_REC_DIR)
if not os.path.exists(KEYPT_TOP_CAM_REC_DIR):
	os.mkdir(KEYPT_TOP_CAM_REC_DIR)
if not os.path.exists(KEYPT_HEAD_CAM_REC_DIR):
	os.mkdir(KEYPT_HEAD_CAM_REC_DIR)

class DetectBase():
	"""
		@param camera_ns Camera namespace precceding 'image_raw' and 'camera_info'
		@type str
		@param vis Show detection images
		@type bool
		@param filter_type
		@type str
		@param filter_iters
		@type int
		@param f_ctrl
		@type float
		@param use_aruco
		@type bool
		@param plt_id
		@type int
		@param refine_pose
		@type bool
		@param flip_outliers
		@type bool
		@param fps
		@type float
		@param test
		@type bool
		@param cv_window
		@type bool

	"""

	FONT_THCKNS = 3
	FONT_SCALE = 1
	FONT_CLR =  (0,0,0)
	TXT_OFFSET = 30
	
	def __init__(self,
			  	marker_length: float=0.010,
				camera_ns: Optional[str]='',
				vis: Optional[bool]=True,
				cv_window:Optional[bool]=True,
				use_reconfigure: Optional[bool]=False,
				filter_type: Optional[str]='none',
				filter_iters: Optional[int]=10,
				f_ctrl: Optional[int]=30,
				use_aruco: Optional[bool]=False,
				plt_id: Optional[int]=-1,
				test: Optional[bool]=False,
				refine_pose: Optional[bool]=False,
				flip_outliers: Optional[bool]=False,
				fps: Optional[float]=30.0,
				) -> None:
		
		self.vis = vis
		self.cv_window = cv_window
		self.test = test
		self.plt_id = plt_id
		self.f_loop = f_ctrl
		self.use_aruco = use_aruco
		self.filter_type = filter_type
		self.refine_pose = refine_pose
		self.flip_outliers = flip_outliers
		self.filter_iters = filter_iters if (filter_type != 'none' and filter_iters > 0) else 1
		self.frame_cnt = 0

		# dummies
		self.rgb_info= sensor_msgs.msg.CameraInfo()
		self.rgb_info.K = np.array([1396.5938720703125, 0.0, 944.5514526367188, 0.0, 1395.5264892578125, 547.0949096679688, 0.0, 0.0, 1.0], dtype=np.float64)
		self.rgb_info.D = np.array([0,0,0,0,0], dtype=np.float64)
		self.img = cv2.imread(os.path.join(DATA_PTH, 'test_img.jpg'), cv2.IMREAD_COLOR)
		# init ros
		if not test:
			self.img = None
			self.bridge = cv_bridge.CvBridge()
			self.img_topic = camera_ns + '/image_raw'
			rospy.loginfo("Waiting for camera_info from %s", camera_ns + '/camera_info')
			self.rgb_info = rospy.wait_for_message(camera_ns + '/camera_info', sensor_msgs.msg.CameraInfo, 5)
			print("Camera height:", self.rgb_info.height, "width:", self.rgb_info.width)

		# init detector
		if use_aruco:
			self.det = ArucoDetector(marker_length=marker_length, 
									K=self.rgb_info.K, 
									D=self.rgb_info.D,
									dt=1/fps,
									invert_pose=False,
									filter_type=filter_type)
		else:
			self.det = AprilDetector(marker_length=marker_length, 
									K=self.rgb_info.K, 
									D=self.rgb_info.D,
									dt=1/fps,
									invert_pose=False,
									filter_type=filter_type)
			
		# init vis	
		if vis and cv_window:
			cv2.namedWindow("Processed", cv2.WINDOW_NORMAL)
			cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
			if plt_id > -1:
				cv2.namedWindow("Plot", cv2.WINDOW_NORMAL)
				setPosePlot()
	
		# init dynamic reconfigure
		if use_reconfigure:
			print("Using reconfigure server")
			self.det_config_server = dynamic_reconfigure.server.Server(ArucoDetectorConfig, self.det.setDetectorParams)

	def flipOutliers(self, detections: dict, tolerance: float=0.5) -> bool:
		"""Check if all Z axes are oriented similarly and 
			  flip orientation for outliers. 
		"""
		marker_ids = list(detections.keys())
		# extract filtered rotations
		rotations = [getRotation(marker_det['frot'], RotTypes.EULER, RotTypes.MAT)  for marker_det in detections.values()]
		# find outliers
		outliers, axis_avg = findAxisOrientOutliers(rotations, tolerance=tolerance, axis='z')

		# correct outliers
		fixed = []
		for idx in outliers:
			mid = marker_ids[idx]
			print(f"Marker {mid} orientation is likely flipped ...", end=" ")
			# find possible PnP solutions
			num_sols, rvecs, tvecs, repr_err = cv2.solvePnPGeneric(detections[mid]['points'], 
														  																np.array(detections[mid]['corners'], dtype=np.float32), 
																														self.det.cmx, 
																														self.det.dist,
																														getRotation(rotations[idx], RotTypes.MAT, RotTypes.RVEC), 
																														detections[mid]['tvec'], 
																														flags=cv2.SOLVEPNP_IPPE_SQUARE)
			# find solution that matches the average
			for rvec, tvec in zip(rvecs, tvecs):
				# normalize rotation
				mat = getRotation(rvec.flatten(), RotTypes.RVEC, RotTypes.MAT)
				axs = mat[:, 2] / np.linalg.norm(mat[:, 2])
				# check angular distance to average
				if abs( np.dot(axs, axis_avg) ) > tolerance:
					# set other rot
					detections[mid]['rot_mat'] = mat
					detections[mid]['rvec'] = rvec.flatten()
					detections[mid]['frot'] = getRotation(mat, RotTypes.MAT, RotTypes.EULER)
					# set other trans
					detections[mid]['ftrans'] = tvec.flatten()
					print("fixed")
					fixed.append(idx)
				
		return all([o in fixed for o in outliers])

	def refineDetection(self, detections: dict) -> None:
		"""Minimizes the projection error with respect to the rotation and the translation vectors, 
			 according to a Levenberg-Marquardt iterative minimization process.
		"""
		for id in detections.keys():
			det = detections[id]
			(tvec, rvec) = refinePose(tvec=det['ftrans'], 
						   				rvec=getRotation(det['frot'], RotTypes.EULER, RotTypes.RVEC), 
										corners=det['corners'], 
										obj_points=det['points'], 
										cmx=self.det.cmx, 
										dist=self.det.dist,
										)
			detections[id]['ftrans'] = tvec
			detections[id]['frot'] = getRotation(rvec, RotTypes.RVEC, RotTypes.EULER)
			detections[id]['rot_mat'] = getRotation(rvec, RotTypes.RVEC, RotTypes.MAT)
			detections[id]['rvec'] = rvec
	
	def preProcImage(self, vis: bool=True) -> Tuple[dict, cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike]:
		""" Put num filter_iters images into
			fresh detection filter and get last
			detection.
		"""
		# test img
		raw_img = self.img
		self.det.resetFilters()
		for i in range(self.filter_iters):
			self.frame_cnt += 1
			# real image
			if not self.test:
				rgb = rospy.wait_for_message(self.img_topic, Image)
				raw_img = self.bridge.imgmsg_to_cv2(rgb, 'bgr8')
			(marker_det, det_img, proc_img) = self.det.detMarkerPoses(raw_img.copy(), vis if (i >= self.filter_iters-1 and self.vis) else False)

		# align rotations by consens
		if self.flip_outliers:
			if not self.flipOutliers(marker_det):
				beep()
		# improve detection
		if self.refine_pose:
			self.refineDetection(marker_det)

		return marker_det, det_img, proc_img, raw_img
	
	def runDebug(self) -> None:
		rate = rospy.Rate(self.f_loop)
		try:
			while not rospy.is_shutdown():
				(marker_det, det_img, proc_img, img) = self.preProcImage()
				if self.vis:
					# frame counter
					cv2.putText(det_img, str(self.frame_cnt), (det_img.shape[1]-100, 50), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.FONT_CLR, self.FONT_THCKNS, cv2.LINE_AA)
					if self.cv_window:
						cv2.imshow('Processed', proc_img)
						cv2.imshow('Detection', det_img)
						if cv2.waitKey(1) == ord("q"):
							break
				try:
					rate.sleep()
				except:
					pass
		except Exception as e:
			rospy.logerr(e)
		finally:
			cv2.destroyAllWindows()

	def detectionRoutine(self, arg: Any) -> Union[Tuple[dict, cv2.typing.MatLike, cv2.typing.MatLike, int], dict]:
		raise NotImplementedError
	
	def run(self) -> None:
		raise NotImplementedError

class KeypointDetect(DetectBase):
	"""
		Detect keypoints from marker poses.

		@param rh8d_port
		@type str
		@param rh8d_baud
		@type int
		@param step_div
		@type int
		@param epochs
		@type int
		@param use_tf
		@type bool
		@param attached
		@type bool
		@param save_imgs
		@type bool
		@param use_eye_cameras
		@type bool
		@param use_top_camera
		@type bool
		@param depth_enabled
		@type bool
		@param top_camera_name
		@type str
		@param left_eye_camera_name
		@type str
		@param right_eye_camera_name
		@type str
		@param joint_state_topic
		@type str
		@param use_head_camera
		@type bool
		@param head_camera_name
		@type str
		@param waypoint_set
		@type str
		@param waypoint_start_idx
		@type int

	"""

	FIRST_LINE_INTERPOL_FACTOR = 0.5 # arc points interpolation
	SEC_LINE_INTERPOL_FACTOR = 0.85 # arc points interpolation
	ARC_SHIFT = 10 # ellipse param
	SAGITTA = 20 # radius size
	AXES_LEN = 0.015 # meter
	X_AXIS = np.array([[-AXES_LEN,0,0], [AXES_LEN, 0, 0]], dtype=np.float32)
	UNIT_AXIS_Y = np.array([0, 1, 0], dtype=np.float32)
	ELIPSE_COLOR = (255,255,255)
	ELIPSE_THKNS = 2
	X_EL_TXT_OFFSET = 0.95
	Y_EL_TXT_OFFSET = 1

	TF_TARGET_FRAME = 'world'
	RH8D_SOURCE_FRAME = 'r_forearm' # dynamic
	RH8D_TCP_SOURCE_FRAME = 'r_laser'
	MARKER_CAM_SOURCE_FRAME = 'markers_camera_base_link' # TODO: use camera base frame
	MARKER_CAM_OPTICAL_SOURCE_FRAME = 'markers_camera_color_optical_frame'
	TOP_RS_SOURCE_FRAME = 'realsense_link'
	TOP_RS_OPTICAL_SOURCE_FRAME = 'realsense_color_optical_frame'
	HEAD_RS_SOURCE_FRAME = 'head_realsense_link' # dynamic
	LEFT_EYE_SOURCE_FRAME = 'left_eye_cam' # dynamic
	RIGHT_EYE_SOURCE_FRAME = 'right_eye_cam' # dynamic

	JS_COLS = ['name', 'position', 'velocity', 'effort', 'timestamp']
	TF_COLS = ['frame_id', 'child_frame_id', 'trans', 'quat', 'timestamp']

	def __init__(self,
			  	marker_length: float=0.010,
				camera_ns: Optional[str]='',
				vis :Optional[bool]=True,
				use_reconfigure: Optional[bool]=False,
				filter_type: Optional[str]='none',
				filter_iters: Optional[int]=10,
				f_ctrl: Optional[int]=30,
				use_aruco: Optional[bool]=False,
				plt_id: Optional[int]=-1,
				rh8d_port: Optional[str]="/dev/ttyUSB1",
				rh8d_baud: Optional[int]=1000000,
				step_div: Optional[int]=100,
				epochs: Optional[int]=10,
				use_tf: Optional[bool]=False,
				test: Optional[bool]=False,
				flip_outliers: Optional[bool]=True,
				refine_pose: Optional[bool]=True,
				fps: Optional[float]=30.0,
				save_imgs: Optional[bool]=True,
				attached: Optional[bool]=False,
				use_eye_cameras: Optional[bool]=False,
				use_top_camera: Optional[bool]=False,
				use_head_camera: Optional[bool]=False,
				depth_enabled: Optional[bool]=False,
				top_camera_name: Optional[str]='top_camera',
				left_eye_camera_name: Optional[str]='left_eye_camera',
				right_eye_camera_name: Optional[str]='right_eye_camera',
				head_camera_name: Optional[str]='head_camera',
				joint_state_topic: Optional[str]='joint_states',
				waypoint_set: Optional[str]='waypoints.json',
				waypoint_start_idx: Optional[int]=0,
				) -> None:
		
		super().__init__(marker_length=marker_length,
						camera_ns=camera_ns,
						use_reconfigure=use_reconfigure,
						refine_pose=refine_pose,
						flip_outliers=flip_outliers,
						filter_type=filter_type,
						filter_iters=filter_iters,
						use_aruco=use_aruco,
						plt_id=plt_id,
						f_ctrl=f_ctrl,
						test=test,
						vis=vis,
						fps=fps,
						cv_window=not attached,
						)

		self.epochs = epochs
		self.use_tf = use_tf
		self.attached = attached
		self.step_div = step_div
		self.save_imgs = save_imgs
		self.use_eye_cameras = use_eye_cameras
		self.use_top_camera = use_top_camera
		self.use_head_camera = use_head_camera
		self.depth_enabled = depth_enabled
		self.camera_ns = camera_ns
		self.top_camera_name = top_camera_name
		self.head_camera_name = head_camera_name
		self.left_eye_camera_name = left_eye_camera_name
		self.right_eye_camera_name = right_eye_camera_name
		self.joint_state_topic = joint_state_topic
		self.record_cnt = 0
		self.subs = []

		# message buffers
		self.det_img_buffer = deque(maxlen=self.filter_iters)
		self.depth_img_buffer = deque(maxlen=1)
		self.top_img_buffer = deque(maxlen=1)
		self.top_depth_img_buffer = deque(maxlen=1)
		self.head_img_buffer = deque(maxlen=1)
		self.head_depth_img_buffer = deque(maxlen=1)
		self.left_img_buffer = deque(maxlen=1)
		self.right_img_buffer = deque(maxlen=1)
		self.joint_state_buffer = deque(maxlen=1)
		self.buf_lock = threading.Lock()

		# data frames
		self.js_df = pd.DataFrame(columns=self.JS_COLS)
		self.rh8d_tf_df = pd.DataFrame(columns=self.TF_COLS)
		self.head_rs_tf_df = pd.DataFrame(columns=self.TF_COLS)
		self.left_eye_tf_df = pd.DataFrame(columns=self.TF_COLS)
		self.right_eye_tf_df = pd.DataFrame(columns=self.TF_COLS)

		# init ros
		if use_tf:
			self.buf = tf2_ros.Buffer()
			self.listener = tf2_ros.TransformListener(self.buf)

		# subscribe img topics
		if not test and attached:
			self.initSubscriber()

		# init controller
		if attached:
			self.rh8d_ctrl = MoveRobot()
		else:
			self.rh8d_ctrl = RH8DSerial(rh8d_port, rh8d_baud) if not self.test else RH8DSerialStub()

		# init vis
		if vis and not attached:
			cv2.namedWindow("Output", cv2.WINDOW_NORMAL)

		# load hand marker ids
		self.fl = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cfg/hand_ids.yaml")
		with open(self.fl, 'r') as fr:
			self.hand_ids = yaml.safe_load(fr)

		self.group_ids = self.hand_ids['names']['index'] # joint names
		self.target_ids = self.hand_ids['ids']['index'] # target marker ids
		self.df = pd.DataFrame(columns=self.group_ids) # dataset
		self.start_angles = {}

		# load waypoints
		fl = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets/detection/keypoint", waypoint_set)
		self.waypoint_df = pd.read_json(fl, orient='index')
		if waypoint_start_idx > 0:
			self.waypoint_df = self.waypoint_df.iloc[waypoint_start_idx :]
			self.record_cnt = waypoint_start_idx

	def saveCamInfo(self, info: sensor_msgs.msg.CameraInfo, fn: str, extr: Extrinsics=None, tf: dict=None) -> None:
		if not self.save_imgs:
			return
		
		info_dict = {"frame_id": info.header.frame_id,
			   		 "width": info.width,
					 "height": info.height,
					 "distortion_model": info.distortion_model,
					 "D": list(info.D),
					 "K": list(info.K),					 
					 "R": list(info.R),
					 "P": list(info.P),
					 "binning_x": info.binning_x,
					 "binning_y": info.binning_y,
					 "roi": {
						"x_offset": info.roi.x_offset,
						"y_offset": info.roi.y_offset,
						"height": info.roi.height,
						"width": info.roi.width,
						"do_rectify": info.roi.do_rectify,
					 	}, 
					}

		if extr is not None:
			extr_dict = {extr.header.frame_id: {'rotation': list(extr.rotation), 'translation': list(extr.translation)}}
			info_dict.update(extr_dict)

		if tf is not None:
			info_dict.update(tf)

		with open(fn, "w") as fw:
			yaml.dump(info_dict, fw, default_flow_style=None, sort_keys=False)

	def initSubscriber(self) -> None:
		rospy.wait_for_message(self.joint_state_topic, JointState, 5)
		self.joint_states_sub = Subscriber(self.joint_state_topic, JointState)
		self.subs.append(self.joint_states_sub)

		self.det_img_sub = Subscriber(self.img_topic, Image)
		self.subs.append(self.det_img_sub)
		# save camera info
		tf_dict = {}
		if self.use_tf:
			tf_dict = self.lookupTF(rospy.Time(0), self.MARKER_CAM_SOURCE_FRAME, self.MARKER_CAM_OPTICAL_SOURCE_FRAME)
		self.saveCamInfo(self.rgb_info, os.path.join(KEYPT_DET_REC_DIR, "rs_d415_info_rgb.yaml"), tf={'optical_extrinsics': tf_dict})

		if self.depth_enabled:
			self.depth_topic = self.camera_ns.replace('color', 'depth')
			depth_img_topic = self.depth_topic + '/image_rect_raw'	
			self.depth_img_sub = Subscriber(depth_img_topic,  Image)
			self.subs.append(self.depth_img_sub)
			# save camera info
			print("Waiting for camera_info from", self.depth_topic + '/camera_info')
			depth_info = rospy.wait_for_message(self.depth_topic + '/camera_info', sensor_msgs.msg.CameraInfo, 5)
			extr_info = rospy.wait_for_message(self.camera_ns.replace('/color', '') + '/extrinsics/depth_to_color', Extrinsics, 5)
			self.saveCamInfo(depth_info, os.path.join(KEYPT_ORIG_REC_DIR, "rs_d415_info_depth.yaml"), extr_info)

		if self.use_top_camera:
			self.top_img_topic = self.top_camera_name + '/image_raw'
			self.top_img_sub = Subscriber(self.top_img_topic, Image)
			self.subs.append(self.top_img_sub)
			# save camera info
			print("Waiting for camera_info from", self.top_camera_name + '/camera_info')
			rgb_info = rospy.wait_for_message(self.top_camera_name + '/camera_info', sensor_msgs.msg.CameraInfo, 5)
			tf_dict = {}
			if self.use_tf:
				world_rs_tf_dict = self.lookupTF(rospy.Time(0), self.TF_TARGET_FRAME, self.TOP_RS_SOURCE_FRAME)
				opt_tf_dict = self.lookupTF(rospy.Time(0), self.TOP_RS_SOURCE_FRAME, self.TOP_RS_OPTICAL_SOURCE_FRAME)
				tf_dict = {'world_realsense': world_rs_tf_dict, 'realsense_color_optical': opt_tf_dict}
			self.saveCamInfo(rgb_info, os.path.join(KEYPT_TOP_CAM_REC_DIR, "rs_d435IF_info_rgb.yaml"), tf={'optical_extrinsics': tf_dict})
			if self.depth_enabled:
				self.top_depth_topic = self.top_camera_name.replace('color', 'depth')
				top_depth_img_topic = self.top_depth_topic + '/image_rect_raw'	
				self.top_depth_img_sub = Subscriber(top_depth_img_topic, Image)
				self.subs.append(self.top_depth_img_sub)
				# save camera info
				print("Waiting for camera_info from", self.top_depth_topic + '/camera_info')
				depth_info = rospy.wait_for_message(self.top_depth_topic + '/camera_info', sensor_msgs.msg.CameraInfo, 5)
				extr_info = rospy.wait_for_message(self.top_camera_name.replace('/color', '') + '/extrinsics/depth_to_color', Extrinsics, 5)
				self.saveCamInfo(depth_info, os.path.join(KEYPT_TOP_CAM_REC_DIR, "rs_d435IF_info_depth.yaml"), extr_info)

		if self.use_head_camera:
			self.head_img_topic = self.head_camera_name + '/image_raw'
			self.head_img_sub = Subscriber(self.head_img_topic, Image)
			self.subs.append(self.head_img_sub)
			# save camera info
			print("Waiting for camera_info from", self.head_camera_name + '/camera_info')
			rgb_info = rospy.wait_for_message(self.head_camera_name + '/camera_info', sensor_msgs.msg.CameraInfo, 5)
			self.saveCamInfo(rgb_info, os.path.join(KEYPT_HEAD_CAM_REC_DIR, "rs_d435I_info_rgb.yaml"))
			if self.depth_enabled:
				self.head_depth_topic = self.head_camera_name.replace('color', 'depth')
				head_depth_img_topic = self.head_depth_topic + '/image_rect_raw'	
				self.head_depth_img_sub = Subscriber(head_depth_img_topic, Image)
				self.subs.append(self.head_depth_img_sub)
				# save camera info
				print("Waiting for camera_info from", self.head_depth_topic + '/camera_info')
				depth_info = rospy.wait_for_message(self.head_depth_topic + '/camera_info', sensor_msgs.msg.CameraInfo, 5)
				extr_info = rospy.wait_for_message(self.head_camera_name.replace('/color', '') + '/extrinsics/depth_to_color', Extrinsics, 5)
				self.saveCamInfo(depth_info, os.path.join(KEYPT_HEAD_CAM_REC_DIR, "rs_d435I_info_depth.yaml"), extr_info)

		if self.use_eye_cameras:
			# right eye
			self.right_img_topic = self.right_eye_camera_name + '/image_raw'
			self.right_img_sub = Subscriber(self.right_img_topic, Image)
			self.subs.append(self.right_img_sub)
			# save camera info
			print("Waiting for camera_info from", self.right_eye_camera_name + '/camera_info')
			rgb_info = rospy.wait_for_message(self.right_eye_camera_name + '/camera_info', sensor_msgs.msg.CameraInfo, 5)
			self.saveCamInfo(rgb_info, os.path.join(KEYPT_R_EYE_REC_DIR, "See3CAM_CU135_info_rgb.yaml"))
			# left eye
			self.left_img_topic = self.left_eye_camera_name + '/image_raw'
			self.left_img_sub = Subscriber(self.left_img_topic, Image)
			self.subs.append(self.left_img_sub)
			# save camera info
			print("Waiting for camera_info from", self.left_eye_camera_name + '/camera_info')
			rgb_info = rospy.wait_for_message(self.left_eye_camera_name + '/camera_info', sensor_msgs.msg.CameraInfo, 5)
			self.saveCamInfo(rgb_info, os.path.join(KEYPT_L_EYE_REC_DIR, "See3CAM_CU135_info_rgb.yaml"))
		
		# trigger condition
		self.depth_frame_id = self.camera_ns.replace('color', 'depth').replace('/', '_')
		self.top_img_frame_id = self.top_camera_name.replace('/', '_')
		self.top_depth_frame_id = self.top_camera_name.replace('color', 'depth').replace('/', '_')
		self.head_img_frame_id = self.head_camera_name.replace('/', '_')
		self.head_depth_frame_id = self.head_camera_name.replace('color', 'depth').replace('/', '_')

		# sync callback
		self.sync = ApproximateTimeSynchronizer(self.subs, queue_size=1000, slop=1)
		self.sync.registerCallback(self.recCB)
		for s in self.subs:
			print("Subscribed to topic", s.topic)
		print("Syncing all aubscibers")

	def recCB(self, 
		   		joint_state: JointState, 
		   		det_img: Image, 
				msg3: Image=None,  # depth_img
				msg4: Image=None,  # top_img
				msg5: Image=None,  # top_depth_img
				msg6: Image=None,  # head_img
				msg7: Image=None,  # head_depth_img
				msg8: Image=None,  # right_img
				msg9: Image=None,  # left_img
				) -> None:
		# try lock
		if not self.buf_lock.acquire(blocking=False):
					return  
		try:
			self.joint_state_buffer.append(joint_state)
			self.det_img_buffer.append(det_img)

			for msg in [msg3, msg4, msg5, msg6, msg7, msg8, msg9]:
				if msg is not None:
					if self.depth_frame_id in msg.header.frame_id:
						self.depth_img_buffer.append(msg)
					elif self.top_img_frame_id in msg.header.frame_id:
						self.top_img_buffer.append(msg)
					elif self.top_depth_frame_id in msg.header.frame_id:
						self.top_depth_img_buffer.append(msg)
					elif self.head_img_frame_id in msg.header.frame_id:
						self.head_img_buffer.append(msg)
					elif self.head_depth_frame_id in msg.header.frame_id:
						self.head_depth_img_buffer.append(msg)
					elif self.right_eye_camera_name in msg.header.frame_id:
						self.right_img_buffer.append(msg)
					elif self.left_eye_camera_name in msg.header.frame_id:
						self.left_img_buffer.append(msg)
		finally:
			self.buf_lock.release()

	def lookupTF(self, stamp: rospy.Time, target_frame: str, source_frame: str) -> dict:
		try:
			tf = self.buf.lookup_transform(target_frame, source_frame, stamp, rospy.Duration(1.0))
			trans = tf.transform.translation
			rot = tf.transform.rotation
			tf_dict = { 'frame_id': tf.header.frame_id,
			  			'child_frame_id': tf.child_frame_id,
						'trans': [trans.x, trans.y, trans.z],
						'quat': [rot.x, rot.y, rot.z, rot.w],
						'timestamp': stamp.secs + stamp.nsecs*1e-9
						}
			return tf_dict
		except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
			print(e)
			return {}

	def saveRecord(self) -> None:	
		num_saved = 0

		# joint states 
		if len(self.joint_state_buffer):
			num_saved += 1
			msg = self.joint_state_buffer.pop()
			state_dict = {'name': msg.name,
							'position': msg.position,
							'velocity': msg.velocity,
							'effort': msg.effort,
							'timestamp': msg.header.stamp.secs + msg.header.stamp.nsecs*1e-9
						 }
			self.js_df = pd.concat([self.js_df, pd.DataFrame([state_dict])], ignore_index=True)
			# save rh8d tf
			if self.use_tf:
				tf_dict = self.lookupTF(msg.header.stamp, self.TF_TARGET_FRAME, self.RH8D_SOURCE_FRAME)
				if tf_dict:
					self.rh8d_tf_df = pd.concat([self.rh8d_tf_df, pd.DataFrame([tf_dict])], ignore_index=True)

		# marker cam depth img
		if self.depth_enabled and len(self.depth_img_buffer):
			num_saved += 1
			depth_img = self.bridge.imgmsg_to_cv2(self.depth_img_buffer.pop(), 'passthrough')
			depth_img = (depth_img).astype('float32')
			cv2.imwrite(os.path.join(KEYPT_ORIG_REC_DIR, str(self.record_cnt) + '.tiff'), depth_img)

		# top realsense
		if self.use_top_camera and len(self.top_img_buffer):
			num_saved += 1
			raw_img = self.bridge.imgmsg_to_cv2(self.top_img_buffer.pop(), 'bgr8')
			cv2.imwrite(os.path.join(KEYPT_TOP_CAM_REC_DIR, str(self.record_cnt) + '.jpg'), raw_img, [int(cv2.IMWRITE_JPEG_QUALITY), JPG_QUALITY])
			# depth
			if self.depth_enabled and len(self.top_depth_img_buffer):
				depth_img = self.bridge.imgmsg_to_cv2(self.top_depth_img_buffer.pop(), 'passthrough')
				depth_img = (depth_img).astype('float32')
				cv2.imwrite(os.path.join(KEYPT_TOP_CAM_REC_DIR, str(self.record_cnt) + '.tiff'), depth_img)

		# head realsense imgs & tf
		if self.use_head_camera and len(self.head_img_buffer):
			num_saved += 1
			msg = self.head_img_buffer.pop()
			raw_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
			cv2.imwrite(os.path.join(KEYPT_HEAD_CAM_REC_DIR, str(self.record_cnt) + '.jpg'), raw_img, [int(cv2.IMWRITE_JPEG_QUALITY), JPG_QUALITY])
			# save head rs tf
			if self.use_tf:
				tf_dict = self.lookupTF(msg.header.stamp, self.TF_TARGET_FRAME, self.HEAD_RS_SOURCE_FRAME)
				if tf_dict:
					self.head_rs_tf_df = pd.concat([self.head_rs_tf_df, pd.DataFrame([tf_dict])], ignore_index=True)
			# depth
			if self.depth_enabled and len(self.head_depth_img_buffer):
				depth_img = self.bridge.imgmsg_to_cv2(self.head_depth_img_buffer.pop(), 'passthrough')
				depth_img = (depth_img).astype('float32')
				cv2.imwrite(os.path.join(KEYPT_HEAD_CAM_REC_DIR, str(self.record_cnt) + '.tiff'), depth_img)

		# eyes imgs & tf
		if self.use_eye_cameras:
			num_saved += 1
			if len(self.left_img_buffer):
				msg = self.left_img_buffer.pop()
				raw_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
				cv2.imwrite(os.path.join(KEYPT_L_EYE_REC_DIR, str(self.record_cnt) + '.jpg'), raw_img, [int(cv2.IMWRITE_JPEG_QUALITY), JPG_QUALITY])
				# save left eye tf
				if self.use_tf:
					tf_dict = self.lookupTF(msg.header.stamp, self.TF_TARGET_FRAME, self.LEFT_EYE_SOURCE_FRAME)
					if tf_dict:
						self.left_eye_tf_df = pd.concat([self.left_eye_tf_df, pd.DataFrame([tf_dict])], ignore_index=True)

			if len(self.right_img_buffer):
				msg = self.right_img_buffer.pop()
				raw_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
				cv2.imwrite(os.path.join(KEYPT_R_EYE_REC_DIR, str(self.record_cnt) + '.jpg'), raw_img, [int(cv2.IMWRITE_JPEG_QUALITY), JPG_QUALITY])
				# save right eye tf
				if self.use_tf:
					tf_dict = self.lookupTF(msg.header.stamp, self.TF_TARGET_FRAME, self.RIGHT_EYE_SOURCE_FRAME)
					if tf_dict:
						self.right_eye_tf_df = pd.concat([self.right_eye_tf_df, pd.DataFrame([tf_dict])], ignore_index=True)

		# check df lengths
		if self.record_cnt != self.js_df.last_valid_index() and self.record_cnt:
			print("Record size deviates js dataframe index:", self.record_cnt, "!=", self.js_df.last_valid_index())
		if self.use_tf:
			if self.record_cnt != self.rh8d_tf_df.last_valid_index() and self.record_cnt:
				print("Record size deviates from rh8d tf dataframe index:", self.record_cnt, "!=", self.rh8d_tf_df.last_valid_index())
			if self.use_eye_cameras:
				if self.record_cnt != self.right_eye_tf_df.last_valid_index() and self.record_cnt:
					print("Record size deviates from right eye dataframe index:", self.record_cnt, "!=", self.right_eye_tf_df.last_valid_index())
				if self.record_cnt != self.left_eye_tf_df.last_valid_index() and self.record_cnt:
					print("Record size deviates from left eye dataframe index:", self.record_cnt, "!=", self.left_eye_tf_df.last_valid_index())
			if self.use_head_camera:
				if self.record_cnt != self.head_rs_tf_df.last_valid_index() and self.record_cnt:
					print("Record size deviates from head rs dataframe index:", self.record_cnt, "!=", self.head_rs_tf_df.last_valid_index())
		
		if num_saved > 0:
			self.record_cnt += 1

	def preProcImage(self, vis: bool=True) -> Tuple[dict, cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike]:
		""" Put num filter_iters images into fresh detection filter and save last images."""

		with self.buf_lock:
			
			if len(self.det_img_buffer) != self.filter_iters:
				print("Record size deviates from filter size:", len(self.det_img_buffer), " !=", self.filter_iters)

			self.det.resetFilters()
			while len(self.det_img_buffer):
				self.frame_cnt += 1
				raw_img = self.bridge.imgmsg_to_cv2(self.det_img_buffer.pop(), 'bgr8')
				(marker_det, det_img, proc_img) = self.det.detMarkerPoses(raw_img.copy(), vis and (not len(self.det_img_buffer) and self.vis))
			# dummy img
			if self.test:
				(marker_det, det_img, proc_img) = self.det.detMarkerPoses(self.img.copy(), vis and self.vis)

			# align rotations by consens
			if self.flip_outliers:
				if not self.flipOutliers(marker_det):
					beep()
			# improve detection
			if self.refine_pose:
				self.refineDetection(marker_det)

			# save additional resources
			if not self.test and self.save_imgs:
				self.saveRecord()

			return marker_det, det_img, proc_img, raw_img

	def normalXZ(self, rot: np.ndarray, rot_t: RotTypes) -> np.ndarray:
		""" Get the normal to the XZ plane in the markee frame.
		"""
		mat = getRotation(rot, rot_t, RotTypes.MAT)
		normal_xz = mat @ self.UNIT_AXIS_Y
		return normal_xz

	def normalXZAngularDispl(self, base_rot: np.ndarray, target_rot: np.ndarray, rot_t: RotTypes=RotTypes.EULER) -> float:
		""" Calculate the angle between normal vectors.
		"""
		# get normal vectors for the XZ planes
		base_normal = self.normalXZ(base_rot, rot_t)
		target_normal = self.normalXZ(target_rot, rot_t)
		# normalize vectors to make sure they are unit vectors
		base_normal = base_normal / np.linalg.norm(base_normal)
		target_normal = target_normal / np.linalg.norm(target_normal)
		# get angle
		dot_product = base_normal @ target_normal
		# ensure the dot product is within the valid range for arccos due to numerical precision
		dot_product = np.clip(dot_product, -1.0, 1.0)
		# angle in radians
		return np.arccos(dot_product)
	
	def tfPoints(self, tf: np.ndarray, points: np.ndarray) -> np.ndarray:
		"""Transform marker corners to some frame.
		""" 
		homog_points = np.hstack((points, np.ones((points.shape[0], 1))))
		tf_points = tf @ homog_points.T
		tf_points = tf_points.T 
		return tf_points[:, :3]
	
	def drawAngle(self, id: int, img: cv2.typing.MatLike, pt1: np.ndarray, pt2: np.ndarray, angle_deg: float) -> None:
		# extract point coordinates
		x1, y1 = pt1
		x2, y2 = pt2
		# find normal from midpoint, follow by length sagitta
		n = np.array([y2 - y1, x1 - x2])
		n_dist = np.sqrt(np.sum(n**2))
		# catch error here, d(pt1, pt2) ~ 0
		if np.isclose(n_dist, 0):
			print('Error: The distance between pt1 and pt2 is too small.')
			return
		
		n = n/n_dist
		x3, y3 = (np.array(pt1) + np.array(pt2))/2 + self.SAGITTA * n
		# calculate the circle from three points
		# see https://math.stackexchange.com/a/1460096/246399
		A = np.array([
			[x1**2 + y1**2, x1, y1, 1],
			[x2**2 + y2**2, x2, y2, 1],
			[x3**2 + y3**2, x3, y3, 1]])
		M11 = np.linalg.det(A[:, (1, 2, 3)])
		M12 = np.linalg.det(A[:, (0, 2, 3)])
		M13 = np.linalg.det(A[:, (0, 1, 3)])
		M14 = np.linalg.det(A[:, (0, 1, 2)])
		# catch error here, the points are collinear (sagitta ~ 0)
		if np.isclose(M11, 0):
			print('Error: The third point is collinear.')
			return

		cx = 0.5 * M12/M11
		cy = -0.5 * M13/M11
		radius = np.sqrt(cx**2 + cy**2 + M14/M11)
		# calculate angles of pt1 and pt2 from center of circle
		pt1_angle = 180*np.arctan2(y1 - cy, x1 - cx)/np.pi
		pt2_angle = 180*np.arctan2(y2 - cy, x2 - cx)/np.pi

		# draw ellipse
		center = (cx, cy)
		axes = (radius, radius)
		arc_center = (int(round(center[0] * 2**self.ARC_SHIFT)),
				  int(round(center[1] * 2**self.ARC_SHIFT)))
		axes = (int(round(axes[0] * 2**self.ARC_SHIFT)),
				int(round(axes[1] * 2**self.ARC_SHIFT)))
		center = tuple(map(int, center))
		cv2.ellipse(img, arc_center, axes, 0, pt1_angle, pt2_angle, self.ELIPSE_COLOR, self.ELIPSE_THKNS, cv2.LINE_AA, self.ARC_SHIFT)
		# cv2.circle(img, pt1, 5, self.ELIPSE_COLOR, -1)
		# cv2.circle(img, pt2, 5, self.ELIPSE_COLOR, -1)
		# cv2.circle(img, center, 5, self.ELIPSE_COLOR, -1)
		# draw angle text
		txt_center = (int(center[0]*self.Y_EL_TXT_OFFSET), int(center[1]*self.X_EL_TXT_OFFSET))
		cv2.putText(img, f'{angle_deg:.2f}', txt_center, cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.ELIPSE_COLOR, self.FONT_THCKNS)

	def labelDetection(self, img: cv2.typing.MatLike, id: int, detection: dict) -> None:
		angle = detection[id].get('angle')
		base_id = detection[id].get('base_id')
		if angle is None: return

		# get detected transforms
		base_rot = getRotation(detection[base_id].get('frot'), RotTypes.EULER, RotTypes.MAT)
		target_rot = getRotation(detection[id].get('frot'), RotTypes.EULER, RotTypes.MAT)
		base_trans = detection[base_id].get('ftrans')
		target_trans = detection[id].get('ftrans')

		# project the x-axes of both markers
		x_axis_base_marker, _ = cv2.projectPoints(self.X_AXIS, base_rot, base_trans, self.det.cmx, self.det.dist)
		x_axis_target_marker, _ = cv2.projectPoints(self.X_AXIS, target_rot, target_trans, self.det.cmx, self.det.dist)
		x_axis_base_marker = np.int32(x_axis_base_marker).reshape(-1, 2)
		x_axis_target_marker = np.int32(x_axis_target_marker).reshape(-1, 2)
		cv2.line(img, x_axis_base_marker[0], x_axis_base_marker[1], self.ELIPSE_COLOR, 2)
		cv2.line(img, x_axis_target_marker[0], x_axis_target_marker[1], self.ELIPSE_COLOR, 2)

		# draw angle between the x-axes
		if np.abs(angle) > 0:
			# interpolate points on the axes
			interpolated_point_base_ax = (1 - self.FIRST_LINE_INTERPOL_FACTOR) * x_axis_base_marker[0] + self.FIRST_LINE_INTERPOL_FACTOR * x_axis_base_marker[1]
			interpolated_point_target_ax = (1 - self.SEC_LINE_INTERPOL_FACTOR) * x_axis_target_marker[0] + self.SEC_LINE_INTERPOL_FACTOR * x_axis_target_marker[1]
			# draw ellipse
			p1 = tuple(map(int, interpolated_point_base_ax))
			p2 = tuple(map(int, interpolated_point_target_ax))
			# self.drawAngle(id, img, p1, p2, np.rad2deg(angle))

	def detectionRoutine(self, init: bool, pos_cmd: int, epoch: int, direction: int) -> bool:
		res = False
		entry = {} # data entry
		# get filtered detection and save other resources
		(marker_det, det_img, proc_img, img) = self.preProcImage()
		out_img = img.copy()

		if marker_det:
			# check all ids detected
			if all([id in marker_det.keys() for id in self.target_ids]):
				
				for idx in range(1, len(self.target_ids)):
					base_idx = idx-1 # predecessor index
					base_id = self.target_ids[base_idx] # predecessor marker
					target_id = self.target_ids[idx] # target marker
					base_marker = marker_det[base_id] # predecessor detection
					target_marker = marker_det[target_id] # target detection

					# detected angle in rad
					angle = self.normalXZAngularDispl(base_marker['frot'], target_marker['frot'], RotTypes.EULER)
					# save initially detected angle
					if init:
						self.start_angles.update({base_idx: angle})
					# substract initial angle
					angle = angle - self.start_angles[base_idx]

					# data entry
					data = {'angle': angle, 'base_id': base_id, 'target_id': target_id, 'frame':self.frame_cnt, 'rec_cnt': self.record_cnt,  'cmd': pos_cmd, 'epoch': epoch, 'direction': direction}
					marker_det[target_id].update(data)
					entry.update({self.group_ids[base_idx]: data})
					# print(f"angle: {np.rad2deg(angle)}, qdec_angle: {np.rad2deg(qdec_angle)}, error: {np.rad2deg(error)}, base_id: {base_id}")

				res = True
			else:
				print("Cannot detect all required ids, missing: ", end=" ")
				[print(id, end=" ") if id not in marker_det.keys() else None for id in self.target_ids]
				beep()
		else:
			print("No detection")
			
		if self.vis:
			# frame counter
			cv2.putText(det_img, str(self.frame_cnt) + " " + str(self.record_cnt), (det_img.shape[1]-100, 50), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.FONT_CLR, self.FONT_THCKNS, cv2.LINE_AA)
			# actuator position
			cv2.putText(out_img, f"actuator position: {pos_cmd}", (self.TXT_OFFSET, self.TXT_OFFSET), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.FONT_CLR, self.FONT_THCKNS, cv2.LINE_AA)
			# draw angle labels and possibly fixed detections 
			for id, detection in marker_det.items():
				self.labelDetection(out_img, id, marker_det) # cv angle
				self.det._drawMarkers(id, detection['corners'], out_img) # square
				out_img = cv2.drawFrameAxes(out_img, self.det.cmx, self.det.dist, detection['rvec'], detection['tvec'], self.det.marker_length*self.det.AXIS_LENGTH, self.det.AXIS_THICKNESS) # CS
				out_img = self.det._projPoints(out_img, detection['points'], detection['rvec'], detection['tvec']) # corners
			
			if self.cv_window:
				cv2.imshow('Processed', proc_img)
				cv2.imshow('Detection', det_img)
				cv2.imshow('Output', out_img)
			if self.save_imgs and not self.test and res:
				try:
					self.df = pd.concat([self.df, pd.DataFrame([entry])], ignore_index=True) # save data entry
					cv2.imwrite(os.path.join(KEYPT_ORIG_REC_DIR, str(self.record_cnt) + '.jpg'), img, [int(cv2.IMWRITE_JPEG_QUALITY), JPG_QUALITY]) # save original
					cv2.imwrite(os.path.join(KEYPT_DET_REC_DIR, str(self.record_cnt) + '.jpg'), det_img, [int(cv2.IMWRITE_JPEG_QUALITY), JPG_QUALITY]) # save detection
					# self.saveImgs() # save additional resources
					if self.df.last_valid_index() != str(self.record_cnt):
						print("RECORD DEVIATION data index: ", self.df.last_valid_index(), " img index:", str(self.record_cnt))
				except Exception as e:
					print(e)
					return False

		return res
		
	def run(self) -> None:
		rate = rospy.Rate(self.f_loop)
		
		try:
			# epoch
			for e in range(self.epochs):
				print("Epoch", e)
				self.rh8d_ctrl.moveHeadHome(2.0)
				head_home = True
				
				for idx in self.waypoint_df.index.tolist():
					if not rospy.is_shutdown():
						# get waypoint
						waypoint = self.waypoint_df.iloc[idx]
						direction = waypoint[-3]
						description = waypoint[-2]
						move_time = waypoint[-1]
						waypoint = waypoint[: -3].to_dict()

						# get head out of range if arm moves
						states = self.rh8d_ctrl.jointStates()
						if states is not None:
							if not all( np.isclose(self.rh8d_ctrl.angularDistanceOMP(states, waypoint.values), 0.0) ):
								self.rh8d_ctrl.moveHeadHome(1.5)
								time.sleep(1.5)
								head_home = True
						else:
							self.rh8d_ctrl.moveHeadHome(1.5)
							time.sleep(1.5)
							head_home = True

						# move arm and hand
						print(f"Reaching waypoint number {idx}: {description} in {move_time}s", end=" ... ")
						print(waypoint.values)
						success, _ = self.rh8d_ctrl.reachPositionBlocking(waypoint, move_time, 0.1)
						print("done\n") if success else print("fail\n")

						# look towards hand
						tf = self.lookupTF(rospy.Time(0), self.TF_TARGET_FRAME, self.RH8D_TCP_SOURCE_FRAME)
						self.rh8d_ctrl.moveHeadTaskSpace(tf['trans'][0], tf['trans'][1], tf['trans'][2], 2.0 if head_home else 1.0)			
						time.sleep(2.0 if head_home else 1.0)
						head_home = False

						# detect angles
						success = self.detectionRoutine(init, pos_cmd, e, direction)
						# with self.buf_lock:
						# 	self.saveRecord()

						if self.vis and not self.attached:
							if cv2.waitKey(1) == ord("q"):
								return
						
						try:
							rate.sleep()
						except:
							pass

		except None as e:
			rospy.logerr(e)
		finally:
			self.df.to_json(KEYPT_DET_PTH, orient="index", indent=4)
			self.js_df.to_json(os.path.join(KEYPT_REC_DIR, 'actuator_states.json'), orient="index", indent=4)
			if self.use_tf:
				self.rh8d_tf_df.to_json(os.path.join(KEYPT_DET_REC_DIR, 'tf.json'), orient="index", indent=4)
				if self.use_head_camera:
					self.head_rs_tf_df.to_json(os.path.join(KEYPT_HEAD_CAM_REC_DIR, 'tf.json'), orient="index", indent=4)
				if self.use_eye_cameras:
					self.left_eye_tf_df.to_json(os.path.join(KEYPT_L_EYE_REC_DIR, 'tf.json'), orient="index", indent=4)
					self.right_eye_tf_df.to_json(os.path.join(KEYPT_R_EYE_REC_DIR, 'tf.json'), orient="index", indent=4)
			if self.vis and not self.attached:
				cv2.destroyAllWindows()
			rospy.signal_shutdown(0)

class HybridDetect(KeypointDetect):
	"""
		Detect keypoints from marker poses and quadrature encoders 
		while moving the hand.

 		@param qdec_port
		@type str
		@param qdec_baud
		@type int
		@param qdec_tout
		@type int
		@param qdec_filter_iters
		@type int
		@param actuator
		@type int

	"""

	def __init__(self,
			  	marker_length: float=0.010,
				camera_ns: Optional[str]='',
				vis :Optional[bool]=True,
				use_reconfigure: Optional[bool]=False,
				filter_type: Optional[str]='none',
				filter_iters: Optional[int]=10,
				f_ctrl: Optional[int]=30,
				use_aruco: Optional[bool]=False,
				plt_id: Optional[int]=-1,
				rh8d_port: Optional[str]="/dev/ttyUSB1",
				rh8d_baud: Optional[int]=1000000,
 				qdec_port: Optional[str]='/dev/ttyUSB0',
				qdec_baud: Optional[int]=19200,
				qdec_tout: Optional[int]=1,
				qdec_filter_iters: Optional[int]=200,
				actuator: Optional[int]=32,
				step_div: Optional[int]=100,
				epochs: Optional[int]=10,
				test: Optional[bool]=False,
				save_imgs: Optional[bool]=True,
				flip_outliers: Optional[bool]=True,
				refine_pose: Optional[bool]=True,
				fps: Optional[float]=30.0,
				) -> None:
		
		super().__init__(marker_length=marker_length,
						use_reconfigure=use_reconfigure,
						flip_outliers=flip_outliers,
						refine_pose=refine_pose,
						camera_ns=camera_ns,
						filter_type=filter_type,
						filter_iters=filter_iters,
						save_imgs=save_imgs,
						use_aruco=use_aruco,
						rh8d_port=rh8d_port,
						rh8d_baud=rh8d_baud,
						step_div=step_div,
						plt_id=plt_id,
						use_tf=False,
						f_ctrl=f_ctrl,
						epochs=epochs,
						test=test,
						vis=vis,
						fps=fps,
						)
		
		# actuator control
		self.actuator = actuator
		# index finger data
		self.group_ids = self.hand_ids['names']['index'] # joint names
		self.df = pd.DataFrame(columns=self.group_ids) # dataset
		self.target_ids = self.hand_ids['ids']['index'] # target marker ids
		self.qdec = QdecSerial(qdec_port, qdec_baud, qdec_tout, qdec_filter_iters) if not self.test else QdecSerialStub()

	def labelQdecAngles(self, img: cv2.typing.MatLike, id: int, marker_det: dict) -> None:
		names = ["spare", "root", "proximal", "medial", "distal"]
		angle = marker_det[id].get('angle') 
		if angle is None: 
			return
		txt = "{} {} cv: {:.2f} deg, qdec: {:.2f} deg, err: {:.2f} deg".format(id, names[id%len(names)], np.rad2deg(angle), np.rad2deg(marker_det[id]['qdec_angle']), np.rad2deg(marker_det[id]['error']))
		xpos = self.TXT_OFFSET
		ypos = (id+2)*self.TXT_OFFSET
		cv2.putText(img, txt, (xpos, ypos), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.FONT_CLR, self.FONT_THCKNS, cv2.LINE_AA)
		
	def detectionRoutine(self, init: bool, pos_cmd: int, epoch: int, direction: int) -> bool:
		res = False
		# get filtered detection
		(marker_det, det_img, proc_img, img) = self.preProcImage()
		out_img = img.copy()

		if marker_det:
			# check all ids detected
			if all([id in marker_det.keys() for id in self.target_ids]):

				# encoder angles in rad
				qdec_angles = self.qdec.readMedianAnglesRad()
				if not qdec_angles:
					print("No qdec angles")
					beep()
					return False
				
				# cv angles
				entry = {}
				for idx in range(1, len(self.target_ids)):
					base_idx = idx-1 # predecessor index
					base_id = self.target_ids[base_idx] # predecessor marker
					target_id = self.target_ids[idx] # target marker
					base_marker = marker_det[base_id] # predecessor detection
					target_marker = marker_det[target_id] # target detection

					# detected angle in rad
					angle = self.normalXZAngularDispl(base_marker['frot'], target_marker['frot'], RotTypes.EULER)
					# save initially detected angle
					if init:
						self.start_angles.update({base_idx: angle})
					# substract initial angle
					angle = angle - self.start_angles[base_idx]
					qdec_angle = qdec_angles[base_idx]
					error = np.abs(qdec_angle - angle)

					# data entry
					data = {'angle': angle, 'qdec_angle': qdec_angle, 'error': error, 'base_id': base_id, 'target_id': target_id, 'frame':self.frame_cnt, 'cmd': pos_cmd, 'epoch': epoch, 'direction': direction}
					marker_det[target_id].update(data)
					entry.update({self.group_ids[base_idx]: data})
					# print(f"angle: {np.rad2deg(angle)}, qdec_angle: {np.rad2deg(qdec_angle)}, error: {np.rad2deg(error)}, base_id: {base_id}")

				res = True
				self.df = pd.concat([self.df, pd.DataFrame([entry])], ignore_index=True)
			else:
				print("Cannot detect all required ids, missing: ", end=" ")
				[print(id, end=" ") if id not in marker_det.keys() else None for id in self.target_ids]
				beep()
		else:
			print("No detection")
			
		if self.vis:
			# frame counter
			cv2.putText(det_img, str(self.frame_cnt), (det_img.shape[1]-100, 50), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.FONT_CLR, self.FONT_THCKNS, cv2.LINE_AA)
			# actuator position
			cv2.putText(out_img, f"actuator position: {pos_cmd}", (self.TXT_OFFSET, self.TXT_OFFSET), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.FONT_CLR, self.FONT_THCKNS, cv2.LINE_AA)
			# draw angle labels and possibly fixed detections 
			for id, detection in marker_det.items():
				self.labelDetection(out_img, id, marker_det) # cv angle
				self.labelQdecAngles(out_img, id, marker_det) # qdec angle
				self.det._drawMarkers(id, detection['corners'], out_img) # square
				out_img = cv2.drawFrameAxes(out_img, self.det.cmx, self.det.dist, detection['rvec'], detection['tvec'], self.det.marker_length*self.det.AXIS_LENGTH, self.det.AXIS_THICKNESS) # CS
				out_img = self.det._projPoints(out_img, detection['points'], detection['rvec'], detection['tvec']) # corners
			
			if self.cv_window:
				cv2.imshow('Processed', proc_img)
				cv2.imshow('Detection', det_img)
				cv2.imshow('Output', out_img)
			if self.save_imgs:
				cv2.imwrite(os.path.join(QDEC_ORIG_REC_DIR, str(self.frame_cnt) + '.jpg'), img, [int(cv2.IMWRITE_JPEG_QUALITY), JPG_QUALITY])
				cv2.imwrite(os.path.join(QDEC_DET_REC_DIR, str(self.frame_cnt) + '.jpg'), out_img, [int(cv2.IMWRITE_JPEG_QUALITY), JPG_QUALITY])

		return res
		
	def run(self) -> None:
		max_pos = 2700 # encoder covers marker
		min_pos = RH8D_MIN_POS
		step = RH8D_MAX_POS // self.step_div
		rate = rospy.Rate(self.f_loop)
		# zero encoder angles
		self.qdec.qdecReset()
		# save initial cv angles once
		init = True

		# initially closing
		direction = 1
		# 0: n/a, 1: closing, -1: opening
		conditions = [False, lambda cmd: cmd <= max_pos, lambda cmd: cmd >= min_pos]
		
		try:
			# epoch
			for e in range(self.epochs):
				print("Epoch", e)
				pos_cmd = step
				fails = 0
				# move to zero
				self.rh8d_ctrl.rampMinPos(self.actuator)
				if e:
					time.sleep(1)
				
				# closing and opening cycle
				for i in range(2):
					while not rospy.is_shutdown() and conditions[direction](pos_cmd):
						# detect angles
						success = self.detectionRoutine(init, pos_cmd, e, direction)
						
						if cv2.waitKey(1) == ord("q"):
							return
						
						# move finger
						self.rh8d_ctrl.setPos(self.actuator, pos_cmd)
						print()
						
						# increment position if detection was successful
						if success:
							pos_cmd += direction * step
							init = False
						else:
							if fails > 3:
								break
							fails += 1

						try:
							rate.sleep()
						except:
							pass

					# invert direction
					direction *= -1

		except Exception as e:
			rospy.logerr(e)
		finally:
			self.df.to_json(QDEC_DET_PTH, orient="index", indent=4)
			self.rh8d_ctrl.setMinPos(self.actuator, 1)
			cv2.destroyAllWindows()
			rospy.signal_shutdown(0)
		
class CameraPoseDetect(DetectBase):
	"""
		Detect camera world pose from marker 
		poses in static environment.

		@param err_term
		@type float
		@param cart_bound_low
		@type float
		@param cart_bound_high
		@type float
		@param fn
		@type str

	"""

	def __init__(self,
			  	marker_length: float=0.010,
				camera_ns: Optional[str]='',
				vis :Optional[bool]=True,
				use_reconfigure: Optional[bool]=False,
				filter_type: Optional[str]='none',
				filter_iters: Optional[int]=10,
				f_ctrl: Optional[int]=30,
				use_aruco: Optional[bool]=False,
				plt_id: Optional[int]=-1,
				err_term: Optional[float]=2.0,
				cart_bound_low: Optional[float]=-3.0,
				cart_bound_high: Optional[float]=3.0,
				flip_outliers: Optional[bool]=True,
				refine_pose: Optional[bool]=True,
				fn: Optional[str]= 'marker_holder_poses.yml',
				fps: Optional[float]=30.0,
				) -> None:
		
		super().__init__(marker_length=marker_length,
						use_reconfigure=use_reconfigure,
						flip_outliers=flip_outliers,
						refine_pose=refine_pose,
						camera_ns=camera_ns,
						filter_type=filter_type,
						use_aruco=use_aruco,
						filter_iters=filter_iters,
						plt_id=plt_id,
						f_ctrl=f_ctrl,
						test=False,
						vis=vis,
						fps=fps,
						)
		
		self.CAM_LABEL_YPOS = 20
		self.err_term = err_term
		self.reprojection_errors = {}
		self.lower_bounds = [cart_bound_low, cart_bound_low, cart_bound_low, -np.pi, -np.pi, -np.pi]
		self.upper_bounds = [cart_bound_high, cart_bound_high, cart_bound_high, np.pi, np.pi, np.pi]
		# load marker poses
		self.fl = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cfg/"+fn)
		with open(self.fl, 'r') as fr:
			self.marker_table_poses = yaml.safe_load(fr)

	def labelDetection(self, img: cv2.typing.MatLike, trans: np.ndarray, rot: np.ndarray, corners: np.ndarray) -> None:
			pos_txt = "X: {:.4f} Y:  {:.4f} Z:  {:.4f}".format(trans[0], trans[1], trans[2])
			ori_txt = "R: {:.4f} P:  {:.4f} Y:  {:.4f}".format(rot[0], rot[1], rot[2])
			x_max = int(np.max(corners[:, 0]))
			y_max = int(np.max(corners[:, 1]))
			y_min = int(np.min(corners[:, 1]))
			x_offset = 0 if x_max <= img.shape[1]/2 else -int(len(pos_txt)*20*self.FONT_SCALE)
			y_offset1 = self.TXT_OFFSET if y_max <= img.shape[0]/2 else -self.TXT_OFFSET-(y_max-y_min)
			y_offset2 = y_offset1 + int(self.FONT_SCALE*50) if y_offset1 > 0 else y_offset1 - int(self.FONT_SCALE*50)
			cv2.putText(img, pos_txt, (x_max+x_offset, y_max+(y_offset1 if y_offset1 > 0 else y_offset2)), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.FONT_CLR, self.FONT_THCKNS, cv2.LINE_AA)
			cv2.putText(img, ori_txt, (x_max+x_offset, y_max+(y_offset2 if y_offset1 > 0 else y_offset1)), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.FONT_CLR, self.FONT_THCKNS, cv2.LINE_AA)

	def labelDetection(self, img: cv2.typing.MatLike, id: int, trans: np.ndarray, rot: np.ndarray, err: Optional[Union[float, None]]=None) -> None:
			if id > -1:
				repr_error = self.reprojection_errors.get(id)
				if repr_error is None:
					repr_error = -1.0
				pos_txt = "{} X: {:.4f} Y: {:.4f} Z: {:.4f} R: {:.4f} P: {:.4f} Y: {:.4f}, err {:.2f}".format(id, trans[0], trans[1], trans[2], rot[0], rot[1], rot[2], repr_error)
				xpos = self.TXT_OFFSET
				ypos = (id+1)*self.TXT_OFFSET
				cv2.putText(img, pos_txt, (xpos, ypos), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.det.RED, self.FONT_THCKNS, cv2.LINE_AA)
			else:
				xpos = self.TXT_OFFSET
				ypos = self.CAM_LABEL_YPOS*self.TXT_OFFSET
				cv2.putText(img, "CAMERA", (xpos, ypos), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.det.GREEN, self.FONT_THCKNS, cv2.LINE_AA)
				cv2.putText(img, "X {:.4f}".format(trans[0]), (xpos, ypos+2*self.TXT_OFFSET), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.det.GREEN, self.FONT_THCKNS, cv2.LINE_AA)
				cv2.putText(img, "Y {:.4f}".format(trans[1]), (xpos, ypos+3*self.TXT_OFFSET), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.det.GREEN, self.FONT_THCKNS, cv2.LINE_AA)
				cv2.putText(img, "Z {:.4f}".format(trans[2]), (xpos, ypos+4*self.TXT_OFFSET), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.det.GREEN, self.FONT_THCKNS, cv2.LINE_AA)
				cv2.putText(img, "roll {:.4f}".format(rot[0]), (xpos, ypos+5*self.TXT_OFFSET), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.det.GREEN, self.FONT_THCKNS, cv2.LINE_AA)
				cv2.putText(img, "pitch {:.4f}".format(rot[1]), (xpos, ypos+6*self.TXT_OFFSET), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.det.GREEN, self.FONT_THCKNS, cv2.LINE_AA)
				cv2.putText(img, "yaw {:.4f}".format(rot[2]), (xpos, ypos+7*self.TXT_OFFSET), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.det.GREEN, self.FONT_THCKNS, cv2.LINE_AA)
				if err is not None and err is not np.inf:
					cv2.putText(img, "mean reprojection error: {:.4f}".format(err), (xpos, ypos+8*self.TXT_OFFSET), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.det.GREEN, self.FONT_THCKNS, cv2.LINE_AA)

	def reprojectionError(self, det_corners: np.ndarray, proj_corners: np.ndarray) -> float:
		error = np.linalg.norm(det_corners - proj_corners, axis=1)
		return np.mean(error)
	
	def projectSingleMarker(self, detection:dict, id: int, camera_pose: np.ndarray, img: cv2.typing.MatLike=None) -> float:
		if self.marker_table_poses.get(id) is None:
			print(f"id {id} not present in marker poses!")
			return np.inf
		# tf marker corners wrt. world
		T_world_marker = self.getWorldMarkerTF(id)
		world_corners = self.tagWorldCorners(T_world_marker, self.det.square_points)
		# project corners to image plane
		projected_corners, _ = cv2.projectPoints(world_corners, camera_pose[:3, :3], camera_pose[:3, 3], self.det.cmx, self.det.dist)
		projected_corners = np.int32(projected_corners).reshape(-1, 2)
		if img is not None:
			cv2.polylines(img, [projected_corners], isClosed=True, color=self.det.BLUE, thickness=2)
			cv2.putText(img, str(id), (projected_corners[0][0]+5, projected_corners[0][1]+5), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.FONT_CLR, self.FONT_THCKNS, cv2.LINE_AA)
		return self.reprojectionError(detection['corners'], projected_corners)
	
	def projectMarkers(self, detection:dict, camera_pose: np.ndarray, img: cv2.typing.MatLike=None) -> float:
		err = []
		# invert world to camera tf for reprojection
		tvec_inv, euler_inv = invPersp(tvec=camera_pose[:3], rot=camera_pose[3:], rot_t=RotTypes.EULER)
		T_cam_world = pose2Matrix(tvec_inv, euler_inv, RotTypes.EULER)
		# iter measured markers
		for id, det in detection.items():
			# reprojection error
			e = self.projectSingleMarker(det, id, T_cam_world, img)
			self.reprojection_errors.update({id: e})
			err.append(e)
		return err
	
	def tagWorldCorners(self, world_tag_tf: np.ndarray, tag_corners: np.ndarray) -> np.ndarray:
		"""Transform marker corners to world frame""" 
		homog_corners = np.hstack((tag_corners, np.ones((tag_corners.shape[0], 1))))
		world_corners = world_tag_tf @ homog_corners.T
		world_corners = world_corners.T 
		return world_corners[:, :3]
	
	def getWorldMarkerTF(self, id: int) -> np.ndarray:
		# marker root tf
		root = self.marker_table_poses.get('root')
		T_world_root = pose2Matrix(root['xyz'], root['rpy'], RotTypes.EULER) if root is not None else np.eye(4)
		# marker tf
		marker = self.marker_table_poses.get(id)
		assert(marker) # marker id entry in yaml?
		T_root_marker = pose2Matrix(marker['xyz'], marker['rpy'], RotTypes.EULER)
		# worldTmarker
		return T_world_root @ T_root_marker
	
	def camTF(self, detection: dict, id: int) -> np.ndarray:
		tf = np.zeros(6)
		det = detection.get(id)
		if det is None:
			print(f"Cannot find id {id} in detection!")
			return tf
		# get markerTcamera
		inv_tvec, inv_euler = invPersp(tvec=det['ftrans'], rot=det['frot'], rot_t=RotTypes.EULER)
		T_marker_cam = pose2Matrix(inv_tvec, inv_euler, RotTypes.EULER)
		# get worldTcamera
		T_world_marker = self.getWorldMarkerTF(id)
		T_world_cam = T_world_marker @ T_marker_cam
		tf[:3] = T_world_cam[:3, 3]
		tf[3:] = R.from_matrix(T_world_cam[:3, :3]).as_euler('xyz')
		return tf
	
	def initialGuess(self, detection: dict) -> np.ndarray:
		if self.use_aruco:
			print(f"Cannot guess for id Aruco, return zero initilization")
			return np.zeros(6)
		# get pose for id with min detection error
		errs = [val['pose_err'] for val in detection.values()]
		min_err_idx = errs.index(min(errs))
		return self.camTF(detection, min_err_idx)

	def residuals(self, camera_pose: np.ndarray, marker_poses: dict, detection: dict) -> np.ndarray:
		"""Compute the residual (error) between world and detected poses.
			Rotations are extr. xyz euler angles.
		"""
		error = []
		# estimate
		T_world_camera = pose2Matrix(tvec=camera_pose[:3], rot=camera_pose[3:], rot_t=RotTypes.EULER)
		# invert for reprojection
		tvec_inv, euler_inv = invPersp(tvec=camera_pose[:3], rot=camera_pose[3:], rot_t=RotTypes.EULER)
		T_camera_world = pose2Matrix(tvec=tvec_inv, rot=euler_inv, rot_t=RotTypes.EULER)

		for id in marker_poses:
			det = detection.get(id)

			if det is not None:
				# detected tag pose wrt camera frame
				T_camera_marker = pose2Matrix(det['ftrans'], det['frot'], RotTypes.EULER)
				T_world_marker_est = T_world_camera @ T_camera_marker
				# measured tag pose wrt world 
				T_world_marker = self.getWorldMarkerTF(id)

				# errors
				position_error = np.linalg.norm(T_world_marker_est[:3, 3] - T_world_marker[:3, 3])
				orientation_error = np.linalg.norm(T_world_marker_est[:3, :3] - T_world_marker[:3, :3])
				error.append(position_error)  
				error.append(orientation_error)		

				# reprojection_error
				repr_err = self.projectSingleMarker(det, id, T_camera_world)
				# error.append(repr_err)

		return np.hstack(error) if len(error) else np.array(error)

	def estimatePoseLS(self, img: cv2.typing.MatLike, err: float, est_camera_pose: np.ndarray, detection: dict) -> np.ndarray:
		res = least_squares(self.residuals, 
							est_camera_pose, 
							args=(self.marker_table_poses, detection),
							method='trf', 
							bounds=(self.lower_bounds, self.upper_bounds),
							max_nfev=5000, # max iterations
							ftol=1e-8,    # tolerance for the cost function
							xtol=1e-8,    # tolerance for the solution parameters
							gtol=1e-8     # tolerance for the gradient
							)
		if res.success:
			opt_cam_pose = res.x
			# reproject markers
			errors = self.projectMarkers(detection, opt_cam_pose, img)
			reserr = np.mean(errors) if len(errors) else np.inf
			txt = f"Result: {res.status} {res.message}\n"
			txt += f"camera world pose trans: {opt_cam_pose[:3]}, rot (extr. xyz euler): {opt_cam_pose[3:]}\n"
			txt += f"reprojection error: {reserr}\n"
			txt += f"cost: {res.cost}\n"
			txt += f"evaluations: {res.nfev}\n"
			txt += f"optimality: {res.optimality}\n"
			print(txt)

			for id, error in self.reprojection_errors.items():
				if error > self.err_term:
					print("id {} reprojection error: {:.2f} > {} threshold".format(id, error, self.err_term))

			# put pose label
			self.labelDetection(img, -1, opt_cam_pose[:3], opt_cam_pose[3:], reserr)

			return reserr, opt_cam_pose
		
		print(f"Least squares failed: {res.status} {res.message}")
		return err, est_camera_pose
	
	def estimatePoseFL(self, img: cv2.typing.MatLike, err: float, detection: dict) -> np.ndarray:
		filter = None
		filtered_pose = np.zeros(6)
		for id in detection:
			T_world_cam = self.camTF(detection, id)
			if filter is None:
				filter = createFilter(self.filter_type, PoseFilterBase.poseToMeasurement(tvec=T_world_cam[:3], rot=T_world_cam[3:], rot_t=RotTypes.EULER), self.f_loop)
			else:
				filter.updateFilter(PoseFilterBase.poseToMeasurement(tvec=T_world_cam[:3], rot=T_world_cam[3:], rot_t=RotTypes.EULER))
		if filter is not None:
			filtered_pose[:3] = filter.est_translation
			filtered_pose[3:] = filter.est_rotation_as_euler
			self.labelDetection(img, 30, filtered_pose[:3], filtered_pose[3:])
			err = self.projectMarkers(detection, filtered_pose, img)
		print(f"camera world pose trans: {filtered_pose[:3]}, rot (extr. xyz euler): {filtered_pose[3:]}")
		return err, filtered_pose
		
	def run(self) -> None:
		init = True
		success = False
		err = np.inf
		est_camera_pose = np.zeros(6)
		rate = rospy.Rate(self.f_loop)

		try:
			while not rospy.is_shutdown():
					
					# detect markers 
					(marker_det, det_img, proc_img, _) = self.preProcImage()

					for id, det in marker_det.items():
						success, tvec, rvec, _ = ransacPose(det['ftrans'], getRotation(det['frot'], RotTypes.EULER, RotTypes.RVEC), det['corners'], det['points'], self.det.cmx, self.det.dist)
						marker_det[id]['ftrans'] = tvec
						marker_det[id]['frot'] = getRotation(rvec, RotTypes.RVEC, RotTypes.EULER)

					# initially show 
					if self.vis and self.cv_window:
						cv2.imshow('Processed', proc_img)
						cv2.imshow('Detection', det_img)
						if cv2.waitKey(10000 if init else 1) == ord("q"):
							break

					# estimate cam pose
					if marker_det:

						initial_guess = est_camera_pose
						if init:
							init = False
							initial_guess = self.initialGuess(marker_det)
						
						print("Running estimation")
						(err, est_camera_pose) = self.estimatePoseLS(det_img, err, initial_guess, marker_det)

						if err <= self.err_term:
							print(f"Estimated camera pose xyz (m): {est_camera_pose[:3]}, extr. xyz Euler angles (rad): {est_camera_pose[3:]}, mean reprojection error: {err}")
							success = True

					if self.vis:
						# frame counter
						cv2.putText(det_img, str(self.frame_cnt), (det_img.shape[1]-40, 20), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.FONT_CLR, self.FONT_THCKNS, cv2.LINE_AA)
						for id, det in marker_det.items():
							# label marker pose
							self.labelDetection(det_img, id, det['ftrans'], det['frot'])
							# plot pose by id
							if id == self.plt_id and self.cv_window:
								cv2.imshow('Plot', cv2.cvtColor(visPose(det['ftrans'], getRotation(det['frot'], RotTypes.EULER, RotTypes.MAT), det['frot'], self.frame_cnt, cla=True), cv2.COLOR_RGBA2BGR))

						if self.cv_window:
							cv2.imshow('Processed', proc_img)
							cv2.imshow('Detection', det_img)
							if cv2.waitKey(100000 if success else 1) == ord("q"):
								break
					
					if success:
						break

					try:
						rate.sleep()
					except:
						pass

		except Exception as e:
			print(e)

		finally:
			cv2.destroyAllWindows()

def main() -> None:
	rospy.init_node('dataset_collector')
	if rospy.get_param('~debug', False):
		DetectBase(camera_ns=rospy.get_param('~markers_camera_name', ''),
				   marker_length=rospy.get_param('~marker_length', 0.010),
				   use_reconfigure=rospy.get_param('~use_reconfigure', False),
				   vis=rospy.get_param('~vis', True),
				   filter_type=rospy.get_param('~filter', 'none'),
				   filter_iters=rospy.get_param('~filter_iters', 10),
				   f_ctrl=rospy.get_param('~f_ctrl', 30),
				   use_aruco=rospy.get_param('~use_aruco', False),
				   plt_id=rospy.get_param('~plot_id', -1),
				   test=rospy.get_param('~test', False),
				   fps=rospy.get_param('~fps', 30.0),
				   refine_pose=True,
				   flip_outliers=True,
				   ).runDebug()
	elif rospy.get_param('~camera_pose', False):
		CameraPoseDetect(camera_ns=rospy.get_param('~markers_camera_name', ''),
				   		fn=rospy.get_param('~marker_poses_file', 'marker_holder_poses.yml'),
						marker_length=rospy.get_param('~marker_length', 0.010),
						use_reconfigure=rospy.get_param('~use_reconfigure', False),
						vis=rospy.get_param('~vis', True),
						filter_type=rospy.get_param('~filter', 'none'),
						filter_iters=rospy.get_param('~filter_iters', 10),
						f_ctrl=rospy.get_param('~f_ctrl', 30),
						use_aruco=rospy.get_param('~use_aruco', False),
						plt_id=rospy.get_param('~plot_id', -1),
				   		fps=rospy.get_param('~fps', 30.0),
						err_term=rospy.get_param('~err_term', 2.0),
						).run()
	elif rospy.get_param('~qdec_detect', False):
		HybridDetect(camera_ns=rospy.get_param('~markers_camera_name', ''),
					 marker_length=rospy.get_param('~marker_length', 0.010),
					 use_reconfigure=rospy.get_param('~use_reconfigure', False),
					 vis=rospy.get_param('~vis', True),
					 filter_type=rospy.get_param('~filter', 'none'),
					 filter_iters=rospy.get_param('~filter_iters', 10),
					 f_ctrl=rospy.get_param('~f_ctrl', 30),
					 use_aruco=rospy.get_param('~use_aruco', False),
					 plt_id=rospy.get_param('~plot_id', -1),
					 rh8d_port=rospy.get_param('~rh8d_port', "/dev/ttyUSB1"),
					 rh8d_baud=rospy.get_param('~rh8d_baud', 1000000),
 					 qdec_port=rospy.get_param('~qdec_port', '/dev/ttyUSB0'),
					 qdec_baud=rospy.get_param('~qdec_baud', 19200),
					 qdec_filter_iters=rospy.get_param('~qdec_filter_iters', 200),
					 actuator=rospy.get_param('~actuator', 32),
					 step_div=rospy.get_param('~step_div', 100),
					 epochs=rospy.get_param('~epochs', 100),
					 test=rospy.get_param('~test', False),
					 fps=rospy.get_param('~fps', 30.0),
					 ).run()
	else:
		KeypointDetect(camera_ns=rospy.get_param('~markers_camera_name', ''),
						marker_length=rospy.get_param('~marker_length', 0.010),
						use_reconfigure=rospy.get_param('~use_reconfigure', False),
						vis=rospy.get_param('~vis', True),
						filter_type=rospy.get_param('~filter', 'none'),
						filter_iters=rospy.get_param('~filter_iters', 10),
						f_ctrl=rospy.get_param('~f_ctrl', 30),
						use_aruco=rospy.get_param('~use_aruco', False),
						plt_id=rospy.get_param('~plot_id', -1),
						use_tf=rospy.get_param('~use_tf', False),
						step_div=rospy.get_param('~step_div', 100),
					 	epochs=rospy.get_param('~epochs', 100),
						rh8d_port=rospy.get_param('~rh8d_port', "/dev/ttyUSB1"),
					 	rh8d_baud=rospy.get_param('~rh8d_baud', 1000000),
						test=rospy.get_param('~test', False),
				   		fps=rospy.get_param('~fps', 30.0),
						attached=rospy.get_param('~attached', False),
						use_eye_cameras=rospy.get_param('~use_eye_cameras', False),
						use_top_camera=rospy.get_param('~use_top_camera', False),
						use_head_camera=rospy.get_param('~use_head_camera', False),
						save_imgs=rospy.get_param('~save_imgs', False),
						depth_enabled=rospy.get_param('~depth_enabled', False),
						top_camera_name=rospy.get_param('~top_camera_name', 'top_camera'),
						head_camera_name=rospy.get_param('~head_camera_name', 'head_camera'),
						left_eye_camera_name=rospy.get_param('~left_eye_camera_name', 'left_eye_camera'),
						right_eye_camera_name=rospy.get_param('~right_eye_camera_name', 'right_eye_camera'),
						joint_state_topic=rospy.get_param('~joint_state_topic', 'joint_states'),
						waypoint_set=rospy.get_param('~waypoint_set', 'waypoints.json'),
						waypoint_start_idx=rospy.get_param('~waypoint_start_idx', 0),
						).run()
	
if __name__ == "__main__":
	main()
