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
import pybullet as pb
import sensor_msgs.msg
from pynput import keyboard
from collections import deque
from std_msgs.msg import Bool
import dynamic_reconfigure.server
from typing import Optional, Any, Tuple
from scipy.optimize import least_squares
from sensor_msgs.msg import Image, JointState
from realsense2_camera.msg import Extrinsics
from scipy.spatial.transform import Rotation as R
from message_filters import Subscriber, ApproximateTimeSynchronizer
from util import *
from move import *
from plot_record import *
from pose_filter import *
from qdec_serial import *
from rh8d_serial import *
from nicol_rh8d.cfg import DetectorConfig
from marker_detector import ArucoDetector, AprilDetector
np.set_printoptions(threshold=sys.maxsize, suppress=True)

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
		self.cv_window = cv_window and vis
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
			self.rgb_info = rospy.wait_for_message(camera_ns + '/camera_info', sensor_msgs.msg.CameraInfo, 25)
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
		if cv_window:
			cv2.namedWindow("Processed", cv2.WINDOW_NORMAL)
			cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
			if plt_id > -1:
				cv2.namedWindow("Plot", cv2.WINDOW_NORMAL)
				setPosePlot()
	
		# init dynamic reconfigure
		if use_reconfigure:
			print("Using reconfigure server")
			self.det_config_server = dynamic_reconfigure.server.Server(DetectorConfig, self.det.setDetectorParams)

	def flipOutliers(self, marker_detections: dict, tolerance: float=0.5, exclude_ids: list=[6,7,8,9], normal_type: NormalTypes=NormalTypes.XZ) -> bool:
		"""Check if all Z axes are oriented similarly and 
			  flip orientation for outliers. 
		"""
		# TODO: divide between fingers and thumb

		# exclude markers from check
		detections = {id: det for id, det in marker_detections.items() if id not in exclude_ids}
		# get ids
		marker_ids = list(detections.keys())
		# extract filtered rotations
		rotations = [getRotation(marker_det['frot'], RotTypes.EULER, RotTypes.MAT)  for marker_det in detections.values()]

		# get axis idx
		axis_idx = NORMAL_IDX_MAP[normal_type]
		# find outliers
		outliers, axis_avg = findAxisOrientOutliers(rotations, tolerance=tolerance, axis_idx=axis_idx)

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
				axs = mat[:, axis_idx] / np.linalg.norm(mat[:, axis_idx])
				# check angular distance to average
				if abs( np.dot(axs, axis_avg) ) > tolerance:
					# set other rot
					marker_detections[mid]['rot_mat'] = mat
					marker_detections[mid]['rvec'] = rvec.flatten()
					marker_detections[mid]['frot'] = getRotation(mat, RotTypes.MAT, RotTypes.EULER)
					# set other trans
					marker_detections[mid]['ftrans'] = tvec.flatten()
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
		@param sync_slop
		@type float
		@param urdf_pth
		@type str
		@param make_noise
		@type bool
		@param self_reset_angles
		@type bool
		@param topic_wait_secs
		@type float
		@param vis_ros_tf
		@type bool

	"""

	# angle comp.
	UNIT_AXIS_X = np.array([1, 0, 0], dtype=np.float32)
	UNIT_AXIS_Y = np.array([0, 1, 0], dtype=np.float32)
	UNIT_AXIS_Z = np.array([0, 0, 1], dtype=np.float32)

	# cv stuff
	FIRST_LINE_INTERPOL_FACTOR = 0.5 # arc points interpolation
	SEC_LINE_INTERPOL_FACTOR = 0.85 # arc points interpolation
	ARC_SHIFT = 10 # ellipse param
	SAGITTA = 20 # radius size
	AXES_LEN = 0.015 # meter
	X_AXIS = np.array([[-AXES_LEN,0,0], [AXES_LEN, 0, 0]], dtype=np.float32)
	ELIPSE_COLOR = (255,255,255)
	ELIPSE_THKNS = 2
	X_EL_TXT_OFFSET = 0.95
	Y_EL_TXT_OFFSET = 1

	KEYPT_THKNS = 10
	KEYPT_LINE_THKNS = 2
	CHAIN_COLORS = [(255,255,255), (0,0,255), (0,255,0), (255,0,0), (0,255,255), (255,255,0)]
	PLOT_COLORS = ['w', 'r', 'g', 'b', 'y', 'c']

	# tf rames to track
	TF_TARGET_FRAME = 'world'
	RH8D_SOURCE_FRAME = 'r_forearm' 
	RH8D_TCP_SOURCE_FRAME = 'r_laser' 
	HEAD_RS_SOURCE_FRAME = 'head_realsense_link' # optical frame?
	LEFT_EYE_SOURCE_FRAME = 'left_eye_cam'
	RIGHT_EYE_SOURCE_FRAME = 'right_eye_cam'

	# dataframe columns
	JS_COLS = ['name', 'position', 'velocity', 'effort', 'timestamp']
	TF_COLS = ['frame_id', 'child_frame_id', 'trans', 'quat', 'timestamp']
	FK_COLS = ['timestamp', 'trans', 'quat']
	KEYPT_COLS = ['timestamp', 'trans', 'rot_mat']
	DET_COLS = [ 'angle', 
			 					'start_angle',
								'base_id', 
								'target_id', 
								'frame_cnt', 
								'rec_cnt', 
								'cmd', 
								'state',
								'epoch', 
								'direction', 
								'description', 
								'target_trans', 
								'target_rot', 
								'base_trans', 
								'base_rot', 
								'parent_frame_id',
								'child_frame_id',
								'timestamp',
								]

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
				epochs: Optional[int]=10,
				use_tf: Optional[bool]=False,
				test: Optional[bool]=False,
				flip_outliers: Optional[bool]=False,
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
				actuator_state_topic: Optional[str]='actuator_states',
				waypoint_set: Optional[str]='waypoints.json',
				waypoint_start_idx: Optional[int]=0,
				sync_slop: Optional[float]=0.08,
				urdf_pth: Optional[str]='rh8d.urdf',
				make_noise: Optional[bool]=False,
				self_reset_angles: Optional[bool]=True,
				topic_wait_secs: Optional[float]=15,
				vis_ros_tf: Optional[bool]=False,
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
						test=False,
						vis=vis,
						fps=fps,
						cv_window=not attached,
						)

		self.fps = fps
		self.test = test
		self.use_tf = use_tf
		self.epochs = epochs
		self.attached = attached
		self.sync_slop = sync_slop
		self.save_record = save_imgs
		self.make_noise = make_noise
		self.use_eye_cameras = use_eye_cameras
		self.use_top_camera = use_top_camera
		self.use_head_camera = use_head_camera
		self.depth_enabled = depth_enabled
		self.camera_ns = camera_ns
		self.self_reset_angles = self_reset_angles
		self.top_camera_name = top_camera_name
		self.head_camera_name = head_camera_name
		self.left_eye_camera_name = left_eye_camera_name
		self.right_eye_camera_name = right_eye_camera_name
		self.actuator_state_topic = actuator_state_topic
		self.joint_state_topic = actuator_state_topic.replace('actuator', 'joint')
		self.record_cnt = 0
		self.data_cnt = 0
		self.subs = []
		self.topic_wait_secs = topic_wait_secs
		self.vis_ros_tf = vis_ros_tf

		# message buffers
		self.det_img_buffer = deque(maxlen=self.filter_iters)
		self.depth_img_buffer = deque(maxlen=1)
		self.top_img_buffer = deque(maxlen=1)
		self.top_depth_img_buffer = deque(maxlen=1)
		self.head_img_buffer = deque(maxlen=1)
		self.head_depth_img_buffer = deque(maxlen=1)
		self.left_img_buffer = deque(maxlen=1)
		self.right_img_buffer = deque(maxlen=1)
		self.actuator_state_buffer = deque(maxlen=1)
		self.joint_state_buffer = deque(maxlen=1)
		self.buf_lock = threading.Lock()

		# data frames
		self.as_df = pd.DataFrame(columns=self.JS_COLS)
		self.js_df = pd.DataFrame(columns=self.JS_COLS)
		self.rh8d_tf_df = pd.DataFrame(columns=self.TF_COLS)
		self.rh8d_tcp_tf_df = pd.DataFrame(columns=self.TF_COLS)
		self.head_rs_tf_df = pd.DataFrame(columns=self.TF_COLS)
		self.left_eye_tf_df = pd.DataFrame(columns=self.TF_COLS)
		self.right_eye_tf_df = pd.DataFrame(columns=self.TF_COLS)

		# init ros
		if use_tf:
			self.buf = tf2_ros.Buffer(cache_time=rospy.Duration(180))
			self.listener = tf2_ros.TransformListener(self.buf, tcp_nodelay=True)

		# init controller
		if attached:
			self.rh8d_ctrl = MoveRobot()
		else:
			self.rh8d_ctrl = RH8DSerial(rh8d_port, rh8d_baud) if not test else RH8DSerialStub()
		
		# record directories
		if save_imgs and not self.test:
			mkDirs()

		# load waypoints
		fl = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets/detection/keypoint/waypoints", waypoint_set)
		self.waypoint_df = pd.read_json(fl, orient='index')
		if waypoint_start_idx > 0:
			self.waypoint_df = self.waypoint_df.iloc[waypoint_start_idx :]
			
		# load camera extrinsics
		fl = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cfg/camera_poses.yaml")
		with open(fl, 'r') as fr:
			extrinsics = yaml.safe_load(fr)
			# marker cam
			marker_cam_holder_to_cam_extr = extrinsics['marker_camera_optical_frame']
			self.marker_cam_extr = Extrinsics()
			self.marker_cam_extr.header.stamp = rospy.Time.now()
			self.marker_cam_extr.header.frame_id = 'marker_camera_optical_frame'
			self.marker_cam_extr.translation = np.array(marker_cam_holder_to_cam_extr['xyz'], dtype=np.float32)
			self.marker_cam_extr.rotation = getRotation(marker_cam_holder_to_cam_extr['rpy'], RotTypes.EULER, RotTypes.MAT)
			# top cam
			if use_top_camera:
				world_to_top_cam_extr = extrinsics['top_camera_optical_frame']
				self.top_cam_extr = Extrinsics()
				self.top_cam_extr.header.stamp = rospy.Time.now()
				self.top_cam_extr.header.frame_id = 'top_camera_optical_frame'
				self.top_cam_extr.translation = np.array(world_to_top_cam_extr['xyz'], dtype=np.float32)
				self.top_cam_extr.rotation = getRotation(world_to_top_cam_extr['rpy'], RotTypes.EULER, RotTypes.MAT)

		# load marker config			
		self.marker_config = loadMarkerConfig()
		self.start_angles = {} # initial angles
		self.root_joint = list(self.marker_config.keys())[0] # first joint
		self.det_df_dict = {joint:  pd.DataFrame(columns=self.DET_COLS) for joint in self.marker_config.keys()} # angles dataset
		
		# init 3D keypoints dataset
		self.keypt_keys = [self.marker_config[self.root_joint]['parent']] # base link
		self.keypt_keys.extend(list(self.marker_config.keys())) # (joint) links
		self.keypt_keys.extend(self.initRH8DFK(urdf_pth)) # end links
		self.rh8d_tf_df_dict = {link:  pd.DataFrame(columns=self.FK_COLS) for link in self.keypt_keys} 
		self.keypt_df_dict = {link:  pd.DataFrame(columns=self.KEYPT_COLS) for link in self.keypt_keys[1:]} 
		self.keypt_plot = KeypointPlot()

		# get tf links
		self.tf_links = []
		for config in self.marker_config.values():
			link = config['child']
			self.tf_links.append(link)
			end_link = config.get('fixed_end')
			if end_link is not None:
				self.tf_links.append(end_link.replace('r_', '').replace('joint', 'r'))
			
		# ros topics
		if attached:
			self.initSubscriber()
			self.clear_sub = rospy.Subscriber("clear_angles", Bool, self.clearAngles, queue_size=1)
			# init ros vis
			if vis:
				self.proc_pub = rospy.Publisher('processed_image', Image, queue_size=10)
				self.det_pub = rospy.Publisher('marker_detection', Image, queue_size=10)
				self.out_pub = rospy.Publisher('angle_detection', Image, queue_size=10)
				self.plot_pub = rospy.Publisher('keypoint_plot', Image, queue_size=10)
		# cv vis
		if self.cv_window:
			cv2.namedWindow("Plot", cv2.WINDOW_NORMAL)
			cv2.namedWindow("Output", cv2.WINDOW_NORMAL)

	def saveCamInfo(self, info: sensor_msgs.msg.CameraInfo, fn: str, extr: Extrinsics=None, tf: dict=None) -> None:
		if not self.save_record:
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
			rot = list(extr.rotation)
			trans = list(extr.translation)
			if isinstance(extr.rotation, np.ndarray):
				rot = extr.rotation.tolist()
				trans = extr.translation.tolist()
			extr_dict = {extr.header.frame_id: {'rotation': rot, 'translation': trans}}
			info_dict.update(extr_dict)

		if tf is not None:
			info_dict.update(tf)

		if self.save_record and not self.test:
			with open(fn, "w") as fw:
				yaml.dump(info_dict, fw, default_flow_style=None, sort_keys=False)

	def initRH8DFK(self, urdf_pth: str) -> list:
		# init pybullet
		self.physicsClient = pb.connect(pb.DIRECT)
		self.robot_id = pb.loadURDF(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'urdf/' + urdf_pth), useFixedBase=True)

		# get all present chains
		self.kinematic_chains = [[]]
		dfsKinematicChain(self.root_joint, self.kinematic_chains, self.marker_config)
		self.chain_complete = [False for _ in range(len(self.kinematic_chains))]
		# revolute end joints to fixed end joint mapping
		end_joints = [chain[-1] for chain in self.kinematic_chains]
		fixed_end_joints = [self.marker_config[joint]['fixed_end'] for joint in end_joints]

		# get joint index aka link index
		self.link_info_dict = {}
		self.joint_info_dict = {}
		for idx in range(pb.getNumJoints(self.robot_id)):
			joint_info = pb.getJointInfo(self.robot_id, idx)

			if joint_info[2] == pb.JOINT_REVOLUTE:
				joint_name = joint_info[1].decode('utf-8')
				# revolute joint index
				self.joint_info_dict.update( {joint_name: idx} )
			elif joint_info[2] == pb.JOINT_FIXED:
				joint_name = joint_info[1].decode('utf-8')
				if joint_name in fixed_end_joints:
					# fixed joint index matches very last link
					self.link_info_dict.update( {end_joints[fixed_end_joints.index(joint_name)]: {'index': idx, 'fixed_end': joint_name}} )
		
		return fixed_end_joints

	def initSubscriber(self) -> None:
		rospy.wait_for_message(self.actuator_state_topic, JointState, self.topic_wait_secs)
		self.actuator_states_sub = Subscriber(self.actuator_state_topic, JointState)
		self.subs.append(self.actuator_states_sub)
		
		rospy.wait_for_message(self.joint_state_topic, JointState, self.topic_wait_secs)
		self.joint_states_sub = Subscriber(self.joint_state_topic, JointState)
		self.subs.append(self.joint_states_sub)

		self.det_img_sub = Subscriber(self.img_topic, Image)
		self.subs.append(self.det_img_sub)
		# save camera info
		self.saveCamInfo(self.rgb_info, os.path.join(KEYPT_DET_REC_DIR, "rs_d415_info_rgb.yaml"), self.marker_cam_extr)

		if self.depth_enabled:
			self.depth_topic = self.camera_ns.replace('color', 'depth')
			depth_img_topic = self.depth_topic + '/image_rect_raw'	
			self.depth_img_sub = Subscriber(depth_img_topic,  Image)
			self.subs.append(self.depth_img_sub)
			# save camera info
			print("Waiting for camera_info from", self.depth_topic + '/camera_info')
			depth_info = rospy.wait_for_message(self.depth_topic + '/camera_info', sensor_msgs.msg.CameraInfo, self.topic_wait_secs)
			extr_info = rospy.wait_for_message(self.camera_ns.replace('/color', '') + '/extrinsics/depth_to_color', Extrinsics, self.topic_wait_secs)
			self.saveCamInfo(depth_info, os.path.join(KEYPT_ORIG_REC_DIR, "rs_d415_info_depth.yaml"), extr_info)

		if self.use_top_camera:
			self.top_img_topic = self.top_camera_name + '/image_raw'
			self.top_img_sub = Subscriber(self.top_img_topic, Image)
			self.subs.append(self.top_img_sub)
			# save camera info
			print("Waiting for camera_info from", self.top_camera_name + '/camera_info')
			rgb_info = rospy.wait_for_message(self.top_camera_name + '/camera_info', sensor_msgs.msg.CameraInfo, self.topic_wait_secs)
			self.saveCamInfo(rgb_info, os.path.join(KEYPT_TOP_CAM_REC_DIR, "rs_d435IF_info_rgb.yaml"), self.top_cam_extr)
			if self.depth_enabled:
				self.top_depth_topic = self.top_camera_name.replace('color', 'depth')
				top_depth_img_topic = self.top_depth_topic + '/image_rect_raw'	
				self.top_depth_img_sub = Subscriber(top_depth_img_topic, Image)
				self.subs.append(self.top_depth_img_sub)
				# save camera info
				print("Waiting for camera_info from", self.top_depth_topic + '/camera_info')
				depth_info = rospy.wait_for_message(self.top_depth_topic + '/camera_info', sensor_msgs.msg.CameraInfo, self.topic_wait_secs)
				extr_info = rospy.wait_for_message(self.top_camera_name.replace('/color', '') + '/extrinsics/depth_to_color', Extrinsics, self.topic_wait_secs)
				self.saveCamInfo(depth_info, os.path.join(KEYPT_TOP_CAM_REC_DIR, "rs_d435IF_info_depth.yaml"), extr_info)

		if self.use_head_camera:
			self.head_img_topic = self.head_camera_name + '/image_raw'
			self.head_img_sub = Subscriber(self.head_img_topic, Image)
			self.subs.append(self.head_img_sub)
			# save camera info
			print("Waiting for camera_info from", self.head_camera_name + '/camera_info')
			rgb_info = rospy.wait_for_message(self.head_camera_name + '/camera_info', sensor_msgs.msg.CameraInfo, self.topic_wait_secs)
			self.saveCamInfo(rgb_info, os.path.join(KEYPT_HEAD_CAM_REC_DIR, "rs_d435I_info_rgb.yaml")) # dynamic extrinsics
			if self.depth_enabled:
				self.head_depth_topic = self.head_camera_name.replace('color', 'depth')
				head_depth_img_topic = self.head_depth_topic + '/image_rect_raw'	
				self.head_depth_img_sub = Subscriber(head_depth_img_topic, Image)
				self.subs.append(self.head_depth_img_sub)
				# save camera info
				print("Waiting for camera_info from", self.head_depth_topic + '/camera_info')
				depth_info = rospy.wait_for_message(self.head_depth_topic + '/camera_info', sensor_msgs.msg.CameraInfo, self.topic_wait_secs)
				extr_info = rospy.wait_for_message(self.head_camera_name.replace('/color', '') + '/extrinsics/depth_to_color', Extrinsics, self.topic_wait_secs)
				self.saveCamInfo(depth_info, os.path.join(KEYPT_HEAD_CAM_REC_DIR, "rs_d435I_info_depth.yaml"), extr_info)

		if self.use_eye_cameras:
			# right eye
			self.right_img_topic = self.right_eye_camera_name + '/image_raw'
			self.right_img_sub = Subscriber(self.right_img_topic, Image)
			self.subs.append(self.right_img_sub)
			# save camera info
			print("Waiting for camera_info from", self.right_eye_camera_name + '/camera_info')
			rgb_info = rospy.wait_for_message(self.right_eye_camera_name + '/camera_info', sensor_msgs.msg.CameraInfo, self.topic_wait_secs)
			self.saveCamInfo(rgb_info, os.path.join(KEYPT_R_EYE_REC_DIR, "See3CAM_CU135_info_rgb.yaml")) # dynamic extrinsics
			# left eye
			self.left_img_topic = self.left_eye_camera_name + '/image_raw'
			self.left_img_sub = Subscriber(self.left_img_topic, Image)
			self.subs.append(self.left_img_sub)
			# save camera info
			print("Waiting for camera_info from", self.left_eye_camera_name + '/camera_info')
			rgb_info = rospy.wait_for_message(self.left_eye_camera_name + '/camera_info', sensor_msgs.msg.CameraInfo, self.topic_wait_secs)
			self.saveCamInfo(rgb_info, os.path.join(KEYPT_L_EYE_REC_DIR, "See3CAM_CU135_info_rgb.yaml")) # dynamic extrinsics
		
		# trigger condition
		self.depth_frame_id = self.camera_ns.replace('color', 'depth').replace('/', '_')
		self.top_img_frame_id = self.top_camera_name.replace('/', '_')
		self.top_depth_frame_id = self.top_camera_name.replace('color', 'depth').replace('/', '_')
		self.head_img_frame_id = self.head_camera_name.replace('/', '_')
		self.head_depth_frame_id = self.head_camera_name.replace('color', 'depth').replace('/', '_')

		# sync callback
		self.sync = ApproximateTimeSynchronizer(self.subs, queue_size=10, slop=self.sync_slop)
		self.sync.registerCallback(self.recCB)
		for s in self.subs:
			print("Subscribed to topic", s.topic)
		print("Syncing all subscibers")

	def recCB(self, 
				actuator_state: JointState, 
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
			self.actuator_state_buffer.append(actuator_state)
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

	def clearAngles(self, msg: Bool) -> None:
		if msg.data:
			self.start_angles.clear()

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

	def saveRecord(self) -> Tuple[dict, dict]:	
		num_saved = 0
		tf_rh8d_dict = {}
		actuator_states_dict = {}

		# actuator states and rh8d tf
		if len(self.actuator_state_buffer):
			num_saved += 1
			msg = self.actuator_state_buffer.pop()
			state_dict = {'name': msg.name,
							'position': msg.position,
							'velocity': msg.velocity,
							'effort': msg.effort,
							'timestamp': msg.header.stamp.secs + msg.header.stamp.nsecs*1e-9
						 }
			self.as_df = pd.concat([self.as_df, pd.DataFrame([state_dict])], ignore_index=True)
			actuator_states_dict = dict(zip(msg.name, msg.position))
			
			# save rh8d tf
			if self.use_tf:
				# forearm tf
				tf_rh8d_dict = self.lookupTF(msg.header.stamp, self.TF_TARGET_FRAME, self.RH8D_SOURCE_FRAME)
				if tf_rh8d_dict:
					self.rh8d_tf_df = pd.concat([self.rh8d_tf_df, pd.DataFrame([tf_rh8d_dict])], ignore_index=True)
				# tcp tf
				tf_rh8d_tcp_dict = self.lookupTF(msg.header.stamp, self.TF_TARGET_FRAME, self.RH8D_TCP_SOURCE_FRAME)
				if tf_rh8d_tcp_dict:
					self.rh8d_tcp_tf_df = pd.concat([self.rh8d_tcp_tf_df, pd.DataFrame([tf_rh8d_tcp_dict])], ignore_index=True)
			
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

		# marker cam depth img
		if self.depth_enabled and len(self.depth_img_buffer):
			num_saved += 1
			depth_img = self.bridge.imgmsg_to_cv2(self.depth_img_buffer.pop(), 'passthrough')
			depth_img = (depth_img).astype('float32')
			if self.save_record and not self.test:
				cv2.imwrite(os.path.join(KEYPT_ORIG_REC_DIR, str(self.record_cnt) + '.tiff'), depth_img)

		# top realsense
		if self.use_top_camera and len(self.top_img_buffer):
			num_saved += 1
			raw_img = self.bridge.imgmsg_to_cv2(self.top_img_buffer.pop(), 'bgr8')
			if self.save_record and not self.test:
				cv2.imwrite(os.path.join(KEYPT_TOP_CAM_REC_DIR, str(self.record_cnt) + '.jpg'), raw_img, [int(cv2.IMWRITE_JPEG_QUALITY), JPG_QUALITY])
			# depth
			if self.depth_enabled and len(self.top_depth_img_buffer):
				depth_img = self.bridge.imgmsg_to_cv2(self.top_depth_img_buffer.pop(), 'passthrough')
				depth_img = (depth_img).astype('float32')
				if self.save_record and not self.test:
					cv2.imwrite(os.path.join(KEYPT_TOP_CAM_REC_DIR, str(self.record_cnt) + '.tiff'), depth_img)
			# static tf

		# head realsense imgs & tf
		if self.use_head_camera and len(self.head_img_buffer):
			num_saved += 1
			msg = self.head_img_buffer.pop()
			raw_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
			if self.save_record and not self.test:
				cv2.imwrite(os.path.join(KEYPT_HEAD_CAM_REC_DIR, str(self.record_cnt) + '.jpg'), raw_img, [int(cv2.IMWRITE_JPEG_QUALITY), JPG_QUALITY])
			# depth
			if self.depth_enabled and len(self.head_depth_img_buffer):
				depth_img = self.bridge.imgmsg_to_cv2(self.head_depth_img_buffer.pop(), 'passthrough')
				depth_img = (depth_img).astype('float32')
				if self.save_record and not self.test:
					cv2.imwrite(os.path.join(KEYPT_HEAD_CAM_REC_DIR, str(self.record_cnt) + '.tiff'), depth_img)
			# save head rs tf
			if self.use_tf:
				tf_dict = self.lookupTF(msg.header.stamp, self.TF_TARGET_FRAME, self.HEAD_RS_SOURCE_FRAME)
				if tf_dict:
					self.head_rs_tf_df = pd.concat([self.head_rs_tf_df, pd.DataFrame([tf_dict])], ignore_index=True)

		# eyes imgs & tf
		if self.use_eye_cameras:
			num_saved += 1
			if len(self.left_img_buffer):
				msg = self.left_img_buffer.pop()
				raw_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
				if self.save_record and not self.test:
					cv2.imwrite(os.path.join(KEYPT_L_EYE_REC_DIR, str(self.record_cnt) + '.jpg'), raw_img, [int(cv2.IMWRITE_JPEG_QUALITY), JPG_QUALITY])
				# save left eye tf
				if self.use_tf:
					tf_dict = self.lookupTF(msg.header.stamp, self.TF_TARGET_FRAME, self.LEFT_EYE_SOURCE_FRAME)
					if tf_dict:
						self.left_eye_tf_df = pd.concat([self.left_eye_tf_df, pd.DataFrame([tf_dict])], ignore_index=True)

			if len(self.right_img_buffer):
				msg = self.right_img_buffer.pop()
				raw_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
				if self.save_record and not self.test:
					cv2.imwrite(os.path.join(KEYPT_R_EYE_REC_DIR, str(self.record_cnt) + '.jpg'), raw_img, [int(cv2.IMWRITE_JPEG_QUALITY), JPG_QUALITY])
				# save right eye tf
				if self.use_tf:
					tf_dict = self.lookupTF(msg.header.stamp, self.TF_TARGET_FRAME, self.RIGHT_EYE_SOURCE_FRAME)
					if tf_dict:
						self.right_eye_tf_df = pd.concat([self.right_eye_tf_df, pd.DataFrame([tf_dict])], ignore_index=True)

		# check df lengths
		if self.record_cnt != self.as_df.last_valid_index():
			print("Record size deviates as dataframe index:", self.record_cnt, "!=", self.as_df.last_valid_index())
		if self.record_cnt != self.js_df.last_valid_index():
			print("Record size deviates js dataframe index:", self.record_cnt, "!=", self.js_df.last_valid_index())
		if self.use_tf:
			if self.record_cnt != self.rh8d_tf_df.last_valid_index():
				print("Record size deviates from rh8d tf dataframe index:", self.record_cnt, "!=", self.rh8d_tf_df.last_valid_index())
			if self.record_cnt != self.rh8d_tcp_tf_df.last_valid_index():
				print("Record size deviates from rh8d tf dataframe index:", self.record_cnt, "!=", self.rh8d_tcp_tf_df.last_valid_index())
			if self.use_eye_cameras:
				if self.record_cnt != self.right_eye_tf_df.last_valid_index():
					print("Record size deviates from right eye dataframe index:", self.record_cnt, "!=", self.right_eye_tf_df.last_valid_index())
				if self.record_cnt != self.left_eye_tf_df.last_valid_index():
					print("Record size deviates from left eye dataframe index:", self.record_cnt, "!=", self.left_eye_tf_df.last_valid_index())
			if self.use_head_camera:
				if self.record_cnt != self.head_rs_tf_df.last_valid_index():
					print("Record size deviates from head rs dataframe index:", self.record_cnt, "!=", self.head_rs_tf_df.last_valid_index())
		
		# increment image counter
		if num_saved > 0:
			self.record_cnt += 1

		return tf_rh8d_dict, actuator_states_dict
	
	def waitForImgs(self) -> None:
		# wait for buffer to be filled
		while len(self.det_img_buffer) != self.filter_iters:
			if not rospy.is_shutdown():
				rospy.logwarn_throttle(1.0, f"Image buffer size deviates from filter size: {str(self.filter_iters-len(self.det_img_buffer))}")
				rospy.sleep(1/self.fps)
		# wait for buffers to be filled
		while not len(self.actuator_state_buffer):
			if not rospy.is_shutdown():
				rospy.logwarn_throttle(1.0, "Actuator state buffer not updated")
				rospy.sleep(0.01)
		while not len(self.joint_state_buffer):
			if not rospy.is_shutdown():
				rospy.logwarn_throttle(1.0, "Joint state buffer not updated")
				rospy.sleep(0.01)
		if self.use_eye_cameras:
			while not len(self.left_img_buffer):
				if not rospy.is_shutdown():
					rospy.logwarn_throttle(1.0, "Left img buffer not updated")
					rospy.sleep(1/self.fps)
			while not len(self.right_img_buffer):
				if not rospy.is_shutdown():
					rospy.logwarn_throttle(1.0, "Right img buffer not updated")
					rospy.sleep(1/self.fps)
		if self.use_top_camera:
			while not len(self.top_img_buffer):
				if not rospy.is_shutdown():
					rospy.logwarn_throttle(1.0, "Top img buffer not updated")
					rospy.sleep(1/self.fps)
		if self.use_head_camera:
			while not len(self.head_img_buffer):
				if not rospy.is_shutdown():
					rospy.logwarn_throttle(1.0, "Head img buffer not updated")
					rospy.sleep(1/self.fps)

	def preProcImage(self, vis: bool=True) -> Tuple[dict, cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike, dict, dict, float]:
		""" Put num filter_iters images into fresh detection filter and save last images."""

		tf_dict = {}
		states_dict = {}
		timestamp = 0.0

		# blocking
		self.waitForImgs()

		# lock updates on img queues
		with self.buf_lock:

			# process images
			self.det.resetFilters()
			while len(self.det_img_buffer):
				self.frame_cnt += 1
				msg = self.det_img_buffer.pop()
				timestamp =  msg.header.stamp.secs + msg.header.stamp.nsecs*1e-9
				raw_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
				(marker_det, det_img, proc_img) = self.det.detMarkerPoses(raw_img.copy(), vis and (not len(self.det_img_buffer) and self.vis))

			# align rotations by consens
			if self.flip_outliers:
				if not self.flipOutliers(marker_det):
					print("Not all flipped markers were fixed!")
					beep(self.make_noise)

			# improve detection
			if self.refine_pose:
				self.refineDetection(marker_det)

			# save additional resources
			if marker_det:
				tf_dict, states_dict = self.saveRecord()

			return marker_det, det_img, proc_img, raw_img, tf_dict, states_dict, timestamp
		
	def normalXY(self, rot: np.ndarray, rot_t: RotTypes) -> np.ndarray:
		""" Get the normal to the XY plane in the marker frame.
		"""
		mat = getRotation(rot, rot_t, RotTypes.MAT)
		normal = mat @ self.UNIT_AXIS_Z
		return normal

	def normalXZ(self, rot: np.ndarray, rot_t: RotTypes) -> np.ndarray:
		""" Get the normal to the XZ plane in the marker frame.
		"""
		mat = getRotation(rot, rot_t, RotTypes.MAT)
		normal = mat @ self.UNIT_AXIS_Y
		return normal
	
	def normalYZ(self, rot: np.ndarray, rot_t: RotTypes) -> np.ndarray:
		""" Get the normal to the YZ plane in the marker frame.
		"""
		mat = getRotation(rot, rot_t, RotTypes.MAT)
		normal = mat @ self.UNIT_AXIS_X
		return normal

	def normalAngularDispl(self, joint: str, base_rot: np.ndarray, target_rot: np.ndarray, rot_t: RotTypes=RotTypes.EULER, normal_type: NormalTypes=NormalTypes.XY) -> float:
		""" Calculate the angle between normal vectors.
		"""
		# get normal vectors for the axis planes
		base_normal = self.normalXZ(base_rot, rot_t) if normal_type == NormalTypes.XZ else \
			self.normalYZ(base_rot, rot_t) if normal_type == NormalTypes.YZ else \
				self.normalXY(base_rot, rot_t)
		target_normal = self.normalXZ(target_rot, rot_t) if normal_type == NormalTypes.XZ else \
			self.normalYZ(target_rot, rot_t) if normal_type == NormalTypes.YZ else \
				self.normalXY(target_rot, rot_t)

		# normalize vectors to make sure they are unit vectors
		base_normal = base_normal / np.linalg.norm(base_normal)
		target_normal = target_normal / np.linalg.norm(target_normal)

		# get angle
		dot_product = base_normal @ target_normal
		# ensure the dot product is within the valid range for arccos due to numerical precision
		cos_theta = np.clip(dot_product, -1.0, 1.0)
		# angle in radians, always return values in the range [0, pi]
		angle = np.arccos(cos_theta)

		# get sign
		cross_product = np.cross(base_normal, target_normal)
		if normal_type == NormalTypes.XY:
			if cross_product[NORMAL_IDX_MAP[normal_type]] < 0.0:
				return  -1 * angle
		elif joint == 'joint7':
			if cross_product[NORMAL_IDX_MAP[normal_type]] > 0.0:
				return  -1 * angle
		
		return angle
	
	def tfBaseMarker(self, base_marker: dict, virtual_base_tf: dict) -> dict:
		trans = base_marker['ftrans']
		rot = base_marker['frot']
		T_cam_base_marker = pose2Matrix(trans, rot, RotTypes.EULER)
		
		virt_trans = virtual_base_tf['trans']
		virt_rot = virtual_base_tf['rot']
		T_base_marker_virtual_marker = pose2Matrix(virt_trans, virt_rot, RotTypes.EULER)

		T_cam_virtual_marker = T_cam_base_marker @ T_base_marker_virtual_marker
		res_marker = base_marker.copy()
		res_marker['ftrans'] = T_cam_virtual_marker[:3, 3]
		res_marker['frot'] = getRotation(T_cam_virtual_marker[:3, :3], RotTypes.MAT, RotTypes.EULER)
		# TODO: visualize
		return res_marker
	
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
			self.drawAngle(id, img, p1, p2, np.rad2deg(angle))

	def labelAngle(self, img: cv2.typing.MatLike, name: str, id: int, angle: float) -> None:
		if angle is None: 
			return
		txt = "{} {} {:.2f} deg".format(id, name, np.rad2deg(angle))
		xpos = self.TXT_OFFSET
		ypos = (id+2)*self.TXT_OFFSET
		cv2.putText(img, txt, (xpos, ypos), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.CHAIN_COLORS[-1], self.FONT_THCKNS, cv2.LINE_AA)

	def visKeypoints(self, img: cv2.typing.MatLike, fk_dict: dict) -> None:
		# root joints not detected
		if not self.chain_complete[0] or not fk_dict:
			return 
		
		joints_visualized = []
		center_point = np.zeros(3, dtype=np.float32)
		
		# world to cam holder (joint6) tf (dynamic)
		base_trans = fk_dict[self.marker_config[self.root_joint]['parent']]['trans']
		base_rot = fk_dict[self.marker_config[self.root_joint]['parent']]['quat']
		T_world_cam_holder = pose2Matrix(base_trans, base_rot, RotTypes.QUAT)
		# cam holder to cam tf (static)
		T_cam_holder_cam = pose2Matrix(self.marker_cam_extr.translation,  self.marker_cam_extr.rotation, RotTypes.MAT)
		# world to cam tf
		T_world_cam = T_world_cam_holder @ T_cam_holder_cam
		# cam to world tf
		(inv_trans, inv_rot) = invPersp(T_world_cam[:3, 3],  T_world_cam[:3, :3], RotTypes.MAT)

		# draw base keypoint
		transformed_point = T_world_cam_holder[:3, :3] @ center_point + T_world_cam_holder[:3, 3]
		projected_point_parent, _ = cv2.projectPoints(transformed_point, inv_rot, inv_trans, self.det.cmx, self.det.dist)
		projected_point_parent =  np.int32(projected_point_parent.flatten())
		cv2.circle(img, projected_point_parent, self.KEYPT_THKNS, self.CHAIN_COLORS[0], -1)
		
		# draw keypoints
		for idx, chain in enumerate(self.kinematic_chains):
			# draw only complete detections
			if self.chain_complete[idx]:
				last_projected_point = projected_point_parent
				
				for joint in chain:
					if joint not in joints_visualized:
						joints_visualized.append(joint)
						
						# transform point
						trans = fk_dict[joint]['trans']
						rot_mat = getRotation(fk_dict[joint]['quat'], RotTypes.QUAT, RotTypes.MAT)
						transformed_point = rot_mat @ center_point + trans
						
						# draw projected point
						color = self.CHAIN_COLORS[idx%len(self.CHAIN_COLORS)]
						projected_point, _ = cv2.projectPoints(transformed_point, inv_rot, inv_trans, self.det.cmx, self.det.dist)
						projected_point = np.int32(projected_point.flatten())
						cv2.circle(img, projected_point, self.KEYPT_THKNS, color, -1)
						
						# connect points
						if last_projected_point is not None:
							cv2.line(img, last_projected_point, projected_point, color, self.KEYPT_LINE_THKNS)
						last_projected_point = projected_point
						
						# root connection
						if idx == 0:
							projected_point_parent = projected_point
							
						# get end link
						link_info = self.link_info_dict.get(joint)
						if link_info is not None:
							# end link tf
							trans = fk_dict[link_info['fixed_end']]['trans']
							rot_mat = getRotation(fk_dict[link_info['fixed_end']]['quat'], RotTypes.QUAT, RotTypes.MAT)
							# draw end link
							transformed_point = rot_mat @ center_point + trans
							projected_point, _ = cv2.projectPoints(transformed_point, inv_rot, inv_trans, self.det.cmx, self.det.dist)
							projected_point = np.int32(projected_point.flatten())
							cv2.circle(img, projected_point, self.KEYPT_THKNS, color, -1)
							cv2.line(img, last_projected_point, projected_point, color, self.KEYPT_LINE_THKNS)
							
	def plotKeypoints(self, keypt_dict: dict) -> Union[None, cv2.typing.MatLike]:
		plot_img = None
		if not self.chain_complete[0] or not keypt_dict:
			# root joints not detected
			return plot_img

		self.keypt_plot.clear()
		parent_joint = self.kinematic_chains[0][-1]

		# plot keypoints
		for idx, chain in enumerate(self.kinematic_chains):
			# plot only complete detections
			if self.chain_complete[idx]:
				plt_dict = {}
				for joint in chain:
					plt_dict.update( {joint: keypt_dict[joint]} )
					plt_dict[joint].update( {'color': self.PLOT_COLORS[idx%len(self.PLOT_COLORS)]} )
					# end link
					link_info = self.link_info_dict.get(joint)
					if link_info is not None:
						link_name = link_info['fixed_end']
						plt_dict.update( {link_name: keypt_dict[link_name]} )
						plt_dict[link_name].update( {'color': self.PLOT_COLORS[idx%len(self.PLOT_COLORS)]} )
				# plot
				plot_img = self.keypt_plot.plotKeypoints(plt_dict, parent_joint)

		return cv2.cvtColor(plot_img, cv2.COLOR_RGBA2BGR)

	def rh8dFK(self, joint_angles: dict, base_tf: dict, timestamp: float) -> Tuple[dict, dict]:
		# reset flag
		self.chain_complete = [True for _ in range(len(self.chain_complete))]

		fk_dict = {}
		keypt_dict = {}
		if base_tf:
			# applied to link's com
			pb.resetBasePositionAndOrientation(self.robot_id, base_tf['trans'], base_tf['quat'])

		# save base link pose
		base_pos, base_orn = pb.getBasePositionAndOrientation(self.robot_id)
		fk_dict.update({self.marker_config[self.root_joint]['parent']: {'timestamp': timestamp, 'trans': base_pos, 'quat': base_orn}})
		
		# set detected angles
		for joint, angle in joint_angles.items():
			if angle is not None:
				# revolute joint index
				pb.resetJointState(self.robot_id, self.joint_info_dict[joint], angle)
				
		# world to root joint tf inverse
		(_, _, _, _, trans, quat) = pb.getLinkState(self.robot_id, self.joint_info_dict[self.root_joint], computeForwardKinematics=True)
		(inv_trans, inv_quat) = invPersp(trans, quat, RotTypes.QUAT)
		T_root_joint_world = pose2Matrix(inv_trans, inv_quat, RotTypes.QUAT)

		# get fk
		for joint, angle in joint_angles.items():
			if angle is not None:
				# revolute joint's attached link
				(_, _, _, _, trans, quat) = pb.getLinkState(self.robot_id, self.joint_info_dict[joint], computeForwardKinematics=True)
				# add world pose
				fk_dict.update( {joint: {'timestamp': timestamp, 'trans': trans, 'quat': quat}} )
				# compute relative pose
				T_world_keypoint = pose2Matrix(trans, quat, RotTypes.QUAT)
				T_root_joint_keypoint = T_root_joint_world @ T_world_keypoint
				keypt_dict.update( {joint: {'timestamp': timestamp, 'trans': T_root_joint_keypoint[:3, 3], 'rot_mat': T_root_joint_keypoint[:3, :3]}} )
				# additional fk for tip frame
				if joint in self.link_info_dict.keys():
					# end link attached to last fixed joint 
					(_, _, _, _, trans, quat) = pb.getLinkState(self.robot_id, self.link_info_dict[joint]['index'], computeForwardKinematics=True)
					fk_dict.update( {self.link_info_dict[joint]['fixed_end']: {'timestamp': timestamp, 'trans': trans, 'quat': quat}} )
					# compute relative pose
					T_world_keypoint = pose2Matrix(trans, quat, RotTypes.QUAT)
					T_root_joint_keypoint = T_root_joint_world @ T_world_keypoint
					keypt_dict.update( {self.link_info_dict[joint]['fixed_end']: {'timestamp': timestamp, 'trans': T_root_joint_keypoint[:3, 3], 'rot_mat': T_root_joint_keypoint[:3, :3]}} )
			else:
				fk_dict.update( {joint: {'timestamp': timestamp, 'trans': np.nan, 'quat': np.nan}} )
				keypt_dict.update( {joint: {'timestamp': timestamp, 'trans': np.nan, 'rot_mat': np.nan}} )
				if joint in self.link_info_dict.keys():
					fk_dict.update( {self.link_info_dict[joint]['fixed_end']: {'timestamp': timestamp, 'trans': np.nan, 'quat': np.nan}} )
					keypt_dict.update( {self.link_info_dict[joint]['fixed_end']: {'timestamp': timestamp, 'trans': np.nan, 'rot_mat': np.nan}} )
				# disable chain
				for idx, chain in enumerate(self.kinematic_chains):
					if joint in chain:
						self.chain_complete[idx] = False

		return fk_dict, keypt_dict

	def visRosTF(self, img: cv2.typing.MatLike, fk_dict: dict) -> None:
		# world to cam holder (joint6) tf (dynamic)
		base_trans = fk_dict[self.marker_config[self.root_joint]['parent']]['trans']
		base_rot = fk_dict[self.marker_config[self.root_joint]['parent']]['quat']
		timestamp = fk_dict[self.marker_config[self.root_joint]['parent']]['timestamp']
		timestamp = rospy.Time(timestamp)
		T_world_cam_holder = pose2Matrix(base_trans, base_rot, RotTypes.QUAT)
		# cam holder to cam tf (static)
		T_cam_holder_cam = pose2Matrix(self.marker_cam_extr.translation,  self.marker_cam_extr.rotation, RotTypes.MAT)
		# world to cam tf
		T_world_cam = T_world_cam_holder @ T_cam_holder_cam
		# cam to world tf
		(inv_trans, inv_rot) = invPersp(T_world_cam[:3, 3],  T_world_cam[:3, :3], RotTypes.MAT)

		center_point = np.zeros(3, dtype=np.float32)
		for link in self.tf_links:
			tf_dict = self.lookupTF(timestamp, self.TF_TARGET_FRAME, link)
			trans = tf_dict['trans']
			quat = tf_dict['quat']
			rot_mat = getRotation(quat, RotTypes.QUAT, RotTypes.MAT)
			transformed_point = rot_mat @ center_point + trans
			# draw projected point
			projected_point, _ = cv2.projectPoints(transformed_point, inv_rot, inv_trans, self.det.cmx, self.det.dist)
			projected_point = np.int32(projected_point.flatten())
			cv2.circle(img, projected_point, self.KEYPT_THKNS, (203,192,255), -1)

	def detectionRoutine(self, pos_cmd: dict, epoch: int, direction: int, description: str) -> bool:
		# get filtered detection and save other resources
		(marker_det, det_img, proc_img, img, base_tf, joint_states, timestamp) = self.preProcImage()
		out_img = img.copy()

		res = True
		fk_dict = {}
		keypt_dict = {}
		joint_angles = {joint: None for joint in self.marker_config.keys()}
		
		# at least one detection
		if marker_det:
			detected_ids = marker_det.keys()

			# iter marker config
			for joint, config in self.marker_config.items():
				# check if and which joint marker set was detected
				marker_ids = config['marker_ids'] if all( [id in detected_ids for id in config['marker_ids']] ) else \
					config['alt_marker_ids'] if all( [id in detected_ids for id in config['alt_marker_ids']] ) else False
				
				if marker_ids:
					base_id = marker_ids[0] # base marker id
					target_id = marker_ids[1] # target marker id
					base_marker = marker_det[base_id] # base marker detection
					target_marker = marker_det[target_id] # target marker detection
					
					# apply static tf
					virtual_base_tf = config.get('virtual_base_tf')
					if virtual_base_tf is not None:
						base_marker = self.tfBaseMarker(base_marker, virtual_base_tf)
					
					# detected angle in rad
					angle = self.normalAngularDispl(joint, base_marker['frot'], target_marker['frot'], RotTypes.EULER, NORMAL_TYPES_MAP[config['plane']])
					# save initially detected angle
					if self.start_angles.get(joint) is None:
						self.start_angles.update({joint: angle})
						print("Updating start angle", {joint: angle})
						if len(self.start_angles.keys()) == len(self.marker_config.keys()):
							print("All start angles updated")
							
					# substract initial angle
					angle = angle - self.start_angles[joint]
					# physical limits
					angle = np.clip(angle, a_min=config['lim_low'], a_max=config['lim_high'])

					# data entry
					data = { 'angle': angle, 
			 						'start_angle': self.start_angles[joint],
									'base_id': base_id, 
									'target_id': target_id, 
									'frame_cnt': self.frame_cnt, 
									'rec_cnt': self.record_cnt, 
									'cmd': pos_cmd[config['actuator']], 
									'state': joint_states[config['actuator']],
									'epoch': epoch,
									'direction': direction,
									'description': description,
									'target_trans': target_marker['ftrans'],
									'target_rot': target_marker['frot'],
									'base_trans':  base_marker['ftrans'],
									'base_rot': base_marker['frot'],
									'parent_frame_id': config['parent'],
									'child_frame_id': config['child'],
									'timestamp': timestamp,
									}
					joint_angles[joint] = angle # add angle for keypoint vis
					# TODO: fix this permanently
					marker_det[target_id if joint != 'joint7' else base_id].update({'angle': angle, 'base_id': base_id, 'joint': joint}) # add to detection for drawings
					self.det_df_dict[joint] = pd.concat([self.det_df_dict[joint], pd.DataFrame([data])], ignore_index=True) # add to results
				else:
					res = False
					beep(self.make_noise)
					nan_entry = dict(zip(self.DET_COLS, [np.nan for _ in self.DET_COLS]))
					self.det_df_dict[joint] = pd.concat([self.det_df_dict[joint], pd.DataFrame([nan_entry])], ignore_index=True) # add nan to results
					print(f"Cannot detect all required ids for {joint}, missing: { [id for id in config['marker_ids'] if id not in detected_ids] }") # , alt missing:  { [id for id in config['alt_marker_ids'] if id not in detected_ids] }")

				# check data counter
				if self.det_df_dict[joint].index[-1] != self.data_cnt:
					print("DATA RECORD DEVIATION, data index: ", str(self.det_df_dict[joint].index[-1] ), " record index:", str(self.data_cnt), "for joint", joint)

			# compute 3D keypoints
			(fk_dict, keypt_dict) = self.rh8dFK(joint_angles, base_tf, timestamp)
			for link, fk in fk_dict.items():
				self.rh8d_tf_df_dict[link] = pd.concat([self.rh8d_tf_df_dict[link], pd.DataFrame([fk])], ignore_index=True) # add to results
			for link, keypt in keypt_dict.items():
				self.keypt_df_dict[link] = pd.concat([self.keypt_df_dict[link], pd.DataFrame([keypt])], ignore_index=True) # add to results

		# no marker detected
		else:
			res = False
			print("No detection")
			beep(self.make_noise)
		
		# draw detections
		if self.vis:
			# frame counter
			cv2.putText(det_img, str(self.frame_cnt) + " " + str(self.record_cnt), (det_img.shape[1]-100, 50), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.FONT_CLR, self.FONT_THCKNS, cv2.LINE_AA)
			# draw keypoints
			self.visKeypoints(out_img, fk_dict)
			kpt_plt = self.plotKeypoints(keypt_dict)
			if self.vis_ros_tf:
				self.visRosTF(out_img, fk_dict)
			# draw angle labels and possibly fixed detections 
			for id, detection in marker_det.items():
				# self.labelDetection(out_img, id, marker_det) # cv angle
				self.labelAngle(out_img, detection.get('joint'), id, detection.get('angle'))
				self.det._drawMarkers(id, detection['corners'], out_img) # square
				out_img = cv2.drawFrameAxes(out_img, self.det.cmx, self.det.dist, detection['rvec'], detection['tvec'], self.det.marker_length*self.det.AXIS_LENGTH, self.det.AXIS_THICKNESS) # CS
				out_img = self.det._projPoints(out_img, detection['points'], detection['rvec'], detection['tvec']) # corners
			if self.cv_window:
				cv2.imshow('Processed', proc_img)
				cv2.imshow('Detection', det_img)
				cv2.imshow('Output', out_img)
				if kpt_plt is not None:
					cv2.imshow('Plot', kpt_plt)
			elif self.attached:
				self.proc_pub.publish(self.bridge.cv2_to_imgmsg(proc_img, encoding="mono8"))
				self.det_pub.publish(self.bridge.cv2_to_imgmsg(det_img, encoding="bgr8"))
				self.out_pub.publish(self.bridge.cv2_to_imgmsg(out_img, encoding="bgr8"))
				if kpt_plt is not None:
					self.plot_pub.publish(self.bridge.cv2_to_imgmsg(kpt_plt, encoding="bgr8"))

			# save drawings and original
			if self.save_record and not self.test and marker_det:
				try:
					cv2.imwrite(os.path.join(KEYPT_ORIG_REC_DIR, str(self.data_cnt) + '.jpg'), img, [int(cv2.IMWRITE_JPEG_QUALITY), JPG_QUALITY]) # save original
					cv2.imwrite(os.path.join(KEYPT_DET_REC_DIR, str(self.data_cnt) + '.jpg'), out_img, [int(cv2.IMWRITE_JPEG_QUALITY), JPG_QUALITY]) # save detection
					if kpt_plt is not None:
						cv2.imwrite(os.path.join(KEYPT_DET_REC_DIR, 'plot3D_' + str(self.data_cnt) + '.jpg'), kpt_plt, [int(cv2.IMWRITE_JPEG_QUALITY), JPG_QUALITY]) # save plot
				except Exception as e:
					print(e)
					return False
				
		# increment data counter 
		if marker_det:
			self.data_cnt += 1
			if self.data_cnt != self.record_cnt:
				print("IMAGE RECORD DEVIATION, record index: ", str(self.data_cnt), " img index:", str(self.record_cnt))

		return res
	
	def run(self) -> None:
		if self.test:
			self.runTest()
		elif self.attached:
			self.runAttached()
		else:
			raise NotImplementedError
	# 		self.runDetached()
	
	# def runDetached(self) -> None:
	# 	rate = rospy.Rate(self.f_loop)
		
	# 	try:
	# 		# epoch
	# 		for e in range(self.epochs):
	# 			print("Epoch", e)
				
	# 			for idx in self.waypoint_df.index.tolist():
	# 				if not rospy.is_shutdown():
	# 					# get waypoint
	# 					waypoint = self.waypoint_df.iloc[idx]
	# 					reset_angles = waypoint[-4]
	# 					direction = waypoint[-3]
	# 					description = waypoint[-2]
	# 					move_time = waypoint[-1]
	# 					waypoint = waypoint[: -4].to_dict()
	# 					# reset initially detected angles
	# 					if reset_angles:
	# 						self.start_angles.clear()

	# 					# move arm and hand
	# 					print(f"Reaching waypoint number {idx}: {description} with direction {direction} in {move_time}s", end=" ... ")
	# 					print(waypoint)
	# 					success = True
	# 					for id, val in zip(RH8D_IDS, list(waypoint.values())[MoveRobot.ROBOT_JOINTS_INDEX['joint7'] :]):
	# 						cmd = ((val + np.pi) / (2 * np.pi)) * RH8D_MAX_POS
	# 						success, _ = self.rh8d_ctrl.setPos(id, cmd) and success
	# 					print("done\n") if success else print("fail\n")

	# 					# detect angles
	# 					success = False
	# 					while not success:
	# 						success = self.detectionRoutine(waypoint, e, direction, description)
	# 						if not success:
	# 							beep(self.make_noise)
	# 							success = input("Enter 'r' to repeat recording or other key to skip.") != 'r'

	# 					if self.vis and not self.attached:
	# 						if cv2.waitKey(1) == ord("q"):
	# 							return
						
	# 					try:
	# 						rate.sleep()
	# 					except:
	# 						pass

	# 	except Exception as e:
	# 		rospy.logerr(e)
	# 	finally:
	# 		if self.save_record and not self.test:
	# 			# angles
	# 			det_df = pd.DataFrame({joint: [df] for joint, df in self.det_df_dict.items()})
	# 			det_df.to_json(KEYPT_DET_PTH, orient="index", indent=4)
	# 			if not self.test:
	# 				# keypoints wrt world
	# 				fk_df = pd.DataFrame({link: [df] for link, df in self.rh8d_tf_df_dict.items()})
	# 				fk_df.to_json(KEYPT_FK_PTH, orient="index", indent=4)
	# 				# relative keypoints
	# 				kypt_df = pd.DataFrame({link: [df] for link, df in self.keypt_df_dict.items()})
	# 				kypt_df.to_json(KEYPT_3D_PTH, orient="index", indent=4)
	# 				pb.disconnect()
	# 		if self.vis and not self.attached:
	# 			cv2.destroyAllWindows()
	# 		rospy.signal_shutdown(0)

	def moveHeadConditioned(self, waypoint: dict, t_move: float=1.5) -> bool:
		# wait for new states
		while not len(self.rh8d_ctrl.positions):
			pass

		# arm moving
		robot_positions = np.round(self.rh8d_ctrl.positions.pop(), 2)
		position_command = np.round(list(waypoint.values()), 2)
		if not all( np.isclose(robot_positions[: self.rh8d_ctrl.ROBOT_JOINTS_INDEX['joint7']], position_command[: self.rh8d_ctrl.ROBOT_JOINTS_INDEX['joint7']]) ):
			self.rh8d_ctrl.moveHeadHome(t_move)
			rospy.sleep(t_move)
			return True
		
		return False
	
	def initAngles(self) -> bool:
		"""Initial angles are the difference between two marker planes when all joints are at zero
			  position. This angle difference is substracted at any further reading.
		"""
		topic_info = f"Visualization topic: {self.det_pub.resolved_name}" if not self.cv_window else ""
		input(f"Illuminate all markers for angle initialization and press any key to proceed! {topic_info}")
		last_joint =  list(self.marker_config.keys())[-1]

		while True:
			# wait for buffer to be filled
			while len(self.det_img_buffer) != self.filter_iters:
				print("Record size deviates from filter size:", len(self.det_img_buffer), " !=", self.filter_iters)
				rospy.sleep(1/self.fps)

			# lock updates on img queues
			# with self.buf_lock:
			# process images
			self.det.resetFilters()
			# while len(self.det_img_buffer):
			for _ in range(self.filter_iters):
				msg = self.det_img_buffer.pop()
				raw_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
				(marker_det, det_img, _) = self.det.detMarkerPoses(raw_img.copy(), (not len(self.det_img_buffer) and self.vis))
			# align rotations by consens
			if self.flip_outliers:
				if not self.flipOutliers(marker_det):
					beep(self.make_noise)
			# improve detection
			if self.refine_pose:
				self.refineDetection(marker_det)

			# detect angles
			detected_ids = marker_det.keys()
			for joint, config in self.marker_config.items():
				# check if and which joint marker set was detected
				marker_ids = config['marker_ids'] if all( [id in detected_ids for id in config['marker_ids']] ) else \
					config['alt_marker_ids'] if all( [id in detected_ids for id in config['alt_marker_ids']] ) else False
				if marker_ids:
					base_marker = marker_det[marker_ids[0]] # base marker detection
					target_marker = marker_det[marker_ids[1]] # target marker detection
					# detected angle in rad
					angle = self.normalAngularDispl(joint, base_marker['frot'], target_marker['frot'], RotTypes.EULER, NORMAL_TYPES_MAP[config['plane']])
					self.start_angles.update({joint: angle})
					# show
					print(f"Updating start angle {joint}: {angle} rad, {np.rad2deg(angle)} deg")
					self.labelAngle(det_img, joint, marker_ids[1], angle)
					if self.cv_window:
						cv2.imshow('Detection', det_img)
						if cv2.waitKey(1) == ord("q"):
							return False
					else:
						self.det_pub.publish(self.bridge.cv2_to_imgmsg(det_img, encoding="bgr8"))
					# exit
					if joint == last_joint:
						if input("All angles initialized. Press any key to exit or r to repeat.") != 'r':
							return True
				else:
					if input(f"{joint} markers not detected, check illumination and press any key to repeat or q to exit!") == 'q':
						return False
					break

	def runTest(self) -> None:
		rate = rospy.Rate(self.f_loop)
		
		try:
			if not rospy.is_shutdown():
				self.rh8d_ctrl.moveHeadHome(2.0)
				self.rh8d_ctrl.reachHomeBlocking(15.0)
				if not self.self_reset_angles:
					if not self.initAngles():
						return
				else:
					self.start_angles.clear()

			# epoch
			while not rospy.is_shutdown():
					
				(marker_det, det_img, proc_img, img, base_tf, joint_states, timestamp) = self.preProcImage()
				out_img = img.copy()

				fk_dict = {}
				keypt_dict = {}
				joint_angles = {joint: None for joint in self.marker_config.keys()}
				
				# at least one detection
				if marker_det:
					detected_ids = marker_det.keys()

					# iter marker config
					for joint, config in self.marker_config.items():
						# check if and which joint marker set was detected
						marker_ids = config['marker_ids'] if all( [id in detected_ids for id in config['marker_ids']] ) else \
							config['alt_marker_ids'] if all( [id in detected_ids for id in config['alt_marker_ids']] ) else False
						
						if marker_ids:
							base_id = marker_ids[0] # base marker id
							target_id = marker_ids[1] # target marker id
							base_marker = marker_det[base_id] # base marker detection
							target_marker = marker_det[target_id] # target marker detection

							# apply static tf
							virtual_base_tf = config.get('virtual_base_tf')
							if virtual_base_tf is not None:
								base_marker = self.tfBaseMarker(base_marker, virtual_base_tf)

							# detected angle in rad
							angle = self.normalAngularDispl(joint, base_marker['frot'], target_marker['frot'], RotTypes.EULER, NORMAL_TYPES_MAP[config['plane']])
							# save initially detected angle
							if self.start_angles.get(joint) is None:
								self.start_angles.update({joint: angle})
								print("Updating start angle", {joint: angle})
								if len(self.start_angles.keys()) == len(self.marker_config.keys()):
									print("All start angles updated")
							
							# substract initial angle
							angle = angle - self.start_angles[joint] # TODO: check
							# physical limits
							angle = np.clip(angle, a_min=config['lim_low'], a_max=config['lim_high'])

							joint_angles[joint] = angle # add angle for keypoint vis
							# TODO: fix this permanently
							marker_det[target_id if joint != 'joint7' else base_id].update({'angle': angle, 'base_id': base_id, 'joint': joint}) # add to detection for drawings
						else:
							beep(self.make_noise)
							print(f"Cannot detect all required ids for {joint}, missing: { [id for id in config['marker_ids'] if id not in detected_ids] }") # , alt missing:  { [id for id in config['alt_marker_ids'] if id not in detected_ids] }")

					# compute 3D keypoints
					(fk_dict, keypt_dict) = self.rh8dFK(joint_angles, base_tf, timestamp)

				# no marker detected
				else:
					beep(self.make_noise)
					print("No detection")
				
				# draw detections
				if self.vis:
					# frame counter
					cv2.putText(det_img, str(self.frame_cnt) + " " + str(self.record_cnt), (det_img.shape[1]-100, 50), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.FONT_CLR, self.FONT_THCKNS, cv2.LINE_AA)
					# draw keypoints
					self.visKeypoints(out_img, fk_dict)
					kpt_plt = self.plotKeypoints(keypt_dict)
					if self.vis_ros_tf:
						self.visRosTF(out_img, fk_dict)
					# draw angle labels and possibly fixed detections 
					for id, detection in marker_det.items():
						self.labelAngle(out_img, detection.get('joint'), id, detection.get('angle'))
						self.det._drawMarkers(id, detection['corners'], out_img) # square
						out_img = cv2.drawFrameAxes(out_img, self.det.cmx, self.det.dist, detection['rvec'], detection['tvec'], self.det.marker_length*self.det.AXIS_LENGTH, self.det.AXIS_THICKNESS) # CS
						out_img = self.det._projPoints(out_img, detection['points'], detection['rvec'], detection['tvec']) # corners
					if self.cv_window:
						cv2.imshow('Processed', proc_img)
						cv2.imshow('Detection', det_img)
						cv2.imshow('Output', out_img)
						if kpt_plt is not None:
							cv2.imshow('Plot', kpt_plt)
					elif self.attached:
						self.proc_pub.publish(self.bridge.cv2_to_imgmsg(proc_img, encoding="mono8"))
						self.det_pub.publish(self.bridge.cv2_to_imgmsg(det_img, encoding="bgr8"))
						self.out_pub.publish(self.bridge.cv2_to_imgmsg(out_img, encoding="bgr8"))
						if kpt_plt is not None:
							self.plot_pub.publish(self.bridge.cv2_to_imgmsg(kpt_plt, encoding="bgr8"))

				if self.cv_window:
					if cv2.waitKey(1) == ord("q"):
						return
				
				try:
					rate.sleep()
				except:
					pass

		except Exception as e:
			rospy.logerr(e)
			
		finally:
			pb.disconnect()
			if self.cv_window:
				cv2.destroyAllWindows()
			rospy.signal_shutdown(0)
		
	def runAttached(self) -> None:
		rate = rospy.Rate(self.f_loop)
		
		try:
			# epoch
			for e in range(self.epochs):
				print("New epoch", e)
				print("Moving home!")
				head_home = self.rh8d_ctrl.moveHeadHome(2.0)
				self.rh8d_ctrl.reachHomeBlocking(15.0)
				# save initially detected angles
				if not self.self_reset_angles:
					if not self.initAngles():
						return
				
				for idx in self.waypoint_df.index.tolist():
					if not rospy.is_shutdown():
						# get waypoint
						waypoint = self.waypoint_df.iloc[idx]
						reset_angles = waypoint[-4]
						direction = waypoint[-3]
						description = waypoint[-2]
						move_time = waypoint[-1]
						waypoint = waypoint[: -4].to_dict()
						# reset initially detected angles
						if reset_angles and self.self_reset_angles:
							self.start_angles.clear()

						# get head out of range if arm moves
						head_home = self.moveHeadConditioned(waypoint)

						# move arm and hand
						print(f"Epoch {e}. Reaching waypoint number {idx}: {description} with direction {direction} in {move_time}s", end="... ")
						success = self.rh8d_ctrl.reachPositionBlocking(waypoint, move_time, description)
						print("done\n") if success else print("fail\n")

						# look towards hand and settle 
						tf = self.lookupTF(rospy.Time(0), self.TF_TARGET_FRAME, self.RH8D_TCP_SOURCE_FRAME)
						self.rh8d_ctrl.moveHeadTaskSpace(tf['trans'][0], tf['trans'][1], tf['trans'][2], 2.0 if head_home else 1.0)			
						rospy.sleep(3.0 if head_home else 1.5)
						head_home = False

						# detect angles
						self.detectionRoutine(waypoint, e, direction, description)

						if self.cv_window:
							if cv2.waitKey(1) == ord("q"):
								return
						
						try:
							rate.sleep()
						except:
							pass

						if idx % 10 == 0:
							self.writeData()

		except None as e:
			rospy.logerr(e)
			
		finally:
			self.writeData()
			if self.cv_window:
				cv2.destroyAllWindows()
			pb.disconnect()
			rospy.signal_shutdown(0)

	def writeData(self) -> None:
		if self.save_record and not self.test:
			# angles
			det_df = pd.DataFrame({joint: [df] for joint, df in self.det_df_dict.items()})
			det_df.to_json(KEYPT_DET_PTH, orient="index", indent=4)
			# keypoints wrt world
			fk_df = pd.DataFrame({link: [df] for link, df in self.rh8d_tf_df_dict.items()})
			fk_df.to_json(KEYPT_FK_PTH, orient="index", indent=4)
			# relative keypoints
			kypt_df = pd.DataFrame({link: [df] for link, df in self.keypt_df_dict.items()})
			kypt_df.to_json(KEYPT_3D_PTH, orient="index", indent=4)
			if self.attached:
				self.as_df.to_json(os.path.join(KEYPT_REC_DIR, 'actuator_states.json'), orient="index", indent=4)
				self.js_df.to_json(os.path.join(KEYPT_REC_DIR, 'joint_states.json'), orient="index", indent=4)
				if self.use_tf:
					self.rh8d_tf_df.to_json(os.path.join(KEYPT_DET_REC_DIR, 'tf.json'), orient="index", indent=4)
					self.rh8d_tcp_tf_df.to_json(os.path.join(KEYPT_DET_REC_DIR, 'tcp_tf.json'), orient="index", indent=4)
					if self.use_head_camera:
						self.head_rs_tf_df.to_json(os.path.join(KEYPT_HEAD_CAM_REC_DIR, 'tf.json'), orient="index", indent=4)
					if self.use_eye_cameras:
						self.left_eye_tf_df.to_json(os.path.join(KEYPT_L_EYE_REC_DIR, 'tf.json'), orient="index", indent=4)
						self.right_eye_tf_df.to_json(os.path.join(KEYPT_R_EYE_REC_DIR, 'tf.json'), orient="index", indent=4)

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
		@param joint_list
		@type list
		@param step_div
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
				joint_list: Optional[list]=["jointI1", "jointI2", "jointI3"],
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
						plt_id=plt_id,
						use_tf=False,
						f_ctrl=f_ctrl,
						epochs=epochs,
						test=test,
						vis=vis,
						fps=fps,
						attached=False,
						)
		
		# actuator control
		self.actuator = actuator
		self.step_div = step_div

		# finger data
		self.group_ids = joint_list # joint names
		self.df = pd.DataFrame(columns=self.group_ids) # dataset

		# target marker ids
		self.target_ids = [] 
		self.normal_plane = []
		# assume ordered marker ids per joint
		# -> 1st id is base marker and 
		# 2nd id is target marker for angle computation
		for joint, config in self.marker_config.items():
			if joint in self.group_ids:
				self.target_ids.append(config['marker_ids'])
				self.normal_plane.append(config['plane'])
		self.target_ids = np.unique(self.target_ids) # rm duplicates
		# all planes are equal?
		assert(all(plane == self.normal_plane[0] for plane in self.normal_plane))
		self.normal_plane = self.normal_plane[0]

		# establish serial com
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
		(marker_det, det_img, proc_img, img) = DetectBase.preProcImage(self)
		out_img = img.copy()

		if marker_det:
			# check all ids detected
			if all([id in marker_det.keys() for id in self.target_ids]):

				# encoder angles in rad
				qdec_angles = self.qdec.readMedianAnglesRad()
				if not qdec_angles:
					print("No qdec angles")
					beep(self.make_noise)
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
					angle = self.normalAngularDispl("jointI1", base_marker['frot'], target_marker['frot'], RotTypes.EULER, NORMAL_TYPES_MAP[self.normal_plane])
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
				beep(self.make_noise)
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
			if self.save_record:
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

	CAM_LABEL_YPOS = 20

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
				# repr_err = self.projectSingleMarker(det, id, T_camera_world)
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
				   flip_outliers=False,
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
					 save_imgs=rospy.get_param('~save_imgs', False),
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
						use_tf=rospy.get_param('~use_tf', False),
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
						actuator_state_topic=rospy.get_param('~actuator_state_topic', 'actuator_states'),
						waypoint_set=rospy.get_param('~waypoint_set', 'waypoints.json'),
						waypoint_start_idx=rospy.get_param('~waypoint_start_idx', 0),
						).run()
	
if __name__ == "__main__":
	main()
