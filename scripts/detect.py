#!/usr/bin/env python3

import os, sys
import yaml
import cv2
import rospy
import tf2_ros
import cv_bridge
import numpy as np
import sensor_msgs.msg
from time import sleep
from typing import Optional, Any, Tuple
import dynamic_reconfigure.server
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import PoseStamped
from tf2_geometry_msgs import PointStamped
import open_manipulator_msgs.msg
import open_manipulator_msgs.srv
from util import *
from plot_record import *
from pose_filter import *
from qdec_serial import QdecSerial
from rh8d_serial import RH8DSerial
from nicol_rh8d.cfg import ArucoDetectorConfig
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

	"""

	FONT_THCKNS = 2
	FONT_SCALE = 0.5
	FONT_CLR =  (255,0,0)
	TXT_OFFSET = 25
	
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
				) -> None:
		
		self.vis = vis
		self.plt_id = plt_id
		self.f_loop = f_ctrl
		self.use_aruco = use_aruco
		self.filter_type = filter_type
		self.filter_iters = filter_iters
		self.frame_cnt = 0

		# init ros
		self.img_topic = camera_ns + '/image_raw'
		self.bridge = cv_bridge.CvBridge()

		# init detector
		rospy.loginfo("Waiting for camera_info from %s", camera_ns + '/camera_info')
		rgb_info = rospy.wait_for_message(camera_ns + '/camera_info', sensor_msgs.msg.CameraInfo)
		if use_aruco:
			self.det = ArucoDetector(marker_length=marker_length, 
															K=rgb_info.K, 
															D=rgb_info.D,
															f_ctrl=f_ctrl,
															invert_pose=False,
															filter_type=filter_type)
		else:
			self.det = AprilDetector(marker_length=marker_length, 
															K=rgb_info.K, 
															D=rgb_info.D,
															f_ctrl=f_ctrl,
															invert_pose=False,
															filter_type=filter_type)
			
		# init vis	
		if vis:
			cv2.namedWindow("Processed", cv2.WINDOW_NORMAL)
			cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
			if plt_id > -1:
				cv2.namedWindow("Plot", cv2.WINDOW_NORMAL)
				setPosePlot()

		# init dynamic reconfigure
		if use_reconfigure:
			print("Using reconfigure server")
			self.det_config_server = dynamic_reconfigure.server.Server(ArucoDetectorConfig, self.det.setDetectorParams)

	def euler2Matrix(self, euler: np.ndarray) -> np.ndarray:
		return R.from_euler('xyz', euler).as_matrix()
	
	def pose2Matrix(self, tvec: np.ndarray, rot: np.ndarray, rot_t: RotTypes) -> np.ndarray:
		transformation_matrix = np.eye(4)
		transformation_matrix[:3, :3] = getRotation(rot, rot_t, RotTypes.MAT)
		transformation_matrix[:3, 3] = tvec
		return transformation_matrix
	
	def preProcImage(self, vis: bool=True) -> Tuple[dict, cv2.typing.MatLike, cv2.typing.MatLike, cv2.typing.MatLike]:
		""" Put num filter_iters images into
			fresh detection filter and get last
			detection.
		"""
		self.det.resetFilters()
		iters = self.filter_iters if self.filter_iters > 0 else 1
		for i in range(iters):
			self.frame_cnt += 1
			rgb = rospy.wait_for_message(self.img_topic, sensor_msgs.msg.Image)
			raw_img = self.bridge.imgmsg_to_cv2(rgb, 'bgr8')
			(marker_det, det_img, proc_img) = self.det.detMarkerPoses(raw_img.copy(), vis if i >= iters-1 else False)
		return marker_det, det_img, proc_img, raw_img
	
	def runDebug(self) -> None:
		rate = rospy.Rate(self.f_loop)
		try:
			while not rospy.is_shutdown():
				(marker_det, det_img, proc_img, img) = self.preProcImage()
				if self.vis:
					# frame counter
					cv2.putText(det_img, str(self.frame_cnt), (det_img.shape[1]-40, 20), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.FONT_CLR, self.FONT_THCKNS, cv2.LINE_AA)
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

	def detectionRoutine(self, cnt: int) -> Union[Tuple[dict, cv2.typing.MatLike, cv2.typing.MatLike, int], dict]:
		raise NotImplementedError
	
	def run(self) -> None:
		raise NotImplementedError

class KeypointDetect(DetectBase):
	"""
		Detect keypoints from marker poses.

		@param err_term
		@type float
		@param cart_bound_low
		@type float
		@param cart_bound_high
		@type float
		@param use_tf
		@type bool

	"""

	POINT_FACTOR = 0.85 # arc points interpolation
	ARC_SHIFT = 10 # ellipse param
	SAGITTA = 50 # arc size
	AXES_LEN = 0.03 # meter
	X_AXIS = np.array([[-AXES_LEN,0,0], [AXES_LEN, 0, 0]], dtype=np.float32)
	UNIT_AXIS_Y = np.array([0, 1, 0], dtype=np.float32)

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
				use_tf: Optional[bool]=False,
				) -> None:
		
		super().__init__(marker_length=marker_length,
						camera_ns=camera_ns,
						vis=vis,
						use_reconfigure=use_reconfigure,
						filter_type=filter_type,
						filter_iters=filter_iters,
						f_ctrl=f_ctrl,
						use_aruco=use_aruco,
						plt_id=plt_id,
						)
		
		# init ros
		if use_tf:
			self.buf = tf2_ros.Buffer()
			self.listener = tf2_ros.TransformListener(self.buf)
		self.use_tf = use_tf

		# load hand marker ids
		self.fl = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cfg/hand_ids.yaml")
		with open(self.fl, 'r') as fr:
			self.hand_ids = yaml.safe_load(fr)

		# init vis
		if vis:
			cv2.namedWindow("Angles", cv2.WINDOW_NORMAL)

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
	
	def drawAngle(self, img: cv2.typing.MatLike, pt1: np.ndarray, pt2: np.ndarray, angle_deg: float) -> None:
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
		cv2.ellipse(img, arc_center, axes, 0, pt1_angle, pt2_angle, self.det.BLUE, 1, cv2.LINE_AA, self.ARC_SHIFT)
		# cv2.circle(img, pt1, 5, self.det.BLUE, -1)
		# cv2.circle(img, pt2, 5, self.det.BLUE, -1)
		# cv2.circle(img, center, 5, self.det.BLUE, -1)
		# draw angle text
		cv2.putText(img, f'{angle_deg:.2f} deg', center, cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.det.BLUE, self.FONT_THCKNS)

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
		cv2.line(img, x_axis_base_marker[0], x_axis_base_marker[1], self.det.RED, 2)
		cv2.line(img, x_axis_target_marker[0], x_axis_target_marker[1], self.det.RED, 2)

		# draw angle between the x-axes
		if np.abs(angle) > 0:
			# interpolate points on the axes
			interpolated_point_base_ax = (1 - self.POINT_FACTOR) * x_axis_base_marker[0] + self.POINT_FACTOR * x_axis_base_marker[1]
			interpolated_point_target_ax = (1 - self.POINT_FACTOR) * x_axis_target_marker[1] + self.POINT_FACTOR * x_axis_target_marker[0]
			# draw ellipse
			p1 = tuple(map(int, interpolated_point_base_ax))
			p2 = tuple(map(int, interpolated_point_target_ax))
			self.drawAngle(img, p1, p2, np.rad2deg(angle))

	def detectionRoutine(self) -> dict:
		(marker_det, det_img, proc_img, img) = self.preProcImage()
		out_img = img.copy()

		if marker_det:
			ids = self.hand_ids['index']
			# check all ids detected
			if not all([id in marker_det.keys() for id in ids]):
				print("Cannot detect all required ids: ", ids)
				return False
			# compute angle
			for idx in range(1, len(ids)):
				base_id = ids[idx-1]
				target_id = ids[idx]
				base_marker = marker_det[base_id]
				target_marker = marker_det[target_id]
				angle = self.normalXZAngularDispl(base_marker['frot'], target_marker['frot'], RotTypes.EULER)
				marker_det[target_id].update({'angle': angle, 'base_id': base_id})

		if self.vis:
			# frame counter
			cv2.putText(out_img, str(self.frame_cnt), (out_img.shape[1]-40, 20), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.FONT_CLR, self.FONT_THCKNS, cv2.LINE_AA)
			for id in marker_det.keys():
				# label marker angle
				self.labelDetection(out_img, id, marker_det)
			cv2.imshow('Processed', proc_img)
			cv2.imshow('Detection', det_img)
			cv2.imshow('Angles', out_img)

		return marker_det
		
	def run(self) -> None:
		rate = rospy.Rate(self.f_loop)
		try:
			while not rospy.is_shutdown():

				self.detectionRoutine()
					
				if cv2.waitKey(1) == ord("q"):
					break
					
				try:
					rate.sleep()
				except:
					pass

		except ValueError as e:
			rospy.logerr(e)
		finally:
			cv2.destroyAllWindows()

class HybridDetect(KeypointDetect):
	"""
		Detect keypoints from marker poses and quadrature encoders 
		while moving the hand.

		@param rh8d_port
		@type str
        @param rh8d_baud
		@type int
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
		@param step_div
		@type int
		@param epochs
		@type

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
				) -> None:
		
		super().__init__(marker_length=marker_length,
						camera_ns=camera_ns,
						vis=vis,
						use_reconfigure=use_reconfigure,
						filter_type=filter_type,
						filter_iters=filter_iters,
						f_ctrl=f_ctrl,
						use_aruco=use_aruco,
						plt_id=plt_id,
						use_tf=False,
						)
		
		self.epochs = epochs
		self.step_div = step_div
		self.actuator = actuator
		self.target_ids = self.hand_ids['index']
		self.rh8d_ctrl = RH8DSerial(rh8d_port, rh8d_baud)
		self.qdec = QdecSerial(qdec_port, qdec_baud, qdec_tout, qdec_filter_iters)

	def labelQdecAngles(self, img: cv2.typing.MatLike, id: int, marker_det: dict) -> None:
		ang = marker_det[id].get('det_angle') 
		if ang is None: 
			return
		txt = "{} cv: {:.2f}° qdec: {:.2f}° err: {:.2f}°".format(id, ang, marker_det[id]['qdec_angle'], marker_det[id]['error'])
		xpos = self.TXT_OFFSET
		ypos = (id+1)*self.TXT_OFFSET
		cv2.putText(img, txt, (xpos, ypos), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.FONT_CLR, self.FONT_THCKNS, cv2.LINE_AA)
		
	def detectionRoutine(self) -> dict:
		# get filtered detection
		(marker_det, det_img, proc_img, _) = self.preProcImage()
		if marker_det:
			# check all ids detected
			if all([id in marker_det.keys() for id in self.target_ids]):
				# encoder angles
				qdec_angles = self.qdec.readMedianAnglesRad()
				# compute cv angle
				for idx in range(1, len(self.target_ids)):
					base_id = self.target_ids[idx-1]
					target_id = self.target_ids[idx]
					base_marker = marker_det[base_id]
					target_marker = marker_det[target_id]
					det_angle = self.normalXZAngularDispl(base_marker['frot'], target_marker['frot'], RotTypes.EULER)
					qdec_angle = qdec_angles[idx-1]
					error = np.abs(qdec_angle - det_angle)
					marker_det[target_id].update({'det_angle': det_angle, 'qdec_angle': qdec_angle, 'error': error, 'base_id': base_id})
			else:
				print("Cannot detect all required ids, missing: ", [id if id not in marker_det.keys() else None for id in self.target_ids])

		if self.vis:
			# frame counter
			cv2.putText(det_img, str(self.frame_cnt), (det_img.shape[1]-40, 20), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.FONT_CLR, self.FONT_THCKNS, cv2.LINE_AA)
			for id in marker_det.keys():
				# label marker angle
				self.labelDetection(det_img, id, marker_det)
				self.labelQdecAngles(det_img, id, marker_det)
			cv2.imshow('Processed', proc_img)
			cv2.imshow('Detection', det_img)

		return marker_det
		
	def run(self) -> None:
		lim = 300
		max_pos = self.rh8d_ctrl.MAX_POS - lim
		step = self.rh8d_ctrl.MAX_POS // self.step_div
		pos_cmd = step

		rate = rospy.Rate(self.f_loop)
		try:
			for e in range(self.epochs):
				print("Epoch", e+1)
				# move to zero
				self.rh8d_ctrl.setMinPos(self.actuator, 1)
				while not rospy.is_shutdown() and (pos_cmd <= max_pos):

					self.detectionRoutine()

					if cv2.waitKey(1) == ord("q"):
						break
					
					print("Setting position", pos_cmd)
					self.rh8d_ctrl.setPos(self.actuator, pos_cmd)
					pos_cmd += step

					try:
						rate.sleep()
					except:
						pass

		except ValueError as e:
			rospy.logerr(e)
		finally:
			self.rh8d_ctrl.setMinPos(self.actuator, 1)
			cv2.destroyAllWindows()
		
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

	"""

	def __init__(self,
			  	marker_length: float=0.010,
				camera_ns: Optional[str]='',
				vis :Optional[bool]=True,
				use_reconfigure: Optional[bool]=False,
				filter_type: Optional[str]='none',
				f_ctrl: Optional[int]=30,
				use_aruco: Optional[bool]=False,
				plt_id: Optional[int]=-1,
				err_term: Optional[float]=1e-8,
				cart_bound_low: Optional[float]=-3.0,
				cart_bound_high: Optional[float]=3.0,
				) -> None:
		
		super().__init__(marker_length=marker_length,
						camera_ns=camera_ns,
						vis=vis,
						use_reconfigure=use_reconfigure,
						filter_type=filter_type,
						filter_iters=0, # n/a
						f_ctrl=f_ctrl,
						use_aruco=use_aruco,
						plt_id=plt_id,
						)
		self.err_term = err_term
		self.lower_bounds = [cart_bound_low, cart_bound_low, cart_bound_low, -np.pi, -np.pi, -np.pi]
		self.upper_bounds = [cart_bound_high, cart_bound_high, cart_bound_high, np.pi, np.pi, np.pi]
		# load marker poses
		self.fl = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cfg/marker_poses.yml")
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

	def labelDetection(self, img: cv2.typing.MatLike, id: int, trans: np.ndarray, rot: np.ndarray) -> None:
			pos_txt = "{} X: {:.4f} Y: {:.4f} Z: {:.4f} R: {:.4f} P: {:.4f} Y: {:.4f}".format(id, trans[0], trans[1], trans[2], rot[0], rot[1], rot[2])
			xpos = self.TXT_OFFSET
			ypos = (id+1)*self.TXT_OFFSET
			cv2.putText(img, pos_txt, (xpos, ypos), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.FONT_CLR, self.FONT_THCKNS, cv2.LINE_AA)

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
		err = 0
		# invert world to camera tf for reprojection
		tvec_inv, euler_inv = self.det.invPersp(tvec=camera_pose[:3], rot=camera_pose[3:], rot_t=RotTypes.EULER)
		T_cam_world = self.pose2Matrix(tvec_inv, euler_inv, RotTypes.EULER)
		# iter measured markers
		for id, det in detection.items():
			# reprojection error
			err += self.projectSingleMarker(det, id, T_cam_world, img)
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
		T_world_root = self.pose2Matrix(root['xyz'], root['rpy'], RotTypes.EULER) if root is not None else np.eye(4)
		# marker tf
		marker = self.marker_table_poses.get(id)
		assert(marker) # marker id entry in yaml?
		T_root_marker = self.pose2Matrix(marker['xyz'], marker['rpy'], RotTypes.EULER)
		# worldTmarker
		return T_world_root @ T_root_marker
	
	def camTF(self, detection: dict, id: int) -> np.ndarray:
		tf = np.zeros(6)
		det = detection.get(id)
		if det is None:
			print(f"Cannot find id {id} in detection!")
			return tf
		# get markerTcamera
		inv_tvec, inv_euler = self.det.invPersp(tvec=det['ftrans'], rot=det['frot'], rot_t=RotTypes.EULER)
		T_marker_cam = self.pose2Matrix(inv_tvec, inv_euler, RotTypes.EULER)
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
		T_world_camera = self.pose2Matrix(tvec=camera_pose[:3], rot=camera_pose[3:], rot_t=RotTypes.EULER)
		# invert for reprojection
		tvec_inv, euler_inv = self.det.invPersp(tvec=camera_pose[:3], rot=camera_pose[3:], rot_t=RotTypes.EULER)
		T_camera_world = self.pose2Matrix(tvec=tvec_inv, rot=euler_inv, rot_t=RotTypes.EULER)
		for id in marker_poses:
			det = detection.get(id)
			if det is not None:
				# detected tag pose wrt camera frame
				T_camera_marker = self.pose2Matrix(det['ftrans'], det['frot'], RotTypes.EULER)
				T_world_marker_est = T_world_camera @ T_camera_marker
				# measured tag pose wrt world 
				T_world_marker = self.getWorldMarkerTF(id)

				# position error (euclidean distance)
				position_error = np.linalg.norm(T_world_marker_est[:3, 3] - T_world_marker[:3, 3])
				# Orientation error (angle between rotations)
				# R_world_marker_est = T_world_marker_est[:3, :3]
				# R_world_marker = T_world_marker[:3, :3]
				# R_error = R_world_marker_est.T @ R_world_marker  # relative rotation
				# orientation_error = np.arccos((np.trace(R_error) - 1) / 2)
				orientation_error = np.linalg.norm(T_world_marker_est[:3, :3] - T_world_marker[:3, :3])
				error.append(position_error)  
				error.append(orientation_error)		
				# reprojection_error
				# error.append(self.projectSingleMarker(det, id, T_camera_world))
		return np.hstack(error) if len(error) else np.array(error)

	def estimatePoseLS(self, img: cv2.typing.MatLike, err: float, est_camera_pose: np.ndarray, detection: dict) -> np.ndarray:
		res = least_squares(self.residuals, 
							est_camera_pose, 
							args=(self.marker_table_poses, detection),
							method='trf', 
							bounds=(self.lower_bounds, self.upper_bounds),
							max_nfev=700, # max iterations
							ftol=1e-8,    # tolerance for the cost function
							xtol=1e-8,    # tolerance for the solution parameters
							gtol=1e-8     # tolerance for the gradient
							)
		if res.success:
			opt_cam_pose = res.x
			# put pose label
			self.labelDetection(img, 30, opt_cam_pose[:3], opt_cam_pose[3:])
			# reproject markers
			reserr = self.projectMarkers(detection, opt_cam_pose, img)
			txt = f"Result: {res.status} {res.message}\n"
			txt += f"camera world pose trans: {opt_cam_pose[:3]}, rot (extr. xyz euler): {opt_cam_pose[3:]}\n"
			txt += f"reprojection error: {reserr}\n"
			txt += f"cost: {res.cost}\n"
			txt += f"evaluations: {res.nfev}\n"
			txt += f"optimality: {res.optimality}\n"
			print(txt)
			return reserr, opt_cam_pose
		print(f"Least squares failed: {res.status}")
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
		"""	Use only overwritten run() as 
			we detect from quasi static image. 
		"""
		err = np.inf
		est_camera_pose = np.zeros(6)
		rate = rospy.Rate(self.f_loop)
		try:
			while not rospy.is_shutdown():
					
					# detected markers 
					(marker_det, det_img, proc_img, img) = self.preProcImage()
					# estimate cam pose
					if marker_det:
						(err, est_camera_pose) = self.estimatePoseLS(det_img, 
												 				   err, 
																   self.initialGuess(marker_det) if self.frame_cnt == 0 else est_camera_pose, 
																   marker_det)
						# (err, est_camera_pose) = self.estimatePoseFL(det_img, err, marker_det)
						if err < self.err_term:
							self.marker_table_poses.update({"camera": {'xyz': est_camera_pose[:3], 'rpy': est_camera_pose[3:]}})
							with open(self.fl, 'w') as fw:
								yaml.dump(self.marker_table_poses, fw)
							print("Final pose est:", est_camera_pose)
							break

					if self.vis:
						# frame counter
						cv2.putText(det_img, str(self.frame_cnt), (det_img.shape[1]-40, 20), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.FONT_CLR, self.FONT_THCKNS, cv2.LINE_AA)
						for id, det in marker_det.items():
							# label marker pose
							self.labelDetection(det_img, id, det['ftrans'], det['frot'])
							# plot pose by id
							if id == self.plt_id and self.frame_cnt > 10:
								cv2.imshow('Plot', cv2.cvtColor(visPose(det['ftrans'], self.euler2Matrix(det['frot']), det['frot'], self.frame_cnt, cla=True), cv2.COLOR_RGBA2BGR))
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
				   ).runDebug()
	elif rospy.get_param('~camera_pose', False):
		CameraPoseDetect(camera_ns=rospy.get_param('~markers_camera_name', ''),
						marker_length=rospy.get_param('~marker_length', 0.010),
						use_reconfigure=rospy.get_param('~use_reconfigure', False),
						vis=rospy.get_param('~vis', True),
						filter_type=rospy.get_param('~filter', 'none'),
						f_ctrl=rospy.get_param('~f_ctrl', 30),
						use_aruco=rospy.get_param('~use_aruco', False),
						plt_id=rospy.get_param('~plot_id', -1),
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
						).run()
	
if __name__ == "__main__":
	main()
