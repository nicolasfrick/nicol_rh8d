#!/usr/bin/env python3

import os, sys
import yaml
import cv2
import rospy
import tf2_ros
import cv_bridge
import numpy as np
import sensor_msgs.msg
from typing import Optional, Any, Tuple
import dynamic_reconfigure.server
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import PoseStamped
from tf2_geometry_msgs import PointStamped
import open_manipulator_msgs.msg
import open_manipulator_msgs.srv
from plot_record import *
from pose_filter import *
from nicol_rh8d.cfg import ArucoDetectorConfig
from marker_detector import ArucoDetector, AprilDetector
np.set_printoptions(threshold=sys.maxsize, suppress=True)

		# buf = tf2_ros.Buffer()
		# listener = tf2_ros.TransformListener(buf)
					# stamp = rgb.header.stamp
					# frame_id = rgb.header.frame_id
					# raw_img_size = (raw_img.shape[1], raw_img.shape[0])
					
class DetectBase():
	
	def __init__(self) -> None:
		pass


class CameraPoseDetect():
	"""
		@param camera_ns Camera namespace precceding 'image_raw' and 'camera_info'
		@type str
		@param vis Show detection images
		@type bool
		@param filter_type
		@type str
		@param filter_steps
		@type int
		@param f_ctrl
		@type float
		@param use_aruco
		@type bool
		@param plt_id
		@type int
		@param err_term
		@type float
		@param cart_bound_low
		@type float
		@param cart_bound_high
		@type float

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
							f_ctrl: Optional[int]=30,
							use_aruco: Optional[bool]=False,
							plt_id: Optional[int]=-1,
							err_term: Optional[float]=1e-8,
							cart_bound_low: Optional[float]=-3.0,
							cart_bound_high: Optional[float]=3.0,
							) -> None:
		
		self.vis = vis
		self.plt_id = plt_id
		self.f_loop = f_ctrl
		self.err_term = err_term
		self.use_aruco = use_aruco
		self.filter_type = filter_type
		self.img_topic = camera_ns + '/image_raw'
		self.bridge = cv_bridge.CvBridge()
		self.lower_bounds = [cart_bound_low, cart_bound_low, cart_bound_low, -np.pi, -np.pi, -np.pi]
		self.upper_bounds = [cart_bound_high, cart_bound_high, cart_bound_high, np.pi, np.pi, np.pi]

		# load marker poses
		self.fl = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cfg/marker_poses.yml")
		with open(self.fl, 'r') as fr:
			self.marker_table_poses = yaml.safe_load(fr)

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
			
		if vis:
			cv2.namedWindow("Processed", cv2.WINDOW_NORMAL)
			cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
			if plt_id > -1:
				cv2.namedWindow("Plot", cv2.WINDOW_NORMAL)
				setPosePlot()

		# dynamic reconfigure
		if use_reconfigure:
			print("Using reconfigure server")
			self.det_config_server = dynamic_reconfigure.server.Server(ArucoDetectorConfig, self.det.setDetectorParams)

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
		tvec_inv, euler_inv = self.det.invPersp(camera_pose[:3], camera_pose[3:], axis_angle=False)
		T_cam_world = self.pose2Matrix(tvec_inv, euler_inv)
		# iter measured markers
		for id, det in detection.items():
			# reprojection error
			err += self.projectSingleMarker(det, id, T_cam_world, img)
		return err
	
	def euler2Matrix(self, euler: np.ndarray) -> np.ndarray:
		return R.from_euler('xyz', euler).as_matrix()
	
	def pose2Matrix(self, translation: np.ndarray, euler: np.ndarray=None, rot_mat: np.ndarray=None) -> np.ndarray:
		transformation_matrix = np.eye(4)
		transformation_matrix[:3, :3] = self.euler2Matrix(euler) if rot_mat is None else rot_mat
		transformation_matrix[:3, 3] = translation
		return transformation_matrix
	
	def tagWorldCorners(self, world_tag_tf: np.ndarray, tag_corners: np.ndarray) -> np.ndarray:
		"""Transform marker corners to world frame""" 
		homog_corners = np.hstack((tag_corners, np.ones((tag_corners.shape[0], 1))))
		world_corners = world_tag_tf @ homog_corners.T
		world_corners = world_corners.T 
		return world_corners[:, :3]
	
	def getWorldMarkerTF(self, id: int) -> np.ndarray:
		# marker root tf
		root = self.marker_table_poses.get('root')
		T_world_root = self.pose2Matrix(root['xyz'], root['rpy']) if root is not None else np.eye(4)
		# marker tf
		marker = self.marker_table_poses.get(id)
		assert(marker) # marker id entry in yaml?
		T_root_marker = self.pose2Matrix(marker['xyz'], marker['rpy'])
		# worldTmarker
		return T_world_root @ T_root_marker
	
	def camTF(self, detection: dict, id: int) -> np.ndarray:
		tf = np.zeros(6)
		det = detection.get(id)
		if det is None:
			print(f"Cannot find id {id} in detection!")
			return tf
		tvec = det['ftrans']
		euler = det['frot']
		# get markerTcamera
		inv_tvec, inv_euler = self.det.invPersp(tvec, euler, axis_angle=False)
		T_marker_cam = self.pose2Matrix(inv_tvec, inv_euler)
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
		T_world_camera = self.pose2Matrix(camera_pose[:3], camera_pose[3:])
		# invert for reprojection
		tvec_inv, euler_inv = self.det.invPersp(camera_pose[:3], camera_pose[3:], axis_angle=False)
		T_camera_world = self.pose2Matrix(tvec_inv, euler_inv)
		for id in marker_poses:
			det = detection.get(id)
			if det is not None:
				# detected tag pose wrt camera frame
				T_camera_marker = self.pose2Matrix(det['ftrans'], det['frot'])
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
				filter = createFilter(self.filter_type, PoseFilterBase.poseToMeasurement(tvec=T_world_cam[:3], rvec=T_world_cam[3:], axis_angle=False), self.f_loop)
			else:
				filter.updateFilter(PoseFilterBase.poseToMeasurement(tvec=T_world_cam[:3], rvec=T_world_cam[3:], axis_angle=False))
		if filter is not None:
			filtered_pose[:3] = filter.est_translation
			filtered_pose[3:] = filter.est_rotation_as_euler
			self.labelDetection(img, 30, filtered_pose[:3], filtered_pose[3:])
			err = self.projectMarkers(detection, filtered_pose, img)
		print(f"camera world pose trans: {filtered_pose[:3]}, rot (extr. xyz euler): {filtered_pose[3:]}")
		return err, filtered_pose
		
	def run(self) -> None:
		cnt = 0
		err = np.inf
		est_camera_pose = np.zeros(6)
		rate = rospy.Rate(self.f_loop)
		try:
			while not rospy.is_shutdown():
					
					# detected markers 
					rgb = rospy.wait_for_message(self.img_topic, sensor_msgs.msg.Image)
					raw_img = self.bridge.imgmsg_to_cv2(rgb, 'bgr8')
					(marker_det, det_img, proc_img) = self.det.detMarkerPoses(raw_img.copy())

					# estimate cam pose
					if marker_det:
						(err, est_camera_pose) = self.estimatePoseLS(det_img, 
												 				   err, 
																   self.initialGuess(marker_det) if cnt == 0 else est_camera_pose, 
																   marker_det)
						# (err, est_camera_pose) = self.estimatePoseFL(det_img, err, marker_det)
						if err < self.err_term:
							self.marker_table_poses.update({"camera": {'xyz': est_camera_pose[:3], 'rpy': est_camera_pose[3:]}})
							with open(self.fl, 'w') as fw:
								yaml.dump(self.marker_table_poses, fw)
							print("Final pose est:", est_camera_pose)
							break
						cnt+=1

					if self.vis:
						# frame counter
						cv2.putText(det_img, str(cnt), (det_img.shape[1]-40, 20), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.FONT_CLR, self.FONT_THCKNS, cv2.LINE_AA)
						for id, det in marker_det.items():
							# label marker pose
							self.labelDetection(det_img, id, det['ftrans'], det['frot'])
							# plot pose by id
							if id == self.plt_id and cnt > 10:
								cv2.imshow('Plot', cv2.cvtColor(visPose(det['ftrans'], self.euler2Matrix(det['frot']), det['frot'], cnt, cla=True), cv2.COLOR_RGBA2BGR))
						cv2.imshow('Processed', proc_img)
						cv2.imshow('Detection', det_img)
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

def main() -> None:
	rospy.init_node('dataset_collector')
	CameraPoseDetect(camera_ns=rospy.get_param('~markers_camera_name', ''),
											marker_length=rospy.get_param('~marker_length', 0.010),
											use_reconfigure=rospy.get_param('~use_reconfigure', False),
											vis=rospy.get_param('~vis', True),
											filter_type=rospy.get_param('~filter', 'none'),
											f_ctrl=rospy.get_param('~f_ctrl', 30),
											use_aruco=rospy.get_param('~use_aruco', False),
											plt_id=rospy.get_param('~plot_id', -1),
											).run()
	
if __name__ == "__main__":
	main()
