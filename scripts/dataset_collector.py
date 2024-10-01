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
from nicol_rh8d.cfg import ArucoDetectorConfig
from aruco_detector import ArucoDetector, AprilDetector
np.set_printoptions(threshold=sys.maxsize, suppress=True)

		# buf = tf2_ros.Buffer()
		# listener = tf2_ros.TransformListener(buf)
					# stamp = rgb.header.stamp
					# frame_id = rgb.header.frame_id
					# raw_img_size = (raw_img.shape[1], raw_img.shape[0])
					
class CameraPoseBase():
	
	def __init__(self) -> None:
		pass


class CameraPose():
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
							) -> None:
		
		self.vis = vis
		self.plt_id = plt_id
		self.f_loop = f_ctrl
		self.err_term = err_term
		self.img_topic = camera_ns + '/image_raw'
		self.bridge = cv_bridge.CvBridge()

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
	
	def euler2Matrix(self, euler: np.ndarray) -> np.ndarray:
		return R.from_euler('xyz', euler).as_matrix()

	def pose2Matrix(self, translation: np.ndarray, euler: np.ndarray) -> np.ndarray:
		transformation_matrix = np.eye(4)
		transformation_matrix[:3, :3] = self.euler2Matrix(euler)
		transformation_matrix[:3, 3] = translation
		return transformation_matrix
	
	def tagWorldCorners(self, world_tag_tf: np.ndarray, tag_corners: np.ndarray) -> np.ndarray:
		"""Transform marker corners to world frame""" 
		homog_corners = np.hstack((tag_corners, np.ones((tag_corners.shape[0], 1))))
		world_corners = world_tag_tf @ homog_corners.T
		world_corners = world_corners.T 
		return world_corners[:, :3]

	def residuals(self, camera_pose:  np.ndarray, tag_poses: dict, marker_poses: dict) -> np.ndarray:
		"""Compute the residual (error) between world and detected poses.
		Rotations are extr. xyz euler angles."""
		camera_mat = self.pose2Matrix(camera_pose[:3], camera_pose[3:])
		error = []
		# tag root tf
		root = tag_poses.get('root')
		root_mat = self.pose2Matrix(root['xyz'], root['rpy']) if root is not None else np.eye(4)
		for id, tag_pose in tag_poses.items():
			det = marker_poses.get(id)
			if det is not None:
				# given tag pose wrt world 
				tag_mat = self.pose2Matrix(tag_pose['xyz'], tag_pose['rpy'])
				T_world_tag = root_mat @ tag_mat
				# estimated tag pose wrt camera frame12
				det_mat = self.pose2Matrix(det['ftrans'], det['frot'])
				T_world_estimated_tag = camera_mat @ det_mat
				error.append(np.linalg.norm(T_world_estimated_tag[:3, 3] - T_world_tag[:3, 3]))  # position error
				error.append(np.linalg.norm(T_world_estimated_tag[:3, :3] - T_world_tag[:3, :3]))  # orientation error
		return np.hstack(error)
	
	def reprojectionError(self, det_corners: np.ndarray, proj_corners: np.ndarray) -> float:
		error = np.linalg.norm(det_corners - proj_corners, axis=1)
		return np.mean(error)
	
	def projectMarkers(self, img: cv2.typing.MatLike, square_points: np.ndarray, marker_poses:dict, cam_trans: np.ndarray, cam_rot: np.ndarray, cmx: np.ndarray, dist: np.ndarray) -> float:
		err = 0
		for id, det in marker_poses.items():
			# tf marker corners to camera frame
			T_cam_tag = self.pose2Matrix(det['ftrans'], det['frot'])
			world_corners = self.tagWorldCorners(T_cam_tag, square_points)
			# project corners to image plane
			projected_corners, _ = cv2.projectPoints(world_corners, self.euler2Matrix(cam_rot), cam_trans, cmx, dist)
			projected_corners = np.int32(projected_corners).reshape(-1, 2)
			cv2.polylines(img, [projected_corners], isClosed=True, color=self.det.GREEN, thickness=2)
			# reprojection error
			err += self.reprojectionError(marker_poses[id]['corners'], projected_corners)
		return err

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

	def estimatePose(self, img, err, est_camera_pose, marker_poses):
		res = least_squares(self.residuals, est_camera_pose, args=(self.marker_table_poses, marker_poses))
		if res.success:
			opt_cam_pose = res.x
			status = res.status 
			# put pose label
			self.labelDetection(img, 30, opt_cam_pose[:3], opt_cam_pose[3:])
			# reproject markers
			reserr = self.projectMarkers(img, self.det.square_points, marker_poses, opt_cam_pose[:3], opt_cam_pose[3:], self.det.cmx, self.det.dist)
			print(f"Result: {status}\ncamera world pose trans: {opt_cam_pose[:3]}, rot (extr. xyz euler): {opt_cam_pose[3:]}\nreprojection error: {reserr}\n")
			return reserr, opt_cam_pose
		print(f"Least squares failed: {res.status}")
		return err, est_camera_pose
		
	def run(self) -> None:
		cnt = 0
		err = np.inf
		est_camera_pose = np.zeros(6)
		rate = rospy.Rate(self.f_loop)
		try:
			while not rospy.is_shutdown():
					cnt+=1
					rgb = rospy.wait_for_message(self.img_topic, sensor_msgs.msg.Image)
					raw_img = self.bridge.imgmsg_to_cv2(rgb, 'bgr8')

					# detected markers 
					(marker_det, det_img, proc_img) = self.det.detMarkerPoses(raw_img.copy())
					# estimate cam pose
					if marker_det:
						(err, est_camera_pose) = self.estimatePose(det_img, err, est_camera_pose, marker_det)

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
					
					# exit
					if err <= self.err_term:
						print("Final pose est:", est_camera_pose)
						with open(self.fl, 'w') as fw:
							yaml.dump(self.marker_table_poses, fw)
						break

		except ValueError as e:
			rospy.logerr(e)
		finally:
			cv2.destroyAllWindows()

def main() -> None:
	rospy.init_node('dataset_collector')
	CameraPose(camera_ns=rospy.get_param('~markers_camera_name', ''),
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
