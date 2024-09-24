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
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import PoseStamped
from tf2_geometry_msgs import PointStamped
import open_manipulator_msgs.msg
import open_manipulator_msgs.srv
from nicol_rh8d.cfg import ArucoDetectorConfig
from aruco_detector import ArucoDetector, AprilDetector
np.set_printoptions(threshold=sys.maxsize, suppress=True)

		# buf = tf2_ros.Buffer()
		# listener = tf2_ros.TransformListener(buf)
					# stamp = rgb.header.stamp
					# frame_id = rgb.header.frame_id
					# raw_img_size = (raw_img.shape[1], raw_img.shape[0])

class CameraPose():
	"""
		@param camera_ns Camera namespace precceding 'image_raw' and 'camera_info'
		@type str
		@param invert_pose Invert detected marker pose 
		@type bool
		@param vis Show detection images
		@type bool
		@param filter_type
		@type str
		@param filter_steps
		@type int
		@param f_ctrl
		@type
		@param use_aruco
		@type bool

	"""

	FONT_THCKNS = 2
	FONT_SCALE = 0.5
	FONT_CLR =  (255,0,0)
	TXT_OFFSET = 25
	
	def __init__(self,
			  				marker_length: float=0.010,
							camera_ns: Optional[str]='',
							invert_pose: Optional[bool]=False,
							vis :Optional[bool]=True,
							use_reconfigure: Optional[bool]=False,
							filter_type: Optional[str]='none',
							f_ctrl: Optional[int]=30,
							invert_perspective: Optional[bool]=False,
							use_aruco: Optional[bool]=False,
							) -> None:
		
		self.vis = vis
		self.f_loop = f_ctrl
		self.invert_pose = invert_pose
		self.invert_perspective = invert_perspective
		self.img_topic = camera_ns + '/image_raw'
		self.bridge = cv_bridge.CvBridge()
		# load marker poses
		fl = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets/aruco/marker_poses.yml")
		with open(fl, 'r') as fr:
			self.marker_table_poses = yaml.safe_load(fr)
		# init detector
		rospy.loginfo("Waiting for camera_info from %s", camera_ns + '/camera_info')
		rgb_info = rospy.wait_for_message(camera_ns + '/camera_info', sensor_msgs.msg.CameraInfo)
		if use_aruco:
			self.det = ArucoDetector(marker_length=marker_length, 
															K=rgb_info.K, 
															D=rgb_info.D,
															f_ctrl=f_ctrl,
															filter_type=filter_type)
		else:
			self.det = AprilDetector(marker_length=marker_length, 
															K=rgb_info.K, 
															D=rgb_info.D,
															f_ctrl=f_ctrl,
															filter_type=filter_type)
		if vis:
			cv2.namedWindow("Processed", cv2.WINDOW_NORMAL)
			cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
			cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
		if use_reconfigure:
			print("Using reconfigure server")
			self.det_config_server = dynamic_reconfigure.server.Server(ArucoDetectorConfig, self.det.setDetectorParams)
		
	def invPersp(self, tvec: np.ndarray, rvec: np.ndarray, axis_angle: bool=True) -> Tuple[cv2.typing.MatLike, cv2.typing.MatLike]:
		"""Apply the inversion to the given vectors [[R^-1 -R^-1*d][0 0 0 1]]"""
		mat, _ = cv2.Rodrigues(rvec) if axis_angle else (R.from_euler('xyz', np.array(rvec)).as_matrix(), None) # axis-angle or euler to mat
		mat = np.matrix(mat).T # orth. matrix: A.T = A^-1
		inv_rvec = R.from_matrix(mat)
		inv_rvec = inv_rvec.as_euler('xyz')
		inv_tvec = -mat @ tvec # -R^-1*d
		return np.array(inv_tvec.flat), inv_rvec.flatten()
	
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

	def labelDetection(self, img: cv2.typing.MatLike, id: int, trans: np.ndarray, rot: np.ndarray, corners: np.ndarray) -> None:
			pos_txt = "{} X: {:.4f} Y: {:.4f} Z: {:.4f} R: {:.4f} P: {:.4f} Y: {:.4f}".format(id, trans[0], trans[1], trans[2], rot[0], rot[1], rot[2])
			xpos = self.TXT_OFFSET
			ypos = (id+1)*self.TXT_OFFSET
			cv2.putText(img, pos_txt, (xpos, ypos), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.FONT_CLR, self.FONT_THCKNS, cv2.LINE_AA)
		
	def run(self) -> None:
		cnt = 0
		rate = rospy.Rate(self.f_loop)
		try:
			while not rospy.is_shutdown():
					cnt+=1
					rgb = rospy.wait_for_message(self.img_topic, sensor_msgs.msg.Image)
					raw_img = self.bridge.imgmsg_to_cv2(rgb, 'bgr8')
					res_img = raw_img.copy()

					(marker_poses, det_img, proc_img) = self.det.detMarkerPoses(raw_img.copy())
					if self.vis:
						# frame counter
						cv2.putText(det_img, str(cnt), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.FONT_CLR, self.FONT_THCKNS, cv2.LINE_AA)
						for id, vec in marker_poses.items():
							# vec['frot'] = np.array([np.pi, 0, np.pi*0.5])
							(trans, rot) = self.invPersp(vec['ftrans'], vec['frot'], axis_angle=False) if self.invert_perspective else (vec['ftrans'], vec['frot'])
							self.labelDetection(res_img, id, trans, rot, vec['corners'])
						cv2.imshow('Processed', proc_img)
						cv2.imshow('Detection', det_img)
						cv2.imshow('Result', res_img)
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
	CameraPose(camera_ns=rospy.get_param('~markers_camera_name', ''),
							 marker_length=rospy.get_param('~marker_length', 0.010),
							 invert_pose=rospy.get_param('~invert_pose', False),
							 use_reconfigure=rospy.get_param('~use_reconfigure', False),
							 vis=rospy.get_param('~vis', True),
							 filter_type=rospy.get_param('~filter', 'none'),
							 f_ctrl=rospy.get_param('~f_ctrl', 30),
							 use_aruco=rospy.get_param('~use_aruco', False),
			).run()
	
if __name__ == "__main__":
	main()
