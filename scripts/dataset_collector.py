#!/usr/bin/env python3

import os, sys, math
import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
import cv_bridge
import rospy
import tf2_ros
import sensor_msgs.msg
from geometry_msgs.msg import PoseStamped
from tf2_geometry_msgs import PointStamped
import open_manipulator_msgs.msg
import open_manipulator_msgs.srv
import dynamic_reconfigure.server
from nicol_rh8d.cfg import ArucoDetectorConfig
from aruco_detector import ArucoDetector
from typing import Sequence, Optional, Tuple, Union, Any
np.set_printoptions(threshold=sys.maxsize, suppress=True)

def euler_from_quaternion(x, y, z, w):
  """
  Convert a quaternion into euler angles (roll, pitch, yaw)
  roll is rotation around x in radians (counterclockwise)
  pitch is rotation around y in radians (counterclockwise)
  yaw is rotation around z in radians (counterclockwise)
  """
  t0 = +2.0 * (w * x + y * z)
  t1 = +1.0 - 2.0 * (x * x + y * y)
  roll_x = math.atan2(t0, t1)
	  
  t2 = +2.0 * (w * y - z * x)
  t2 = +1.0 if t2 > +1.0 else t2
  t2 = -1.0 if t2 < -1.0 else t2
  pitch_y = math.asin(t2)
	  
  t3 = +2.0 * (w * z + x * y)
  t4 = +1.0 - 2.0 * (y * y + z * z)
  yaw_z = math.atan2(t3, t4)
	  
  return roll_x, pitch_y, yaw_z # in radians
			
def getPose(rvec, tvec):
	  # x-axis points to the right
	  # y-axis points straight down towards your toes
	  # z-axis points straight ahead away from your eye, out of the camera
		print(tvec)
		print(rvec)
		# Store the translation (i.e. position) information
		transform_translation_x = tvec[0]
		transform_translation_y = tvec[1]
		transform_translation_z = tvec[2]
 
		# Store the rotation information
		rotation_matrix = np.eye(4)
		rotation_matrix[0:3, 0:3] = cv2.Rodrigues(np.array(rvec))[0]
		r = R.from_matrix(rotation_matrix[0:3, 0:3])
		print(r.as_euler('xyz'))
		quat = r.as_quat()   
		 
		# Quaternion format     
		transform_rotation_x = quat[0] 
		transform_rotation_y = quat[1] 
		transform_rotation_z = quat[2] 
		transform_rotation_w = quat[3] 
		 
		# Euler angle format in radians
		roll_x, pitch_y, yaw_z = euler_from_quaternion(transform_rotation_x, 
													   transform_rotation_y, 
													   transform_rotation_z, 
													   transform_rotation_w)
		 
		# roll_x = math.degrees(roll_x)
		# pitch_y = math.degrees(pitch_y)
		# yaw_z = math.degrees(yaw_z)
		print("transform_translation_x: {}".format(transform_translation_x))
		print("transform_translation_y: {}".format(transform_translation_y))
		print("transform_translation_z: {}".format(transform_translation_z))
		print("roll_x: {}".format(roll_x))
		print("pitch_y: {}".format(pitch_y))
		print("yaw_z: {}".format(yaw_z))
		# print()
		return transform_translation_x, transform_translation_y, transform_translation_z, roll_x, pitch_y, yaw_z

def create_transform(rotation, translation):
	transform = np.eye(4)
	transform[:3, :3] = rotation.as_matrix()
	transform[:3, 3] = translation
	return transform

class CameraPose():
	"""
        @param camera_ns Camera namespace precceding 'image_raw' and 'camera_info'
		@type str
		@param invert_pose Invert detected marker pose 
		@type bool
		@param vis Show detection images
		@type bool

	"""
	
	def __init__(self,
                            camera_ns: Optional[str]='',
                            invert_pose: Optional[bool]=True,
							vis :Optional[bool]=True) -> None:
		
		self.bridge = cv_bridge.CvBridge()
		# buf = tf2_ros.Buffer()
		# listener = tf2_ros.TransformListener(buf)

		# load marker poses
		fl = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets/aruco/marker_poses.yml")
		with open(fl, 'r') as fr:
			self.marker_table_poses = yaml.safe_load(fr)

		rospy.loginfo("Waiting for camera_info from %s", camera_ns + '/camera_info')
		rgb_info = rospy.wait_for_message(camera_ns + '/camera_info', sensor_msgs.msg.CameraInfo)
		self.det = ArucoDetector(marker_length=0.010, 
														 K=rgb_info.K, 
														 D=rgb_info.D, 
														 dict_yaml='custom_matrix_4x4_32_consider_flipped.yml')
		self.det_config_server = dynamic_reconfigure.server.Server(ArucoDetectorConfig, self.det.setDetectorParams)
		self.invert_pose = invert_pose
		self.img_topic = camera_ns + '/image_raw'
		self.vis = vis
		if vis:
			cv2.namedWindow("aruco_img", cv2.WINDOW_NORMAL)

	def transformToWorld(self, id, x, y, z, r, p, ya):
		# target_pose_from_cam = PoseStamped()
		# target_pose_from_cam.header.frame_id  = "camera_frame"
		# target_pose_from_cam.pose.position.x = x
		# target_pose_from_cam.pose.position.y = y
		# target_pose_from_cam.pose.position.z = z
		# rot = R.from_euler('xyz', [[r, p, ya]], degrees=False)
		# quat = rot.as_quat()[0]
		# target_pose_from_cam.pose.orientation.x = quat[0]
		# target_pose_from_cam.pose.orientation.y = quat[1]
		# target_pose_from_cam.pose.orientation.z = quat[2]
		# target_pose_from_cam.pose.orientation.w = quat[3]
		# try:
		# 	target_pose_from_req = buf.transform(target_pose_from_cam, "base_link", timeout=rospy.Duration(3))
		# 	print(target_pose_from_req)
		# except Exception as e:  
		# 	print(e)

		world_rot = R.from_euler('xyz', np.array([0, 0, 0]))
		W = create_transform(world_rot, np.array([0, 0, 0.8]))
		
		marker_rot = R.from_euler('xyz', np.array(self.marker_table_poses[id]['rpy']))
		M = create_transform(world_rot, np.array(self.marker_table_poses[id]['xyz']))
		
		cam_rot = R.from_euler('xyz', np.array([r, p, ya]))
		# print(cam_rot.as_matrix())
		t = np.array([x, y, z])[0]
		T = create_transform(cam_rot, t)

		res = W@M@T
		res_rot = R.from_matrix(res[:3,:3])
		res_trans = res[:3, 3]
		print(res_rot.as_euler('xyz', degrees=False), res_trans)
		return res

	def getTf(self, tvecs: dict) -> None:
		for id, vecs in tvecs.items():
			if id == 4:
				tx, ty, tz, rx, ry, rz = getPose(vecs['rvec'], vecs['tvec'])
				self.transformToWorld(id, tx, ty, tz, rx, ry, rz)
				print()
				
	def getExtrinsics(self, pose_dct: dict) -> dict:
		"""Create homog. transformations from dictionary
			  and compute transform world -> camera frame"""

		# world -> base tf
		W = np.eye(4)
		W[:3, :3] = R.from_euler('xyz', np.array(self.marker_table_poses[0]['rpy'])).as_matrix()
		W[:3, 3] = self.marker_table_poses[0]['xyz']

		for id, vecs in pose_dct.items():
			# base -> marker tf
			M = np.eye(4)
			M[:3, :3] = R.from_euler('xyz', np.array(self.marker_table_poses[id]['rpy'])).as_matrix()
			M[:3, 3] = self.marker_table_poses[id]['xyz']

			T = np.eye(4)
			# T[:3, :3] = R.from_euler('xyz', np.array([np.pi,0,-np.pi/2])).as_matrix()
			# T[:3, 3] = np.array([-0.3,-0.1789,1])
			roudriges, _ = cv2.Rodrigues(np.array(pose_dct[id]['rvec']))
			mat = R.from_matrix(roudriges)
			cv_euler = mat.as_euler('xyz')
			# print(cv_euler)
			ros_euler = np.array([cv_euler[2], cv_euler[0], cv_euler[1]])
			T[:3, :3] =  R.from_euler('xyz', ros_euler).as_matrix()
			# T[:3, 3] = pose_dct[id]['tvec'][0]
			v = pose_dct[id]['tvec'] # .reshape(3)
			T[:3, 3] = v # np.array([v[2], v[0], v[1]])
			# if id==4:
			# 	print(id, ros_euler, v)
			
			res = W@M@T
			res_rot = R.from_matrix(res[:3,:3])
			res_trans = res[:3, 3]
			print(id, res_rot.as_euler('xyz', degrees=False), res_trans)
			if id==4: print()

		# # marker -> camera tf
		# C = np.eye(4)
		# # axis-angle to rot matrix
		# roudriges = cv2.Rodrigues(np.array(pose_dct[id]['rvec']))[0]
		# mat = R.from_matrix(roudriges)
		# cv_euler = mat.as_euler('xyz')
		# ros_euler = np.array([cv_euler[2], cv_euler[0], cv_euler[1]])
		# mat = R.from_euler('xyz', ros_euler)
		# C[:3, :3] = mat.as_matrix()
		# C[:3, 3] = pose_dct[id]['tvec'][0]

		# res = W@M@C
		# res_rot = R.from_matrix(res[:3,:3])
		# res_trans = res[:3, 3]
		# # print(id, res_rot.as_euler('xyz', degrees=False), res_trans)
		
	def invPersp(self, rvec: cv2.typing.MatLike, tvec: cv2.typing.MatLike) -> tuple[cv2.typing.MatLike, cv2.typing.MatLike]:
		"""Apply the inversion to the given vectors"""
		mat, _ = cv2.Rodrigues(rvec)
		mat = np.matrix(mat).T
		inv_tvec = np.dot(-mat, tvec)
		inv_rvec, _ = cv2.Rodrigues(mat)
		return inv_rvec, inv_tvec
		
	def run(self) -> None:
		rate = rospy.Rate(6)
		try:
			while not rospy.is_shutdown():
					rgb = rospy.wait_for_message(self.img_topic, sensor_msgs.msg.Image)
					raw_img = self.bridge.imgmsg_to_cv2(rgb, 'bgr8')

					stamp = rgb.header.stamp
					frame_id = rgb.header.frame_id
					raw_img_size = (raw_img.shape[1], raw_img.shape[0])
					
					(marker_poses, out_img) = self.det.detMarkerPoses(raw_img)
					if marker_poses:
						# self.getTf(marker_poses)
						# self.getExtrinsics(marker_poses)
						pass

					if self.vis:
						cv2.imshow('aruco_img', out_img)
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
	CameraPose(camera_ns=rospy.get_param('~markers_camera_name', '')
			).run()
	
if __name__ == "__main__":
	main()
