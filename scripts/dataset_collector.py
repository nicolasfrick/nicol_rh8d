#!/usr/bin/env python3

# generate markers with opencv (custom build)
# .opencv_build/opencv/build/bin/example_cpp_aruco_dict_utils /home/nic/catkin_ws/src/nicol_rh8d/datasets/aruco/custom_matrix_4x4_20.yml -nMarkers=20 -markerSize=4 -r

import os, sys, math
import yaml
import numpy as np
import cv2
import cv2.aruco as aru
import cv_bridge
import rospy
import tf2_ros
import sensor_msgs.msg
from geometry_msgs.msg import PoseStamped
from tf2_geometry_msgs import PointStamped
import open_manipulator_msgs.msg
import open_manipulator_msgs.srv
import dynamic_reconfigure.server
from nicol_rh8d.cfg import RH8DDatasetCollectorConfig
from scipy.spatial.transform import Rotation as R
np.set_printoptions(threshold=sys.maxsize, suppress=True)

camera_cv_bridge = cv_bridge.CvBridge()
cv2.namedWindow("im_out", cv2.WINDOW_NORMAL)

rospy.init_node('RH8D_dataset_collector')

buf = tf2_ros.Buffer()
listener = tf2_ros.TransformListener(buf)

NUM_MARKERS = 32
MARKER_SIZE = 6 # >= number of marker cols/rows + 2x BORDER_BITS
MARKER_SCALING = 5 # factor to scale marker matrix img
MARKER_LENGTH = 0.185 # dimension of markers in m
ARUCO_DICT = aru.DICT_4X4_50 # 4 cols, 4 rows, 50 pcs 

def focalMM_to_focalPixel( focalMM, pixelPitch ):
    f = focalMM / pixelPitch
    return f

def loadArucoYaml(filename: str):

	fl = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets/aruco/" + filename)
	dct = aru.Dictionary()

	with open(fl, 'r') as fr:
		txt = yaml.safe_load(fr)
		num_markers = txt.pop('nmarkers')
		dct.markerSize = txt.pop('markersize')
		dct.maxCorrectionBits = txt.pop('maxCorrectionBits')
		nbytes =  int((dct.markerSize * dct.markerSize + 8 - 1) / 8)
		dct.bytesList = np.empty(shape=(num_markers, nbytes , 4), dtype = np.uint8)

		for key, val in txt.items():
			# convert str to int array
			bit_arr = np.array(list(map(int, val)), dtype=np.uint8)
			# size  # TODO: check if this works with n cols > 8
			bit_arr = bit_arr.reshape(dct.markerSize, dct.markerSize)
			compressed = aru.Dictionary.getByteListFromBits(bit_arr)
			# append to index
			idx = int(key.replace('marker_', ""))
			dct.bytesList[idx] = compressed

	return dct

def saveArucoImgMatrix(aruco_dict: dict, show: bool=False, filename: str="aruco_matrix.png", bbits: int=1, num: int=0, scale: float=MARKER_SCALING):
	"""Aligns the markers in a mxn matrix where m >= n ."""

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
				print(idx, " ", end="") if idx < num_markers else print("pad ", end="")
				aruco_img = aru.generateImageMarker(aruco_dict, idx, size) if idx < num_markers else v_border
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
		cv2.imshow("im_out", matrix)
		if cv2.waitKey(0) == ord("q"):
			cv2.destroyAllWindows()
			
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
		#       # x-axis points to the right
#       # y-axis points straight down towards your toes
#       # z-axis points straight ahead away from your eye, out of the camera

							# np_rodrigues = np.asarray(rvec[:,:], np.float64)
							# rmat = cv2.Rodrigues(np_rodrigues)[0]
							# camera_position = -np.matrix(rmat).T @ np.matrix(tvec)
							# r = R.from_rotvec([rvec[0][0],rvec[1][0],rvec[2][0]])
							# rot = r.as_euler('xyz', degrees=True)
							# rx = round(180-rot[0],5) 
							# ry = round(rot[1],5) 
							# rz = round(rot[2],5) 
							# tx = camera_position[0][0]
							# ty = camera_position[1][0]
							# tz = camera_position[2][0]
        # Store the translation (i.e. position) information
        transform_translation_x = tvec[0]
        transform_translation_y = tvec[1]
        transform_translation_z = tvec[2]
 
        # Store the rotation information
        rotation_matrix = np.eye(4)
        rotation_matrix[0:3, 0:3] = cv2.Rodrigues(np.array(rvec))[0]
        r = R.from_matrix(rotation_matrix[0:3, 0:3])
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
         
        roll_x = math.degrees(roll_x)
        pitch_y = math.degrees(pitch_y)
        yaw_z = math.degrees(yaw_z)
        print("transform_translation_x: {}".format(transform_translation_x))
        print("transform_translation_y: {}".format(transform_translation_y))
        print("transform_translation_z: {}".format(transform_translation_z))
        print("roll_x: {}".format(roll_x))
        print("pitch_y: {}".format(pitch_y))
        print("yaw_z: {}".format(yaw_z))
        print()
        return transform_translation_x, transform_translation_y, transform_translation_z, roll_x, pitch_y, yaw_z

def transformToWorld(x, y, z, r, p, ya):
	target_pose_from_cam = PoseStamped()
	target_pose_from_cam.header.frame_id  = "camera_frame"
	target_pose_from_cam.pose.position.x = x
	target_pose_from_cam.pose.position.y = y
	target_pose_from_cam.pose.position.z = z
	rot = R.from_euler('xyz', [[r, p, ya]], degrees=False)
	quat = rot.as_quat()[0]
	target_pose_from_cam.pose.orientation.x = quat[0]
	target_pose_from_cam.pose.orientation.y = quat[1]
	target_pose_from_cam.pose.orientation.z = quat[2]
	target_pose_from_cam.pose.orientation.w = quat[3]
	try:
		target_pose_from_req = buf.transform(target_pose_from_cam, "base_link", timeout=rospy.Duration(3))
		print(target_pose_from_req)
	except Exception as e:  
		print(e)

aruco_dict = loadArucoYaml('custom_matrix_4x4_32_consider_flipped.yml') # aru.getPredefinedDictionary(ARUCO_DICT)
det_params = aru.DetectorParameters()
aruco_det = aru.ArucoDetector(aruco_dict, det_params)

# https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html
# cv::SOLVEPNP_IPPE_SQUARE Special case suitable for marker pose estimation. 
# Number of input points must be 4. Object points must be defined in the following order:
#     point 0: [-squareLength / 2, squareLength / 2, 0]
#     point 1: [ squareLength / 2, squareLength / 2, 0]
#     point 2: [ squareLength / 2, -squareLength / 2, 0]
#     point 3: [-squareLength / 2, -squareLength / 2, 0]
obj_points = np.zeros((4, 3), dtype=np.float32)
obj_points[0,:] = np.array([-MARKER_LENGTH/2, MARKER_LENGTH/2, 0])
obj_points[1,:] = np.array([MARKER_LENGTH/2, MARKER_LENGTH/2, 0])
obj_points[2,:] = np.array([MARKER_LENGTH/2, -MARKER_LENGTH/2, 0])
obj_points[3,:] = np.array([-MARKER_LENGTH/2, -MARKER_LENGTH/2, 0])

def main():
	t_total = 0
	it_total = 0
	rate = rospy.Rate(30)
	try:
		while not rospy.is_shutdown():
				rgb_info = rospy.wait_for_message('marker_realsense/color/camera_info', sensor_msgs.msg.CameraInfo, 10)
				rgb = rospy.wait_for_message('marker_realsense/color/image_raw', sensor_msgs.msg.Image)
				raw_img = camera_cv_bridge.imgmsg_to_cv2(rgb, 'bgr8')
				# img_cpy = np.copy(raw_img)
				stamp = rgb.header.stamp
				frame_id = rgb.header.frame_id
				raw_img_size = (raw_img.shape[1], raw_img.shape[0])
				
				# raw_img = raw_img[382:1000, 615:1326]
				# raw_img = cv2.fastNlMeansDenoisingColored(raw_img,None,10,10,7,21)
                
				
				marker_poses = {}
				tick = cv2.getTickCount()
				(corners, ids, rejected) = aruco_det.detectMarkers(raw_img)
				if len(corners) > 0:
					ids = ids.flatten()
					for id, corner in zip(ids, corners):
						corner_reshaped = corner.reshape(4,2)
						cmx = np.asanyarray(rgb_info.K).reshape(3,3)
						dist =  np.asanyarray(rgb_info.D)
						# estimate camera pose relative to the marker using the unit provided by obj_points
						(res, rvec, tvec) = cv2.solvePnP(obj_points, corner_reshaped, cmx, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
						if res:
							tx, ty, tz, rx, ry, rz = getPose(rvec, tvec)
							transformToWorld(tx, ty, tz, rx, ry, rz)
							marker_poses.update({id: {'rvec': rvec, 'tvec': tvec, 'rpy': [rx, ry, rz], 'xyz': [tx, ty, tz]}})

					t_current = (cv2.getTickCount() - tick) / cv2.getTickFrequency()
					t_total += t_current
					it_total += 1
					if it_total % 30 == 0:
						print("Detection Time = {} ms (Mean = {} ms)".format(t_current * 1000, 1000 * t_total / it_total))

				else:
					rospy.logwarn_throttle(10, "No marker found")
				# print(marker_poses)
				
				out_img = aru.drawDetectedMarkers(raw_img, corners, ids)
				if marker_poses:
					for id, pose in marker_poses.items():
						out_img = cv2.drawFrameAxes(out_img, cmx, dist, pose['rvec'], pose['tvec'], MARKER_LENGTH*1.5, 2)

# #Test the solvePnP by projecting the 3D Points to camera
# projPoints = cv2.projectPoints(points_3D, rvecs, tvecs, K, distCoeffs)[0]
# for p in points_2D:
#  cv2.circle(im, (int(p[0]), int(p[1])), 3, (0,255,0), -1)
# for p in projPoints:
#  cv2.circle(im, (int(p[0][0]), int(p[0][1])), 3, (255,0,0), -1)
# cv2.imshow("image", im)
# cv2.waitKey(0)

				cv2.imshow('im_out', out_img)
				if cv2.waitKey(1) == ord("q"):
					break				

				try:
					rate.sleep()
				except:
					pass
	finally:
	    cv2.destroyAllWindows()
		
# #!/usr/bin/env python
  
# '''
# Welcome to the ArUco Marker Pose Estimator!
  
# This program:
#   - Estimates the pose of an ArUco Marker
# '''
  
# from __future__ import print_function # Python 2/3 compatibility
# import cv2 # Import the OpenCV library
# import numpy as np # Import Numpy library
# from scipy.spatial.transform import Rotation as R
# import math # Math library
 
# # Project: ArUco Marker Pose Estimator
# # Date created: 12/21/2021
# # Python version: 3.8
 
# # Dictionary that was used to generate the ArUco marker
# aruco_dictionary_name = "DICT_ARUCO_ORIGINAL"
 
# # The different ArUco dictionaries built into the OpenCV library. 
# ARUCO_DICT = {
#   "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
#   "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
#   "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
#   "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
#   "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
#   "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
#   "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
#   "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
#   "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
#   "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
#   "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
#   "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
#   "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
#   "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
#   "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
#   "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
#   "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL
# }
 
# # Side length of the ArUco marker in meters 
# aruco_marker_side_length = 0.0785
 
# # Calibration parameters yaml file
# camera_calibration_parameters_filename = 'calibration_chessboard.yaml'

 
# def main():
#   """
#   Main method of the program.
#   """
#   # Check that we have a valid ArUco marker
#   if ARUCO_DICT.get(aruco_dictionary_name, None) is None:
#     print("[INFO] ArUCo tag of '{}' is not supported".format(
#       args["type"]))
#     sys.exit(0)
 
#   # Load the camera parameters from the saved file
#   cv_file = cv2.FileStorage(
#     camera_calibration_parameters_filename, cv2.FILE_STORAGE_READ) 
#   mtx = cv_file.getNode('K').mat()
#   dst = cv_file.getNode('D').mat()
#   cv_file.release()
     
#   # Load the ArUco dictionary
#   print("[INFO] detecting '{}' markers...".format(
#     aruco_dictionary_name))
#   this_aruco_dictionary = cv2.aruco.Dictionary_get(ARUCO_DICT[aruco_dictionary_name])
#   this_aruco_parameters = cv2.aruco.DetectorParameters_create()
   
#   # Start the video stream
#   cap = cv2.VideoCapture(0)
   
#   while(True):
  
#     # Capture frame-by-frame
#     # This method returns True/False as well
#     # as the video frame.
#     ret, frame = cap.read()  
     
#     # Detect ArUco markers in the video frame
#     (corners, marker_ids, rejected) = cv2.aruco.detectMarkers(
#       frame, this_aruco_dictionary, parameters=this_aruco_parameters,
#       cameraMatrix=mtx, distCoeff=dst)
       
#     # Check that at least one ArUco marker was detected
#     if marker_ids is not None:
 
#       # Draw a square around detected markers in the video frame
#       cv2.aruco.drawDetectedMarkers(frame, corners, marker_ids)
       
#       # Get the rotation and translation vectors
#       rvecs, tvecs, obj_points = cv2.aruco.estimatePoseSingleMarkers(
#         corners,
#         aruco_marker_side_length,
#         mtx,
#         dst)
         
#       # Print the pose for the ArUco marker
#       # The pose of the marker is with respect to the camera lens frame.
#       # Imagine you are looking through the camera viewfinder, 
#       # the camera lens frame's:
#       # x-axis points to the right
#       # y-axis points straight down towards your toes
#       # z-axis points straight ahead away from your eye, out of the camera
#       for i, marker_id in enumerate(marker_ids):
       
#         # Store the translation (i.e. position) information
#         transform_translation_x = tvecs[i][0][0]
#         transform_translation_y = tvecs[i][0][1]
#         transform_translation_z = tvecs[i][0][2]
 
#         # Store the rotation information
#         rotation_matrix = np.eye(4)
#         rotation_matrix[0:3, 0:3] = cv2.Rodrigues(np.array(rvecs[i][0]))[0]
#         r = R.from_matrix(rotation_matrix[0:3, 0:3])
#         quat = r.as_quat()   
         
#         # Quaternion format     
#         transform_rotation_x = quat[0] 
#         transform_rotation_y = quat[1] 
#         transform_rotation_z = quat[2] 
#         transform_rotation_w = quat[3] 
         
#         # Euler angle format in radians
#         roll_x, pitch_y, yaw_z = euler_from_quaternion(transform_rotation_x, 
#                                                        transform_rotation_y, 
#                                                        transform_rotation_z, 
#                                                        transform_rotation_w)
         
#         roll_x = math.degrees(roll_x)
#         pitch_y = math.degrees(pitch_y)
#         yaw_z = math.degrees(yaw_z)
#         print("transform_translation_x: {}".format(transform_translation_x))
#         print("transform_translation_y: {}".format(transform_translation_y))
#         print("transform_translation_z: {}".format(transform_translation_z))
#         print("roll_x: {}".format(roll_x))
#         print("pitch_y: {}".format(pitch_y))
#         print("yaw_z: {}".format(yaw_z))
#         print()
         
#         # Draw the axes on the marker
#         cv2.aruco.drawAxis(frame, mtx, dst, rvecs[i], tvecs[i], 0.05)
     
#     # Display the resulting frame
#     cv2.imshow('frame',frame)
          
#     # If "q" is pressed on the keyboard, 
#     # exit this loop
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#       break
  
#   # Close down the video stream
#   cap.release()
#   cv2.destroyAllWindows()
   
# if __name__ == '__main__':
#   print(__doc__)
#   main()

if __name__ == "__main__":
	main()
	# dct = loadArucoYaml('custom_matrix_4x4_32_consider_flipped.yml')
	# saveArucoImgMatrix(dct, False, "custom_marker_scale_50.png", 1, 2, 50)

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
