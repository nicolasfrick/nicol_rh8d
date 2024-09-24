#!/usr/bin/env python3

import os, sys
import numpy as np
import cv2
import cv2.aruco as aru
import cv_bridge
import rospy
import tf2_ros
import sensor_msgs.msg
import geometry_msgs.msg
import open_manipulator_msgs.msg
import open_manipulator_msgs.srv
import dynamic_reconfigure.server
from nicol_rh8d.cfg import RH8DDatasetCollectorConfig
from scipy.spatial.transform import Rotation
np.set_printoptions(threshold=sys.maxsize, suppress=True)

camera_cv_bridge = cv_bridge.CvBridge()
cv2.namedWindow("im_out", cv2.WINDOW_NORMAL)

NUM_MARKERS = 33
MARKER_SIZE = 6 # >= number of marker cols/rows + 2x BORDER_BITS
MARKER_SCALING = 5 # factor to scale marker matrix img
MARKER_LENGTH = 0.0105 # dimension of markers in mm
ARUCO_DICT = aru.DICT_4X4_50 # 4 cols, 4 rows, 50 pcs

def focalMM_to_focalPixel( focalMM, pixelPitch ):
    f = focalMM / pixelPitch
    return f

def saveArucoImgMatrix(aruco_dict: dict, show: bool=False):
	"""Aligns the markers in a mxn matrix where m >= n ."""

	# matrix
	n = int(np.sqrt(NUM_MARKERS))  # cols
	residual = NUM_MARKERS - np.square(n)
	m = n + residual//n + (1 if residual%n > 0 else 0) # rows

	# entries
	v_border = np.ones((MARKER_SIZE, MARKER_SIZE))*255 # white vertical spacing = 1 *marker size
	v_border[:, 1] = v_border[:, -2] = [MARKER_SIZE*30] # add grey border lines
	h_border = np.ones((MARKER_SIZE, (2*n*MARKER_SIZE) + MARKER_SIZE))*255 # white horizontal spacing = n * (marker size + v_border size) + v_border size
	h_border[1, :] = h_border[-2, :] = [MARKER_SIZE*30] # add horizontal grey border lines
	h_border[:, 1::2*MARKER_SIZE] = h_border[:, MARKER_SIZE-2::2*MARKER_SIZE] = [MARKER_SIZE*30]  # add vertical grey border lines
	rows = v_border.copy()
	matrix = h_border.copy()

	# draw
	idx = 0
	print("Order of ", NUM_MARKERS, " Aruco markers:")
	print("-"*NUM_MARKERS)
	# cols
	for _ in range(m):
			# rows
			for _ in range(n):
				print(idx, " ", end="") if idx < NUM_MARKERS else print("pad ", end="")
				aruco_img = aru.generateImageMarker(aruco_dict, idx, MARKER_SIZE) if idx < NUM_MARKERS else v_border
				rows = np.hstack((rows, aruco_img, v_border))
				idx += 1
			matrix = np.vstack((matrix, rows, h_border))
			rows = v_border.copy()
			print()
	print("-"*NUM_MARKERS)
	print("Scaling by factor ", MARKER_SCALING)

	# resize and save
	matrix = cv2.resize(matrix, None, fx=MARKER_SCALING, fy=MARKER_SCALING, interpolation= cv2.INTER_AREA)
	cv2.imwrite(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets/aruco/aruco_matrix_scale_" + str(MARKER_SCALING) + ".png"),  matrix)
	if show:
		cv2.imshow("matrix", matrix)
		if cv2.waitKey(0) == ord("q"):
			cv2.destroyAllWindows()

aruco_dict = aru.getPredefinedDictionary(ARUCO_DICT)
# saveArucoImgMatrix(aruco_dict)
det_params = aru.DetectorParameters()
print(det_params)
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
	rospy.init_node('RH8D_dataset_collector')
	t_total = 0
	it_total = 0
	rate = rospy.Rate(30)
	try:
		while not rospy.is_shutdown():
				rgb_info = rospy.wait_for_message('marker_realsense/color/camera_info', sensor_msgs.msg.CameraInfo, 10)
				rgb = rospy.wait_for_message('marker_realsense/color/image_raw', sensor_msgs.msg.Image)
				raw_img = camera_cv_bridge.imgmsg_to_cv2(rgb, 'bgr8')
				img_cpy = np.copy(raw_img)
				stamp = rgb.header.stamp
				frame_id = rgb.header.frame_id
				raw_img_size = (raw_img.shape[1], raw_img.shape[0])
				
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
							np_rodrigues = np.asarray(rvec[:,:], np.float64)
							rmat = cv2.Rodrigues(np_rodrigues)[0]
							camera_position = -np.matrix(rmat).T @ np.matrix(tvec)
							r = Rotation.from_rotvec([rvec[0][0],rvec[1][0],rvec[2][0]])
							rot = r.as_euler('xyz', degrees=True)
							rx = round(180-rot[0],5) 
							ry = round(rot[1],5) 
							rz = round(rot[2],5) 
							tx = camera_position[0][0]
							ty = camera_position[1][0]
							tz = camera_position[2][0]
							marker_poses.update({id: {'rvec': rvec, 'tvec': tvec, 'rpy': [rx, ry, rz], 'xyz': [tx, ty, tz]}})

					t_current = (cv2.getTickCount() - tick) / cv2.getTickFrequency()
					t_total += t_current
					it_total += 1
					if it_total % 30 == 0:
						print("Detection Time = {} ms (Mean = {} ms)".format(t_current * 1000, 1000 * t_total / it_total))

				else:
					rospy.logwarn_throttle(10, "No marker found")
				# print(marker_poses)
				
				out_img = aru.drawDetectedMarkers(img_cpy, corners, ids)
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

if __name__ == "__main__":
	main()

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