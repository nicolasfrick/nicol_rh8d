#!/usr/bin/env python
 
import os
import cv2 
import argparse
import numpy as np
from pynput import keyboard
from time import sleep

pressed = False
def on_activate():
	global pressed
	pressed = True
terminate = False
def on_terminate():
	global terminate
	terminate = True

def main(video_dev: int, outfile: str, width: int, height: int, num_squares_x: int, num_squares_y: int, sqr_size: float):
	global pressed, terminate

	# Chessboard dimensions
	number_of_squares_X = num_squares_x 
	number_of_squares_Y = num_squares_y  
	nX = number_of_squares_X - 1 # Number of interior corners along x-axis
	nY = number_of_squares_Y - 1 # Number of interior corners along y-axis
	square_size = sqr_size 
	
	# Set termination criteria. We stop either when an accuracy is reached or when
	# we have finished a certain number of iterations.
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) 
	
	# Define real world coordinates for points in the 3D coordinate frame
	# Object points are (0,0,0), (1,0,0), (2,0,0) ...., (5,8,0)
	object_points_3D = np.zeros((nX * nY, 3), np.float32)  
	# These are the x and y coordinates                                              
	object_points_3D[:,:2] = np.mgrid[0:nY, 0:nX].T.reshape(-1, 2) 
	
	object_points_3D = object_points_3D * square_size
	
	# Store vectors of 3D points for all chessboard images (world coordinate frame)
	object_points = []
	# Store vectors of 2D points for all chessboard images (camera coordinate frame)
	image_points = []

	# define a video capture object 
	vid = cv2.VideoCapture(video_dev) 
	vid.set(cv2.CAP_PROP_FRAME_WIDTH, width)
	vid.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
	if not vid.isOpened():
		print(f"Could not open device {video_dev}, exiting!")
		return
	else:
		ret, frame = vid.read() 
		print(f"Video opened with {frame.shape}, detecting chessboard with {number_of_squares_X}x{number_of_squares_Y} squares with length {square_size} m")

	cnt = 0
	with keyboard.GlobalHotKeys({'<enter>': on_activate,
							 									 'q': on_terminate}) as h:
		while vid.isOpened(): 
			# Capture the video frame by frame 
			ret, frame = vid.read() 
			det = False

			if pressed:
				pressed = False
				# Convert the image to grayscale
				gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
				# Find the corners on the chessboard
				success, corners = cv2.findChessboardCorners(gray, (nY, nX), None)
				
				# If the corners are found by the algorithm, draw them
				if success:
					cnt += 1
					# Append object points
					object_points.append(object_points_3D)
					# Find more exact corner pixels       
					corners_2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)       
					# Append image points
					image_points.append(corners_2)
					# Draw the corners
					cv2.drawChessboardCorners(frame, (nY, nX), corners_2, success)
					print(f"Chessboard detected, recorded {cnt} images")
					det = True
				else:
					print("No chessboard detected")

			# display
			cv2.imshow("Image", frame) 
			if cv2.waitKey(1) & 0xFF == ord('q') or terminate: 
				h.stop()
				h.join()
				break
			if det:
				sleep(1)
	
	# After the loop release the cap object 
	vid.release() 
	# Destroy all the windows 
	cv2.destroyAllWindows() 

	if len(image_points):						
		print("Detection completed, computing ...")																		  
		# Perform camera calibration to return the camera matrix, distortion coefficients, rotation and translation vectors etc 
		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, 
															image_points, 
															gray.shape[::-1], 
															None, 
															None)
	
		# Save parameters to a file
		cv_file = cv2.FileStorage(outfile, cv2.FILE_STORAGE_WRITE)
		cv_file.write('K', mtx)
		cv_file.write('D', dist)
		cv_file.release()
	
		# Load the parameters from the saved file
		cv_file = cv2.FileStorage(outfile, cv2.FILE_STORAGE_READ) 
		mtx = cv_file.getNode('K').mat()
		dst = cv_file.getNode('D').mat()
		cv_file.release()
	
		# Display key parameter outputs of the camera calibration process
		print("Camera matrix:") 
		print(mtx) 
		print("\n Distortion coefficient:") 
		print(dist) 
	else:
		print("No detection to compute")
	  
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Camera calibration with a chessboard pattern')
	parser.add_argument('--video_dev', metavar='int', help='Video device', default=0)
	parser.add_argument('--width', metavar='int', help='Video width', default=1280)
	parser.add_argument('--height', metavar='int', help='Video height', default=960)
	parser.add_argument('--num_squares_x', metavar='int', help='Number of vertical squares', default=10)
	parser.add_argument('--num_squares_y', metavar='int', help='Number of horizontal squares', default=7)
	parser.add_argument('--sqr_size', metavar='float', help='Sidelength of a square in meter', default=0.025)
	parser.add_argument('--outfile', metavar='str', help='Filepath for calib results', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cfg/camcalib.yaml"))
	args = parser.parse_args()
	main(args.video_dev, args.outfile, args.width, args.height, args.num_squares_x, args.num_squares_y, args.sqr_size)

# REALSENSE D415:
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
