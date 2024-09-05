#!/usr/bin/env python
 
import os
import cv2 
import numpy as np
import argparse
  
def main(video_dev: int, outfile: str, width: int, height: int, num_squares_x: int, num_squares_y: int, sqr_size: float):
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
	inp = 'start'
	while vid.isOpened(): 
		
		# Capture the video frame by frame 
		ret, frame = vid.read() 

		inp = input("Position the chessboard and press 'r' to record, 'q' to abort and any other key to show new img." if inp != '' else '')		

		if inp == "r":
			# Convert the image to grayscale
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
			# Find the corners on the chessboard
			success, corners = cv2.findChessboardCorners(gray, (nY, nX), None)
			
			# If the corners are found by the algorithm, draw them
			if success == True:
				# Append object points
				object_points.append(object_points_3D)
				# Find more exact corner pixels       
				corners_2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)       
				# Append image points
				image_points.append(corners_2)
				# Draw the corners
				cv2.drawChessboardCorners(frame, (nY, nX), corners_2, success)
				print(f"Chessboard detected, recorded {cnt} images")
				cnt += 1

		elif inp=="q":
			break

		# display
		cv2.imshow("Image", frame) 
		if cv2.waitKey(1) & 0xFF == ord('q'): 
			break
	
	# After the loop release the cap object 
	vid.release() 
	# Destroy all the windows 
	cv2.destroyAllWindows() 

	if len(image_points):																								  
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
	  
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Camera calibration with a chessboard pattern')
	parser.add_argument('--video_dev', metavar='int', help='Video device', default=0)
	parser.add_argument('--width', metavar='int', help='Video width', default=1280)
	parser.add_argument('--height', metavar='int', help='Video height', default=960)
	parser.add_argument('--num_squares_x', metavar='int', help='Number of vertical squares', default=10)
	parser.add_argument('--num_squares_y', metavar='int', help='Number of horizontal squares', default=7)
	parser.add_argument('--sqr_size', metavar='float', help='Sidelength of a square in meter', default=0.025)
	parser.add_argument('--outfile', metavar='str', help='Filepath for calib results', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets/aruco/camcalib.yaml"))
	args = parser.parse_args()
	main(args.video_dev, args.outfile, args.width, args.height, args.num_squares_x, args.num_squares_y, args.sqr_size)