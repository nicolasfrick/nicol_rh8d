import cv2
import numpy as np
import pyrealsense2 as rs
from dt_apriltags import Detector
from scipy.spatial.transform import Rotation as R

cv2.namedWindow("AprilTag Detection", cv2.WINDOW_NORMAL)
cv2.namedWindow("Proc", cv2.WINDOW_NORMAL)

# Initialize camera settings
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
pipeline.start(config)

# Define AprilTag detector parameters optimized for small tags (10mm)
at_detector = Detector(
    families='tag16h5',  # Optimized for small markers
    nthreads=4,
    quad_decimate=0.0,    # Lower this value for small markers
    quad_sigma=0.0,       # Improve image contrast
    refine_edges=True,    # Refine edges for better detection
    decode_sharpening=0.25, # Sharpen image before decoding
)

# Camera intrinsic parameters (from RealSense D415 calibration or manual setup)
fx, fy =  1396.59387207031, 1395.52648925781  # Focal lengths (example values)
cx, cy =  944.551452636719, 547.094909667969   # Principal points (center of image)
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
dist_coeffs = np.zeros(5)  # Assuming no distortion for simplicity

# Define marker size (10mm = 0.01 meters)
marker_size = 0.069

# Known world coordinates of 5 AprilTags (example positions in meters)
world_tag_positions = {
    0: np.array([[0, 0, 0], [0.1, 0, 0], [0.1, 0.1, 0], [0, 0.1, 0]], dtype=np.float32),  # Tag ID 0
    1: np.array([[0.2, 0, 0], [0.3, 0, 0], [0.3, 0.1, 0], [0.2, 0.1, 0]], dtype=np.float32),  # Tag ID 1
    2: np.array([[0.4, 0, 0], [0.5, 0, 0], [0.5, 0.1, 0], [0.4, 0.1, 0]], dtype=np.float32),  # Tag ID 2
    3: np.array([[0.6, 0, 0], [0.7, 0, 0], [0.7, 0.1, 0], [0.6, 0.1, 0]], dtype=np.float32),  # Tag ID 3
    4: np.array([[0.8, 0, 0], [0.9, 0, 0], [0.9, 0.1, 0], [0.8, 0.1, 0]], dtype=np.float32),  # Tag ID 4
}

# Function to invert pose and return Euler angles and inverted rotation matrix
def invert_pose(rot_mat, tvec):
    R_inv = rot_mat.T
    tvec_inv = -R_inv @ tvec
    euler = R.from_matrix(R_inv).as_euler('xyz')
    return R_inv, tvec_inv, euler

# Function to draw coordinate axes on the image for the AprilTag pose
# def draw_axes(image, camera_matrix, dist_coeffs, rvec, tvec, marker_size=0.01):
#     # Define axis points (3D points in the marker coordinate system)
#     axis_length = marker_size * 2  # You can adjust this to make the axes more visible
#     axis = np.float32([[axis_length, 0, 0], [0, axis_length, 0], [0, 0, -axis_length]]).reshape(-1, 3)
    
#     # Project 3D points to 2D image points
#     imgpts, _ = cv2.projectPoints(axis, rvec.flatten(), tvec.flatten(), camera_matrix, dist_coeffs)
    
#     # Draw the axes (X: Red, Y: Green, Z: Blue)
#     corner = tuple(np.int32(imgpts[0]).ravel())
#     image = cv2.line(image, corner, tuple(np.int32(imgpts[1]).ravel()), (0, 255, 0), 5)  # Y-Axis (Green)
#     image = cv2.line(image, corner, tuple(np.int32(imgpts[2]).ravel()), (255, 0, 0), 5)  # Z-Axis (Blue)
#     image = cv2.line(image, corner, tuple(np.int32(imgpts[0]).ravel()), (0, 0, 255), 5)  # X-Axis (Red)
    
#     return image

def drawMarkers(detection, image):
    corners = detection.corners.astype(int)
    # id
    cv2.putText(image, str(detection.tag_id), tuple(corners[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    # boundary
    for i in range(4):
        cv2.line(image, tuple(corners[i]), tuple(corners[(i + 1) % 4]), (0, 255, 0), 2)
    # axes
    cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, detection.pose_R, detection.pose_t, 0.05)

# Function to compute camera pose from detected markers
def estimate_camera_pose(detections, camera_matrix, dist_coeffs):
    obj_points = []
    img_points = []

    # Iterate through detected tags
    for detection in detections:
        tag_id = detection.tag_id
        if tag_id in world_tag_positions:
            # 3D world coordinates of the tag corners
            world_points = world_tag_positions[tag_id]
            # 2D image points of the detected tag corners
            image_points = np.array(detection.corners, dtype=np.float32)

            # Collect points for PnP
            obj_points.append(world_points)
            img_points.append(image_points)

    if len(obj_points) > 0:
        obj_points = np.vstack(obj_points)  # Combine all 3D points
        img_points = np.vstack(img_points)  # Combine all 2D points

        # Estimate camera pose using solvePnP
        success, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)
        if success:
            return rvec, tvec
    return None, None

def validateDetection(detection):
    corners = detection.corners.astype(int)
    marker_width = np.linalg.norm(corners[0] - corners[1])
    marker_height = np.linalg.norm(corners[1] - corners[2])
    return detection.decision_margin > 50 and detection.hamming < 5 and marker_width > 50 and marker_height > 50

# Main loop
try:
    while True:
        # Capture image frame
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        image = np.asanyarray(color_frame.get_data())
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Histogram equalization to improve contrast
        gray = cv2.equalizeHist(gray)

        detections = at_detector.detect(gray, estimate_tag_pose=True, camera_params=(fx, fy, cx, cy), tag_size=marker_size)

        for detection in detections:
            if validateDetection(detection):
                # Extract pose data
                pose_r = detection.pose_R
                xang, _ = cv2.Rodrigues(pose_r)
                euler = R.from_matrix(pose_r).as_euler('xyz')
                pose_t = detection.pose_t
                pose_err = detection.pose_err

                # Draw the detected tag's coordinate system
                drawMarkers(detection, image)

                # Invert the pose (optional, for marker to camera transformation)
                _, tvec_inv, euler_inv = invert_pose(pose_r,  pose_t)

                print(f"ID: {detection.tag_id}, Translation Vector (inv): {tvec_inv.flatten()}, Euler Angles (inv): {euler_inv}, Euler Angles: {euler}, Pose error: {pose_err}")
            else:
                drawMarkers(detection, gray)
        print()
            
        # Estimate the camera pose based on the detected markers
        # rvec, tvec = estimate_camera_pose(detections, camera_matrix, dist_coeffs)
        # if rvec is not None and tvec is not None:
        #     # Display camera pose in the world frame
        #     print(f"Rotation Vector: {rvec.flatten()}")
        #     print(f"Translation Vector: {tvec.flatten()}")

        # Show the image with detected tags and coordinate systems
        cv2.imshow("Proc", gray)
        cv2.imshow("AprilTag Detection", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Clean up
    pipeline.stop()
    cv2.destroyAllWindows()

# import numpy as np
# import cv2
# from dt_apriltags import Detector
# from scipy.spatial.transform import Rotation as R

# # Initialize the AprilTag detector with optimized parameters
# at_detector = Detector(
#     families='tag36h11',  # Choose the tag family; 'tag36h11' is a good choice for robustness
#     nthreads=1,  # Number of threads for parallel processing
#     quad_decimate=1.0,  # Full resolution
#     quad_sigma=0.0,  # No Gaussian blur
#     refine_edges=1,  # Refine marker edges
#     decode_sharpening=0.25,  # Sharpen the decoded image
#     debug=0  # Set to 1 if you want to see debug output
# )

# # Function to invert a 4x4 transformation matrix
# def invert_transformation_matrix(matrix):
#     r = matrix[:3, :3]
#     t = matrix[:3, 3]
#     r_inv = r.T
#     t_inv = -r_inv @ t
#     matrix_inv = np.eye(4)
#     matrix_inv[:3, :3] = r_inv
#     matrix_inv[:3, 3] = t_inv
#     return matrix_inv

# # Function to detect AprilTags and compute poses
# def detect_apriltags_and_invert_pose(image):
#     # Convert image to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # Detect AprilTags in the image
#     tags = at_detector.detect(gray)
    
#     for tag in tags:
#         print(f"Tag ID: {tag.tag_id}")
#         print(f"Pose (tag to camera):\n{tag.pose_T_tag_camera}")
        
#         # Compute the inverted pose (camera to tag)
#         pose_T_camera_tag = invert_transformation_matrix(tag.pose_T_tag_camera)
#         print(f"Inverted Pose (camera to tag):\n{pose_T_camera_tag}")
    
#     return tags

# # Example usage
# if __name__ == "__main__":
#     # Load an example image (replace with your image path or capture from camera)
#     image = cv2.imread('example_image.jpg')
    
#     # Detect AprilTags and compute poses
#     detected_tags = detect_apriltags_and_invert_pose(image)
    
#     # Draw detected tags on the image (for visualization)
#     for tag in detected_tags:
#         # Draw bounding box
#         corners = np.array(tag.corners, dtype=np.int32)
#         cv2.polylines(image, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
#         # Draw tag ID
#         cv2.putText(image, str(tag.tag_id), tuple(corners[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
#     # Show the result
#     cv2.imshow('Detected AprilTags', image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# 	import cv2
# import numpy as np
# from dt_apriltags import Detector

# # Initialize the AprilTag detector from Duckietown
# at_detector = Detector(
#     families='tag36h11',  # Tag family (Duckietown uses 36h11)
#     nthreads=1,           # Number of threads
#     quad_decimate=1.0,     # Decimation factor (for faster detection at lower resolution)
#     quad_sigma=0.0,        # Gaussian blur (0.0 means no blur)
#     refine_edges=1,        # Whether to refine edges
#     decode_sharpening=0.25, # Sharpening for decoding
#     debug=0                # No debug output
# )

# # Camera intrinsic parameters (for pose estimation) - adjust these according to your camera calibration
# fx = 600  # Focal length in x direction (in pixels)
# fy = 600  # Focal length in y direction (in pixels)
# cx = 320  # Principal point x-coordinate (in pixels)
# cy = 240  # Principal point y-coordinate (in pixels)

# camera_params = (fx, fy, cx, cy)

# # The physical size of your AprilTag marker (in meters)
# tag_size = 0.162  # Example: 16.2 cm wide

# # Open a video capture (0 for webcam)
# cap = cv2.VideoCapture(0)

# while True:
#     # Read a frame from the video stream
#     ret, frame = cap.read()

#     if not ret:
#         print("Failed to grab frame")
#         break

#     # Convert the frame to grayscale (AprilTag detection works on grayscale images)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Detect AprilTags in the image
#     detections = at_detector.detect(gray, estimate_tag_pose=True, camera_params=camera_params, tag_size=tag_size)

#     # Draw the detections and calculate 6D pose for each detected tag
#     for detection in detections:
#         # Draw a bounding box around the detected tag
#         for i in range(4):
#             pt1 = (int(detection.corners[i][0]), int(detection.corners[i][1]))
#             pt2 = (int(detection.corners[(i + 1) % 4][0]), int(detection.corners[(i + 1) % 4][1]))
#             cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

#         # Translation and rotation in the camera frame
#         translation = detection.pose_t.flatten()  # Translation vector (x, y, z)
#         rotation_matrix = detection.pose_R  # Rotation matrix (3x3)

#         # Convert rotation matrix to Euler angles (roll, pitch, yaw)
#         rvec, _ = cv2.Rodrigues(rotation_matrix)
#         rvec_deg = np.degrees(rvec)

#         # Print marker-to-camera pose (translation and rotation)
#         print(f"Tag ID: {detection.tag_id}")
#         print(f"Marker-to-Camera Translation (x, y, z): {translation}")
#         print(f"Marker-to-Camera Rotation (roll, pitch, yaw) in degrees: {rvec_deg.ravel()}")

#         # ----------- Inverted Pose Calculation (Camera-to-Marker) ----------- #
#         # Invert the rotation matrix (transpose)
#         rotation_matrix_inv = rotation_matrix.T

#         # Invert the translation
#         translation_inv = -np.dot(rotation_matrix_inv, translation)

#         # Convert the inverted rotation matrix to Euler angles
#         rvec_inv, _ = cv2.Rodrigues(rotation_matrix_inv)
#         rvec_inv_deg = np.degrees(rvec_inv)

#         # Print camera-to-marker pose (inverted transformation)
#         print(f"Camera-to-Marker Translation (x, y, z): {translation_inv}")
#         print(f"Camera-to-Marker Rotation (roll, pitch, yaw) in degrees: {rvec_inv_deg.ravel()}")

#     # Display the result
#     cv2.imshow('AprilTag Detection', frame)

#     # Press 'q' to quit the loop
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the video capture and close windows
# cap.release()
# cv2.destroyAllWindows()

# Detecting a 10-millimeter marker from a distance of 0.4 meters can be challenging because the marker will cover only a small portion of the image. However, there are several strategies you can use to improve detection:
# 1. Increase Image Resolution

#     Higher Resolution Camera: A camera with a higher resolution will provide more pixels for the marker, which helps in detecting smaller markers.
#     Higher Resolution Images: If possible, capture images at higher resolutions and then downscale them for processing.

# 2. Improve Camera Calibration

#     Accurate Calibration: Ensure that your camera is well-calibrated to reduce distortion and improve the accuracy of pose estimation. Proper calibration also improves the detection reliability.

# 3. Enhance Image Quality

#     Good Lighting: Ensure adequate and uniform lighting to minimize shadows and reflections on the marker, which helps in better detection.
#     Sharp Focus: Make sure the camera is in focus. A blurry image can significantly affect detection accuracy.

# 4. Optimize Detection Parameters

#     Adjust Detection Parameters: Tune parameters such as quad_decimate, quad_sigma, and decode_sharpening to balance detection speed and accuracy. For very small markers, you might need to adjust these parameters to be more sensitive.
#     Use High-Contrast Markers: Ensure that your AprilTag markers have high contrast with the background to make them more distinguishable.

# 5. Image Preprocessing

#     Image Enhancement: Apply preprocessing techniques such as sharpening or contrast enhancement to make the marker edges more distinct.
#     Thresholding: Use thresholding techniques to increase the contrast of the marker against the background.

# Here’s a combined example applying some of these techniques:

# python

# import cv2
# import numpy as np

# # Load the image
# image = cv2.imread('your_image.jpg')

# # Convert to YUV color space
# yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
# yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])  # Histogram equalization on Y channel

# # Convert back to BGR color space
# equalized_image = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

# # Apply Gaussian Blur
# blurred_image = cv2.GaussianBlur(equalized_image, (5, 5), 0)

# # Apply sharpening filter
# sharpening_kernel = np.array([[0, -1, 0],
#                               [-1, 5,-1],
#                               [0, -1, 0]])
# sharpened_image = cv2.filter2D(blurred_image, -1, sharpening_kernel)

# # Adjust brightness and contrast
# alpha = 1.2
# beta = 30
# adjusted_image = cv2.convertScaleAbs(sharpened_image, alpha=alpha, beta=beta)

# # Save or display the result
# cv2.imshow('Enhanced Image', adjusted_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Proper illumination is crucial for achieving accurate marker detection, especially when working with small markers or in varying lighting conditions. Here are some strategies to enhance scene illumination for optimal marker detection:
# **1. Even Lighting:

#     Diffuse Light: Use diffuse lighting to reduce harsh shadows and reflections. Soft, even lighting helps the camera capture clearer images and improves marker detection. Lightboxes or softboxes can be effective for diffusing light.
#     Avoid Direct Light: Position light sources to avoid direct glare on the markers. Glare can interfere with marker detection and pose estimation.

# **2. Control Shadows:

#     Minimize Shadows: Use multiple light sources or reflectors to reduce shadows. Shadows can obscure parts of the marker and affect detection accuracy.
#     Light Placement: Position lights at different angles to minimize the creation of shadows on the marker.

# **3. Use High-Quality Lighting:

#     Consistent Color Temperature: Use lights with a consistent color temperature to avoid color shifts in your images. LED lights with adjustable color temperatures can be beneficial.
#     High CRI (Color Rendering Index): Choose lights with a high CRI for better color accuracy, which can help in distinguishing markers more effectively.

# **4. Lighting Techniques:

#     Backlighting: For certain applications, backlighting can enhance marker visibility by creating a silhouette effect, which can make the marker stand out more.
#     Side Lighting: Side lighting can help highlight the edges of markers and improve their visibility.

# **5. Use Controlled Lighting Environments:

#     Lighting Control: Create a controlled lighting environment to ensure consistent and optimal lighting conditions for marker detection. Avoid changes in ambient light that could affect detection accuracy.
#     Light Modifiers: Use light modifiers like diffusers, reflectors, and flags to control and shape the light.

# **6. Adjust Camera Settings:

#     Exposure and Gain: Adjust the camera’s exposure and gain settings to match the lighting conditions. Properly configured settings can enhance image quality and marker detection.
#     White Balance: Set the correct white balance on the camera to ensure accurate color representation and reduce color distortions.

# **7. Additional Illumination Tools:

#     Ring Lights: Ring lights provide even illumination around the camera lens, reducing shadows and glare. They can be particularly useful for close-up shots.
#     LED Panels: Adjustable LED panels can provide consistent and controllable lighting, suitable for various scenes and distances.

# **8. Test and Calibrate:

#     Calibration: Regularly test and calibrate your lighting setup to ensure it remains consistent and effective for marker detection.
#     Test Images: Capture test images under different lighting conditions to identify the best setup for your specific markers and environment.

# Summary

# To achieve good marker detection, focus on providing even, diffuse lighting to minimize shadows and glare. Use high-quality, consistent lighting and adjust your camera settings to complement the lighting conditions. Controlled lighting environments and additional tools like ring lights or LED panels can further enhance detection accuracy. Regular testing and calibration ensure that your lighting setup remains effective over time.


# import cv2
# import numpy as np
# import glob

# # Define the dimensions of the chessboard
# chessboard_size = (9, 6)  # Number of internal corners in the chessboard (columns, rows)

# # Prepare object points based on the chessboard size
# object_points = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
# object_points[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# # Arrays to store object points and image points
# obj_points = []
# img_points = []

# # Path to your calibration images
# images_path = 'calibration_images/*.jpg'  # Change this path to your images

# # Load images
# images = glob.glob(images_path)

# for image_path in images:
#     img = cv2.imread(image_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     # Find the chessboard corners
#     ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    
#     if ret:
#         obj_points.append(object_points)
#         img_points.append(corners)
        
#         # Draw and display the corners
#         cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
#         cv2.imshow('Chessboard', img)
#         cv2.waitKey(500)

# cv2.destroyAllWindows()

# # Calibration
# ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

# # Save calibration data
# np.savez('camera_calibration.npz', camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)

# print("Calibration done. Camera matrix and distortion coefficients saved.")

# # Load calibration data
# with np.load('camera_calibration.npz') as X:
#     camera_matrix = X['camera_matrix']
#     dist_coeffs = X['dist_coeffs']

# # Undistort an image
# img = cv2.imread('test_image.jpg')  # Change this to a test image
# h, w = img.shape[:2]
# new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
# undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

# # Crop the image
# x, y, w, h = roi
# undistorted_img = undistorted_img[y:y+h, x:x+w]

# # Display and save undistorted image
# cv2.imshow('Undistorted Image', undistorted_img)
# cv2.imwrite('undistorted_image.jpg', undistorted_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
# import glob

# # Define chessboard size
# chessboard_size = (9, 6)

# # Prepare object points
# object_points = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
# object_points[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# # Arrays to store object points and image points
# obj_points = []
# img_points = []

# # Path to calibration images
# images_path = 'calibration_images/*.jpg'

# # Load images
# images = glob.glob(images_path)

# for image_path in images:
#     img = cv2.imread(image_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     # Find chessboard corners
#     ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    
#     if ret:
#         obj_points.append(object_points)
#         img_points.append(corners)
#         cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
#         cv2.imshow('Chessboard', img)
#         cv2.waitKey(500)

# cv2.destroyAllWindows()

# # Camera calibration
# ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

# # Compute reprojection error
# total_error = 0
# for i in range(len(obj_points)):
#     img_points2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
#     error = cv2.norm(img_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
#     total_error += error

# mean_error = total_error / len(obj_points)

# print(f"Reprojection error: {mean_error}")

# Explanation:

#     Object and Image Points: obj_points and img_points are the object points (3D coordinates) and image points (2D coordinates) from the calibration images.
#     Project Points: cv2.projectPoints() projects the 3D object points into the image plane using the estimated parameters.
#     Error Calculation: cv2.norm() calculates the Euclidean distance between the observed image points and the projected points.
#     Mean Error: The mean reprojection error is averaged over all images and points.

# 	Got it! If you have the camera extrinsics (i.e., the rotation and translation vectors for the camera relative to the world or object coordinate system) and the object poses, you can still compute the reprojection error. Here’s how you can approach this problem:
# **1. Understanding the Setup

#     Camera Extrinsics: Define the position and orientation of the camera relative to a reference coordinate system (usually the world or object coordinate system). These are typically represented by rotation vectors and translation vectors.

#     Object Poses: These define the position and orientation of the calibration pattern or object in the world coordinate system.

# **2. Compute the Reprojection Error

# To compute the reprojection error using the camera extrinsics, follow these steps:

#     Transform Object Points to Camera Coordinates:
#         Given the object poses and camera extrinsics, you can transform the 3D object points from the object coordinate system to the camera coordinate system.

#     Project 3D Points to Image Plane:
#         Use the camera intrinsic parameters (camera matrix and distortion coefficients) to project these transformed 3D points onto the 2D image plane.

#     Compute Reprojection Error:
#         Compare the projected points with the observed image points to compute the reprojection error.

# Example Code

# Here’s a Python script to compute the reprojection error using camera extrinsics:

# python

# import cv2
# import numpy as np

# # Define known camera parameters
# camera_matrix = np.array([[fx, 0, cx],  # Replace with actual values
#                           [0, fy, cy],
#                           [0, 0, 1]])
# dist_coeffs = np.array([k1, k2, p1, p2, k3])  # Replace with actual values

# # Define known object points (e.g., for a 9x6 chessboard)
# object_points = np.zeros((9*6, 3), np.float32)  # Replace with the number of points in your pattern
# object_points[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# # Known camera extrinsics (rotation and translation vectors)
# # For each object pose, we need the rotation vector and translation vector
# rvecs = [...]  # List of rotation vectors for each object pose
# tvecs = [...]  # List of translation vectors for each object pose

# # Observed image points (for each object pose)
# img_points = [...]  # List of observed image points for each object pose

# # Calculate reprojection error
# total_error = 0
# for i in range(len(object_points)):
#     # Transform object points to camera coordinates using extrinsics
#     # Assuming object points are in world coordinates and need to be transformed
#     rvec = rvecs[i]
#     tvec = tvecs[i]
    
#     # Project 3D object points to 2D image points
#     img_points2, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    
#     # Calculate the error
#     error = cv2.norm(img_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
#     total_error += error

# mean_error = total_error / len(object_points)
# print(f"Reprojection error: {mean_error}")

# Explanation:

#     Camera Parameters: camera_matrix and dist_coeffs should be replaced with the actual camera intrinsic parameters.

#     Object Points: Define the 3D object points based on your calibration pattern.

#     Camera Extrinsics: rvecs and tvecs are the rotation and translation vectors representing the camera's pose for each image.

#     Projection and Error Calculation: Use cv2.projectPoints() to project the object points to the image plane and then calculate the reprojection error by comparing these projections to the observed image points.

# Summary

# By using the camera extrinsics (rotation and translation vectors) along with the camera intrinsic parameters, you can project 3D object points onto the image plane and compute the reprojection error. This error measures how well the estimated camera and object poses align with the actual image data.