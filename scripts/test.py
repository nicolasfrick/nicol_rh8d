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
from plot_record import *
from nicol_rh8d.cfg import ArucoDetectorConfig
from marker_detector import ArucoDetector, AprilDetector
np.set_printoptions(threshold=sys.maxsize, suppress=True)

# Result: 2
# camera world pose trans: [-0.36190126 -0.02421061  0.22947128], rot (extr. xyz euler): [-1.72959518  8.45835075 -1.81152342]
# reprojection error: 2057.8595611344845

fl = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cfg/marker_poses.yml")
with open(fl, 'r') as fr:
	marker_table_poses = yaml.safe_load(fr)

cmx = np.array([[1396.59387207,    0. ,         944.55145264],
                                [   0.,         1395.52648926,  547.09490967],
                                [   0. ,           0. ,           1.,        ]])

dist = np.zeros(5)

det = {0: {'rvec': np.array([-0.34859441,  0.34294092, -1.5501623 ]), 
                  'rot_mat': np.array([[-0.0061339 ,  0.90453091,  0.42636395],
                                                        [-0.99996852, -0.00769446,  0.00193771],
                                                        [ 0.00503336, -0.42633864,  0.90454963]]), 
                     'tvec': np.array([0.09840697, 0.00955485, 0.35067685]), 
                     'points': np.array([[-0.005,  0.005,  0.   ],
                                                        [ 0.005,  0.005,  0.   ],
                                                        [ 0.005, -0.005,  0.   ],
                                                        [-0.005, -0.005,  0.   ]], dtype=np.float32), 
                    'corners': np.array([[1357.10681152,  605.07189941],
                                                        [1356.81872559,  565.41137695],
                                                        [1315.96875   ,  564.99816895],
                                                        [1316.45568848,  605.00286865]]), 
                    'ftrans': np.array([0.09860986, 0.00958476, 0.35138461]), 
                    'frot': np.array([-0.28734761, -0.0175594 , -1.57673204]), 
                    'homography': np.array([[  -3.39992236,   26.13019818, 1336.67569002],
                                                                    [ -21.31959977,    2.63975201,  585.16937542],
                                                                    [  -0.00239893,    0.00430577,    1.        ]]), 
                    'center': np.array([1336.67569002,  585.16937542]), 
                    'pose_err': 1.250541538370862e-08},

                1: {   'rvec': np.array([-0.21304749,  0.09302966, -1.57006448]), 
                        'rot_mat': np.array([[ 0.00192813,  0.9810859 ,  0.19356326],
                                                                [-0.997079  , -0.012893  ,  0.07528104],
                                                                [ 0.07635278, -0.19314301,  0.97819529]]), 
                        'tvec': np.array([0.06815567, 0.03027054, 0.35310987]), 
                        'points': np.array([[-0.005,  0.005,  0.   ],
                                                            [ 0.005,  0.005,  0.   ],
                                                            [ 0.005, -0.005,  0.   ],
                                                            [-0.005, -0.005,  0.   ]], dtype=np.float32), 
                        'corners': np.array([[1234.26989746,  686.90454102],
                                                                [1233.95263672,  646.72668457],
                                                                [1193.88720703,  646.72692871],
                                                                [1194.57006836,  686.63873291]]), 
                        'ftrans': np.array([0.06747529, 0.02992297, 0.34939522]), 
                        'frot': np.array([-0.00175197, -0.1017238 , -1.5665549 ]),
                        'homography': np.array([[  -5.86623596,   15.88888297, 1214.10455937],
                                                                        [ -23.10610844,   -2.15842135,  666.84160992],
                                                                        [  -0.0046253 ,   -0.00333727,    1.        ]]), 
                        'center': np.array([1214.10455937,  666.84160992]), 
                        'pose_err': 2.2060953381335076e-08},

                    2: {'rvec': np.array([-0.18324325, -0.02206657, -2.3540988 ]), 
                        'rot_mat': np.array([[-0.70042176,  0.70255953,  0.1257754 ],
                                                                [-0.70007833, -0.71057445,  0.07052853],
                                                                [ 0.13892328, -0.03865291,  0.98954852]]), 
                        'tvec': np.array([ 0.06763991, -0.01064927,  0.35301528]), 
                        'points': np.array([[-0.005,  0.005,  0.   ],
                                                                [ 0.005,  0.005,  0.   ],
                                                                [ 0.005, -0.005,  0.   ],
                                                                [-0.005, -0.005,  0.   ]], dtype=np.float32), 
                        'corners': np.array([[1240.80053711,  504.76480103],
                                                                [1211.71850586,  477.209198  ],
                                                                [1183.80688477,  505.37640381],
                                                                [1212.40869141,  532.7131958 ]]), 
                        'ftrans': np.array([ 0.06788883, -0.01067734,  0.35188456]), 
                        'frot': np.array([-0.14895455, -0.1267767 , -2.35946512]), 
                        'homography': np.array([[ -11.78825544,    6.55301352, 1212.06499093],
                                                                            [ -12.62624133,  -17.16305625,  505.07316414],
                                                                            [   0.00217125,   -0.00620582,    1.        ]]), 
                            'center': np.array([1212.06499093,  505.07316414]),
                            'pose_err': 4.4718176547699155e-09},

                        3: {'rvec': np.array([-0.82216547,  0.18824519, -2.26289372]), 
                            'rot_mat': np.array([[-0.54489079,  0.57614129,  0.60922512],
                                                                    [-0.66888566, -0.73680493,  0.09854175],
                                                                    [ 0.50565404, -0.35380745,  0.78685086]]), 
                            'tvec': np.array([0.12962051, 0.03003794, 0.34938389]), 
                            'points': np.array([[-0.005,  0.005,  0.   ],
                                                                [ 0.005,  0.005,  0.   ],
                                                                [ 0.005, -0.005,  0.   ],
                                                                [-0.005, -0.005,  0.   ]], dtype=np.float32), 
                            'corners': np.array([[1491.68139648,  667.15661621],
                                                                    [1462.32373047,  638.82159424],
                                                                    [1434.13513184,  666.9397583 ],
                                                                    [1463.3059082 ,  695.50335693]]), 
                            'ftrans': np.array([0.13036756, 0.03018568, 0.35144779]), 
                            'frot': np.array([-0.03619044, -0.0928497 , -2.35405159]), 
                            'homography': np.array([[  -9.24732959,   14.67379583, 1462.81283272],
                                                                            [ -11.76901747,  -13.87297769,  667.04782763],
                                                                            [   0.003681  ,    0.00036432,    1.        ]]), 
                            'center': np.array([1462.81283272,  667.04782763]), 
                            'pose_err': 3.8707294748556104e-09},

                        4: { 'rvec': np.array([-0.11330249,  0.02223655, -1.55990409]),
                                'rot_mat': np.array([[ 0.01183694,  0.99622691,  0.08597579],
                                                                        [-0.99827279,  0.00682548,  0.05835103],
                                                                        [ 0.05754404, -0.08651799,  0.99458701]]), 
                            'tvec': np.array([ 0.12987763, -0.01095672,  0.35212431]), 
                            'points': np.array([[-0.005,  0.005,  0.   ],
                                                                [ 0.005,  0.005,  0.   ],
                                                                [ 0.005, -0.005,  0.   ],
                                                                [-0.005, -0.005,  0.   ]], dtype=np.float32), 
                            'corners': np.array([[1480.05603027,  523.64697266],
                                                                [1480.03942871,  483.87466431],
                                                                [1438.99963379,  483.93960571],
                                                                [1439.69140625,  523.2902832 ]]), 
                            'ftrans': np.array([ 0.13054188, -0.01108089,  0.35404608]), 
                            'frot': np.array([-0.06958664, -0.0357204 , -1.57375526]), 
                            'homography': np.array([[ -12.35284736,   12.52636813, 1459.58902891],
                                                                            [ -23.98127639,   -2.62572825,  503.85247526],
                                                                            [  -0.00834067,   -0.00535956,    1.        ]]),
                            'center': np.array([1459.58902891,  503.85247526]), 
                            'pose_err': 9.963025148258912e-09}}

def euler2Matrix(euler: np.ndarray) -> np.ndarray:
	return R.from_euler('xyz', euler).as_matrix()

def pose2Matrix(translation: np.ndarray, euler: np.ndarray=None, rot_mat: np.ndarray=None) -> np.ndarray:
	transformation_matrix = np.eye(4)
	transformation_matrix[:3, :3] = euler2Matrix(euler) if rot_mat is None else rot_mat
	transformation_matrix[:3, 3] = translation
	return transformation_matrix

def tagWorldCorners(world_tag_tf: np.ndarray, tag_corners: np.ndarray) -> np.ndarray:
	"""Transform marker corners to world frame""" 
	homog_corners = np.hstack((tag_corners, np.ones((tag_corners.shape[0], 1))))
	world_corners = world_tag_tf @ homog_corners.T
	world_corners = world_corners.T 
	return world_corners[:, :3]

def residuals(camera_pose:  np.ndarray, tag_poses: dict, marker_poses: dict) -> np.ndarray:
	"""Compute the residual (error) between world and detected poses.
	Rotations are extr. xyz euler angles."""
	camera_mat = pose2Matrix(camera_pose[:3], camera_pose[3:])
	error = []
	# tag root tf
	root = tag_poses.get('root')
	root_mat = pose2Matrix(root['xyz'], root['rpy']) if root is not None else np.eye(4)
	for id, tag_pose in tag_poses.items():
		det = marker_poses.get(id)
		if det is not None:
			# given tag pose wrt world 
			tag_mat = pose2Matrix(tag_pose['xyz'], tag_pose['rpy'])
			T_world_tag = root_mat @ tag_mat
			# estimated tag pose wrt camera frame12
			det_mat = pose2Matrix(det['ftrans'], det['frot'])
			T_world_estimated_tag = camera_mat @ det_mat
			error.append(np.linalg.norm(T_world_estimated_tag[:3, 3] - T_world_tag[:3, 3]))  # position error
			error.append(np.linalg.norm(T_world_estimated_tag[:3, :3] - T_world_tag[:3, :3]))  # orientation error
	return np.hstack(error)

def reprojectionError(det_corners: np.ndarray, proj_corners: np.ndarray) -> float:
	error = np.linalg.norm(det_corners - proj_corners, axis=1)
	return np.mean(error)

def projectMarkers(img: cv2.typing.MatLike, square_points: np.ndarray, marker_poses:dict, cam_trans: np.ndarray, cam_rot: np.ndarray, cmx: np.ndarray, dist: np.ndarray) -> float:
	err = 0
	for id, det in marker_poses.items():
		# tf marker corners to camera frame
		T_cam_tag = pose2Matrix(det['ftrans'], det['frot'])
		world_corners = tagWorldCorners(T_cam_tag, square_points)
		# project corners to image plane
		projected_corners, _ = cv2.projectPoints(world_corners, euler2Matrix(cam_rot), cam_trans, cmx, dist)
		projected_corners = np.int32(projected_corners).reshape(-1, 2)
		cv2.polylines(img, [projected_corners], isClosed=True, color=det.GREEN, thickness=2)
		# reprojection error
		err += reprojectionError(marker_poses[id]['corners'], projected_corners)
	return err

def estimatePose(img, err, est_camera_pose, marker_poses):
	res = least_squares(residuals, est_camera_pose, args=(marker_table_poses, marker_poses))
	if res.success:
		opt_cam_pose = res.x
		status = res.status 
		# reproject markers
		reserr = projectMarkers(img, det.square_points, marker_poses, opt_cam_pose[:3], opt_cam_pose[3:], det.cmx, det.dist)
		print(f"Result: {status}\ncamera world pose trans: {opt_cam_pose[:3]}, rot (extr. xyz euler): {opt_cam_pose[3:]}\nreprojection error: {reserr}\n")
		return reserr, opt_cam_pose
	print(f"Least squares failed: {res.status}")
	return err, est_camera_pose

m_rvec = det[0]['rvec']
m_tvec = det[0]['tvec']
m_mat = cv2.Rodrigues(m_rvec)[0]
m_mat = det[0]['rot_mat']
T_cam_marker = pose2Matrix(m_tvec, rot_mat=m_mat)

c_mat = np.array([ [0.0024861,  0.0000000,  0.9999969],
                                    [ 0.0000000,  1.0000000,  0.0000000],
                                    [ -0.9999969,  0.0000000,  0.0024861] ])
c_tvec = np.array([-0.35252, 0.00089723, 0.27555])
T_world_cam =  pose2Matrix(c_tvec, rot_mat=c_mat)

T_world_marker = T_world_cam @ T_cam_marker
world_marker_tvec = T_world_marker[:3, 3]
world_marker_rvec = R.from_matrix(T_world_marker[:3, :3] ).as_euler('xyz')
print(world_marker_tvec, world_marker_rvec)

r_tvec = marker_table_poses['root']['xyz']
r_euler = marker_table_poses['root']['rpy']
T_world_root = pose2Matrix(r_tvec, r_euler)

w_tvec = marker_table_poses[0]['xyz']
w_euler = marker_table_poses[0]['rpy']
T_root_marker = pose2Matrix(w_tvec, w_euler)

T_world_marker = T_world_root @ T_root_marker
world_marker_tvec = T_world_marker[:3, 3]
world_marker_rvec = R.from_matrix(T_world_marker[:3, :3] ).as_euler('xyz')
print(world_marker_tvec, world_marker_rvec)

# img = cv2.imread('/home/nic/catkin_ws/src/nicol_rh8d/datasets/aruco/det_image.jpg', cv2.IMREAD_COLOR)
# cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
# cv2.imshow('Detection', img)
# while 1:
#     if cv2.waitKey(1) == ord("q"):
#         break
# cv2.destroyAllWindows()