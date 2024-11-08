import os
import cv2
import yaml
import subprocess
import numpy as np
# import pandas as pd	
from enum import Enum
from typing import Tuple
from cv2 import Rodrigues
from datetime import datetime
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R

dt_now = datetime.now()
dt_now = dt_now.strftime("%H_%M_%S")
# data records
DATA_PTH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets/detection')
QDEC_DET_PTH = os.path.join(DATA_PTH, 'qdec/detection_' + dt_now + '.json')
KEYPT_DET_PTH = os.path.join(DATA_PTH, 'keypoint/detection_' + dt_now + '.json')
KEYPT_3D_PTH = os.path.join(DATA_PTH, 'keypoint/kpts3D_' + dt_now + '.json')
KEYPT_FK_PTH = os.path.join(DATA_PTH, 'keypoint/kptsFK_' + dt_now + '.json')
# image records
JPG_QUALITY = 60
REC_DIR = os.path.join(os.path.expanduser('~'), 'rh8d_dataset')
# quadrature encoder
QDEC_REC_DIR = os.path.join(REC_DIR, 'qdec')
QDEC_ORIG_REC_DIR = os.path.join(QDEC_REC_DIR, 'orig')
QDEC_DET_REC_DIR = os.path.join(QDEC_REC_DIR, 'det')
# keypoints
KEYPT_REC_DIR = os.path.join(REC_DIR, 'keypoint_'+ dt_now)
KEYPT_ORIG_REC_DIR = os.path.join(KEYPT_REC_DIR, 'orig')
KEYPT_DET_REC_DIR = os.path.join(KEYPT_REC_DIR, 'det')
KEYPT_R_EYE_REC_DIR = os.path.join(KEYPT_REC_DIR, 'right_eye')
KEYPT_L_EYE_REC_DIR = os.path.join(KEYPT_REC_DIR, 'left_eye')
KEYPT_TOP_CAM_REC_DIR = os.path.join(KEYPT_REC_DIR, 'top_cam')
KEYPT_HEAD_CAM_REC_DIR = os.path.join(KEYPT_REC_DIR, 'head_cam')

# training
TRAIN_PTH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'train')
MLP_LOG_PTH = os.path.join(TRAIN_PTH, 'mlp/log')
	
def mkDirs() -> None:
    if not os.path.exists(REC_DIR):
        os.mkdir(REC_DIR)
    print("Writing images to", REC_DIR)
    if not os.path.exists(QDEC_REC_DIR):
        os.mkdir(QDEC_REC_DIR)
    if not os.path.exists(QDEC_ORIG_REC_DIR):
        os.mkdir(QDEC_ORIG_REC_DIR)
    if not os.path.exists(QDEC_DET_REC_DIR):
        os.mkdir(QDEC_DET_REC_DIR)
    if not os.path.exists(KEYPT_REC_DIR):
        os.mkdir(KEYPT_REC_DIR)
    if not os.path.exists(KEYPT_ORIG_REC_DIR):
        os.mkdir(KEYPT_ORIG_REC_DIR)
    if not os.path.exists(KEYPT_DET_REC_DIR):
        os.mkdir(KEYPT_DET_REC_DIR)
    if not os.path.exists(KEYPT_R_EYE_REC_DIR):
        os.mkdir(KEYPT_R_EYE_REC_DIR)
    if not os.path.exists(KEYPT_L_EYE_REC_DIR):
        os.mkdir(KEYPT_L_EYE_REC_DIR)
    if not os.path.exists(KEYPT_TOP_CAM_REC_DIR):
        os.mkdir(KEYPT_TOP_CAM_REC_DIR)
    if not os.path.exists(KEYPT_HEAD_CAM_REC_DIR):
        os.mkdir(KEYPT_HEAD_CAM_REC_DIR)
	
class NormalTypes(Enum):
	XY='xy'
	XZ='xz'
	YZ='yz'
NORMAL_TYPES_MAP={  NormalTypes.XY.value : NormalTypes.XY, 
				                                    NormalTypes.XZ.value : NormalTypes.XZ, 
													NormalTypes.YZ.value : NormalTypes.YZ 
												}

class RotTypes(Enum):
	RVEC='rvec'
	EULER='xyz_euler'
	MAT='matrix'
	QUAT='quaternions'

def getRotation(rot: np.ndarray, rot_type: RotTypes, out_type: RotTypes) -> np.ndarray:
	if rot_type == out_type:
		return rot
	
	# convert to mat
	mat = rot
	if rot_type == RotTypes.RVEC:
		(mat, _) = Rodrigues(rot)
	elif rot_type == RotTypes.EULER:
		# gimble lock?
		if np.abs(np.abs(rot[1]) - np.pi*0.5) < 1e-6:
			print("Incoming gimble lock detected")
		mat = R.from_euler('xyz', rot).as_matrix()
	elif rot_type == RotTypes.QUAT:
		# x y z w
		mat = R.from_quat(rot).as_matrix() 
	elif rot_type == RotTypes.MAT:
		pass
	else:
		raise NotImplementedError 
	
	# convert to output format
	res = mat
	if out_type == RotTypes.RVEC:
		(res, _) = Rodrigues(res)
		res = res.flatten()
	elif out_type == RotTypes.EULER:
		res = R.from_matrix(res).as_euler('xyz')
		res = res.flatten()
		# gimble lock?
		if np.abs(np.abs(res[1]) - np.pi*0.5) < 1e-6:
			print("Outgoing gimble lock detected")
	elif out_type == RotTypes.QUAT:
		res = R.from_matrix(mat).as_quat()
	elif out_type == RotTypes.MAT:
		pass
	else:
		raise NotImplementedError 
	
	return res

def rotDiff(rot1: np.ndarray, rot2: np.ndarray) -> float:
		diff = rot1.T @ rot2  # relative rotation
		angle_diff = np.arccos((np.trace(diff) - 1) / 2)  # angle difference
		return angle_diff

def avgRotMat(rotation_matrices: np.ndarray) -> np.ndarray:
	quats = np.array([getRotation(mat, RotTypes.MAT, RotTypes.QUAT) for mat in rotation_matrices])
	avg_quat = np.mean(quats, axis=0) 	# average
	avg_quat /= np.linalg.norm(avg_quat)    # normalize
	return getRotation(avg_quat, RotTypes.QUAT, RotTypes.MAT)

def pose2Matrix(tvec: np.ndarray, rot: np.ndarray, rot_t: RotTypes) -> np.ndarray:
	transformation_matrix = np.eye(4)
	transformation_matrix[:3, :3] = getRotation(rot, rot_t, RotTypes.MAT)
	transformation_matrix[:3, 3] = tvec
	return transformation_matrix

def invPersp(tvec: np.ndarray, rot: np.ndarray, rot_t: RotTypes) -> Tuple[np.ndarray, np.ndarray]:
	"""Apply the inversion to the given vectors [[R^-1 -R^-1*d][0 0 0 1]]"""
	mat = getRotation(rot, rot_t, RotTypes.MAT)
	mat = np.matrix(mat).T # orth. matrix: A.T = A^-1
	inv_tvec = -mat @ tvec # -R^-1*d
	inv_rot = getRotation(mat, RotTypes.MAT, rot_t)
	if rot_t != RotTypes.MAT:
		inv_rot = inv_rot.flatten()
	return np.array(inv_tvec.flat), inv_rot

def consensusRot(rot_mats: np.ndarray, n_iterations: int=100, threshold: float=0.2) -> Tuple[np.ndarray, list]:
	"""Find the rotation that has a minimal distance to all rotations."""
	best_inliers = []
	best_rotation = None
	for _ in range(n_iterations):
		# random sample two rotations
		sample_indices = np.random.choice(len(rot_mats), 2, replace=False)
		sample_rotations = [rot_mats[i] for i in sample_indices]
		
		# find consensus rotation 
		consens_rot = avgRotMat(sample_rotations)
		# find inliers 
		current_inliers = []
		for idx, mat in enumerate(rot_mats):
			if rotDiff(consens_rot, mat) < threshold:
				current_inliers.append(idx)

		# update best inliers
		if len(current_inliers) > len(best_inliers):
			best_inliers = current_inliers
			best_rotation = consens_rot

	return best_rotation, best_inliers

def refRotZ(rot_mat: np.ndarray, ref_rotations: np.ndarray, threshold: float=1e-6) -> np.ndarray:
	"""Rotate the z axis of rot_mat by the angular difference to ref_rotations average."""
	# compute average z-axis from the reference rotations
	avg_z_axis = np.mean([r[:3, 2] for r in ref_rotations], axis=0)
	avg_z_axis /= np.linalg.norm(avg_z_axis)
	# get z-axis of target rotation
	current_z_axis = rot_mat[:3, 2]
	current_z_axis /= np.linalg.norm(current_z_axis)  

	# compute the axis- angle to rotate the current z-axis to the average z-axis
	rotation_axis = np.cross(current_z_axis, avg_z_axis) # axis of rotation
	if np.linalg.norm(rotation_axis) < threshold:  # already aligned
		return rot_mat
	rotation_axis /= np.linalg.norm(rotation_axis)  
	# compute angle of rotation
	rotation_angle = np.arccos(np.dot(current_z_axis, avg_z_axis)) 

	# compute the correction rotation matrix around the rotation_axis
	correction_rotation = R.from_rotvec(rotation_angle * rotation_axis).as_matrix()
	# align the z-axis while keeping x and y as intact as possible
	return rot_mat @ correction_rotation

import numpy as np

def findAxisOrientOutliers(rot_mats: np.ndarray, tolerance: float=1e-6, axis_idx: int=ord('x')) -> Tuple[list, np.ndarray]:
	"""Average the axes of all matrices and find the index
		for matrices that do not match the avg orientation.
	"""	
	# compute average axes
	axs_avg = np.mean( [r[:3, axis_idx] for r in rot_mats], axis=0 )
	axs_avg /= np.linalg.norm(axs_avg)

	# compare each axis to the average
	outliers = []
	for idx, mat in enumerate(rot_mats):
		# normalize
		axs = mat[:, axis_idx] / np.linalg.norm(mat[:, axis_idx])
		# check degree of alignment
		if abs( np.dot(axs, axs_avg) ) < tolerance:
			outliers.append(idx)
	
	return outliers, axs_avg

def ransacPose(tvec: np.ndarray, rvec: np.ndarray, corners: np.ndarray, obj_points: np.ndarray, cmx: np.ndarray, dist: np.ndarray, solver_flag: int=cv2.SOLVEPNP_IPPE_SQUARE) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
	"""RANSAC over given pose can improve the accuracy."""
	(success, out_rvec, out_tvec, inliers) = cv2.solvePnPRansac(objectPoints=obj_points,
																														imagePoints=np.array(corners, dtype=np.float32),
																														cameraMatrix=cmx,
																														distCoeffs=dist,
																														rvec=rvec,
																														tvec=tvec,
																														useExtrinsicGuess=False,
																														flags=solver_flag,
																														)
	if success:
		out_tvec = out_tvec.reshape(3)
		out_rvec = out_rvec.reshape(3)
		inliers = inliers.flatten()
		return success, out_tvec, out_rvec,  inliers
	
	return success, tvec, rvec,  None

def refinePose(tvec: np.ndarray, rvec: np.ndarray, corners: np.ndarray, obj_points: np.ndarray, cmx: np.ndarray, dist: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	""" Non-linear Levenberg-Marquardt minimization scheme and the current implementation 
			computes the rotation update as a perturbation and not on SO(3).
	"""
	# TermCriteria(TermCriteria::EPS+TermCriteria::COUNT, 20, FLT_EPSILON) 
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) 
	(out_rvec, out_tvec) = cv2.solvePnPRefineLM(objectPoints=obj_points,
										   													imagePoints=np.array(corners, dtype=np.float32),
																							cameraMatrix=cmx,
																							distCoeffs=dist,
																							rvec=rvec,
																							tvec=tvec,
																							criteria=criteria,
																							)
	return out_tvec.flatten(), out_rvec.flatten()

def dfsKinematicChain(joint_name: str, kinematic_chains: list, marker_config:dict, branched: bool=False) -> None:
	"""Find all joint chains in config."""
	kinematic_chains[-1].append(joint_name)
	joint_children = marker_config[joint_name]['joint_children']
	if len(joint_children) > 1:
		assert(not branched)
		branched = True
	for child in joint_children:
		if len(joint_children) > 1:
			kinematic_chains.append(kinematic_chains[0].copy())
		dfsKinematicChain(child, kinematic_chains, marker_config, branched)

def loadMarkerConfig() -> dict:
	# load marker configuration
	fl = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cfg/marker_config.yaml")
	with open(fl, 'r') as fr:
		config = yaml.safe_load(fr)
		marker_config = config['marker_config']
		return marker_config
	
def loadNetConfig(net: str) -> dict:
	# load net configuration
	fl = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cfg/net_config.yaml")
	with open(fl, 'r') as fr:
		config = yaml.safe_load(fr)
		net_config = config[net]
		return net_config

def beep(do_beep: bool=True) -> None:
	if do_beep:
	    subprocess.run(['paplay', '/usr/share/sounds/gnome/default/alerts/sonar.ogg'])

def greenScreen(img: cv2.typing.MatLike):
	repl = np.ones(img.shape, dtype=np.float32) * 255
	u_green = np.array([104, 153, 70])
	l_green = np.array([30, 30, 0])
	
	mask = cv2.inRange(img, l_green, u_green)
	res = cv2.bitwise_and(img, img, mask = mask)
	f = img - res
	f = np.where(f == 0, repl, f)
	return f

def visTiff(pth: str) -> None:
   img = cv2.imread(pth, cv2.IMREAD_UNCHANGED)
   # only for matplotlib to display
   img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   plt.imshow(img_rgb)
   plt.axis('off')
   plt.show()

if __name__ == "__main__":
	# cv2.namedWindow("gs", cv2.WINDOW_NORMAL)
	# img = cv2.imread(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets/detection/test_img.jpg'), cv2.IMREAD_COLOR)
	# img = greenScreen(img)
	# cv2.imshow("gs", img)
	# while 1:
	# 	if cv2.waitKey(1) == ord('q'):
	# 		break
	# cv2.destroyAllWindows()

	visTiff(os.path.join(REC_DIR, "keypoint_17_03_57/top_cam/20.tiff"))
