import os
import re
import cv2
import json
import glob
import yaml
import subprocess
import numpy as np
import tifffile as tiff
import pandas as pd
from enum import Enum
from cv2 import Rodrigues
from datetime import datetime
from send2trash import send2trash
from typing import Tuple, Union, Any
from matplotlib import pyplot as plt
from send2trash.plat_other import HOMETRASH_B
from scipy.spatial.transform import Rotation as R

dt_now = datetime.now()
dt_now = "validation" # dt_now.strftime("%m_%d_%H_%M")
# data records
DATA_PTH = os.path.join(os.path.dirname(os.path.dirname(
	os.path.abspath(__file__))), 'datasets/detection')
QDEC_DET_PTH = os.path.join(DATA_PTH, 'qdec/detection_' + dt_now + '.json')
KEYPT_DET_PTH = os.path.join(
	DATA_PTH, 'keypoint/detection_' + dt_now + '.json')
KEYPT_3D_PTH = os.path.join(DATA_PTH, 'keypoint/kpts3D_' + dt_now + '.json')
KEYPT_FK_PTH = os.path.join(DATA_PTH, 'keypoint/kptsFK_' + dt_now + '.json')
# image records
JPG_QUALITY = 60
REC_DIR = os.path.join(os.path.expanduser('~'), 'rh8d_dataset')
# REC_DIR = '/data/rh8d_dataset'
# quadrature encoder
QDEC_REC_DIR = os.path.join(REC_DIR, 'qdec')
QDEC_ORIG_REC_DIR = os.path.join(QDEC_REC_DIR, 'orig')
QDEC_DET_REC_DIR = os.path.join(QDEC_REC_DIR, 'det')
# keypoints
KEYPT_REC_DIR = os.path.join(REC_DIR, 'keypoint_' + dt_now)
KEYPT_ORIG_REC_DIR = os.path.join(KEYPT_REC_DIR, 'orig')
KEYPT_DET_REC_DIR = os.path.join(KEYPT_REC_DIR, 'det')
KEYPT_R_EYE_REC_DIR = os.path.join(KEYPT_REC_DIR, 'right_eye')
KEYPT_L_EYE_REC_DIR = os.path.join(KEYPT_REC_DIR, 'left_eye')
KEYPT_TOP_CAM_REC_DIR = os.path.join(KEYPT_REC_DIR, 'top_cam')
KEYPT_HEAD_CAM_REC_DIR = os.path.join(KEYPT_REC_DIR, 'head_cam')

# training
TRAIN_PTH = os.path.join(os.path.dirname(os.path.dirname(
	os.path.abspath(__file__))), 'datasets/detection/keypoint/train')
MLP_LOG_PTH = os.path.join(TRAIN_PTH, 'mlp/log')
MLP_SCLRS_PTH = os.path.join(TRAIN_PTH, 'mlp/scalers')
MLP_CHKPT_PTH = os.path.join(TRAIN_PTH, 'mlp/checkpoints')

# indices for translation data
TRANS_COLS = ["x", "y", "z"]
# indices for froce data
FORCE_COLS = ["fx", "fy", "fz"]
# indices for orientation data
QUAT_COLS = ["x", "y", "z", "w"]
# generate names for a training dataset
# ['cmdA1', ..., 'cmdZ1', 'dirA1', ..., 'dirZ1', 'quat', 'angleA1', ..., 'angleAn', ..., 'angleZn', 'transA', ..., 'transZ']
GEN_TRAIN_COLS = lambda feature_names, target_names, trans_names: [f"cmd{name.replace('joint', '')}" for name in feature_names] + \
																																				[f"dir{name.replace('joint', '')}" for name in feature_names] + \
																																				["quat"] + \
																																				[f"angle{name.replace('joint', '')}" for name in target_names] + \
																																				[f"trans{name.replace('joint_', '').replace('bumper', '')}" for name in trans_names]
# dynamic index generation for translation data
format_trans_cols = lambda idx: [f"x_{idx}", f"y_{idx}", f"z_{idx}"]

# generate names for a training dataset
# ['cmdA1', ..., 'cmdZ1', 'dirA1', ..., 'dirZ1', 'quat', 'force', 'angleA1', ..., 'angleAn', ..., 'angleZn', 'transA', ..., 'transZ']
GEN_TRAIN_COLS_FORCE = lambda feature_names, target_names, trans_names: [f"cmd{name.replace('joint', '')}" for name in feature_names] + \
                                                                                                                                                                [f"dir{name.replace('joint', '')}" for name in feature_names] + \
                                                                                                                                                                ["quat"] + \
                                                                                                                                                                ["force"] + \
                                                                                                                                                                [f"angle{name.replace('joint', '')}" for name in target_names] + \
                                                                                                                                                                [f"trans{name.replace('joint_', '').replace('bumper', '')}" for name in trans_names]


def mkDirs() -> None:
	"""Create paths on fs"""

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
	XY = 'xy'
	XZ = 'xz'
	YZ = 'yz'


NORMAL_TYPES_MAP = {NormalTypes.XY.value: NormalTypes.XY,
					NormalTypes.XZ.value: NormalTypes.XZ,
					NormalTypes.YZ.value: NormalTypes.YZ,
					}
NORMAL_IDX_MAP = {NormalTypes.XY: 2,
				  NormalTypes.XZ: 1,
				  NormalTypes.YZ: 0,
				  }


class RotTypes(Enum):
	RVEC = 'rvec'
	EULER = 'xyz_euler'
	MAT = 'matrix'
	QUAT = 'quaternions'


class Normalization(Enum):
	NONE = 'none'
	Z_SCORE = 'z_score'
	MINMAX_POS = 'minmax_pos'
	MINMAX_CENTERED = 'minmax_centered'


NORMALIZATION_MAP = {Normalization.NONE.value: Normalization.NONE,
					 Normalization.Z_SCORE.value: Normalization.Z_SCORE,
					 Normalization.MINMAX_POS.value: Normalization.MINMAX_POS,
					 Normalization.MINMAX_CENTERED.value: Normalization.MINMAX_CENTERED,
					 }
NORMS = f"{Normalization.NONE.value}, {Normalization.MINMAX_CENTERED.value}, {Normalization.MINMAX_POS.value}, {Normalization.Z_SCORE.value}"


def clean(args: Any) -> None:
	if args.clean_all:
		if not args.y:
			if input("Cleaning all training data? Type y to proceed cleaning") != 'y':
				return
		args.clean_log = True
		args.clean_scaler = True
		args.clean_checkpoint = True

	# cleanup
	if args.clean_log:
		if os.path.exists(MLP_LOG_PTH):
			print("Moving directory", MLP_LOG_PTH, "to", HOMETRASH_B.decode())
			send2trash(MLP_LOG_PTH)
	if args.clean_scaler:
		if os.path.exists(MLP_SCLRS_PTH):
			print("Moving directory", MLP_SCLRS_PTH,
				  "to", HOMETRASH_B.decode())
			send2trash(MLP_SCLRS_PTH)
	if args.clean_checkpoint:
		if os.path.exists(MLP_CHKPT_PTH):
			print("Moving directory", MLP_CHKPT_PTH,
				  "to", HOMETRASH_B.decode())
			send2trash(MLP_CHKPT_PTH)

	if args.clean_log or args.clean_scaler or args.clean_checkpoint:
		if args.y:
			print("Cleaned..")
		else:
			print("Cleaned.. exiting")
			exit(0)


def parseIntTuple(value: str) -> Union[None, Tuple]:
	if not ',' in value:
		return None
	t = tuple(map(int, value.split(',')))
	return t


def parseFloatTuple(value: str) -> Union[None, Tuple]:
	if not ',' in value:
		return None
	t = tuple(map(float, value.split(',')))
	return t


def parseNorm(value: str) -> Normalization:
	if not value in NORMS:
		raise ValueError
	return NORMALIZATION_MAP[value]

def parseDict(value: str) -> dict:
	assert('{' in value and '}' in value)
	return json.loads(value)

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
	quats = np.array([getRotation(mat, RotTypes.MAT, RotTypes.QUAT)
					  for mat in rotation_matrices])
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
	mat = np.matrix(mat).T  # orth. matrix: A.T = A^-1
	inv_tvec = -mat @ tvec  # -R^-1*d
	inv_rot = getRotation(mat, RotTypes.MAT, rot_t)
	if rot_t != RotTypes.MAT:
		inv_rot = inv_rot.flatten()
	return np.array(inv_tvec.flat), inv_rot


def consensusRot(rot_mats: np.ndarray, n_iterations: int = 100, threshold: float = 0.2) -> Tuple[np.ndarray, list]:
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


def refRotZ(rot_mat: np.ndarray, ref_rotations: np.ndarray, threshold: float = 1e-6) -> np.ndarray:
	"""Rotate the z axis of rot_mat by the angular difference to ref_rotations average."""
	# compute average z-axis from the reference rotations
	avg_z_axis = np.mean([r[:3, 2] for r in ref_rotations], axis=0)
	avg_z_axis /= np.linalg.norm(avg_z_axis)
	# get z-axis of target rotation
	current_z_axis = rot_mat[:3, 2]
	current_z_axis /= np.linalg.norm(current_z_axis)

	# compute the axis- angle to rotate the current z-axis to the average z-axis
	rotation_axis = np.cross(current_z_axis, avg_z_axis)  # axis of rotation
	if np.linalg.norm(rotation_axis) < threshold:  # already aligned
		return rot_mat
	rotation_axis /= np.linalg.norm(rotation_axis)
	# compute angle of rotation
	rotation_angle = np.arccos(np.dot(current_z_axis, avg_z_axis))

	# compute the correction rotation matrix around the rotation_axis
	correction_rotation = R.from_rotvec(
		rotation_angle * rotation_axis).as_matrix()
	# align the z-axis while keeping x and y as intact as possible
	return rot_mat @ correction_rotation


def findAxisOrientOutliers(rot_mats: np.ndarray, tolerance: float = 1e-6, axis_idx: int = ord('x')) -> Tuple[list, np.ndarray]:
	"""Average the axes of all matrices and find the index
									for matrices that do not match the avg orientation.
	"""
	# compute average axes
	axs_avg = np.mean([r[:3, axis_idx] for r in rot_mats], axis=0)
	axs_avg /= np.linalg.norm(axs_avg)

	# compare each axis to the average
	outliers = []
	for idx, mat in enumerate(rot_mats):
		# normalize
		axs = mat[:, axis_idx] / np.linalg.norm(mat[:, axis_idx])
		# check degree of alignment
		if abs(np.dot(axs, axs_avg)) < tolerance:
			outliers.append(idx)

	return outliers, axs_avg


def tfForce(quats: np.ndarray) -> np.ndarray:
	return getRotation(quats, RotTypes.QUAT, RotTypes.MAT).T @ np.array([0,0,-9.81])


def ransacPose(tvec: np.ndarray, rvec: np.ndarray, corners: np.ndarray, obj_points: np.ndarray, cmx: np.ndarray, dist: np.ndarray, solver_flag: int = cv2.SOLVEPNP_IPPE_SQUARE) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
	"""RANSAC over given pose can improve the accuracy."""
	(success, out_rvec, out_tvec, inliers) = cv2.solvePnPRansac(objectPoints=obj_points,
																imagePoints=np.array(
																	corners, dtype=np.float32),
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
												imagePoints=np.array(
													corners, dtype=np.float32),
												cameraMatrix=cmx,
												distCoeffs=dist,
												rvec=rvec,
												tvec=tvec,
												criteria=criteria,
												)
	return out_tvec.flatten(), out_rvec.flatten()


def dfsKinematicChain(joint_name: str, kinematic_chains: list, marker_config: dict, branched: bool = False) -> None:
	"""Find all joint chains in config."""
	kinematic_chains[-1].append(joint_name)
	joint_children = marker_config[joint_name]['joint_children']
	if len(joint_children) > 1:
		assert (not branched)
		branched = True
	for child in joint_children:
		if len(joint_children) > 1:
			kinematic_chains.append(kinematic_chains[0].copy())
		dfsKinematicChain(child, kinematic_chains, marker_config, branched)


def loadMarkerConfig() -> dict:
	# load marker configuration
	fl = os.path.join(os.path.dirname(os.path.dirname(
		os.path.abspath(__file__))), "cfg/marker_config.yaml")
	with open(fl, 'r') as fr:
		config = yaml.safe_load(fr)
		marker_config = config['marker_config']
		return marker_config


def loadNetConfig(net: str) -> dict:
	# load net configuration
	fl = os.path.join(os.path.dirname(os.path.dirname(
		os.path.abspath(__file__))), "cfg/net_config.yaml")
	with open(fl, 'r') as fr:
		config = yaml.safe_load(fr)
		net_config = config[net]
		return net_config


def beep(do_beep: bool = True) -> None:
	if do_beep:
		subprocess.run(
			['paplay', '/usr/share/sounds/gnome/default/alerts/sonar.ogg'])


def greenScreen(img: cv2.typing.MatLike):
	repl = np.ones(img.shape, dtype=np.float32) * 255
	u_green = np.array([104, 153, 70])
	l_green = np.array([30, 30, 0])

	mask = cv2.inRange(img, l_green, u_green)
	res = cv2.bitwise_and(img, img, mask=mask)
	f = img - res
	f = np.where(f == 0, repl, f)
	return f


def mosaicImg(num: int, save_pth: str, target_size: Tuple = (1920, 1080)) -> None:
	jpg = str(num)+'.jpg'
	npz = str(num)+'.npz'
	pth = os.path.join(REC_DIR, 'joined')

	imgs = [os.path.join(pth, 'orig', jpg),
			os.path.join(pth, 'head_cam', jpg),
			os.path.join(pth, 'top_cam', jpg),
			os.path.join(pth, 'left_eye', jpg),
			os.path.join(pth, 'right_eye', jpg)
			]
	npzs = [os.path.join(pth, 'orig', npz),
			os.path.join(pth, 'head_cam', npz),
			os.path.join(pth, 'top_cam', npz),
			]

	images = [cv2.imread(path) for path in imgs]
	depth_images = [cv2.cvtColor(
					np.load(path)['array'], cv2.COLOR_GRAY2BGR) for path in npzs]

	h, w = images[0].shape[:2]
	r = target_size[0] / float(w)
	dim = (target_size[0], int(h * r))
	resized_images = [cv2.resize(
		img, dim, interpolation=cv2.INTER_AREA) for img in images]

	h, w = depth_images[0].shape[:2]
	r = target_size[0] / float(w)
	dim = (target_size[0], int(h * r))
	resized_depth_images = [cv2.resize(
		img, dim, interpolation=cv2.INTER_AREA) for img in depth_images]

	tmp = []
	for idx in range(len(resized_images)):
		tmp.append(resized_images[idx])
		if idx < len(resized_depth_images):
			tmp.append(resized_depth_images[idx])
	resized_images = tmp

	resized_images = [img.astype(np.uint8) for img in resized_images]
	while len(resized_images) < 8:  # pad black
		resized_images.append(np.zeros_like(resized_images[0]))

	rows = [
		np.hstack(resized_images[:2]),  # first row: 2 images
		np.hstack(resized_images[2:4]),   # second row: 2 images
		np.hstack(resized_images[4:6]),   # third row: 2 images
		np.hstack(resized_images[6:]),   # fourth row: 2 images
	]
	mosaic = np.vstack(rows)  # combine vertically

	cv2.namedWindow("Mosaic", cv2.WINDOW_NORMAL)
	cv2.imwrite(save_pth, mosaic)
	cv2.imshow("Mosaic", mosaic)
	if cv2.waitKey(0) == 'q':
		cv2.destroyAllWindows()


def visTiff(pth: str) -> None:
	img = cv2.imread(pth, cv2.IMREAD_UNCHANGED)
	# only for matplotlib to display
	img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	plt.imshow(img_rgb)
	plt.axis('off')
	plt.show()


def compressTiff(pth: str) -> None:
	# read tiff
	image_array = tiff.imread(pth)
	split = pth.split('/')
	filename = split[-1].replace(".tiff", "")
	save_pth = split[:-1]
	save_pth = "/".join(save_pth)
	# lossless compression and save np array
	np.savez_compressed(os.path.join(
		save_pth, filename + '.npz'), array=image_array)


def visCompressedPC(pth: str) -> None:
	data = np.load(pth)
	image_array = data['array']
	cv2.imshow('PC vis', image_array)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def compressTiffFromFolder(folder_pth: str) -> None:
	pattern = os.path.join(folder_pth, f'*.tiff*')
	data_files = glob.glob(pattern, recursive=False)
	cnt_all = len(data_files)
	cnt = 0
	for file in data_files:
		if cnt % 100 == 0:
			print("Compressing", cnt, "of", cnt_all, " ", file)
		compressTiff(file)
		cnt += 1
	print("Compressed", cnt_all, "tiff images, clean with: rm *.tiff")


def mvImgs(folder_pth: str, file_type: str, increment: int):
	# find all files with type file_type
	files = [f for f in os.listdir(folder_pth) if f.endswith(file_type)]
	# revert order, assuming file names are integers
	files.sort(key=lambda x: int(os.path.splitext(x)[0]), reverse=True)

	for file in files:
		# extract the numeric part
		current_num = int(os.path.splitext(file)[0])
		# increment filename
		new_filename = f"{current_num+increment}{file_type}"

		# rename
		os.rename(os.path.join(folder_pth, file),
				  os.path.join(folder_pth, new_filename))
		print(f"Renamed {file} to {new_filename}")


def extract_number(filename: str) -> int:
	match = re.search(r'(\d+)', filename)
	return int(match.group(1)) if match else float('inf')


def img2Video(img_path: str, output_name: str, img_format: str = '.jpg', video_format: str = 'mp4v', fps: float = 30):

	images = [img for img in os.listdir(img_path) if img.endswith(
		img_format) and not 'plot3D_' in img]
	images.sort(key=extract_number)
	print("Converting", len(images), "images to video", output_name,
		  "with format", video_format, "from", img_path)

	first_image = cv2.imread(os.path.join(img_path, images[0]))
	height, width, layers = first_image.shape
	fourcc = cv2.VideoWriter_fourcc(*video_format)
	video = cv2.VideoWriter(output_name, fourcc, fps, (width, height))

	cnt = 0
	for image in images:
		pth = os.path.join(img_path, image)
		frame = cv2.imread(pth)
		video.write(frame)
		cnt += 1
		if cnt % 100 == 0:
			print("Converting img", pth)

	video.release()
	cv2.destroyAllWindows()


def readDetectionDataset(filepth: str) -> dict:
	"""Read data from json and return a dictionary with
									 joints as keys and pd.DataFrame as values.
	"""
	df = pd.read_json(filepth, orient='index')
	df_dict = {joint:
			   # we want integer index
			   pd.DataFrame.from_dict(
				   data, orient='index')
			   .reset_index(drop=False).rename(columns={'index': 'old_index'}).astype({'old_index': int}).set_index('old_index')
			   for joint, data in df.iloc[0].items()}
	return df_dict


def joinDetectionDataset(pth1: str, pth2: str, save_pth: str) -> None:
	"""Read two dataframes of dictionaries and join them with a 
									 common index  = [0, n1-1, n1, n2-1]
	"""
	df1 = readDetectionDataset(pth1)
	df2 = readDetectionDataset(pth2)
	df_dict_concat = {}

	for joint, df_dict in df1.items():
		n1 = df_dict.index.max() + 1
		df2[joint].index = df2[joint].index + n1
		df_dict_concat.update({joint:  pd.concat([df_dict, df2[joint]])})

	det_df = pd.DataFrame({joint: [df]
						   for joint, df in df_dict_concat.items()})
	det_df.to_json(save_pth, orient="index", indent=4)


def joinDF(pth1: str, pth2: str, save_pth: str) -> None:
	"""Read two dataframes and joint them with a
									common index  = [0, n1-1, n1, n2-1]
	"""
	df1 = pd.read_json(pth1, orient='index')
	df2 = pd.read_json(pth2, orient='index')

	n1 = df1.index.max() + 1
	df2.index = df2.index + n1
	df_concat = pd.concat([df1, df2])

	df_concat.to_json(save_pth, orient="index", indent=4)

if __name__ == "__main__":
	# img2Video(os.path.join(REC_DIR, "joined/det"), os.path.join(REC_DIR, "movies/detection.mp4"), fps=25)
	mosaicImg(3526, os.path.join(REC_DIR, 'joined/mosaic.jpg'))
