import os
import re
import cv2
import glob
import yaml
import subprocess
import numpy as np
import tifffile as tiff
import pandas as pd
import pybullet as pb
from enum import Enum
from cv2 import Rodrigues
from datetime import datetime
from send2trash import send2trash
from typing import Tuple, Union, Any
from matplotlib import pyplot as plt
from send2trash.plat_other import HOMETRASH_B
from scipy.spatial.transform import Rotation as R

dt_now = datetime.now()
dt_now = dt_now.strftime("%m_%d_%H_%M")
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
	# if len(t) != 3:
	#     raise ValueError
	return t


def parseFloatTuple(value: str) -> Union[None, Tuple]:
	if not ',' in value:
		return None
	t = tuple(map(float, value.split(',')))
	# if len(t) != 3:
	#     raise ValueError
	return t


def parseNorm(value: str) -> Normalization:
	if not value in NORMS:
		raise ValueError
	return NORMALIZATION_MAP[value]


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

def findCommonIndices(detection_dct: pd.DataFrame, 
												keypoints_dct: pd.DataFrame,
												out_joints: list,
												tools: list,
												tcp: str,
												) -> list:
	
	# all indices are equal, start with full index of 1st joint
	first_joint = out_joints[0]
	first_joint_df = detection_dct[first_joint]
	common_indices: pd.Index = first_joint_df.index

	# find common indices from detections that contains no nans
	for joint, df in detection_dct.items():
		# consider only specified target joints
		if joint in out_joints:
			valid_detection_df = df.dropna(how='all')
			valid_index = valid_detection_df.index
			inv_idxs = df.index.difference(valid_index).tolist()
			# intersect indices
			old_len = len(common_indices)
			common_indices = common_indices.intersection(valid_index)
			print("Detection", joint, "has", len(inv_idxs), "invalid indices, dropped", len(common_indices) - old_len)

	# find common indices between detections and keypoints
	for joint, df in keypoints_dct.items():
		# consider only specified target joints
		if joint in tools or joint == tcp:
			valid_detection_df = df.dropna(subset=["trans", "rot_mat"], how="all")
			valid_index = valid_detection_df.index
			inv_idxs = df.index.difference(valid_index).tolist()
			# intersect indices
			old_len = len(common_indices)
			common_indices = common_indices.intersection(valid_index)
			print("Keypoint", joint, "has", len(inv_idxs), "invalid indices, dropped", len(common_indices) - old_len)

	invalid_indices = first_joint_df.index.difference(common_indices)
	print("Ignoring", len(invalid_indices), "invalid indices, using", len(
		common_indices), "valid indices out of", len(first_joint_df.index))
	
	return common_indices.tolist()

def joinTrainingData(  in_joints: list,
											out_joints: list,
											tools: list,
											train_cols: list,
											common_indices: list,
											detection_dct: pd.DataFrame, 
											keypoints_dct: pd.DataFrame,
											fk_df: pd.DataFrame,
											tcp: str,
											) -> dict:
	
	col_idx = 0
	train_dct = {}
	
	# copy actuator cmds 
	for joint in in_joints:
		cmd_data = detection_dct[joint].loc[common_indices, 'cmd'].tolist()
		train_dct.update( {train_cols[col_idx] : cmd_data} )
		col_idx += 1

	# copy actuator dirs 
	for joint in in_joints:
		cmd_data = detection_dct[joint].loc[common_indices, 'direction'].tolist()
		train_dct.update( {train_cols[col_idx] : cmd_data} )
		col_idx += 1

	# copy tcp orientation
	quat = fk_df.loc[common_indices, 'quat'].tolist()
	train_dct.update( {train_cols[col_idx] : quat} )
	col_idx += 1

	# copy joint angles
	for joint in out_joints:
		joint_data = detection_dct[joint].loc[common_indices, 'angle'].tolist()
		train_dct.update( {train_cols[col_idx] : joint_data} )
		col_idx += 1

	# copy tip positions
	for tool in tools:
		# we want tf tip relative to tcp
		transformed_trans = []
		for idx in common_indices:
			tip_trans = keypoints_dct[tool]['trans'][idx]
			tip_rot = keypoints_dct[tool]['rot_mat'][idx]
			tcp_trans = keypoints_dct[tcp]['trans'][idx]
			tcp_rot = keypoints_dct[tcp]['rot_mat'][idx]
			T_root_tip = pose2Matrix(tip_trans, tip_rot, RotTypes.MAT)
			(tvec, rot_mat) = invPersp(tcp_trans, tcp_rot, RotTypes.MAT)
			T_tcp_root = pose2Matrix(tvec, rot_mat, RotTypes.MAT)
			T_tcp_tip = T_tcp_root @ T_root_tip
			transformed_trans.append(T_tcp_tip[:3, 3])

		train_dct.update( {train_cols[col_idx] : transformed_trans} )
		col_idx += 1

	return train_dct

def genTrainingData(net_config: str, folder: str, post_proc: bool=False) -> None:
	"""  Load joined datasets and create training data
			for in- and output joints given by the given net_config. 
			Find a common set of valid entries out of the detection frames
			and keypoint frames. Save training data in a single dataframe.
	"""

	# static data path
	data_pth: str = os.path.join(DATA_PTH, f"keypoint/{'post_processed' if post_proc else 'joined'}")

	# load recordings
	detection_dct: dict = readDetectionDataset(os.path.join(data_pth, 'detection.json'))  # contains nans
	keypoints_dct: dict = readDetectionDataset(os.path.join(data_pth, 'kpts3D.json'))  # contains nans
	fk_df: pd.DataFrame = pd.read_json(os.path.join(DATA_PTH, 'keypoint/joined/tcp_tf.json'),orient='index')  # contains no nans

	# load dataset config for a net type
	config: dict = loadNetConfig(net_config)

	in_joints: list = config['input']
	out_joints: list = config['output']
	tools: list = config['tools']
	tcp: str = config['relative_to']
	net_type: str = config['type']

	print("Creating dataset for", net_config, ", type:", net_type, "\ninput:", in_joints,
		  "\noutput:", out_joints, "\ntools:", tools, "\nrelative to", tcp, "\n")
	
	# exclude indices where nan values are present
	common_indices: list = findCommonIndices(detection_dct, 
																							keypoints_dct, 
																							out_joints, 
																							tools, 
																							tcp)

	# create training data
	train_cols: list = GEN_TRAIN_COLS(in_joints, 
																		 out_joints, 
																		 tools)
	train_dct:  dict =  joinTrainingData(in_joints, 
																		out_joints, 
																		tools, 
																		train_cols, 
																		common_indices, 
																		detection_dct, 
																		keypoints_dct, 
																		fk_df, 
																		tcp)
	
	# final data
	train_df = pd.DataFrame(columns=train_cols)

	# arrange per index [0,..,n]
	for idx in range(len(common_indices)):
		# new row
		data = {}
		for key, val_list in train_dct.items():
			# feature/target per index
			data.update( {key : val_list[idx]} )
		# concat row
		train_df = pd.concat([train_df, pd.DataFrame([data])], ignore_index=True)
	
	# save
	train_df.to_json(os.path.join(TRAIN_PTH, folder, f'{net_config}_{net_type}.json'), orient="index", indent=4)
	print("\nFinished dataset generation for", net_config, "with data columns:\n", train_cols)

def genAllTrainingData(post_proc: bool) -> None:
	fl = os.path.join(os.path.dirname(os.path.dirname(
		os.path.abspath(__file__))), "cfg/net_config.yaml")
	with open(fl, 'r') as fr:
		config = yaml.safe_load(fr)
		for  c in config.keys():
			genTrainingData(net_config=c, folder='config_processed' if post_proc else 'config', post_proc=post_proc)
			print()

def replaceNanData() -> None:
	"""Postproc data records by replacing nan values in detection and
		  keypoints. If a detection for a joint contains nans where the actuator
		  is not moved, we use the zero position values for replacement. 
	"""

	# load recordings
	detection_dct: dict = readDetectionDataset(os.path.join(DATA_PTH, "keypoint/joined/detection.json")) 

	# load config
	fl = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cfg/post_proc.yaml")
	with open(fl, 'r') as fr:
		config = yaml.safe_load(fr)
	
	# map joint names to command 
	# descriptions  in detection entries
	descr_mapping = config['description_mapping']
	# replacement values
	joint_repl_values = config['joint_repl_values']

	# process data joint-wise
	for joint in detection_dct.keys():
		print("\n\nProcessing", joint)
		df = detection_dct[joint]
		# find indices where all values are nan
		nan_indices = df[df.isna().all(axis=1)].index
		# break
		if len(nan_indices) == 0:
			print(joint, "data has no nans")
			continue
		# split into contiguous sequences
		nan_seq = np.split(nan_indices, np.where(np.diff(nan_indices) != 1)[0] + 1)
		print("Found", len(nan_seq), "sequences. Processing...")

		for seq_idx, seq in enumerate(nan_seq):
			# discard 1st entry
			if seq[0] > 0:
				# access row before 1st nan row
				predecessor_idx = seq[0] -1
				preceeding_row = df.loc[predecessor_idx]

				# manipulate only entries where this joint is idle
				if descr_mapping[joint] != preceeding_row['description']:
					# copy row and replace values
					rplc_row = preceeding_row.copy()
					rplc_row['cmd'] =  joint_repl_values[joint]['zero']['cmd']
					rplc_row['direction'] = -1.0 if preceeding_row['direction'] <= 0.0 else 1.0
					angle_dct = joint_repl_values[joint]['zero']['angle']
					# check for other cmd than zero
					if rplc_row['cmd'] != preceeding_row['cmd']:
						# only some joints possible
						if joint_repl_values[joint].get('moved') is not None:
							rplc_row['cmd'] =  joint_repl_values[joint]['moved']['cmd']
							angle_dct = joint_repl_values[joint]['moved']['angle']
							# must match
							if rplc_row['cmd'] != preceeding_row['cmd']:
								raise RuntimeError("Alt replacement cmd does not match predecessor cmd for {} at idx {}".format(joint, predecessor_idx))
						else:
							raise RuntimeError("Alt replacement cmd not found in config for {} at idx {}".format(joint, predecessor_idx))
						
					for nan_idx in seq:
						# replace nan row
						df.loc[nan_idx] = rplc_row
						# randomize angle slightly
						df.loc[nan_idx, 'angle'] = np.random.uniform(angle_dct['min'], angle_dct['max'])
					
				else:
					print(joint, "nan sequence:\n", seq_idx, "\nis actuated")
					
			else:
				raise RuntimeWarning("Cannot replace first element in data for {}".format(joint))

	# save processed detections
	det_df = pd.DataFrame({joint: [df] for joint, df in detection_dct.items()})
	det_df.to_json(os.path.join(DATA_PTH, "keypoint/post_processed/detection.json"), orient="index", indent=4)
	print("... done")

def fk(joint_info_dict: dict, link_info_dict: dict, joint_angles: dict, robot_id: int, T_root_joint_world: np.ndarray, valid_entry: bool) -> dict:
	# set detected angles
	if valid_entry:
		for joint, angle in joint_angles.items():
			# revolute joint index
			pb.resetJointState(robot_id, joint_info_dict[joint], angle)

	# get fk
	keypt_dict = {}
	for joint in joint_angles.keys():
		# non-nan entry
		if valid_entry:
			# revolute joint's attached link
			(_, _, _, _, trans, quat) = pb.getLinkState(robot_id, joint_info_dict[joint], computeForwardKinematics=True)
			# compute pose relative to root joint
			T_world_keypoint = pose2Matrix(trans, quat, RotTypes.QUAT)
			T_root_joint_keypoint = T_root_joint_world @ T_world_keypoint
			keypt_dict.update( {joint: {'timestamp': 0.0, 'trans': T_root_joint_keypoint[:3, 3], 'rot_mat': T_root_joint_keypoint[:3, :3]}} )
			# additional fk for tip frame
			if joint in link_info_dict.keys():
				# end link attached to last fixed joint 
				(_, _, _, _, trans, quat) = pb.getLinkState(robot_id, link_info_dict[joint]['index'], computeForwardKinematics=True)
				# compute pose relative to root joint
				T_world_keypoint = pose2Matrix(trans, quat, RotTypes.QUAT)
				T_root_joint_keypoint = T_root_joint_world @ T_world_keypoint
				keypt_dict.update( {link_info_dict[joint]['fixed_end']: {'timestamp': 0.0, 'trans': T_root_joint_keypoint[:3, 3], 'rot_mat': T_root_joint_keypoint[:3, :3]}} )

		# invalid entry
		else:
			keypt_dict.update( {joint: {'timestamp': np.nan, 'trans': np.nan, 'rot_mat': np.nan}} )
			if joint in link_info_dict.keys():
				keypt_dict.update( {link_info_dict[joint]['fixed_end']: {'timestamp': np.nan, 'trans': np.nan, 'rot_mat': np.nan}} )

	return keypt_dict

def fkInit() -> Tuple[dict, dict, int]:
	# init pybullet
	pb.connect(pb.DIRECT)
	robot_id = pb.loadURDF(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'urdf/rh8d.urdf'), useFixedBase=True)

	# load config
	fl = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cfg/post_proc.yaml")
	with open(fl, 'r') as fr:
		config = yaml.safe_load(fr)
	joint_repl_values = config['joint_repl_values']
	# map tip link names for fk target
	tcp_dct = {vals['tcp']: joint for joint, vals in joint_repl_values.items() if vals.get('tcp') is not None}

	# get joint index aka link index
	link_info_dict = {}
	joint_info_dict = {}
	for idx in range(pb.getNumJoints(robot_id)):
		joint_info = pb.getJointInfo(robot_id, idx)
		if joint_info[2] == pb.JOINT_REVOLUTE:
			joint_name = joint_info[1].decode('utf-8')
			# revolute joint index
			joint_info_dict.update( {joint_name: idx} )
		elif joint_info[2] == pb.JOINT_FIXED:
			joint_name = joint_info[1].decode('utf-8')
			if joint_name in tcp_dct.keys():
				# fixed joint index matches very last link
				link_info_dict.update( {tcp_dct[joint_name]: {'index': idx, 'fixed_end': joint_name}} )

	return joint_info_dict, link_info_dict, robot_id

def findCommonIndicesAllJoints(detection_dct: dict) -> Tuple[list, list]:
	first_joint_df: pd.DataFrame = detection_dct[list(detection_dct.keys())[0]]
	common_indices: pd.Index = first_joint_df.index

	# find common indices w/o nans from all detections
	for joint, df in detection_dct.items():
		valid_detection_df = df.dropna(how='all')
		valid_index = valid_detection_df.index
		inv_idxs = df.index.difference(valid_index).tolist()
		# intersect indices
		old_len = len(common_indices)
		common_indices = common_indices.intersection(valid_index)
		print("Detection", joint, "has", len(inv_idxs), "invalid indices, dropped", old_len - len(common_indices))

	invalid_indices = first_joint_df.index.difference(common_indices)
	print("Ignoring", len(invalid_indices), "invalid indices, using", len(
		common_indices), "valid indices out of", len(first_joint_df.index))
	
	# cast
	return common_indices.tolist(), first_joint_df.index.tolist()

def fkFromDetection() -> None:
	"""Compute keypoints from post-processed
		 detections. Comput. criteria: all joints have valid 
		 detection entries/ no nans for complete fk. Invalid
		 detections generate nan entries in the resulting dataset.
	"""

	# init 
	(joint_info_dict, link_info_dict, robot_id) = fkInit()
	# inverse of rh8d base to root joint tf 
	root_joint = list(joint_info_dict.keys())[0]
	(_, _, _, _, trans, quat) = pb.getLinkState(robot_id, joint_info_dict[root_joint], computeForwardKinematics=True)
	(inv_trans, inv_quat) = invPersp(trans, quat, RotTypes.QUAT)
	T_root_joint_world = pose2Matrix(inv_trans, inv_quat, RotTypes.QUAT)
	print("Init fk with root joint", root_joint, "root trans", trans, "root ori", getRotation(quat, RotTypes.QUAT, RotTypes.EULER))

	joint_info_dict = {'joint7': 1, 'joint8': 2, 'jointI1': 10, 'jointI2': 11, 'jointI3': 12, 'jointM1': 14, 'jointM2': 15, 'jointM3': 16, 'jointR1': 18, 'jointR2': 19, 'jointR3': 20, 'jointL1': 22, 'jointL2': 23, 'jointL3': 24, 'jointT0': 5, 'jointT1': 6, 'jointT2': 7, 'jointT3': 8}
	link_info_dict = {'joint8': {'index': 4, 'fixed_end': 'joint_r_laser'}, 'jointI3': {'index': 13, 'fixed_end': 'joint_Ibumper'}, 'jointM3': {'index': 17, 'fixed_end': 'joint_Mbumper'}, 'jointR3': {'index': 21, 'fixed_end': 'joint_Rbumper'}, 'jointL3': {'index': 25, 'fixed_end': 'joint_Lbumper'}, 'jointT3': {'index': 9, 'fixed_end': 'joint_Tbumper'}}
	
	# create dataset keys
	keypt_joint_keys = []
	keypt_tcp_keys = []
	for joint in joint_info_dict.keys():
		keypt_joint_keys.append(joint)
		if joint in link_info_dict.keys():
			keypt_tcp_keys.append(link_info_dict[joint]['fixed_end'])
	keypt_joint_keys.extend(keypt_tcp_keys)

	# data structures
	keypt_df_dict = {joint:  pd.DataFrame(columns=['timestamp', 'trans', 'rot_mat']) for joint in keypt_joint_keys} 
	print("\nCreating datastructures for joints:")
	print(keypt_joint_keys)

	# load post-processed recordings
	detection_dct: dict = readDetectionDataset(os.path.join(DATA_PTH, "keypoint/post_processed/detection.json"))  # contains nans

	# find non-nan row indices
	print("\nSearching all non-nan indices")
	(common_indices, all_indices) = findCommonIndicesAllJoints(detection_dct)

	# iter whole detection
	print("\nComputing fk from index ...")
	for idx in all_indices:
		# collect all angles for fk
		angle_dict = {}
		for joint, df in detection_dct.items():
			angle_dict.update( {joint: df.loc[idx, 'angle']} )

		# compute keypoints
		keypt_dict = fk(joint_info_dict, link_info_dict, angle_dict, robot_id, T_root_joint_world, idx in common_indices)
		# add to results
		for joint, keypt in keypt_dict.items():
			keypt_df_dict[joint] = pd.concat([keypt_df_dict[joint], pd.DataFrame([keypt])], ignore_index=True) 

		print(f"\r{idx:5}", end="", flush=True)

	# save 
	print("\ndone. Saving...")
	kypt_df = pd.DataFrame({link: [df] for link, df in keypt_df_dict.items()})
	kypt_df.to_json(os.path.join(DATA_PTH, "keypoint/post_processed/kpts3D.json"), orient="index", indent=4)
	pb.disconnect()
	
def checkDataIntegrity() -> None:

	# init 
	(joint_info_dict, link_info_dict, robot_id) = fkInit()
	# inverse of rh8d base to root joint tf 
	root_joint = list(joint_info_dict.keys())[0]
	(_, _, _, _, trans, quat) = pb.getLinkState(robot_id, joint_info_dict[root_joint], computeForwardKinematics=True)
	(inv_trans, inv_quat) = invPersp(trans, quat, RotTypes.QUAT)
	T_root_joint_world = pose2Matrix(inv_trans, inv_quat, RotTypes.QUAT)
	print("Init fk with root joint", root_joint, "root trans", trans, "root ori", getRotation(quat, RotTypes.QUAT, RotTypes.EULER))

	joint_info_dict = {'joint7': 1, 'joint8': 2, 'jointI1': 10, 'jointI2': 11, 'jointI3': 12, 'jointM1': 14, 'jointM2': 15, 'jointM3': 16, 'jointR1': 18, 'jointR2': 19, 'jointR3': 20, 'jointL1': 22, 'jointL2': 23, 'jointL3': 24, 'jointT0': 5, 'jointT1': 6, 'jointT2': 7, 'jointT3': 8}
	link_info_dict = {'joint8': {'index': 4, 'fixed_end': 'joint_r_laser'}, 'jointI3': {'index': 13, 'fixed_end': 'joint_Ibumper'}, 'jointM3': {'index': 17, 'fixed_end': 'joint_Mbumper'}, 'jointR3': {'index': 21, 'fixed_end': 'joint_Rbumper'}, 'jointL3': {'index': 25, 'fixed_end': 'joint_Lbumper'}, 'jointT3': {'index': 9, 'fixed_end': 'joint_Tbumper'}}

	# load data
	detection_dct: dict = readDetectionDataset(os.path.join(DATA_PTH, "keypoint/joined/detection.json"))
	pp_detection_dct: dict = readDetectionDataset(os.path.join(DATA_PTH, "keypoint/post_processed/detection.json"))
	keypoints_dct: dict = readDetectionDataset(os.path.join(DATA_PTH, "keypoint/joined/kpts3D.json"))
	
	print("\norig cols", list(detection_dct.keys()))
	print("pp cols", list(pp_detection_dct.keys()))
	print()

	for joint, df in detection_dct.items():
		assert(detection_dct[joint].index[-1] == pp_detection_dct[joint].index[-1])

	print("Checking index...")
	for idx in detection_dct[root_joint].index.tolist():
		nan_list = []
		angle_dict = {}
		pp_angle_dict = {}
		for joint, df in detection_dct.items():
			angle = df.loc[idx, 'angle']
			angle_dict.update( {joint: angle} )
			pp_angle = pp_detection_dct[joint].loc[idx, 'angle']
			pp_angle_dict.update( {joint: angle} )
			if np.isnan(angle):
				nan_list.append(joint)
			else:
				assert( np.isclose(angle, pp_angle))

		if len(nan_list) > 0:
			# print(idx, "has nans for", nan_list)
			continue

		# compute keypoints
		keypt_dict = fk(joint_info_dict, link_info_dict, angle_dict, robot_id, T_root_joint_world)
		pp_keypt_dict = fk(joint_info_dict, link_info_dict, pp_angle_dict, robot_id, T_root_joint_world)
		# add to results
		for joint, keypt in keypt_dict.items():
			if 'bumper' in joint or 'laser' in joint:
				if not np.allclose(keypt["trans"], pp_keypt_dict[joint]["trans"]):
					print(joint, "trans", keypt["trans"], "pp_trans", pp_keypt_dict[joint]["trans"])
				# if not np.allclose(keypt["trans"], keypoints_dct[joint].loc[idx, "trans"]):
				# 	print(idx, joint, "has keypoint", keypt["trans"], "data keypoint", keypoints_dct[joint].loc[idx, "trans"])
		
		print(f"\r{idx:5}", end="", flush=True)

	print("\nDone")
	pb.disconnect()

if __name__ == "__main__":
	# img2Video(os.path.join(REC_DIR, "joined/det"), os.path.join(REC_DIR, "movies/detection.mp4"), fps=25)
	# mosaicImg(3526, os.path.join(REC_DIR, 'joined/mosaic.jpg'))

	# replaceNanData()
	fkFromDetection()
	# genAllTrainingData(post_proc=True)
	# checkDataIntegrity()
