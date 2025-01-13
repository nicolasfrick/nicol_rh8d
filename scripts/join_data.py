#!/usr/bin/env python3

import os
import cv2
import glob
import shutil
import numpy as np
import pandas as pd
from typing import Tuple
from datetime import datetime

REC_DIR = os.path.join(os.path.expanduser("~"), "rh8d_dataset")

def readDictDataset(filepth: str) -> dict:
	"""detection.json
		  kpts3D.json
		  kptsFK.json
	"""
	df = pd.read_json(filepth, orient='index')                                           
	df_dict = { joint: 
		 					pd.DataFrame.from_dict(data, orient='index')	# we want integer index
										.reset_index(drop=False).rename(columns={'index': 'old_index'}).astype({'old_index': int}).set_index('old_index')
											for joint, data in df.iloc[0].items() }		
	return df_dict

def readDataset(filepth: str) -> dict:
	"""actuator_states.json
		  joint_states.json
		  det/tcp_tf.json
		  det/tf.json
		  head_cam/tf.json
		  right_eye/tf.json
		   left_eye/tf.json
	"""
	df = pd.read_json(filepth, orient='index')                                           
	return df

def listFiles(filepth: str, pattern: str) -> list:
		pattern = os.path.join(filepth, f'*{pattern}*')
		data_files = glob.glob(pattern, recursive=False)
		sorted_paths = sorted(data_files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].replace('plot3D_', '')), reverse=True)
		# print("\n".join(sorted_paths))
		return sorted_paths

def incFilename(incr: int, files: list) -> None:
	for file_path in files:
		# Extract directory and filename
		directory, filename = os.path.split(file_path)
		# Extract the numerical part of the filename and add the increment
		name, extension = os.path.splitext(filename)
		if 'plot3D_' in name:
			new_name = f"plot3D_{int(name.replace('plot3D_', '')) + incr}{extension}"
		else:
			new_name = f"{int(name) + incr}{extension}"
		
		# Create the new file path
		new_path = os.path.join(directory, new_name)
		
		# Move the file
		shutil.move(file_path, new_path)
		print(f"Moved: {file_path} -> {new_path}")

def mvImgNames(pth: str, start: int, ftype: str) -> None:
	imgs = listFiles(pth, ftype)
	incFilename(start, imgs)

def mvImgs(pth1: str, pth2: str) -> None:
	jpg_imgs = listFiles(pth2, '.jpg')
	npz_imgs = listFiles(pth2, '.npz')
	jpg_imgs.extend(npz_imgs)
	for img_pth in jpg_imgs:
		_, filename = os.path.split(img_pth)
		new_path = os.path.join(pth1, filename)
		shutil.move(img_pth, new_path)
		print(f"Moved: {img_pth} -> {new_path}")

def mvDfIndex(pth: str, start: int) -> None:
	df = readDataset(pth)
	print(df)
	df.index = df.index + start
	print(df)
	df.to_json(pth, orient="index", indent=4)

def mvDfDictIndex(pth: str, start: int) -> None:
	df_dict = readDictDataset(pth)
	for joint in df_dict:
		df_dict[joint].index = df_dict[joint].index + start

	df = pd.DataFrame({joint: [df] for joint, df in df_dict.items()})
	df.to_json(pth, orient="index", indent=4)

def appendDf(pth1: str, pth2: str, end: int) -> None:
	print("Joining", pth2)
	print("to", pth1)
	df1 = readDataset(pth1)
	df2 = readDataset(pth2)
	assert(end == df2.index.tolist()[0])
	print(df1)
	result_df = pd.concat([df1[ : end], df2])
	print(result_df)

	result_df.to_json(pth1, orient="index", indent=4)

def appendDictDf(pth1: str, pth2: str, end: int) -> None:
	df_dict1 = readDictDataset(pth1)
	df_dict2 = readDictDataset(pth2)
	print("Joining", pth2)
	print("to", pth1)
	print(df_dict1['joint7'])
	for joint in df_dict1:
		assert(end == df_dict2[joint].index.tolist()[0])
		df_dict1[joint] = pd.concat([df_dict1[joint][ : end], df_dict2[joint]])
	print(df_dict1['joint7'])

	df = pd.DataFrame({joint: [df] for joint, df in df_dict1.items()})
	df.to_json(pth1, orient="index", indent=4)

def renameAll(pth: str, start: int) -> None:
	# left_eye
	mvImgNames(os.path.join(pth, "left_eye"), start, ".jpg")
	mvDfIndex(os.path.join(pth, "left_eye/tf.json"), start)
	# right_eye
	mvImgNames(os.path.join(pth, "right_eye"), start, ".jpg")
	mvDfIndex(os.path.join(pth, "right_eye/tf.json"), start)

	# top_cam
	mvImgNames( os.path.join(pth, "top_cam"), start, ".jpg")
	mvImgNames( os.path.join(pth, "top_cam"), start, ".npz")
	# head cam
	mvImgNames( os.path.join(pth, "head_cam"), start, ".jpg")
	mvImgNames( os.path.join(pth, "head_cam"), start, ".npz")
	mvDfIndex(os.path.join(pth, "head_cam/tf.json"), start)

	# det
	mvImgNames( os.path.join(pth, "det"), start, ".jpg")
	mvDfIndex(os.path.join(pth, "det/tf.json"), start)
	mvDfIndex(os.path.join(pth, "det/tcp_tf.json"), start)

	# orig
	mvImgNames( os.path.join(pth, "orig"), start, ".jpg")
	mvImgNames( os.path.join(pth, "orig"), start, ".npz")

	# actuator_states/joint_states
	mvDfIndex(os.path.join(pth, "actuator_states.json"), start)
	mvDfIndex(os.path.join(pth, "joint_states.json"), start)

	# detection/kpts3D/kptsFK
	mvDfDictIndex(os.path.join(pth, "detection.json"), start)
	mvDfDictIndex(os.path.join(pth, "kpts3D.json"), start)
	mvDfDictIndex(os.path.join(pth, "kptsFK.json"), start)

def concatAll(top_pth: str, folder_pth: str, df1_end_idx: int) -> None:
	# top level
	detection = 'detection.json'
	appendDictDf(os.path.join(top_pth, detection), os.path.join(top_pth, folder_pth, detection), df1_end_idx)
	kpts3D = 'kpts3D.json'
	appendDictDf(os.path.join(top_pth, kpts3D), os.path.join(top_pth, folder_pth, kpts3D), df1_end_idx)
	kptsFK = 'kptsFK.json'
	appendDictDf(os.path.join(top_pth, kptsFK), os.path.join(top_pth, folder_pth, kptsFK), df1_end_idx)
	# actuator_states
	actuator_states = 'actuator_states.json'
	appendDf(os.path.join(top_pth, actuator_states), os.path.join(top_pth, folder_pth, actuator_states), df1_end_idx)
	# joint_states
	joint_states = 'joint_states.json'
	appendDf(os.path.join(top_pth, joint_states), os.path.join(top_pth, folder_pth, joint_states), df1_end_idx)

	# det
	det_tcp_tf = "det/tcp_tf.json"
	appendDf(os.path.join(top_pth, det_tcp_tf), os.path.join(top_pth, folder_pth, det_tcp_tf), df1_end_idx)
	det_tf = "det/tf.json"
	appendDf(os.path.join(top_pth, det_tf), os.path.join(top_pth, folder_pth, det_tf), df1_end_idx)
	mvImgs(os.path.join(top_pth, "det"), os.path.join(top_pth, folder_pth, "det"))
	# orig
	mvImgs(os.path.join(top_pth, "orig"), os.path.join(top_pth, folder_pth, "orig"))

	# left_eye
	le_tf = "left_eye/tf.json"
	appendDf(os.path.join(top_pth, le_tf), os.path.join(top_pth, folder_pth, le_tf), df1_end_idx)
	mvImgs(os.path.join(top_pth, "left_eye"), os.path.join(top_pth, folder_pth, "left_eye"))
	# right_eye
	re_tf = "right_eye/tf.json"
	appendDf(os.path.join(top_pth, re_tf), os.path.join(top_pth, folder_pth, re_tf), df1_end_idx)
	mvImgs(os.path.join(top_pth, "right_eye"), os.path.join(top_pth, folder_pth, "right_eye"))
	# head_cam
	head_tf = "head_cam/tf.json"
	appendDf(os.path.join(top_pth, head_tf), os.path.join(top_pth, folder_pth, head_tf), df1_end_idx)
	mvImgs(os.path.join(top_pth, "head_cam"), os.path.join(top_pth, folder_pth, "head_cam"))
	# top_cam
	mvImgs(os.path.join(top_pth, "top_cam"), os.path.join(top_pth, folder_pth, "top_cam"))

# keypoint_11_11_13_30_wp_0_to_2004_no_add_imgs
# keypoint_11_18_11_05_wp_2003_to_4409
# keypoint_11_15_10_46_wp_4006_to_9390
# keypoint_11_18_14_13_wp_0_to_2004

# start = 9796
# renameAll(os.path.join(REC_DIR, "joined/tmp"), start)
# concatAll(os.path.join(REC_DIR, "joined/tmp"), start)

def assertDataset() -> None:
	pth = os.path.join(REC_DIR, "joined")

	# single dfs
	actuator_states_df = readDataset(os.path.join(pth, 'actuator_states.json'))
	joint_states_df = readDataset(os.path.join(pth, 'joint_states.json' ))
	det_tcp_df = readDataset(os.path.join(pth, 'det/tcp_tf.json'))
	det_df = readDataset(os.path.join(pth, 'det/tf.json'))
	head_cam_df = readDataset(os.path.join(pth, 'head_cam/tf.json'))
	left_eye_df = readDataset(os.path.join(pth, 'left_eye/tf.json'))
	right_eye_df = readDataset(os.path.join(pth, 'right_eye/tf.json'))

	full_dfs = [actuator_states_df, joint_states_df, det_tcp_df, det_df]
	part_dfs = [head_cam_df, left_eye_df, right_eye_df]

	base_df_index_full = full_dfs[0].index.tolist()
	for df in full_dfs:
		index = df.index.tolist()
		assert(index == base_df_index_full)
	assert(base_df_index_full[-1] == len(base_df_index_full)-1)

	base_df_index_part = part_dfs[0].index.tolist()
	for df in part_dfs:
		index = df.index.tolist()
		assert(index == base_df_index_part)
	missing_index_part =  full_dfs[0].index.difference(part_dfs[0].index).tolist()

	assert(base_df_index_full[0] == missing_index_part[0])
	assert(base_df_index_full[missing_index_part[-1] +1 :] == base_df_index_part)
	assert(missing_index_part[-1] == len(missing_index_part)-1)

	print("Simple dfs have same length and content")
	print("Num elements in full set", len(base_df_index_full))
	print("Num elements in partial set", len(base_df_index_part))
	print("Num missing in part", len(missing_index_part))
	print("Fist index full", base_df_index_full[0], "last", base_df_index_full[-1])
	print("Fist index part", base_df_index_part[0], "last", base_df_index_part[-1])
	print()

	# dict dfs
	detection = readDictDataset(os.path.join(pth, 'detection.json'))
	kpts3D = readDictDataset(os.path.join(pth, 'kpts3D.json'))
	kptsFK = readDictDataset(os.path.join(pth, 'kptsFK.json'))

	for joint, df in detection.items():
		index = df.index.tolist()
		assert(index == base_df_index_full)
	for joint, df in kpts3D.items():
		index = df.index.tolist()
		assert(index == base_df_index_full)
	for joint, df in kptsFK.items():
		index = df.index.tolist()
		assert(index == base_df_index_full)

	dict_df_index = detection['joint7'].index.tolist()
	print("Dict dfs have same length and content")
	print("Num elements in set", len(dict_df_index))
	print("First index", dict_df_index[0], "last", dict_df_index[-1])
	print()

	# timestamps
	full_timestamps_equal =   actuator_states_df['timestamp'].equals(joint_states_df['timestamp']) and \
														joint_states_df['timestamp'].equals(det_tcp_df['timestamp']) and \
														det_tcp_df['timestamp'].equals(det_df['timestamp'])
	print("Timestamps in full set are equal: ", full_timestamps_equal)

	idx = head_cam_df.index.tolist()[0]
	asd = actuator_states_df['timestamp'].values.tolist()

	for i, ts2 in enumerate(head_cam_df['timestamp'].values.tolist()):
		ts1 = asd[idx + i]
		if not np.isclose(ts1, ts2, atol=1e9, rtol=0):
			print("Head ts deviates at index",  i, ts1/1e9, ts2/1e9, (ts1-ts2)/1e9)
		
	for i, ts2 in enumerate(right_eye_df['timestamp'].values.tolist()):
		ts1 = asd[idx + i]
		if not np.isclose(ts1, ts2, atol=1e9, rtol=0):
			print("Right eye ts deviates at index",  i, ts1/1e9, ts2/1e9, (ts1-ts2)/1e9)

	for i, ts2 in enumerate(left_eye_df['timestamp'].values.tolist()):
		ts1 = asd[idx + i]
		if not np.isclose(ts1, ts2, atol=1e9, rtol=0):
			print("Left eye ts deviates at index",  i, ts1/1e9, ts2/1e9, (ts1-ts2)/1e9)

	# float required now
	asd = np.array(asd, dtype=np.float64)
	asd = asd / 1e9

	# iter detection (contains nans)
	for joint, df in detection.items():
		tsd = df['timestamp'].values.tolist()
		for i,( ts1, ts2) in enumerate(zip(asd, tsd)):
			if not np.isnan(ts2):
				if not np.isclose(ts1, ts2, atol=1.0, rtol=0):
					print("Det joint", joint, "deviates at index",  i, ts1, ts2, ts1-ts2)
	print()

	for joint, df in kpts3D.items():
		tsd = df['timestamp'].values.tolist()
		for i,( ts1, ts2) in enumerate(zip(asd, tsd)):
			if not np.isnan(ts2):
				if not np.isclose(ts1, ts2, atol=1.0, rtol=0):
					print("Kpts 3D joint", joint, "deviates at index",  i, ts1, ts2, ts1-ts2)
	print()

	for joint, df in kptsFK.items():
		tsd = df['timestamp'].values.tolist()
		for i,( ts1, ts2) in enumerate(zip(asd, tsd)):
			if not np.isnan(ts2):
				if not np.isclose(ts1, ts2, atol=1.0, rtol=0):
					print("Kpts FK joint", joint, "deviates at index",  i, ts1, ts2, ts1-ts2)
	print()

	# img folders
	det = os.path.join(pth, 'det')
	orig = os.path.join(pth, 'orig')
	top_cam = os.path.join(pth, 'top_cam')
	head_cam = os.path.join(pth, 'head_cam')
	left_eye = os.path.join(pth, 'left_eye')
	right_eye = os.path.join(pth, 'right_eye')

	det_imgs = listFiles(det, '.jpg')
	det_imgs.reverse()
	plt_imgs = [fl for fl in det_imgs if 'plot3D_' in fl]
	det_imgs = [fl for fl in det_imgs if not 'plot3D_' in fl]
	orig_imgs = listFiles(orig, '.jpg')
	orig_imgs.reverse()
	top_cam_imgs = listFiles(top_cam, '.jpg')
	top_cam_imgs.reverse()
	head_cam_imgs = listFiles(head_cam, '.jpg')
	head_cam_imgs.reverse()
	left_eye_imgs = listFiles(left_eye, '.jpg')
	left_eye_imgs.reverse()
	right_eye_imgs = listFiles(right_eye, '.jpg')
	right_eye_imgs.reverse()

	assert(len(det_imgs) ==  len(orig_imgs)) # plot incomplete
	assert(len(top_cam_imgs) == len(head_cam_imgs) == len(left_eye_imgs) == len(right_eye_imgs))

	# reverse order 
	assert(int(os.path.splitext(os.path.basename(det_imgs[0]))[0]) == 0)
	assert(int(os.path.splitext(os.path.basename(orig_imgs[0]))[0]) == 0)
	assert(int(os.path.splitext(os.path.basename(top_cam_imgs[0]))[0]) == missing_index_part[-1] +1)
	assert(int(os.path.splitext(os.path.basename(head_cam_imgs[0]))[0]) == missing_index_part[-1] +1)
	assert(int(os.path.splitext(os.path.basename(left_eye_imgs[0]))[0]) == missing_index_part[-1] +1)
	assert(int(os.path.splitext(os.path.basename(right_eye_imgs[0]))[0]) == missing_index_part[-1] +1)

	assert(int(os.path.splitext(os.path.basename(det_imgs[-1]))[0]) == base_df_index_full[-1])
	assert(int(os.path.splitext(os.path.basename(orig_imgs[-1]))[0]) == base_df_index_full[-1])
	assert(int(os.path.splitext(os.path.basename(top_cam_imgs[-1]))[0]) == base_df_index_full[-1])
	assert(int(os.path.splitext(os.path.basename(head_cam_imgs[-1]))[0]) == base_df_index_full[-1])
	assert(int(os.path.splitext(os.path.basename(left_eye_imgs[-1]))[0]) == base_df_index_full[-1])
	assert(int(os.path.splitext(os.path.basename(right_eye_imgs[-1]))[0]) == base_df_index_full[-1])

	# for i in base_df_index_full:
	# 	ts1 = asd[i]
	# 	ts2 = os.path.getmtime(det_imgs[i])
	# 	if not np.isclose(ts1, ts2,atol=2.5, rtol=0):
	# 		print(det_imgs[i], " ts deviates at index",  i, ts1, ts2, ts1-ts2)

# assertDataset()

def mosaicImg(num: int, target_size: Tuple=(1920, 1080)) -> None:
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
	depth_images = 	[cv2.cvtColor(np.load(path)['array'], cv2.COLOR_GRAY2BGR) for path in npzs]
	
	ts = []
	for p in imgs:
		t = os.path.getmtime(p)
		print(p, t)
		ts.append(t)
	# for p in npzs:
	# 	t = os.path.getmtime(p)
	# 	print(p, t)
	# 	ts.append(t)
	ts = np.array(ts, dtype=np.float64)
	mean = np.mean(ts)
	for t in ts:
		print("diff", t - mean)
	for t in ts:
		print(datetime.fromtimestamp(t))

	h, w = images[0].shape[:2]
	r = target_size[0] / float(w)
	dim = (target_size[0], int(h * r))
	resized_images = [cv2.resize(img, dim, interpolation=cv2.INTER_AREA) for img in images]

	h, w = depth_images[0].shape[:2]
	r = target_size[0] / float(w)
	dim = (target_size[0], int(h * r))
	resized_depth_images = [cv2.resize(img, dim, interpolation=cv2.INTER_AREA) for img in depth_images]

	tmp = []
	for idx in range(len(resized_images)):
		tmp.append(resized_images[idx])
		if idx < len(resized_depth_images):
			tmp.append(resized_depth_images[idx])
	resized_images = tmp

	resized_images = [img.astype(np.uint8) for img in resized_images]
	while len(resized_images) < 8: # pad black
		resized_images.append(np.zeros_like(resized_images[0]))

	rows = [
		np.hstack(resized_images[:2]),  # first row: 2 images
		np.hstack(resized_images[2:4]),   # second row: 2 images
		np.hstack(resized_images[4:6]),   # third row: 2 images
		np.hstack(resized_images[6:]),   # fourth row: 2 images
	]
	mosaic = np.vstack(rows)  # combine vertically

	cv2.namedWindow("Mosaic", cv2.WINDOW_NORMAL)
	cv2.imshow("Mosaic", mosaic)
	if cv2.waitKey(0) == 'q':
		cv2.destroyAllWindows()
		
mosaicImg(11000)
