from util import *
import pybullet as pb

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
	# replaceNanData()
	# fkFromDetection()
	# checkDataIntegrity()
	genAllTrainingData(post_proc=True)
