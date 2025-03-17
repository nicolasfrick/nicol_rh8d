import os
import glob
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import gridspec
from scipy.signal import find_peaks
from typing import Tuple, Optional, Union
from scipy.interpolate import UnivariateSpline
from util import *
matplotlib.use('agg')

PLT_CLRS = ['r', 'g', 'b', 'y', 'c']

fig, ax_3d, ax_z, ax_y, ax_x = None,None,None,None,None
euler_ax = []
t_ax = []

def plotPose(ax, position, rotation_matrix, label=''):
	origin = np.array(position)
	x_axis = rotation_matrix[:, 0]
	y_axis = rotation_matrix[:, 1]
	z_axis = rotation_matrix[:, 2]
	scale = 0.1
	ax.quiver(origin[0], origin[1], origin[2], x_axis[0], x_axis[1], x_axis[2], color='r', length=scale, label=f'{label}_x')
	ax.quiver(origin[0], origin[1], origin[2], y_axis[0], y_axis[1], y_axis[2], color='g', length=scale, label=f'{label}_y')
	ax.quiver(origin[0], origin[1], origin[2], z_axis[0], z_axis[1], z_axis[2], color='b', length=scale, label=f'{label}_z')

def visPose(pos, rot_mat, euler, t, pause=0, cla=False, llim=-1, hlim=1):
	global euler_ax, t_ax, fig, ax_3d, ax_z, ax_y, ax_x
	# clear axis
	if cla:
		ax_3d.cla()
	ax_3d.set_xlabel('X')
	ax_3d.set_ylabel('Y')
	ax_3d.set_zlabel('Z')
	ax_3d.set_xlim([llim, hlim])
	ax_3d.set_ylim([llim, hlim])
	ax_3d.set_zlim([llim, hlim])
	plotPose(ax_3d, pos, rot_mat, label=f'Time {t}')
	ax_3d.set_title(f'Pose at Time {t}')
	t_ax.append(t)
	euler_ax.append(euler)
	ax_x.plot(t_ax, [e[0] for e in euler_ax], color='r')
	ax_y.plot(t_ax, [e[1] for e in euler_ax], color='g')
	ax_z.plot(t_ax, [e[2] for e in euler_ax], color='b')
	if pause > 0.0:
		plt.pause(pause)
	fig.canvas.draw()
	return np.array(fig.canvas.renderer.buffer_rgba())

def setPosePlot(llim=-1, hlim=1):
	global fig, ax_3d, ax_z, ax_y, ax_x
	fig = plt.figure(figsize=(30, 18))
	gs = gridspec.GridSpec(2, 2, width_ratios=[1.5, 1], height_ratios=[1, 1])
	ax_3d = fig.add_subplot(gs[0, 0], projection='3d')
	ax_z = fig.add_subplot(gs[0, 1])
	ax_y = fig.add_subplot(gs[1, 0])
	ax_x = fig.add_subplot(gs[1, 1])
	ax_z.set_xlabel('Time')
	ax_z.set_ylabel('Z-axis')
	ax_y.set_xlabel('Time')
	ax_y.set_ylabel('Y-axis')
	ax_x.set_xlabel('Time')
	ax_x.set_ylabel('X-axis')
	ax_3d.set_xlim([llim, hlim])
	ax_3d.set_ylim([llim, hlim])
	ax_3d.set_zlim([llim, hlim])
	ax_3d.set_xlabel('X')
	ax_3d.set_ylabel('Y')
	ax_3d.set_zlabel('Z')
	return fig, ax_3d, ax_x, ax_y, ax_z

class KeypointPlot():
	def __init__(self,
							dpi: int=300,
							shape: Tuple=(1920, 1080),
							x_llim: float=-0.05, 
							x_hlim: float=0.075,
							y_llim: float=0.0,
							y_hlim: float=0.15,
							z_llim: float=0.0,
							z_hlim: float=0.04,
							ax_scale: float=0.005,
							linewidth: float=1.0,
							grid_shape: Tuple=(1, 1),
							width_ratios: list=[1], 
							height_ratios: list=[1],
							) -> None:
		
		self.shape = (shape[1], shape[0], 3) # rgb
		figsize = (shape[0] / dpi, shape[1] / dpi)
		self.fig = plt.figure(figsize=figsize)
		self.gs = gridspec.GridSpec(grid_shape[0], grid_shape[1], width_ratios=width_ratios, height_ratios=height_ratios)
		self.ax_3d = self.fig.add_subplot(self.gs[0, 0], projection='3d')
		self.ax_3d.set_xlim([x_llim, x_hlim])
		self.ax_3d.set_ylim([y_llim, y_hlim])
		self.ax_3d.set_zlim([z_llim, z_hlim])
		self.ax_3d.set_xlabel('X')
		self.ax_3d.set_ylabel('Y')
		self.ax_3d.set_zlabel('Z')
		self.x_llim = x_llim
		self.x_hlim = x_hlim
		self.y_llim = y_llim
		self.y_hlim = y_hlim
		self.z_llim = z_llim
		self.z_hlim = z_hlim
		self.scale = ax_scale
		self.linewidth = linewidth
		self.root_center_point = None

	@property
	def img_shape(self) -> np.ndarray:
		return self.shape

	def clear(self) -> None:
		self.ax_3d.cla()
		self.root_center_point = None

	def _plotKeypoint(self, origin: np.array, rotation_matrix: np.array, label: str='') -> None:
		x_axis = rotation_matrix[:, 0]
		y_axis = rotation_matrix[:, 1]
		z_axis = rotation_matrix[:, 2]
		self.ax_3d.quiver(origin[0], origin[1], origin[2], x_axis[0], x_axis[1], x_axis[2], color='r', length=self.scale, label=f'{label}_x')
		self.ax_3d.quiver(origin[0], origin[1], origin[2], y_axis[0], y_axis[1], y_axis[2], color='g', length=self.scale, label=f'{label}_y')
		self.ax_3d.quiver(origin[0], origin[1], origin[2], z_axis[0], z_axis[1], z_axis[2], color='b', length=self.scale, label=f'{label}_z')

	def plotKeypoints(self, 
				   						fk_dict: dict, 
										parent_joint: str, 
										pause: Optional[float]=0.0, 
										cla: Optional[bool]=False, 
										line: Optional[bool]=True, 
										elev: Optional[Union[float, None]]=None, 
										azim: Optional[Union[float, None]]=None, 
										text:  Optional[Union[str, None]]=None, 
										title:  Optional[str]='', 
										) -> np.ndarray:
		if cla:
			self.clear()

		self.ax_3d.set_xlabel('X')
		self.ax_3d.set_ylabel('Y')
		self.ax_3d.set_zlabel('Z')
		self.ax_3d.set_xlim([self.x_llim, self.x_hlim])
		self.ax_3d.set_ylim([self.y_llim, self.y_hlim])
		self.ax_3d.set_zlim([self.z_llim, self.z_hlim])
		if elev is not None and azim is not None:
			self.ax_3d.view_init(elev=elev, azim=azim)
			self.ax_3d.view_init(elev=elev, azim=azim)
			self.ax_3d.view_init(elev=elev, azim=azim)

		last_center_point = self.root_center_point
		for joint, fk in fk_dict.items():
			# plt cs
			trans = fk['trans']
			self._plotKeypoint(trans, fk['rot_mat'], joint)

			# connect cs
			if last_center_point is not None and line:
				self.ax_3d.plot([last_center_point[0], trans[0]], [last_center_point[1], trans[1]], [last_center_point[2], trans[2]], color=fk['color'], linewidth=self.linewidth)
			last_center_point = trans

			if joint == parent_joint:
				# save last point before branching
				self.root_center_point = trans

		if text:
			text2D = self.ax_3d.text2D(0.0, 0.0, "", transform= self.ax_3d.transAxes, fontsize=10)
			text2D.set_text("Fg: " + text)
			
		self.ax_3d.set_title(title)
		if pause > 0.0:
			plt.pause(pause)
		self.fig.canvas.draw()

		return np.array(self.fig.canvas.renderer.buffer_rgba())
	
	def plotOrientation(self, quats: np.ndarray, f: np.ndarray) -> np.ndarray:
		keypt_dict = {'orientation': {'trans': np.array([0, 0, 0]), 'rot_mat': getRotation(quats, RotTypes.QUAT, RotTypes.MAT), 'color': 'b'}}
		qtxt = f"fx: {f[0]:2f}, fy: {f[1]:2f}, fz: {f[2]:2f}"
		return self.plotKeypoints(keypt_dict, '', pause=0.0, line=False, elev=10, azim=-20, cla=True, text=str(f), title="TCP Orientation")
	
def plotData(x: np.ndarray, y: np.ndarray, name: str, save_pth: str=None, grid: bool=True) -> None:
	fig, ax = plt.subplots()
	ax.plot(x, y)
	plt.xlabel("Index")
	plt.ylabel(name)
	plt.grid(visible=grid)
	if save_pth is not None:
		fig.savefig(save_pth, format='svg')
		
def smoothMagnQuats(quats: np.ndarray, s: int=300) -> np.ndarray:
	angle_radians = 2 * np.arccos(quats[:, 3]) # rotation angle in radians
	angle_radians -= 4
	# spline smoothing
	x = np.arange(len(angle_radians))
	spline = UnivariateSpline(x, angle_radians, s=s) 
	return spline(x)

def smoothMagnTrans(trans: np.ndarray, s: int=0) -> np.ndarray:
	magn = np.linalg.norm(trans, axis=1)
	# spline smoothing
	x = np.arange(len(magn))
	spline = UnivariateSpline(x, magn, s=s) 
	return spline(x)
	
def plotTrainingData(data_pth: str, save_pth: str=None, grid: bool=True, show: bool=True) -> None:
	matplotlib.use('TkAgg') 

	df = pd.read_json(data_pth, orient='index')      

	cols = ["cmd", "angle"]
	if 'thumb' in data_pth and not 'mono' in data_pth:
		cols = ["cmd1", "cmd2", "angle1", "angle2", "angle3"]
	elif 'poly' in data_pth:
		cols = ["cmd", "angle1", "angle2", "angle3"]

	fig, ax = plt.subplots()
	df[cols].plot(title=data_pth.split('/')[-1].replace('.json', ''), ax=ax, grid=True, kind="line", marker='.', markersize=4)
	
	# rotation magnitude
	quats = np.array([np.array(lst) for lst in df['quat']])
	smoothed = smoothMagnQuats(quats)
	# plot
	ax.plot(df.index.to_list(), smoothed, label="magnitude of rotation")
	ax.legend()
	
	# secondary y-axis
	ax2 = ax.twinx()
	ax2.set_ylabel('Mapped actuator range')
	ax2.set_yticks(np.linspace(-np.pi, np.pi, endpoint=True, num=10))
	# ax2.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])

	plt.xlabel("Index")
	plt.ylabel("Values")
	plt.grid(visible=grid)
	if save_pth is not None:
		fig.savefig(save_pth, format='svg')
	if show:
		plt.show()

def plotTrainingDataLogScale(file_pth: str) -> None:
	matplotlib.use('TkAgg') 
	data_pth = os.path.join(DATA_PTH, 'keypoint/train/config', file_pth)
	df = pd.read_json(data_pth, orient='index')      

	cols = ["cmd", "angle"]
	if 'thumb' in file_pth and not 'mono' in file_pth:
		cols = ["cmd1", "cmd2", "angle1", "angle2", "angle3"]
	elif 'poly' in file_pth:
		cols = ["cmd", "angle1", "angle2", "angle3"]

	fig, ax1 = plt.subplots()
	# linear scale
	ax1.plot(df.index, df[cols[1]], label=cols[1], color=PLT_CLRS[-1])
	ax1.tick_params(axis='y', labelcolor=PLT_CLRS[-1])
	# linear or log scale
	for i, c in enumerate(cols):
		ax = ax1.twinx()
		ax.plot(df.index, df[c], label=c, color=PLT_CLRS[i%len(PLT_CLRS)])
		if 'cmd' in c:
			ax.set_yscale('log')
		ax.tick_params(axis='y', labelcolor=PLT_CLRS[i%len(PLT_CLRS)])

	plt.title("")
	plt.grid()
	plt.show()
	
def plotKeypoints(start: int=0, end: int=10000, plt_pp: bool=False) -> None:
	# load keypoints
	keypoints_dct = readDetectionDataset(os.path.join(DATA_PTH, 'keypoint/post_processed/dense_kpts3D.json')) 
	if plt_pp:
		pp_keypoints_dct = readDetectionDataset(os.path.join(DATA_PTH, 'keypoint/post_processed/kpts3D.json')) 
	# get tcp name
	cfg = loadNetConfig('index_flexion')
	tcp = cfg['relative_to']

	# plot data
	cv2.namedWindow("Keypoints", cv2.WINDOW_NORMAL)
	keypt_plot = KeypointPlot(x_llim=0.0, 
														x_hlim=0.075,
														y_llim=-0.1,
														y_hlim=0.03,
														z_llim=-0.04,
														z_hlim=0.06,)
	try:
		for idx in range(start, end):
			keypt_dict = {}
			keypt_plot.clear()

			# get keypoint tcp as root tf
			tcp_trans = keypoints_dct[tcp].loc[idx, 'trans']
			tcp_rot_mat =  keypoints_dct[tcp].loc[idx, 'rot_mat']
			if tcp_trans is None or tcp_rot_mat is None:
				continue
			(inv_tcp_trans, inv_tcp_rot_mat) = invPersp(tcp_trans, tcp_rot_mat, RotTypes.MAT)
			T_tcp_root = pose2Matrix(inv_tcp_trans, inv_tcp_rot_mat, RotTypes.MAT)

			for joint in keypoints_dct:
				trans = keypoints_dct[joint].loc[idx, 'trans']
				rot_mat =  keypoints_dct[joint].loc[idx, 'rot_mat']
				if trans is not None and rot_mat is not None and joint not in ['joint7', 'joint8']:
					T_root_keypt = pose2Matrix(np.array(trans), np.array(rot_mat), RotTypes.MAT) 
					T_tcp_keypt = T_tcp_root @ T_root_keypt 
					keypt_dict.update( {joint: {'trans': T_tcp_keypt[:3, 3], 'rot_mat': T_tcp_keypt[:3, :3], 'color': 'b'}} )

			buffer = keypt_plot.plotKeypoints(keypt_dict, tcp, pause=0.0 if plt_pp else 0.1, line=False, elev=10, azim=10)
			
			# plot 2nd data
			if plt_pp:
				keypt_dict = {}

				# get keypoint tcp as root tf
				tcp_trans = pp_keypoints_dct[tcp].loc[idx, 'trans']
				tcp_rot_mat =  pp_keypoints_dct[tcp].loc[idx, 'rot_mat']
				if tcp_trans is None or tcp_rot_mat is None:
					continue
				(inv_tcp_trans, inv_tcp_rot_mat) = invPersp(tcp_trans, tcp_rot_mat, RotTypes.MAT)
				T_tcp_root = pose2Matrix(inv_tcp_trans, inv_tcp_rot_mat, RotTypes.MAT)

				for joint in pp_keypoints_dct:
					trans = pp_keypoints_dct[joint].loc[idx, 'trans']
					rot_mat =  pp_keypoints_dct[joint].loc[idx, 'rot_mat']
					if trans is not None and rot_mat is not None and joint not in ['joint7', 'joint8']:
						T_root_keypt = pose2Matrix(np.array(trans), np.array(rot_mat), RotTypes.MAT) 
						T_tcp_keypt = T_tcp_root @ T_root_keypt 
						keypt_dict.update( {joint: {'trans': T_tcp_keypt[:3, 3], 'rot_mat': T_tcp_keypt[:3, :3], 'color': 'b'}} )

				buffer = keypt_plot.plotKeypoints(keypt_dict, tcp, pause=0.1, line=False, elev=10, azim=10)

			cv2.imshow("Keypoints", cv2.cvtColor(buffer, cv2.COLOR_RGBA2BGR))
			if cv2.waitKey(1) == 'q':
				return
	finally:
		cv2.destroyAllWindows()

def plotOrientations(quats: pd.Series) -> None:
	keypt_plot = KeypointPlot(x_llim=0.0, 
														x_hlim=10,
														y_llim=-0.1,
														y_hlim=0.075,
														z_llim=-0.04,
														z_hlim=0.06,
														ax_scale = 0.1,
														linewidth = 0.01)
	
	keypt_dict = {}
	for idx, q in enumerate(quats.values[0:10]):
		keypt_dict.update( {f'{idx}': {'trans': np.array([idx, 0, 0]), 'rot_mat': getRotation(q, RotTypes.QUAT, RotTypes.MAT), 'color': 'b'}} )
	
	keypt_plot.plotKeypoints(keypt_dict, '', pause=0.1, line=False, elev=10, azim=-20)
	plt.show()

def plotAllTrainingData(save: bool=False) -> None:
	data_pth = os.path.join(DATA_PTH, 'keypoint/train/config')
	pattern = os.path.join(data_pth, f'*.json*')
	data_files = glob.glob(pattern, recursive=False)
	save_files = [None for _ in range(len(data_files))]
	if save:
		save_files = [os.path.join(REC_DIR, os.path.basename(fl)).replace('json', 'svg') for fl in data_files]

	for dfile, sfile in zip(data_files, save_files):
		plotTrainingData(dfile, sfile)

def plotDataMagn(quat_df: pd.Series, df: pd.DataFrame, title: str='', save_pth: str=None) -> None:
	matplotlib.use('TkAgg') 

	fig, ax = plt.subplots()

	# angles
	joint_angles = df["angle"]  
	ax.plot(df.index, joint_angles, label=title, color='b', marker='.', markersize=1)
	
	# rotation magnitude
	quats = np.array([np.array(lst) for lst in quat_df.values])
	smoothed = smoothMagnQuats(quats)
	ax.plot(df.index.to_list(), smoothed, label="magnitude of rotation")

	# orientations
	scale_factor = 0.02  
	min_angle = min(joint_angles)
	for i in range(0, len(quat_df), 200): 
		q = quat_df.iloc[i]
		rotation_matrix = getRotation(q, RotTypes.QUAT, RotTypes.MAT)

		rotation_axis_xy = rotation_matrix @ np.array([1, 0, 0])  
		rotation_axis_yz = rotation_matrix @ np.array([0, 1, 0])  
		rotation_axis_xy = rotation_axis_xy * scale_factor
		rotation_axis_yz = rotation_axis_yz * scale_factor

		ax.quiver(i, min_angle-0.1, rotation_axis_xy[0], rotation_axis_xy[1], color='r', width=0.0025, scale=0.8, label='X-axis' if i == 0 else "")
		ax.quiver(i, min_angle-0.1, rotation_axis_xy[0], rotation_axis_yz[2], color='g', width=0.0025, scale=0.8, label='Y-axis' if i == 0 else "")

	plt.xlabel("Index")
	plt.ylabel("Values")
	ax.set_title(title)
	ax.legend()
	plt.grid(visible=True)

	if save_pth is not None:
		fig.savefig(save_pth, format='svg')
	
	plt.show()

def plotDataEuler(quat_df: pd.Series, df: pd.DataFrame, title: str='', save_pth: str=None, stats: bool=True) -> None:
	matplotlib.use('TkAgg') 

	fig, axs = plt.subplots(3, 1)

	# angles
	joint_angles = df["angle"] 
	axs[0].plot(df.index.to_list(), joint_angles.values, label=title, color='y', marker='.', markersize=1)
	# amplitudes
	peaks, _ = find_peaks(joint_angles, height=joint_angles.mean(), distance=150) 
	peak_indices = joint_angles.index[peaks]
	peak_values = joint_angles.iloc[peaks]
	axs[0].plot(peak_indices.to_list(), peak_values.values, 'r-', linewidth=1)  

	# smoothed euler rotation components
	eulers = np.array([getRotation(np.array(lst), RotTypes.QUAT, RotTypes.EULER) for lst in quat_df.values])
	x = np.arange(len(eulers))
	spline_r = UnivariateSpline(x, eulers[:,0], s=2000) 
	spline_p = UnivariateSpline(x, eulers[:,1], s=300) 
	spline_y = UnivariateSpline(x, eulers[:,2], s=3000) 
	axs[1].plot(x, spline_r(x), label="roll", color='r')
	axs[1].plot(x, spline_p(x), label="pitch", color='g')
	axs[1].plot(x, spline_y(x), label="yaw", color='b')
	# magnitude
	quats = np.array([np.array(lst) for lst in quat_df.values])
	angles = 2 * np.arccos(quats[:,3].clip(-1, 1))
	spline_angles = UnivariateSpline(x, angles, s=800) 
	axs[1].plot(df.index.to_list(), spline_angles(x), label="angle magn", color='k')

	# solve for all axes and extract Fx in the tcp frame
	Fx = lambda q: tfForce(q)[0]*0.05 +1
	fx = np.array( [Fx(np.array(q)) for q in quat_df.values] )
	spline_fx = UnivariateSpline(x, fx, s=300) 
	axs[0].plot(df.index.to_list(), spline_fx(x), label="Fx_tcp", color='m')

	# zip
	eulers_smoothed = list(zip(spline_r(x), spline_p(x), spline_y(x)))
	quaternions_smoothed = np.array([getRotation(e, RotTypes.EULER, RotTypes.QUAT) for e in eulers_smoothed])
	axs[2].plot(x, quaternions_smoothed[:, 0], label="x", color='r')
	axs[2].plot(x, quaternions_smoothed[:, 1], label="y", color='g')
	axs[2].plot(x, quaternions_smoothed[:, 2], label="z", color='b')
	axs[2].plot(x, quaternions_smoothed[:, 3], label="w", color='k')

	# save for stats
	if stats:
		# smoothed eulers
		df = pd.DataFrame(columns=["euler"])
		for e in eulers_smoothed:
			df = pd.concat([df, pd.DataFrame([{"euler":e}])], ignore_index=True)
		df.to_json(os.path.join(DATA_PTH, "keypoint/post_processed/eulers_smoothed.json"), orient="index", indent=4)
		# orig eulers
		df = pd.DataFrame(columns=["euler"])
		for e in eulers:
			df = pd.concat([df, pd.DataFrame([{"euler":e}])], ignore_index=True)
		df.to_json(os.path.join(DATA_PTH, "keypoint/post_processed/eulers.json"), orient="index", indent=4)

		# smoothed quats
		df = pd.DataFrame(columns=["quat"])
		for q in quaternions_smoothed:
			df = pd.concat([df, pd.DataFrame([{"quat": q}])], ignore_index=True)
		df.to_json(os.path.join(DATA_PTH, "keypoint/post_processed/quaternions_smoothed.json"), orient="index", indent=4)
		# quats
		df = pd.DataFrame(columns=["quat"])
		for q in quat_df.values:
			df = pd.concat([df, pd.DataFrame([{"quat": np.array(q)}])], ignore_index=True)
		df.to_json(os.path.join(DATA_PTH, "keypoint/post_processed/quaternions.json"), orient="index", indent=4)

		# force smoothed
		F = lambda q: getRotation(q, RotTypes.QUAT, RotTypes.MAT)@Fg # R@Fg
		f = np.array( [F(np.array(q)) for q in quat_df.values] )
		# smooth
		spline_fx = UnivariateSpline(x, f[:,0], s=300) 
		spline_fy = UnivariateSpline(x, f[:,1], s=300) 
		spline_fz = UnivariateSpline(x, f[:,2], s=300) 
		# save
		df = pd.DataFrame(columns=["f_tcp"])
		for f_elmt in list(zip(spline_fx(x), spline_fy(x), spline_fz(x))):
			df = pd.concat([df, pd.DataFrame([{"f_tcp":f_elmt}])], ignore_index=True)
		df.to_json(os.path.join(DATA_PTH, "keypoint/post_processed/f_tcp_smoothed.json"), orient="index", indent=4)
		# force
		df = pd.DataFrame(columns=["f_tcp"])
		for f_elmt in f:
			df = pd.concat([df, pd.DataFrame([{"f_tcp" :f_elmt}])], ignore_index=True)
		df.to_json(os.path.join(DATA_PTH, "keypoint/post_processed/f_tcp.json"), orient="index", indent=4)

	plt.xlabel("Index")
	plt.ylabel("Values")
	axs[0].set_title(title)
	axs[0].legend()
	axs[1].legend()
	axs[2].legend()
	plt.grid(visible=True)

	if save_pth is not None:
		fig.savefig(save_pth, format='svg')
	
	plt.show()

def plotHelper(joint: str='jointM2') -> None:
	tcp_df: pd.DataFrame = pd.read_json(os.path.join(DATA_PTH, 'keypoint/joined/tcp_tf.json'), orient='index') 
	detection_dct = readDetectionDataset(os.path.join(DATA_PTH, "keypoint/post_processed/detection_smoothed.json")) # nans and outliers removed
	plotDataEuler(tcp_df['quat'], detection_dct[joint], joint)

# animate orientation
is_paused = False
ani = None
def animPose() -> None:
	global ani 

	matplotlib.use('TkAgg') 
	tcp_df: pd.DataFrame = pd.read_json(os.path.join(DATA_PTH, 'keypoint/joined/tcp_tf.json'), orient='index') 
	fixed_pos = np.array([0.5,-0.5,1.2])

	fig = plt.figure(figsize=(10, 8))
	ax = fig.add_subplot(111, projection='3d')
	axis_length = 0.2

	# lines for the coordinate frame
	x_axis, = ax.plot([], [], [], 'r-', label='X', linewidth=2)  
	y_axis, = ax.plot([], [], [], 'g-', label='Y', linewidth=2) 
	z_axis, = ax.plot([], [], [], 'b-', label='Z', linewidth=2)
	text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes, fontsize=12)
	text2 = ax.text2D(0.05, 0.8, "", transform=ax.transAxes, fontsize=12)
	text3 = ax.text2D(0.05, 0.65, "", transform=ax.transAxes, fontsize=12)

	def init():
		ax.set_xlim(1, 0)
		ax.set_ylim(0, -1)
		ax.set_zlim(0.8, 1.8)
		ax.set_xlabel('X')
		ax.set_ylabel('Y')
		ax.set_zlabel('Z')
		ax.set_title('Animated Pose')
		ax.legend()
		text.set_text("Timestamp: 0.0 s, frame: 0")
		text2.set_text("r: 0, p: 0, y: 0")
		text3.set_text("[]")
		return x_axis, y_axis, z_axis, text, text2, text3

	def update(frame):
		global is_paused
		if is_paused:
			return x_axis, y_axis, z_axis, "", ""

		entry = tcp_df.iloc[frame]
		ts = entry['timestamp']
		position = fixed_pos # entry['trans']
		quaternion = entry['quat']
		R = getRotation(quaternion, RotTypes.QUAT, RotTypes.MAT)
		euler = getRotation(quaternion, RotTypes.QUAT, RotTypes.EULER)
		# force
		Fg = tfForce(quaternion)

		x_local = np.array([1, 0, 0]) * axis_length
		y_local = np.array([0, 1, 0]) * axis_length
		z_local = np.array([0, 0, 1]) * axis_length
		
		x_world = position + R @ x_local
		y_world = position + R @ y_local
		z_world = position + R @ z_local
		
		x_axis.set_data_3d([position[0], x_world[0]], 
						[position[1], x_world[1]], 
						[position[2], x_world[2]])
		y_axis.set_data_3d([position[0], y_world[0]], 
						[position[1], y_world[1]], 
						[position[2], y_world[2]])
		z_axis.set_data_3d([position[0], z_world[0]], 
						[position[1], z_world[1]], 
						[position[2], z_world[2]])
		
		text.set_text(f"Timestamp: {ts.strftime('%H:%M:%S')}, frame: {frame}")
		text2.set_text(f"r: {euler[0]:2f}, p: {euler[1]:2f}, y: {euler[2]:2f}")
		text3.set_text(f"fx: {Fg[0]:2f}, p: {Fg[1]:2f}, y: {Fg[2]:2f}")
		
		return x_axis, y_axis, z_axis, text, text2, text3

	def on_key_press(event):
		global is_paused, ani
		# space
		if event.key == ' ': 
			is_paused = not is_paused
			if is_paused:
				ani.event_source.stop()
				print("Animation paused")
			else:
				ani.event_source.start()
				print("Animation resumed")

	# Bind the key press event
	fig.canvas.mpl_connect('key_press_event', on_key_press)

	ani = animation.FuncAnimation(fig, update, frames=len(tcp_df), init_func=init, blit=True, interval=3, )

	plt.show()
	
def visFeatCont(attributions: Any, target_names: list, feature_names: list) -> None:
	"""Plot feature contribution matrix with heatmap style."""
	matplotlib.use('TkAgg') 
	plt.figure(figsize=(12, 8))
	sns.heatmap(attributions, annot=True, cmap="YlGnBu", 
				xticklabels=target_names, yticklabels=feature_names, fmt=".2f")
	plt.title("Feature Contributions")
	plt.xlabel("Output Dimensions")
	plt.ylabel("Input Features")
	plt.tight_layout()
	plt.show()
	
def visActivationHist(activation_dict: dict) -> None:
	""" Plot histograms of activations per layer """
	matplotlib.use('TkAgg') 
	num_layers = len(activation_dict)
	fig, axes = plt.subplots(1, num_layers, figsize=(5 * num_layers, 5))
	if num_layers == 1:
		axes = [axes]  # single-layer case
	for i, (key, acts) in enumerate(activation_dict.items()):
		acts_flat = np.concatenate(acts)
		# include negative values
		min_val, max_val = min(acts_flat.min(), -1), max(acts_flat.max(), 1)
		axes[i].hist(acts_flat, bins=50, range=(min_val, max_val), log=True)
		axes[i].set_title(f"{key} PReLU Activation Histogram")
		axes[i].set_xlabel("Activation Value")
		axes[i].set_ylabel("Frequency (Log Scale)")
	plt.tight_layout()
	plt.show()

if __name__ == "__main__":
	# plotKeypoints(0, 1000, False)
	# plotAllTrainingData()
	animPose()
	# plotHelper()
