import os
import glob
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
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

		# self.ax_3d.set_title('6D Keypoints')
		if pause > 0.0:
			plt.pause(pause)
		self.fig.canvas.draw()

		return np.array(self.fig.canvas.renderer.buffer_rgba())
	
def plotTrainingData(data_pth: str, save_pth: str=None, grid: bool=True) -> None:
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
	angle_radians = 2 * np.arccos(quats[:, 3]) # rotation angle in radians
	angle_radians -= 4
	# spline smoothing
	x = np.arange(len(angle_radians))
	spline = UnivariateSpline(x, angle_radians, s=300) 
	smoothed = spline(x)
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
	
def plotKeypoints(net: str, start: int=0, end: int=10000) -> None:
	# load keypoints
	data_pth = os.path.join(DATA_PTH, 'keypoint/joined')
	keypoints_dct = readDetectionDataset(os.path.join(data_pth, 'kpts3D.json')) 
	# get tcp name
	cfg = loadNetConfig('finger')
	tcp = cfg[net]['relative_to']

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

			buffer = keypt_plot.plotKeypoints(keypt_dict, tcp, pause=0.1, line=False, elev=10, azim=10)
			cv2.imshow("Keypoints", cv2.cvtColor(buffer, cv2.COLOR_RGBA2BGR))
			if cv2.waitKey(1) == 'q':
				return
	finally:
		cv2.destroyAllWindows()

def plotAllTrainingData(save: bool=False) -> None:
	data_pth = os.path.join(DATA_PTH, 'keypoint/train/config')
	pattern = os.path.join(data_pth, f'*.json*')
	data_files = glob.glob(pattern, recursive=False)
	save_files = [None for _ in range(len(data_files))]
	if save:
		save_files = [os.path.join(REC_DIR, os.path.basename(fl)).replace('json', 'svg') for fl in data_files]

	for dfile, sfile in zip(data_files, save_files):
		plotTrainingData(dfile, sfile)

if __name__ == "__main__":
	# plotKeypoints('index_flexion', 0, 1000)
	plotAllTrainingData()
