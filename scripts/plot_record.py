import matplotlib
import numpy as np
from typing import Tuple
from matplotlib import gridspec
import matplotlib.pyplot as plt
matplotlib.use('agg')

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
                            llim: float=0.0, 
                            hlim: float=3.0,
                            ax_scale: float=0.1,
                            linewidth: float=2.0,
                            grid_shape: Tuple=(1, 1),
                            width_ratios: list=[1], 
                            height_ratios: list=[1],
                            ) -> None:
        
        self.shape = (shape[0], shape[1], 4) # rgba
        figsize = (shape[0] / dpi, shape[1] / dpi)
        self.fig = plt.figure(figsize=figsize)
        self.gs = gridspec.GridSpec(grid_shape[0], grid_shape[1], width_ratios=width_ratios, height_ratios=height_ratios)
        self.ax_3d = self.fig.add_subplot(self.gs[0, 0], projection='3d')
        self.ax_3d.set_xlim([llim, hlim])
        self.ax_3d.set_ylim([llim, hlim])
        self.ax_3d.set_zlim([llim, hlim])
        self.ax_3d.set_xlabel('X')
        self.ax_3d.set_ylabel('Y')
        self.ax_3d.set_zlabel('Z')
        self.llim = llim
        self.hlim = hlim
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

    def plotKeypoints(self, fk_dict: dict, parent_joint: str, pause=0.0, cla=False) -> np.ndarray:
        if cla:
            self.clear()

        self.ax_3d.set_xlabel('X')
        self.ax_3d.set_ylabel('Y')
        self.ax_3d.set_zlabel('Z')
        self.ax_3d.set_xlim([self.llim, self.hlim])
        self.ax_3d.set_ylim([self.llim, self.hlim])
        self.ax_3d.set_zlim([self.llim, self.hlim])

        last_center_point = self.root_center_point
        for joint, fk in fk_dict.items():
            # plt cs
            trans = fk['trans']
            self._plotKeypoint(trans, fk['rot_mat'], joint)

            # connect cs
            if last_center_point is not None:
                self.ax_3d.plot([last_center_point[0], trans[0]], [last_center_point[1], trans[1]], [last_center_point[2], trans[2]], color=fk['color'], linewidth=self.linewidth)
            last_center_point = trans

            if joint == parent_joint:
                # save last point before branching
                self.root_center_point = trans

        self.ax_3d.set_title('3D Keypoints')
        if pause > 0.0:
            plt.pause(pause)
        self.fig.canvas.draw()

        return np.array(self.fig.canvas.renderer.buffer_rgba())

if __name__ == "__main__":
    pass
