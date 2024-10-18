import numpy as np
import matplotlib
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


if __name__ == "__main__":
    pass
