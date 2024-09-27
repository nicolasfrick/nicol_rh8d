import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def plotPose(ax, position, rotation_matrix, label=''):
    origin = np.array(position)
    x_axis = rotation_matrix[:, 0]
    y_axis = rotation_matrix[:, 1]
    z_axis = rotation_matrix[:, 2]
    scale = 0.1
    ax.quiver(origin[0], origin[1], origin[2], x_axis[0], x_axis[1], x_axis[2], color='r', length=scale, label=f'{label}_x')
    ax.quiver(origin[0], origin[1], origin[2], y_axis[0], y_axis[1], y_axis[2], color='g', length=scale, label=f'{label}_y')
    ax.quiver(origin[0], origin[1], origin[2], z_axis[0], z_axis[1], z_axis[2], color='b', length=scale, label=f'{label}_z')

def visPose(ax, pos, rot_mat, t, pause=0, cla=False, llim=-1, hlim=1):
    # clear axis
    if cla:
        ax.cla()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([llim, hlim])
    ax.set_ylim([llim, hlim])
    ax.set_zlim([llim, hlim])
    plotPose(ax, pos, rot_mat, label=f'Time {t}')
    ax.set_title(f'Pose at Time {t}')
    plt.draw()
    if pause > 0.0:
        plt.pause(pause)

def setPosePlot(llim=-1, hlim=1):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([llim, hlim])
    ax.set_ylim([llim, hlim])
    ax.set_zlim([llim, hlim])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    return fig, ax

if __name__ == "__main__":
    fig, ax = setPosePlot()

    for i in range(100):
        sw = i % 2 == 0
        # cluster
        positions = np.random.uniform(0 if sw else -1, 1 if sw else 0, 3)
        euler =  np.random.uniform(0, 1, 3)
        m = R.from_euler('xyz', euler)
        visPose(ax, positions, m.as_matrix(), i, 0.01)
