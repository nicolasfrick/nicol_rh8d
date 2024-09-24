import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to plot a single pose with 3D axes
def plot_pose(ax, position, rotation_matrix, label=''):
    # Origin (position of the 6D pose)
    origin = np.array(position)

    # Define unit vectors for the axes in the local frame
    x_axis = rotation_matrix[:, 0]
    y_axis = rotation_matrix[:, 1]
    z_axis = rotation_matrix[:, 2]

    # Scale for visual clarity (you can adjust this scale factor)
    scale = 0.1

    # Plot the X, Y, Z axes in red, green, blue respectively
    ax.quiver(origin[0], origin[1], origin[2], x_axis[0], x_axis[1], x_axis[2], color='r', length=scale, label=f'{label}_x')
    ax.quiver(origin[0], origin[1], origin[2], y_axis[0], y_axis[1], y_axis[2], color='g', length=scale, label=f'{label}_y')
    ax.quiver(origin[0], origin[1], origin[2], z_axis[0], z_axis[1], z_axis[2], color='b', length=scale, label=f'{label}_z')

# Function to update the plot dynamically
def visualize_pose_over_time(positions, rotation_matrices, time_steps):
    # Create a figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Setting plot limits (adjust according to your data)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Loop over each time step and plot the pose
    for i in range(len(time_steps)):
        ax.cla()  # Clear the current axes
        
        # Set axes labels and limits again
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        
        # Plot the pose at the current time step
        plot_pose(ax, positions[i], rotation_matrices[i], label=f'Time {time_steps[i]}')

        # Add a title indicating the current time step
        ax.set_title(f'Pose at Time {time_steps[i]}')

        # Draw the plot and pause briefly to create animation effect
        plt.draw()
        plt.pause(0.1)

    # Keep the plot open after the animation is done
    plt.show()

# Example usage:
# positions: List of 3D positions [[x1, y1, z1], [x2, y2, z2], ...]
# rotation_matrices: List of 3x3 rotation matrices at each time step
# time_steps: Corresponding time steps for each pose

# Sample data
positions = [[0, 0, 0], [0.1, 0.1, 0], [0.2, 0.1, 0.1], [0.3, 0.1, 0.1]]  # Sample positions
rotation_matrices = [np.eye(3)] * len(positions)  # Identity rotation matrices as an example
time_steps = [0, 1, 2, 3]  # Example time steps

visualize_pose_over_time(positions, rotation_matrices, time_steps)