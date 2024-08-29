
import numpy as np
from scipy.spatial.transform import Rotation as R

# world -> base link
w_xyz = [0,0,0.8]
w_rpy = [0,0,0]
W = np.eye(4)
W[:3, :3] = R.from_euler('xyz', np.array(w_rpy)).as_matrix()
W[:3, 3] = w_xyz

# base link -> marker
m_xyz = [0.673, -0.104, 0]
m_rpy = [0, 0, np.pi/2]
M = np.eye(4)
M[:3, :3] = R.from_euler('xyz', np.array(m_rpy)).as_matrix()
M[:3, 3] = m_xyz

# world -> marker
T = W @ M
res_rot = R.from_matrix(T[:3,:3])
res_trans = T[:3, 3]
print(res_trans)
print(res_rot.as_euler('xyz', degrees=True))
print()

# [ 0.673 -0.104  0.8  ]
# [ 0.  0. 90.]

# marker -> cam
c_xyz = [0.142, 0.027, 1.163]
c_rpy = [3.07,-0.07,-3.13]
C = np.eye(4)
C[:3, :3] = R.from_euler('xyz', np.array(c_rpy)).as_matrix()
C[:3, 3] = c_xyz

# world -> marker
T = W @ M @ C
res_rot = R.from_matrix(T[:3,:3])
res_trans = T[:3, 3]
print(res_trans)
print(res_rot.as_euler('xyz', degrees=True))

# [0.646 0.038 1.963] # ?
# [175.89804311  -4.01070457 -89.33578988]  # ok

# all tf are extrinsic (lower case) xyz euler angles 
