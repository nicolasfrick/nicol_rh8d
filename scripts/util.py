import os, cv2
import numpy as np
from enum import Enum
from typing import Tuple
from cv2 import Rodrigues
from scipy.spatial.transform import Rotation as R

class RotTypes(Enum):
	RVEC='rvec'
	EULER='xyz_euler'
	MAT='matrix'

def getRotation(rot: np.ndarray, rot_type: RotTypes, out_type: RotTypes) -> np.ndarray:
	# convert to mat
	mat = rot
	if rot_type == RotTypes.RVEC:
		(mat, _) = Rodrigues(rot)
	elif rot_type == RotTypes.EULER:
		mat = R.from_euler('xyz', rot).as_matrix()
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
	elif out_type == RotTypes.MAT:
		pass
	else:
		raise NotImplementedError 
	return res

def euler2Matrix(euler: np.ndarray) -> np.ndarray:
	return R.from_euler('xyz', euler).as_matrix()

def pose2Matrix(tvec: np.ndarray, rot: np.ndarray, rot_t: RotTypes) -> np.ndarray:
	transformation_matrix = np.eye(4)
	transformation_matrix[:3, :3] = getRotation(rot, rot_t, RotTypes.MAT)
	transformation_matrix[:3, 3] = tvec
	return transformation_matrix

def invPersp(tvec: np.ndarray, rot: np.ndarray, rot_t: RotTypes) -> Tuple[np.ndarray, np.ndarray]:
	"""Apply the inversion to the given vectors [[R^-1 -R^-1*d][0 0 0 1]]"""
	mat = getRotation(rot, rot_t, RotTypes.MAT)
	mat = np.matrix(mat).T # orth. matrix: A.T = A^-1
	inv_tvec = -mat @ tvec # -R^-1*d
	inv_rot = getRotation(mat, RotTypes.MAT, rot_t)
	if rot_t != RotTypes.MAT:
		inv_rot = inv_rot.flatten()
	return np.array(inv_tvec.flat), inv_rot

def greenScreen(img: cv2.typing.MatLike):
	repl = np.ones(img.shape, dtype=np.float32) * 255
	u_green = np.array([104, 153, 70])
	l_green = np.array([30, 30, 0])
	
	mask = cv2.inRange(img, l_green, u_green)
	res = cv2.bitwise_and(img, img, mask = mask)
	f = img - res
	f = np.where(f == 0, repl, f)
	return f

if __name__ == "__main__":
	cv2.namedWindow("gs", cv2.WINDOW_NORMAL)
	img = cv2.imread(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets/detection/test_img.jpg'), cv2.IMREAD_COLOR)
	img = greenScreen(img)
	cv2.imshow("gs", img)
	while 1:
		if cv2.waitKey(1) == ord('q'):
			break
	cv2.destroyAllWindows()
