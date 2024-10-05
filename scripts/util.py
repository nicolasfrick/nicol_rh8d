import numpy as np
from enum import Enum
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
