import os
import glob
import torch
import numpy as np
from typing import Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from train import DEVICE, MLP, TrainingData as TD
from util import *

class InferMLP():
	""" 
			Inference class for mlp models using plain torch.

			@param checkpoint_path
			@type str
			@param scalers_path
			@type str

	"""

	def __init__(self,
							checkpoint_path: str,
							scalers_path: str,
							) -> None:
		
		# load checkpoint
		checkpoint = torch.load(checkpoint_path, 
														weights_only=False,
						  								map_location= DEVICE,
														)
		
		# setup model
		hparams = checkpoint['hyper_parameters']
		self.model = MLP(input_dim=hparams['input_dim'],
											hidden_dim=hparams['hidden_dim'],
											output_dim=hparams['output_dim'],
											num_layers=hparams['num_layers'],
											dropout_rate=hparams['dropout_rate'],
											)
		# load weights
		model_state = {key.replace("model.model.", "model."): value for key, value in checkpoint['state_dict'].items()}
		self.model.load_state_dict(model_state)
		# setup for inference
		self.model.to(DEVICE)
		self.model.eval()

		# load additionally saved mapping names
		self.feature_names = hparams['feature_names']
		self.target_names = hparams['target_names']
		self.len_features = len(self.feature_names)
		self.len_targets = len(self.target_names)
		
		# load scalers
		self.scaler_dct = {}
		pattern = os.path.join(scalers_path, f'*.pkl*')
		scaler_files = glob.glob(pattern, recursive=False)
		for fl in scaler_files:
			fn = os.path.basename(fl)
			fn = os.path.splitext(fn)[0]
			self.scaler_dct.update( {fn: self.loadScaler(fl)} )

		print("Instantiated model for inference with features:")
		print(self.feature_names)
		print("and targets")
		print(self.target_names)

	def loadScaler(self, pth: str) -> Union[MinMaxScaler, StandardScaler]:
		return TD.loadScaler(pth)
	
	def normalizeInput(self, scaler: Union[MinMaxScaler, StandardScaler], data: np.ndarray) -> np.ndarray:
		return TD.scaleInputOutputData(scaler, data)
	
	def denormalizeOutput(self, scaler: Union[MinMaxScaler, StandardScaler], data: np.ndarray) -> np.ndarray:
		return TD.denormalizeOutputData(scaler, data)
	
	def mapInput(self, input: dict) -> np.ndarray:
		assert(len(input.keys()) ==  self.len_features)
		X = np.zeros(self.len_features, dtype=np.float32)

		# map input to tensor
		for idx, (key, val) in enumerate(input.items()):
			if key in QUAT_COLS:
				# index 0 to 3
				assert(key == QUAT_COLS[idx])
				X[idx] = val

			elif 'joint' in key:
				# index 4 to n
				joint_id = key.replace('joint', 'cmd')
				assert(joint_id == self.feature_names[idx])
				# normalize
				data = np.array([val], dtype=np.float32).reshape(1, -1)
				X[idx] = self.normalizeInput(self.scaler_dct[joint_id], data)

			elif 'dir' in key:
				assert(key == self.feature_names[idx])
				X[idx] = val

		return X
	
	def mapOutput(self, output: np.ndarray) -> dict:
		output_dict = {}
		for idx, val in enumerate(output):
			name = self.target_names[idx]

			# process ordered translations
			if 'x_' in name:
				#  translation x -> new entry
				output_dict.update( {name.replace('x_', 'trans'): [val]} )
			elif 'y_' in name:
				output_dict[name.replace('y_', 'trans')].append(val)
			elif 'z_' in name:
				name = name.replace('z_', 'trans')
				output_dict[name].append(val)
				# scale final values [x,y,z]
				data = np.array(output_dict[name], dtype=np.float32).reshape(1, -1) 
				data = self.denormalizeOutput(self.scaler_dct[name], data)
				output_dict[name] = data.flatten()
			# process angles
			elif 'angle' in name:
				data = np.array([val], dtype=np.float32).reshape(1, -1)
				val_scaled = self.denormalizeOutput(self.scaler_dct[name], data)
				output_dict.update( {name.replace('angle', 'joint'): val_scaled.flatten()[0]} )

		return output_dict

	def forward(self, input: dict) -> dict:
		""" Predict joint angles and tip link positions 
			  from actuator angles, direction and eef orientation.

							                [quaternions, actuator commands, direction of rotations]
			  @param input [x, y, z, w, cmd7, cmd8, cmdT0, cmdT1, cmdI1, cmdM1, cmdL1R1, dirT0, dirT1, dirI1, dirM1, dirL1R1]
			  @type dict
			  								  [translations, joint angles]
			  @param output [transT, transI, transM, transR, transL, 
			  								   joint7, joint8, jointT0, jointT1, jointT2, jointT3, jointI1, jointI2, jointI3, 
											   jointM1, jointM2, jointM3, jointR1, jointR2, jointR3, jointL1, jointL2, jointL3]
			  @type dict

		"""
		X = self.mapInput(input)
		X =  torch.from_numpy(X).float().to(DEVICE)

		with torch.inference_mode():
			y = self.model(X)
			output = y.detach().cpu().numpy()

			return self.mapOutput(output)
		
def test(idx: int) -> None:
	infer = InferMLP(os.path.join(MLP_CHKPT_PTH, 'rh8d_all/01_06_15_01/final_best_model_epoch=24_train_loss=0.033.ckpt'),
				  					 os.path.join(MLP_SCLRS_PTH, 'rh8d_all'),)
	
	dataset = pd.read_json(os.path.join(TRAIN_PTH, 'config_processed/rh8d_all.json'),orient='index')
	row:pd.Series = dataset.loc[idx]
	
	input = {}
	# quaternions 1st
	quat = row.loc['quat']
	for idx, q in enumerate(QUAT_COLS):
		input.update( {q: quat[idx]} )
	# cmd 2nd, dir 3rd
	for key, val in row.items():
		if 'cmd' in key:
			input.update( {key.replace('cmd', 'joint') : val} )
		elif 'dir' in key:
			input.update( {key: val} )

	prediction = infer.forward(input)

	for key, val in prediction.items():
		print(key, "prediction error:", abs(val - row.loc[key]))

if __name__ == "__main__":
	test(8555)
