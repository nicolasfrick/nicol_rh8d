import os
import glob
import torch
import numpy as np
from typing import Union
from captum.attr import IntegratedGradients
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from train import DEVICE, MLP, TrainingData as TD
from plot_record import visFeatCont, visActivationHist
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
		del model_state['cmd_map'] # rm obsolete entry
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
		print()

	def loadScaler(self, pth: str) -> Union[MinMaxScaler, StandardScaler]:
		return TD.loadScaler(pth)
	
	def normalizeInput(self, scaler: Union[MinMaxScaler, StandardScaler], data: np.ndarray) -> np.ndarray:
		return TD.scaleInputOutputData(scaler, data)
	
	def denormalizeOutput(self, scaler: Union[MinMaxScaler, StandardScaler], data: np.ndarray) -> np.ndarray:
		return TD.denormalizeOutputData(scaler, data)
	
	def mapInput(self, input: dict) -> np.ndarray:
		assert(len(input.keys()) ==  self.len_features)
		X = np.array([np.nan for _ in range(self.len_features)], dtype=np.float32)
		# TODO: improve
		data = np.array([input['fx'], input['fy'], input['fz']], dtype=np.float32).reshape(1, -1) 
		Fs = self.normalizeInput(self.scaler_dct["force"], data).flatten()

		# map input to tensor
		for idx, (key, val) in enumerate(input.items()):
			if key in QUAT_COLS:
				# index 0 to 3
				assert(key == QUAT_COLS[idx])
				X[idx] = val

			elif key in FORCE_COLS and key in self.feature_names:
				# index 4 to 6
				fid = idx-len(QUAT_COLS)
				assert(key == FORCE_COLS[fid])
				X[idx] = Fs[fid]

			elif 'cmd' in key:
				# index 4 to n
				assert(key == self.feature_names[idx])
				# normalize
				data = np.array([val], dtype=np.float32).reshape(1, -1)
				X[idx] = self.normalizeInput(self.scaler_dct[key], data)

			elif 'dir' in key:
				assert(key == self.feature_names[idx])
				X[idx] = val

			else:
				raise NotImplementedError(f"Unknown key {key} for inference input map!")
			
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

			else:
				raise NotImplementedError(f"Unknown key {name} for inference output map!")

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
		X = torch.from_numpy(X).float().to(DEVICE)

		with torch.inference_mode():
			y = self.model(X)
			output = y.detach().cpu().numpy()

			return self.mapOutput(output)
		
	def visFeatureCont(self, input: dict, plot: bool = True) -> np.ndarray:
		X = self.mapInput(input)
		X = torch.from_numpy(X).float().to(DEVICE).requires_grad_(True)
		baseline = torch.zeros_like(X)  # zero input no actuation/forces

		# init
		ig = IntegratedGradients(self.model)

		# compute attributions for each output dimension
		num_outputs = self.len_targets 
		attributions_list = []
		for target_idx in range(num_outputs):
			attr = ig.attribute(X.unsqueeze(0), baselines=baseline.unsqueeze(0), target=target_idx)
			# [1, num_features] -> [num_features]
			attr = attr.squeeze(0).abs().detach().cpu().numpy()
			attributions_list.append(attr)

		# [num_features, num_outputs]
		attributions = np.stack(attributions_list, axis=1)

		if plot:
			visFeatCont(attributions, self.target_names, self.feature_names)

		return attributions
	
	def visActivations(self, inputs: list[dict]) -> dict:
		self.model.eval()
		activation_dict = {}

		def hook_fn(module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor, name: str) -> None:
			if name not in activation_dict:
				activation_dict[name] = []
			activation_dict[name].append(output.detach().cpu().numpy().flatten())

		handles = []
		for i, layer in enumerate(self.model.model):
			if isinstance(layer, torch.nn.PReLU):
				handle = layer.register_forward_hook(
					lambda m, i, o, n=f"Layer {i//3 + 1}": hook_fn(m, i, o, n)
				)
				handles.append(handle)

		# forward passes
		for input_dict in inputs:
			X = self.mapInput(input_dict)
			X = torch.from_numpy(X).float().to(DEVICE)
			self.model(X)  # hooks capture activations

		# rm hooks
		for handle in handles:
			handle.remove()

		# utilization stats
		stats = {}
		for layer, acts in activation_dict.items():
			acts_flat = np.concatenate(acts)
			mean_abs = np.mean(np.abs(acts_flat))
			std_abs = np.std(np.abs(acts_flat))
			# percentage of activations with low magnitude 
			low_magnitude = np.sum(np.abs(acts_flat) < 0.01) / len(acts_flat) * 100
			stats[layer] = {"mean_abs": mean_abs, "std_abs": std_abs, "low_magnitude_percent": low_magnitude}

		return {"activations": activation_dict, "stats": stats}
		
def testSingle(idx: int) -> None:
	infer = InferMLP(os.path.join(MLP_CHKPT_PTH, 'rh8d_all/validation/final_best_model_epoch=299_train_loss=0.009.ckpt'),
				  					 os.path.join(MLP_SCLRS_PTH, 'rh8d_all'),)
	
	dataset = pd.read_json(os.path.join(TRAIN_PTH, 'config_processed_dense/all/rh8d_all.json'),orient='index')
	row:pd.Series = dataset.loc[idx]
	
	input = {}
	# quaternions 1st
	quat = row.loc['quat']
	for idx, q in enumerate(QUAT_COLS):
		input.update( {q: quat[idx]} )
	# force 2nd
	F = tfForce(quat)
	for idx, f in enumerate(FORCE_COLS):
		input.update( {f: F[idx]} )
	# cmd 3nd, dir 4th
	for key, val in row.items():
		if 'cmd' in key:
			input.update( {key : val} )
		elif 'dir' in key:
			input.update( {key: val} )

	prediction = infer.forward(input)

	for key, val in prediction.items():
		print(key, "prediction error:", abs(val - row.loc[key.replace('joint', 'angle')]))

	infer.visFeatureCont(input)

def testAll(start_idx: int=0, end_idx: int=0) -> None:
	infer = InferMLP(os.path.join(MLP_CHKPT_PTH, 'rh8d_all/validation/final_best_model_epoch=199_train_loss=0.018.ckpt'),
				  					 os.path.join(MLP_SCLRS_PTH, 'rh8d_all'),)
	
	dataset = pd.read_json(os.path.join(TRAIN_PTH, 'config_processed_dense/all/rh8d_all.json'),orient='index')
	print("Testing model from index", start_idx, "to",  len(dataset) if end_idx == 0 else end_idx)
	inputs = []
	errors_all = {}
	attributions_all = []
	for idx in range(start_idx, len(dataset) if end_idx == 0 else end_idx):
		print(f"\r{idx:5}", end="", flush=True)
		row: pd.Series = dataset.loc[idx]
		input = {}
		# quaternions 1st
		quat = row.loc['quat']
		for idx, q in enumerate(QUAT_COLS):
			input.update( {q: quat[idx]} )
		# force 2nd
		F = row.loc['force']
		for idx, f in enumerate(FORCE_COLS):
			input.update( {f: F[idx]} )
		# cmd 3nd, dir 4th
		for key, val in row.items():
			if 'cmd' in key:
				input.update( {key : val} )
			elif 'dir' in key:
				input.update( {key: val} )

		inputs.append(input)
		prediction = infer.forward(input)

		for key, val in prediction.items():
			err = abs(val - row.loc[key.replace('joint', 'angle')])
			if errors_all.get(key) is None:
				errors_all.update( {key: [err]} )
			else:
				errors_all[key].append(err)

		attributions = infer.visFeatureCont(input, plot=False)
		attributions_all.append(attributions)

	print("\ndone")

	# error
	for key, err in errors_all.items():
		print(key, "mean error", np.mean(err))

	# contributions
	attributions_mean = np.mean(attributions_all, axis=0)
	visFeatCont(attributions_mean, infer.target_names, infer.feature_names)

	# activations
	results = infer.visActivations(inputs)
	visActivationHist(results["activations"])
	print("Utilization Stats:")
	for layer, stat in results["stats"].items():
		print(f"{layer}: Mean |Act| = {stat['mean_abs']:.4f}, Std |Act| = {stat['std_abs']:.4f}, "
			  f"Low Magnitude (< 0.01) = {stat['low_magnitude_percent']:.2f}%")

if __name__ == "__main__":
	testAll()
