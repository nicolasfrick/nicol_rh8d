#!/usr/bin/env python3

import os
import glob
import torch
import joblib
import optuna
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Optional, Union, Tuple
from util import *

# torch at cuda or cpu
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DataLoader():
	"""
		Load training data and apply normalization.
		Split normalized data and move to DEVICE.

		@param data_file
		@type str
		@param test_size
		@type float
		@param random_state
		@type int
		@param validation_size
		@type float
		@param norm_quats 
		@type bool
		@param trans_norm 
		@type Normalization
		@param input_norm 
		@type Normalization
		@param target_norm 
		@type Normalization

	Steps for Inference with Normalized Inputs
		Save the normalization parameters: During training, keep track of the mean, standard deviation, minimum, and maximum values used for normalization.
		Normalize the input data: Before feeding the input data into the model, apply the same normalization using the saved parameters.
		Make predictions: Use the normalized input data for the model inference.
		Inverse transform the output (if needed): If your output targets were also normalized, apply the inverse transformation to interpret the results in the original scale.

	"""

	def __init__(self,
							data_file: str,
							test_size: Optional[float]=0.2, 
							validation_size: Optional[float]=0.2,
							random_state: Optional[int]=42,
							norm_quats: Optional[bool]=True,
							trans_norm: Optional[Normalization]=Normalization.Z_SCORE,
							input_norm: Optional[Normalization]=Normalization.Z_SCORE,
							target_norm: Optional[Normalization]=Normalization.Z_SCORE,
							) -> None:
		
		# path names
		split = data_file.split('/')
		self.file_name =  split[-1]
		self.folder_pth = "/".join(split[: -1])
		self.name = self.file_name.replace('.json', '')
		self.scaler_pth = self.folder_pth + "/" + self.name + "_scalers/"
		if not os.path.exists(self.scaler_pth):
			os.mkdir(self.scaler_pth)
			print("Making directory", self.scaler_pth)

		# dataframe
		data_pth = os.path.join(DATA_PTH, 'keypoint/train', data_file)
		df = pd.read_json(data_pth, orient='index')
		cols = df.columns.tolist()

		print("Loading data for net", self.name, "with columns", cols)

		# conditions
		self.test_size = test_size
		self.random_state = random_state
		self.validation_size = validation_size
		self.norm_quats = norm_quats
		self.trans_norm = trans_norm
		self.input_norm = input_norm
		self.target_norm = target_norm
		
		# input data
		self.X_cmds = {}
		self.X_dirs = {}
		self.X_quats = None
		# target data
		self.y_angles = {}
		self.y_trans = None
		# scalers
		self.X_cmds_scalers = {}
		self.y_angles_scalers = {}
		self.y_trans_scalers = {}
		
		# load, normalize, split and move data
		self.prepare(df, cols)

		# save fitted scalers
		self.saveScalers()

	def prepare(self, df: pd.DataFrame, cols: list) -> None:
		# load and normalize data
		self.loadData(df, cols)
		print( "Added", len(self.X_cmds.keys()), "cmd data frames,",  len(self.X_dirs.keys()), "dir data frames,", \
					len(self.y_angles.keys()), "target angle data frames, one input orientation and target translation data frame.\n")
		
		# stack training data
		(X, y) = self.stackData()
		# split and move data
		self.splitData(X, y)
		print("Loaded features", self.feature_names, "and targets", self.target_names, "to device:", DEVICE)
		print(f"Splitted {self.num_samples} samples into {len(self.X_train_tensor)} training ({100*(1-self.test_size)}%),")
		print(f"{len(self.X_test_tensor)} test ({100*(1-self.test_size)*(self.validation_size)}%) and {len(self.X_val_tensor)} validation ({100*(self.test_size)*(self.validation_size)}%) samples")

	def stackData(self) -> Tuple[np.ndarray, np.ndarray]:
		"""Stack normalized data to a feature vector X and
			  a target vector y.
		"""
		# quaternions first
		X = self.X_quats.copy()
		self.feature_names = ["x", "y", "z", "w"]
		# stack commands
		for name, X_cmd in self.X_cmds.items():
			self.feature_names.append(name)
			X = np.hstack([X, X_cmd.reshape(-1, 1)])
		# stack directions
		for name, X_dir in self.X_dirs.items():
			self.feature_names.append(name)
			X = np.hstack([X, X_dir.reshape(-1, 1)])

		# translation first
		y = self.y_trans.copy()
		self.target_names =  ["x", "y", "z"]
		# stack angles
		for name, y_angle in self.y_angles.items():
			self.target_names.append(name)
			y = np.hstack([y, y_angle.reshape(-1, 1)])

		self.num_features = len(self.feature_names)
		self.num_targets = len(self.target_names)
		self.num_samples = len(X)

		return X, y

	def splitData(self, X: np.ndarray, y: np.ndarray) -> None:
		"""Split features X and targets y into a  train, test 
			  and validation set and move to the present DEVICE.
		"""
		# split off training data
		X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
		# split off test and validation data
		X_test, X_val, y_test, y_val = train_test_split(X_tmp, y_tmp, test_size=self.validation_size, random_state=self.random_state)

		# map features to tensors and move to device
		self.X_train_tensor = torch.from_numpy(X_train).to(DEVICE)
		self.X_test_tensor = torch.from_numpy(X_test).to(DEVICE)
		self.X_val_tensor = torch.from_numpy(X_val).to(DEVICE)

		# map targets to tensors and move to device
		self.y_train_tensor = torch.from_numpy(y_train).to(DEVICE)
		self.y_test_tensor = torch.from_numpy(y_test).to(DEVICE)
		self.y_val_tensor = torch.from_numpy(y_val).to(DEVICE)

	def loadData(self, df: pd.DataFrame, cols: list) -> None:
		"""Fill datastructures with traning data, possibly apply normalization.
		"""
		for col in cols:
			data = df[col].values

			# process input commands
			if 'cmd' in col:
				normalized_data = self.normalizeData(data, self.input_norm, self.X_cmds_scalers, col)
				self.X_cmds.update( {col: normalized_data.flatten()} )
			# process input direction
			elif 'dir' in col:
				self.X_dirs.update( {col: np.array(data, dtype=np.int32)} )
			# process input orientation
			elif 'quat' in col:
				quats = np.array([list(q) for q in data], dtype=np.float32)
				self.X_quats = self.normQuaternions(quats) if self.norm_quats else quats

			# process target angles
			elif 'angle' in col:
				normalized_data = self.normalizeData(data, self.target_norm, self.y_angles_scalers, col)
				self.y_angles.update( {col: normalized_data.flatten()} )
			# process target trans
			elif 'trans' in col:
				self.y_trans  = self.normalizeData(np.array([list(t) for t in data], dtype=np.float32), self.trans_norm, self.y_trans_scalers, col, None)
			else:
				raise NotImplementedError
			
	def normalizeData(self, 
				   						data: np.ndarray, 
				   						norm: Normalization, 
										scaler_dict: dict, 
										scaler_name: str,
										reshape: Tuple=(-1, 1)) -> np.ndarray:
		
		normalized_data = data.copy()
		if reshape is not None: 
			# reshape for single feature
			normalized_data = normalized_data.reshape(reshape[0], reshape[1])

		if norm == Normalization.Z_SCORE:
			(scaler, normalized_data) = self.zScoreScaler(normalized_data)
			scaler_dict.update( {scaler_name: scaler} )
		elif norm == Normalization.MINMAX_POS:
			(scaler, normalized_data) = self.minMaxScaler(normalized_data, (0, 1))
			scaler_dict.update( {scaler_name: scaler} )
		elif norm == Normalization.MINMAX_CENTERED:
			(scaler, normalized_data) = self.minMaxScaler(normalized_data, (-1, 1))
			scaler_dict.update( {scaler_name: scaler} )
		elif norm == Normalization.NONE:
			pass
		else: 
			raise NotImplementedError

		return normalized_data

	def normQuaternions(self, quats: np.ndarray) -> np.ndarray:
		"""Make sure data is unit quaternions"""
		norm = np.linalg.norm(quats)
		return quats / norm
				
	def zScore(self, data: np.ndarray) -> np.ndarray:
		"""z=(x-mu)/sigma, assume normally distributed data
				It helps the model converge faster during training.
				It prevents features with larger magnitudes from dominating the learning process.
				It standardizes the data distribution, making it easier for neural networks to learn.
		"""
		data_reshaped = data.reshape(-1, 1)
		scaler = StandardScaler()
		z_scores = scaler.fit_transform(data_reshaped).flatten()
		return z_scores
	
	def minMaxCentered(self, data: np.ndarray) -> np.ndarray:
		"""  Modified version of min-max scaling that 
				rescales the data to the range [-1, 1].
				When working with centered data or using tanh activation.
		"""
		min_val = np.min(data)
		max_val = np.max(data)
		scaled_values = 2 * (data - min_val) / (max_val - min_val) - 1
		return scaled_values
	
	def minMaxPos(self, data: np.ndarray) -> np.ndarray:
		"""Min-max scaling that rescales the data to the range [0, 1].
			 When bounded output between 0 and 1 is needed (e.g., sigmoid activation)
			 and bounds are known.
		"""
		min_val = np.min(data)
		max_val = np.max(data)
		scaled_values = (data - min_val) / (max_val - min_val)
		return scaled_values
	
	def zScoreScaler(self, data: np.ndarray) -> Tuple[StandardScaler, np.ndarray]: 
		scaler = StandardScaler()
		normalized_data = scaler.fit_transform(data)
		return scaler, normalized_data
	
	def minMaxScaler(self, data: np.ndarray, feature_range: Tuple) -> Tuple[MinMaxScaler, np.ndarray]: 
		scaler = MinMaxScaler(feature_range=feature_range)
		scaled_values = scaler.fit_transform(data) 
		return scaler, scaled_values
	
	def normalizeInputData(self, scaler: Union[MinMaxScaler, StandardScaler], data: np.ndarray) -> np.ndarray: 
		normalized_data = scaler.transform(data) 
		return normalized_data
	
	def denormalizeOutputData(self, scaler: Union[MinMaxScaler, StandardScaler], data: np.ndarray) -> np.ndarray:
		denormalized_data = scaler.inverse_transform(data) 
		return denormalized_data
	
	def saveScalers(self) -> None:
		for name, scaler in self.X_cmds_scalers.items():
			pth = self.scaler_pth + name + ".pkl"
			joblib.dump(scaler, pth)
		for name, scaler in self.y_angles_scalers.items():
			pth = self.scaler_pth + name + ".pkl"
			joblib.dump(scaler, pth)
		for name, scaler in self.y_trans_scalers.items():
			pth = self.scaler_pth + name + ".pkl"
			joblib.dump(scaler, pth)

	def loadScalers(self, pth: str) -> Union[MinMaxScaler, StandardScaler]:
		scaler = joblib.load(pth)
		return scaler
		
class MLP(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
		super(MLP, self).__init__()
		layers = []
		layers.append(nn.Linear(input_dim, hidden_dim))
		layers.append(nn.ReLU())
		for _ in range(num_layers - 1):
			layers.append(nn.Linear(hidden_dim, hidden_dim))
			layers.append(nn.ReLU())
		layers.append(nn.Linear(hidden_dim, output_dim))
		self.model = nn.Sequential(*layers)
	
	def forward(self, x):
		return self.model(x)
	
class Trainer():

	def __init__(self):
		pass

	# Objective function for Optuna
	def objective(self, trial):
		# Suggest hyperparameters
		hidden_dim = trial.suggest_int('hidden_dim', 32, 512, step=32)
		num_layers = trial.suggest_int('num_layers', 1, 3)
		learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
		
		# Instantiate the model and move to device
		model = MLP(input_dim=2, hidden_dim=hidden_dim, output_dim=6, num_layers=num_layers).to(DEVICE)
		
		# Criterion and Optimizer
		criterion = nn.MSELoss()
		optimizer = optim.Adam(model.parameters(), lr=learning_rate)
		
		# TensorBoard logging setup
		run_name = f"trial_{trial.number}_hidden_{hidden_dim}_layers_{num_layers}_lr_{learning_rate:.4f}"
		writer = SummaryWriter(log_dir=os.path.join(MLP_LOG_PTH, run_name))
		
		# Training loop
		model.train()
		epochs = 50
		for epoch in range(epochs):
			optimizer.zero_grad()
			outputs = model(X_train_tensor)
			loss = criterion(outputs, y_train_tensor)
			loss.backward()
			optimizer.step()
			
			# Log the training loss
			writer.add_scalar('Loss/train', loss.item(), epoch)
			
			# Validation
			model.eval()
			with torch.no_grad():
				val_outputs = model(X_val_tensor)
				val_loss = criterion(val_outputs, y_val_tensor).item()
				writer.add_scalar('Loss/val', val_loss, epoch)
		
		writer.close()
		return val_loss

	def run(self):
		# Run Optuna optimization
		study = optuna.create_study(direction='minimize')
		study.optimize(self.objective, n_trials=2000)

		# Retrieve the best trial
		print('Best trial:')
		trial = study.best_trial
		print('  Value: ', trial.value)
		print('  Params: ')
		for key, value in trial.params.items():
			print(f'    {key}: {value}')

if __name__ == '__main__':
	pattern = os.path.join(DATA_PTH, 'keypoint/train/10013', f'*poly*')
	data_files = glob.glob(pattern, recursive=False)
	dl = DataLoader(data_file=data_files[0])
