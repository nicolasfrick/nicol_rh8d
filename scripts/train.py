#!/usr/bin/env python3

import os
import json
import glob
import torch
import joblib
import optuna
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
from optuna.trial import FrozenTrial
from optuna.storages import RDBStorage
from typing import Optional, Union, Tuple, List, Any
from torch.utils.tensorboard import SummaryWriter
from lightning.pytorch.trainer.states import TrainerFn
from torch.utils.data import DataLoader, TensorDataset
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from optuna.integration import PyTorchLightningPruningCallback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from util import *
from plot_record import *

# torch at cuda or cpu
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_DEV = torch.cuda.device_count()
DEVICES = list(range(NUM_DEV))
NAME_DEV = [torch.cuda.get_device_name(cnt) for cnt in range(NUM_DEV)]

class TrainingData():
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
						@param trans_norm 
						@type Normalization
						@param input_norm 
						@type Normalization
						@param target_norm 
						@type Normalization
						@param move_gpu
						@type bool
						@param kfold
						@type int

		"""

		def __init__(self,
								 data_file: str,
								 test_size: Optional[float] = 0.3,
								 validation_size: Optional[float] = 0.5,
								 random_state: Optional[int] = 42,
								 trans_norm: Optional[Normalization] = Normalization.Z_SCORE,
								 input_norm: Optional[Normalization] = Normalization.Z_SCORE,
								 target_norm: Optional[Normalization] = Normalization.Z_SCORE,
								 move_gpu: Optional[bool] = False,
								 kfold: Optional[int] = 0,
								 ) -> None:

				assert(isinstance(random_state, int)) # requires int
				if kfold > 0: # use kfold
					assert(kfold > 1) # min folds = 2

				# path names
				split = data_file.split('/')
				self.file_name = split[-1] # filename
				self.folder_pth = "/".join(split[: -1]) # parent dir
				self.name = self.file_name.replace('.json', '') # net name
				self.scaler_pth = os.path.join(MLP_SCLRS_PTH, self.name) # scalers dir

				# dataframe
				data_pth = os.path.join(DATA_PTH, 'keypoint/train', data_file)
				self.df = pd.read_json(data_pth, orient='index')
				self.cols = self.df.columns.tolist() # net config

				print("Loading data for net", self.name, "with columns", self.cols)

				# conditions
				self.test_size = test_size
				self.random_state = random_state
				self.validation_size = validation_size
				self.trans_norm = trans_norm
				self.input_norm = input_norm
				self.target_norm = target_norm
				self.move_gpu = move_gpu

				# input data
				self.X_quats = None # single
				self.X_cmds = {}
				self.X_dirs = {}
				# target data
				self.y_trans = {}
				self.y_angles = {}
				# scalers
				self.X_cmds_scalers = {}
				self.y_angles_scalers = {}
				self.y_trans_scalers = {}

				# tensors
				self.X_train_tensor = None
				self.X_test_tensor = None
				self.X_val_tensor = None
				self.y_train_tensor = None
				self.y_test_tensor = None
				self.y_val_tensor = None
				# kfold 
				self.use_kfold = kfold > 1
				self.num_folds = kfold
				self.train_index = None
				self.val_index = None
				self.X_train = None
				self.y_train = None
				# size
				self.num_features = 0
				self.num_targets = 0
				self.num_samples = 0

				# load, normalize, split and move data
				self.prepare(self.df, self.cols)

		@property
		def len_train_data(self) -> int:
			assert(self.X_train_tensor is not None)
			assert(self.y_train_tensor is not None)
			assert(len(self.X_train_tensor) == len(self.y_train_tensor))
			return len(self.X_train_tensor)
		
		@property
		def len_val_data(self) -> int:
			assert(self.X_val_tensor is not None)
			assert(self.y_val_tensor is not None)
			assert(len(self.X_val_tensor) == len(self.y_val_tensor))
			return len(self.X_val_tensor)
		
		@property
		def len_test_data(self) -> int:
			assert(self.X_test_tensor is not None)
			assert(self.y_test_tensor is not None)
			assert(len(self.X_test_tensor) == len(self.y_test_tensor))
			return len(self.X_test_tensor)

		def prepare(self, df: pd.DataFrame, cols: list) -> None:
				# load and normalize data
				self.loadData(df, cols)

				# stack training data
				(X, y) = self.stackData()
				print("Loaded features", self.feature_names, "and targets", self.target_names, f"to device: {DEVICE}" if self.move_gpu else "")

				if self.use_kfold:
					self.splitDataKFold(X, y)
					self.kf = KFold(self.num_folds, shuffle=False)
					index_tuples =  list(self.kf.split(self.X_train, self.y_train))
					(self.train_index, self.val_index) = ([t[0] for t in index_tuples], [t[1] for t in index_tuples])
					# print info
					print(f"Splitted {self.num_samples} samples into {len(self.X_train)} ({100*(1-self.test_size)}%) training and {len(self.X_test_tensor)} ({100*self.test_size}%) test samples for cross validation")
				else:
					# split, normalize and move data
					self.splitData3WayHoldout(X, y)
					# print info
					print(f"Splitted {self.num_samples} samples into {len(self.X_train_tensor)} training ({100*(1-self.validation_size)}%),")
					print(f"{len(self.X_test_tensor)} test ({100*self.validation_size*self.test_size}%) and {len(self.X_val_tensor)} validation ({100*self.validation_size - (100*self.validation_size*self.test_size)}%) samples")
				
				print(f"Command features normalization: {self.input_norm.value}, translation target normalization: {self.trans_norm.value}, angle target normalization: {self.target_norm.value}")

		def loadData(self, df: pd.DataFrame, cols: list) -> None:
			"""Fill datastructures with traning data.
			"""
			for col in cols:
					data = df[col].values

					# process input commands
					if 'cmd' in col:
							self.X_cmds.update({col: data})
					# process input direction
					elif 'dir' in col:
							self.X_dirs.update({col: np.array(data, dtype=np.int32)})
					# process input orientation
					elif 'quat' in col:
							self.X_quats  = np.array([list(q) for q in data], dtype=np.float32)
							assert(self.checkUnitQuaternions(self.X_quats)) # require unit quaternions
					# process target angles
					elif 'angle' in col:
							self.y_angles.update({col: data})
					# process target trans
					elif 'trans' in col:
							self.y_trans.update( {col: np.array([list(t) for t in data], dtype=np.float32)} )
					else:
							raise NotImplementedError
					
		def stackData(self) -> Tuple[np.ndarray, np.ndarray]:
				"""Stack data to a feature vector X [quats, cmd_1, .., cmd_n, dir_1, ..., dir_n] and
						a target vector y [[trans_1, ..., trans_n], angle_1, ..., angle_n].
				"""
				# features X:
				# quaternions first, always present
				X = self.X_quats.copy()
				self.feature_names = QUAT_COLS.copy()
				# stack commands
				for name, X_cmd in self.X_cmds.items():
						self.feature_names.append(name)
						X = np.hstack([X, X_cmd.reshape(-1, 1).copy()])
				# stack directions
				for name, X_dir in self.X_dirs.items():
						self.feature_names.append(name)
						X = np.hstack([X, X_dir.reshape(-1, 1).copy()])

				# targets y:
				y = None
				# translation first if present
				for name, y_trans in self.y_trans.items():
					trans_idx = name.replace('trans', '').replace('_', '')
					if y is None:
						# 1st elmt
						self.target_names = [*format_trans_cols(trans_idx)]
						y = y_trans.copy()
					else:
						self.target_names.extend(format_trans_cols(trans_idx))
						y = np.hstack([y, y_trans.copy()])

				# stack angles
				for name, y_angle in self.y_angles.items():
						if y is None:
								# no trans, angle1 is 1st elmt
								self.target_names = [name]
								y = y_angle.copy()
								y = y.reshape(-1, 1)
						else:
								self.target_names.append(name)
								y = np.hstack([y, y_angle.reshape(-1, 1).copy()])

				self.num_features = len(self.feature_names)
				self.num_targets = len(self.target_names)
				self.num_samples = len(X)

				return X, y

		def splitData3WayHoldout(self, X: np.ndarray, y: np.ndarray) -> None:
				"""Split features X and targets y into a  train, test and validation set, 
					 normalize train data and scale val and test data. If move_gpu is true, 
					 move tensors to the present DEVICE, else just map to torch tensors.
				"""

				# split off training data
				(X_train, X_tmp, y_train, y_tmp) = train_test_split(
						X, y, test_size=self.validation_size, shuffle=False if self.random_state == 0 else True, random_state=self.random_state)
				# split off test and validation data
				(X_val, X_test, y_val, y_test) = train_test_split(X_tmp, y_tmp, test_size=self.test_size, shuffle=False)
				
				# normalize training data
				self.normalizeTrainData(X_train, y_train)
				# scale validation data
				self.scaleValTestData(X_val, y_val)
				# scale test data
				self.scaleValTestData(X_test, y_test)

				# map features to tensors
				self.X_train_tensor = torch.from_numpy(X_train).float()
				self.X_test_tensor = torch.from_numpy(X_test).float()
				self.X_val_tensor = torch.from_numpy(X_val).float()
				# map targets to tensors
				self.y_train_tensor = torch.from_numpy(y_train).float()
				self.y_test_tensor = torch.from_numpy(y_test).float()
				self.y_val_tensor = torch.from_numpy(y_val).float()

				# move to gpu
				if self.move_gpu:
						# features
						self.X_train_tensor = self.X_train_tensor.to(DEVICE)
						self.X_test_tensor = self.X_test_tensor.to(DEVICE)
						self.X_val_tensor = self.X_val_tensor.to(DEVICE)
						# targets
						self.y_train_tensor = self.y_train_tensor.to(DEVICE)
						self.y_test_tensor = self.y_test_tensor.to(DEVICE)
						self.y_val_tensor = self.y_val_tensor.to(DEVICE)

		def splitDataKFold(self, X: np.ndarray, y: np.ndarray) -> None:
				"""  Split features X and targets y into a  train and test set. No 
						normalization is applied. If move_gpu is true, move tensors
						to the present DEVICE else use numpy arrays.
				"""

				# split
				(self.X_train, X_test, self.y_train, y_test) = train_test_split(
					X, y, test_size=self.test_size, shuffle=False if self.random_state == 0 else True, random_state=self.random_state)
				
				# normalize copies
				self.normalizeTrainData(self.X_train.copy(), self.y_train.copy())
				# scale test data
				self.scaleValTestData(X_test, y_test)
				# drop saved scalers
				self.clearScalers()

				# map features to tensors
				self.X_test_tensor = torch.from_numpy(X_test).float()
				self.y_test_tensor = torch.from_numpy(y_test).float()

				if self.move_gpu:
					# move to device
					self.X_test_tensor = self.X_test_tensor.to(DEVICE)
					self.y_test_tensor = self.y_test_tensor.to(DEVICE)

		def foldData(self, k: int) -> None:
			""" Set training and validation tensors from the the k'th fold of the training data.
					Train batches are normalized and validation batches are scaled.
			"""
			assert(k < self.kf.get_n_splits()) # k exceeds num splits

			if self.X_train is None or self.y_train is None or not self.use_kfold:
				raise RuntimeError("k-fold is not initialized!")
			
			# views -> don't change original data
			self.X_train_tensor = self.X_train[self.train_index[k]].copy()
			self.y_train_tensor = self.y_train[self.train_index[k]].copy()
			self.X_val_tensor = self.X_train[self.val_index[k]].copy()
			self.y_val_tensor = self.y_train[self.val_index[k]].copy()

			self.normalizeTrainData(self.X_train_tensor, self.y_train_tensor)
			self.scaleValTestData(self.X_val_tensor, self.y_val_tensor)
			self.clearScalers()

			self.X_train_tensor = torch.from_numpy(self.X_train_tensor).float()
			self.y_train_tensor = torch.from_numpy(self.y_train_tensor).float()
			self.X_val_tensor = torch.from_numpy(self.X_val_tensor).float()
			self.y_val_tensor = torch.from_numpy(self.y_val_tensor).float()

			if self.move_gpu:
				self.X_train_tensor = self.X_train_tensor.to(DEVICE)
				self.y_train_tensor = self.y_train_tensor.to(DEVICE)
				self.X_val_tensor = self.X_val_tensor.to(DEVICE)
				self.y_val_tensor = self.y_val_tensor.to(DEVICE)
						
		def normalizeTrainData(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
			""" Normalize the training data by the given method and save scaler for later use.
			"""
			# features:
			# normalize only cmds
			for idx, name in enumerate(self.X_cmds.keys()):
				# norm separated, quaternions always present
				assert(self.feature_names[len(QUAT_COLS)+idx] == name)
				X_train[:, len(QUAT_COLS)+idx] = self.normalizeData(X_train[:, len(QUAT_COLS)+idx].reshape(-1, 1), self.input_norm, self.X_cmds_scalers, name).flatten()

			# targets:
			start_idx = 0
			# normalize translation first if present
			for idx, name in enumerate(self.y_trans.keys()):
				tmp_end_idx = len(TRANS_COLS) + start_idx
				y_train[:, start_idx : tmp_end_idx] = self.normalizeData(y_train[:, start_idx : tmp_end_idx], self.trans_norm, self.y_trans_scalers, name)
				start_idx = tmp_end_idx

			# transform angles secondly
			for idx, name in enumerate(self.y_angles.keys()):
				assert(self.target_names[start_idx+idx] == name)
				y_train[:, start_idx+idx] = self.normalizeData(y_train[:, start_idx+idx].reshape(-1, 1), self.target_norm, self.y_angles_scalers, name).flatten()

		def scaleValTestData(self, X_val_test: np.ndarray, y_val_test: np.ndarray) -> None:
			"""	  Scale validation or test data by the scaler instance from train data normalization.
				 	Skip scaling if no scaler is present
			"""

			# features:
			if self.X_cmds_scalers:
				# scale only commands
				for idx, name in enumerate(self.X_cmds.keys()):
					# scale separated, quaternions always present
					X_val_test[:, len(QUAT_COLS)+idx] = self.scaleInputOutputData(self.X_cmds_scalers[name], X_val_test[:, len(QUAT_COLS)+idx].reshape(-1, 1)).flatten()

			# targets
			start_idx = 0
			# scale translation first if present
			if self.y_trans:
				if self.y_trans_scalers:
					# scale trans data
					for idx, name in enumerate(self.y_trans.keys()):
						tmp_end_idx = len(TRANS_COLS) + start_idx
						y_val_test[:, start_idx : tmp_end_idx] = self.scaleInputOutputData(self.y_trans_scalers[name], y_val_test[:, start_idx : tmp_end_idx])
						start_idx = tmp_end_idx
				else:
					# move index for angles
					start_idx = len(self.y_trans.keys()) * len(TRANS_COLS)

			# scale angles secondly
			if self.y_angles_scalers:
				for idx, name in enumerate(self.y_angles.keys()):
					y_val_test[:, start_idx+idx] = self.scaleInputOutputData(self.y_angles_scalers[name], y_val_test[:, start_idx+idx].reshape(-1, 1)).flatten()

		def normalizeData(self,
											data: np.ndarray,
											norm: Normalization,
											scaler_dict: dict,
											scaler_name: str,
											) -> np.ndarray:

				normalized_data = data.copy()

				if norm == Normalization.Z_SCORE:
						(scaler, normalized_data) = self.zScoreScaler(normalized_data)
						scaler_dict.update({scaler_name: scaler})
				elif norm == Normalization.MINMAX_POS:
						(scaler, normalized_data) = self.minMaxScaler(normalized_data, (0, 1))
						scaler_dict.update({scaler_name: scaler})
				elif norm == Normalization.MINMAX_CENTERED:
						(scaler, normalized_data) = self.minMaxScaler(
								normalized_data, (-1, 1))
						scaler_dict.update({scaler_name: scaler})
				elif norm == Normalization.NONE:
						pass
				else:
						raise NotImplementedError

				return normalized_data
		
		def checkUnitQuaternions(self, quats: np.ndarray) -> bool:
			norm = np.linalg.norm(quats, axis=1)
			return all(np.isclose(norm, 1.0))

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

		def scaleInputOutputData(self, scaler: Union[MinMaxScaler, StandardScaler], data: np.ndarray) -> np.ndarray:
				normalized_data = scaler.transform(data)
				return normalized_data

		def denormalizeOutputData(self, scaler: Union[MinMaxScaler, StandardScaler], data: np.ndarray) -> np.ndarray:
				denormalized_data = scaler.inverse_transform(data)
				return denormalized_data

		def saveScalers(self) -> None:
				# mkdir
				if not os.path.exists(self.scaler_pth):
						os.makedirs(self.scaler_pth, exist_ok=True)

				for name, scaler in self.X_cmds_scalers.items():
						pth = os.path.join(self.scaler_pth, name + ".pkl")
						joblib.dump(scaler, pth)
				for name, scaler in self.y_angles_scalers.items():
						pth = os.path.join(self.scaler_pth, name + ".pkl")
						joblib.dump(scaler, pth)
				for name, scaler in self.y_trans_scalers.items():
						pth = os.path.join(self.scaler_pth, name + ".pkl")
						joblib.dump(scaler, pth)

		def clearScalers(self) -> None:
				self.X_cmds_scalers.clear()
				self.y_angles_scalers.clear()
				self.y_trans_scalers.clear()

		def loadScaler(self, pth: str) -> Union[MinMaxScaler, StandardScaler]:
				scaler = joblib.load(pth)
				return scaler


class MLP(nn.Module):
		"""
						Create a standard feedforward net.

						@property input_dim
						@type int
						@property hidden_dim
						@type int
						@property output_dim
						@type int
						@property num_layers
						@type int
						@property dropout_rate
						@type Union[None, float]
						@property activation
						@type nn.Module
						@property activ_init
						@type float

		"""

		def __init__(self,
								 input_dim: int,
								 hidden_dim: int,
								 output_dim: int,
								 num_layers: int,
								 dropout_rate: Union[None, float],
								 activation: Optional[nn.Module]=nn.PReLU, 
								 activ_init: Optional[float]=0.25
								 ) -> None:

				super().__init__()

				layers: List[nn.Module] = []
				# input
				layers.append(nn.Linear(input_dim, hidden_dim))
				layers.append(activation(init=activ_init))
				if dropout_rate is not None:
						layers.append(nn.Dropout(p=dropout_rate))
				# hidden
				for _ in range(num_layers - 1):
						layers.append(nn.Linear(hidden_dim, hidden_dim))
						layers.append(activation(init=activ_init))
						if dropout_rate is not None:
								layers.append(nn.Dropout(p=dropout_rate))
				# output
				layers.append(nn.Linear(hidden_dim, output_dim))

				# model
				self.model = nn.Sequential(*layers)

				# rnd init weights
				self.model.apply(self.initWeights)

		# overwrite
		def forward(self, data: torch.Tensor) -> torch.Tensor:
				return self.model(data)

		def initWeights(self, module: nn.Module) -> None:
				if isinstance(module, nn.Linear):
						nn.init.kaiming_uniform_(module.weight, nonlinearity='leaky_relu') 
						if module.bias is not None:
								nn.init.zeros_(module.bias)

class WrappedMLP(pl.LightningModule):
		"""
						Lightning wrapper for the MLP impl.

						@property lr
						@type float
						@property weight_decay
						@type float

		"""

		def __init__(self,
								 input_dim: int,
								 hidden_dim: int,
								 output_dim: int,
								 num_layers: int,
								 dropout_rate: Union[None, float],
								 lr: float,
								 weight_decay: float,
								 ) -> None:

				super().__init__()

				self.model = MLP(input_dim=input_dim,
												 hidden_dim=hidden_dim,
												 output_dim=output_dim,
												 num_layers=num_layers,
												 dropout_rate=dropout_rate,
												 )

				# auto save hparams
				self.save_hyperparameters()
				# loss
				self.loss_fn = nn.MSELoss()
				# test stage
				self.test_loss = []

		# overwrite
		def forward(self, data: torch.Tensor) -> torch.Tensor:
				return self.model(data)

		# overwrite
		def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
				X, y = batch
				preds = self(X)
				loss = self.loss_fn(preds, y)
				self.log('train_loss', loss.item(), on_step=False,
								 on_epoch=True, prog_bar=True, sync_dist=True, )
				return loss

		# overwrite
		def validation_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
				X, y = batch
				preds = self(X)
				val_loss = self.loss_fn(preds, y)
				self.log('val_loss', val_loss.item(), on_step=False,
								 on_epoch=True, prog_bar=False, sync_dist=True, )
				self.log("hp_metric", val_loss.item(), on_step=False, on_epoch=True, sync_dist=True, )
				return val_loss
		
		# overwrite
		def test_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
				X, y = batch
				preds = self(X)
				test_loss = self.loss_fn(preds, y)
				# for final evaluation
				self.test_loss.append(test_loss.item())
				self.log('step_test_loss', test_loss.item(), on_step=True,
								 on_epoch=False, prog_bar=True, sync_dist=False, )
				return test_loss

		def on_test_epoch_end(self) -> None:
				self.log_dict( {'test_loss': np.mean(self.test_loss)} )

		# overwrite
		def configure_optimizers(self) -> optim.Optimizer:
				return optim.Adam(self.model.parameters(),
													lr=self.hparams.lr,
													weight_decay=self.hparams.weight_decay,
													)

class MLPDataModule(pl.LightningDataModule):
		""" Load data for lightning trainer.

				@param td
				@type TrainingData
				@param batch_size
				@type int
				@param bgd
				@type bool
				@param k
				@type Union[None, int]

		"""

		def __init__(self,
								td: TrainingData,
								 batch_size: int,
								 bgd: bool,
								 k: Optional[Union[None, int]] = None,
								 ) -> None:

				super().__init__()
				
				assert(not td.move_gpu) # not allowed here

				self.k = k
				self.td = td
				self.bgd = bgd
				self.batch_size = batch_size
				self.train_dataset = None
				self.val_dataset = None
				self.test_dataset = None


				# overwrite
		def setup(self, stage: Optional[str] = None) -> None:
				if stage is not None:

					if stage == TrainerFn.FITTING.value:
						# split and normalize data from k'th fold
						if self.td.use_kfold:
							self.td.foldData(self.k)

						# get tensors
						self.train_dataset = TensorDataset(self.td.X_train_tensor, self.td.y_train_tensor)
						self.val_dataset = TensorDataset(self.td.X_val_tensor, self.td.y_val_tensor)

						# set batch size
						if self.bgd:
							# batch gradient desc.
							self.train_batch_size = self.td.len_train_data
							self.val_batch_size = self.td.len_val_data
						else:
							# mini-batch gradient desc.
							self.train_batch_size = self.td.len_train_data if self.batch_size > self.td.len_train_data else self.batch_size
							self.val_batch_size = self.td.len_val_data if self.batch_size > self.td.len_val_data else self.batch_size

					elif stage == TrainerFn.TESTING.value:
						# always present
						self.test_dataset = TensorDataset(self.td.X_test_tensor, self.td.y_test_tensor)
						# set batch size
						if self.bgd:
							self.test_batch_size = self.td.len_test_data
						else:
							self.test_batch_size = self.td.len_test_data if self.batch_size > self.td.len_test_data else self.batch_size

					else:
						raise NotImplementedError(f"setup({stage}) is not implemented")

		# overwrite
		def train_dataloader(self) -> DataLoader:
				return DataLoader(self.train_dataset,
													batch_size=self.train_batch_size,
													shuffle=False,
													pin_memory=True,
													num_workers=7,
													persistent_workers=True,
													)

		# overwrite
		def val_dataloader(self) -> DataLoader:
				return DataLoader(self.val_dataset,
													batch_size=self.val_batch_size,
													shuffle=False,
													pin_memory=True,
													num_workers=7,
													persistent_workers=True,
													)

		# overwrite
		def test_dataloader(self) -> DataLoader:
				return DataLoader(self.test_dataset,
													batch_size=self.test_batch_size,
													shuffle=False,
													pin_memory=True,
													num_workers=7,
													persistent_workers=True,
													)

class Trainer():
		"""Training class for optimization task

										@param td
										@type TrainingData
										@param epochs
										@type int
										@param model_type
										@type Any
										@param num_layers
										@type Tuple
										@param hidden_dim
										@type Tuple
										@param learning_rate
										@type Tuple
										@param batch_size
										@type Tuple
										@property dropout_rate
										@type Union[None, Tuple]
										@param log_domain
										@type bool
										@param weight_decay
										@type Union[None, Tuple]
										@param pbar
										@type bool
										@param bgd
										@type bool
										@param mdelta_estop
										@type float
										@param save_chkpts
										@type bool
										@param metric
										@type str
										@param mode
										@type str

		"""

		LOW = 0
		HIGH = 1
		STEP = 2

		def __init__(self,
								 td: TrainingData,
								 epochs: Optional[int] = 50,
								 model_type: Optional[Any] = WrappedMLP,
								 num_layers: Optional[Tuple] = (1, 10, 1),
								 hidden_dim: Optional[Tuple] = (1, 32, 2),
								 batch_size: Optional[Tuple] = (16, 32, 64),
								 learning_rate: Optional[Tuple] = (1e-4, 1e-2, 1e-3),
								 dropout_rate: Optional[Union[None, Tuple]] = (0, 0.4, 0.01),
								 log_domain: Optional[bool] = False,
								 weight_decay: Optional[Union[None, Tuple]] = (0,0,0),
								 pbar: Optional[bool] = False,
								 bgd: Optional[bool] = False,
								 mdelta_estop: Optional[float] = 0,
								 save_chkpts: Optional[bool] = True,
								 metric: Optional[str]='val_loss',
								 mode: Optional[str]='min',
								 ) -> None:

				self.td = td
				self.epochs = epochs
				self.ModelType = model_type
				self.num_layers = num_layers
				self.hidden_dim = hidden_dim
				self.learning_rate = learning_rate
				self.dropout_rate = dropout_rate
				self.log_domain = log_domain
				self.weight_decay = weight_decay
				self.batch_size = batch_size
				self.pbar = pbar
				self.bgd = bgd
				self.mdelta_estop = mdelta_estop
				self.save_chkpts = save_chkpts
				self.metric = metric
				self.mode = mode

				self.chkpt_path = os.path.join(MLP_CHKPT_PTH, td.name, dt_now)
				if not os.path.exists(self.chkpt_path):
						os.makedirs(self.chkpt_path, exist_ok=True)
				self.log_pth = os.path.join(MLP_LOG_PTH, td.name, dt_now, "train")
				if not os.path.exists(self.log_pth):
						os.makedirs(self.log_pth, exist_ok=True)
				self.test_log_pth = os.path.join(MLP_LOG_PTH, td.name, dt_now, "test")
				if not os.path.exists(self.test_log_pth):
						os.makedirs(self.test_log_pth, exist_ok=True)

				self.checkpoint_callback = ModelCheckpoint(monitor=metric,
																									mode=mode,
																									save_top_k=1,
																									dirpath=self.chkpt_path,
																									filename='',
																									)
				
				if mdelta_estop > -1.0:
					self.estop_callback = EarlyStopping(monitor=metric,  
																							patience=max(1, int(epochs/10)),         
																							mode=mode,          
																							verbose=True,
																							min_delta=self.mdelta_estop,     
																							)

				print(f"Initialized Trainer for {td.name}. \nConsider running:  tensorboard --logdir={MLP_LOG_PTH}\n")
				if self.save_chkpts:
					print("Saving checkpoints in directory", self.chkpt_path)
				else:
					print("Saving no checkpoints")

		def suggestHParams(self, trial: optuna.Trial) -> Tuple[int, int, float, Union[None, float], int, Union[None, float], Union[None, int]]:
				hidden_dim = trial.suggest_int('hidden_dim',
																			 low=self.hidden_dim[self.LOW],
																			 high=self.hidden_dim[self.HIGH],
																			 step=self.hidden_dim[self.STEP] if not self.log_domain else 1,
																			 log=self.log_domain,
																			 )
				num_layers = trial.suggest_int('num_layers',
																			 low=self.num_layers[self.LOW],
																			 high=self.num_layers[self.HIGH],
																			 step=self.num_layers[self.STEP] if not self.log_domain else 1,
																			 log=self.log_domain,
																			 )
				learning_rate = trial.suggest_float('learning_rate',
																						low=self.learning_rate[self.LOW],
																						high=self.learning_rate[self.HIGH],
																						step=self.learning_rate[self.STEP] if not self.log_domain else 1,
																						log=self.log_domain,
																						)
				dropout_rate = None if self.dropout_rate is None \
						else trial.suggest_float('dropout_rate',
																		 low=self.dropout_rate[self.LOW],
																		 high=self.dropout_rate[self.HIGH],
																		 step=self.dropout_rate[self.STEP] if not self.log_domain else 1,
																		 log=self.log_domain,
																		 )
				
				batch_size = trial.suggest_categorical('batch_size', self.batch_size)

				weight_decay = 0 if self.weight_decay is None \
							else trial.suggest_float('weight_decay',
																						low=self.weight_decay[self.LOW],
																						high=self.weight_decay[self.HIGH],
																						step=self.weight_decay[self.STEP] if not self.log_domain else 1,
																						log=self.log_domain,
																						)

				return hidden_dim, num_layers, learning_rate, dropout_rate, batch_size, weight_decay

		def plObjective(self, trial: optuna.trial.Trial) -> float:
				"""	Train with mini-batch gradient descent, automatic checkpointing and 
												tensorboard logging.
				"""

				# suggest hyperparameters
				(hidden_dim, num_layers, learning_rate, dropout_rate, batch_size, weight_decay) = self.suggestHParams(trial)
				
				# single run on holdout or num_folds runs on cross-val
				loss_lst = []
				for fold in range(self.td.num_folds if self.td.use_kfold else 1):
					fmt_str = f'_fold_{fold}' if self.td.use_kfold else ''
					
					if self.td.use_kfold and self.pbar:
						print(f"/nrunning trial_{trial.number}{fmt_str}, hidden: {hidden_dim}, layers:{num_layers}, lr:{learning_rate:.4f}, batch size: {batch_size}, dropout: {dropout_rate}, weight decay: {weight_decay}")


					# prep data
					data_module = MLPDataModule(td=self.td,
																			batch_size=batch_size,
																			bgd = self.bgd,
																			k=fold,
																			)

					# instantiate the model
					model = self.ModelType(input_dim=self.td.num_features,
																hidden_dim=hidden_dim,
																output_dim=self.td.num_targets,
																num_layers=num_layers,
																dropout_rate=dropout_rate,
																lr=learning_rate,
																weight_decay=weight_decay,
																)

					# inst. logger
					logger = TensorBoardLogger(self.log_pth,
																		name=f"trial_{trial.number}{fmt_str}_hidden_{hidden_dim}_layers_{num_layers}_lr_{learning_rate:.4f}".replace(".", "_"),)

					# automatic checkpointing
					self.checkpoint_callback.filename = f'best_model_trial_{trial.number:02d}{fmt_str}_{{epoch:02d}}_{{val_loss:.3f}}'
					# create callback list
					cbs = [self.checkpoint_callback] if self.save_chkpts else []

					if self.mdelta_estop > -1.0:
						# add early stopping
						cbs.append(self.estop_callback)
					if not self.td.use_kfold:
						# disable pruning for cross-val
						cbs.append(PyTorchLightningPruningCallback(trial, monitor=self.metric))

					# train
					trainer = pl.Trainer(max_epochs=self.epochs,
															logger=logger,
															log_every_n_steps=1,
															callbacks=cbs,
															enable_progress_bar=self.pbar,
															enable_checkpointing=self.save_chkpts,
															devices=1,
															strategy='auto',
															precision="16-mixed" if str(DEVICE) != 'cpu' else 'bf16-mixed',
															accelerator="auto",
															enable_model_summary=True,
															benchmark=True,
															inference_mode=False,
															)
					
					trainer.fit(model, datamodule=data_module)
					
					res = trainer.callback_metrics[self.metric].item()
					loss_lst.append(res)
				
				if self.td.use_kfold and self.pbar:
					print()

				return np.mean(loss_lst)
		
		def plTest(self, trial: optuna.Trial) -> Any:
				hidden_dim = trial.params['hidden_dim']
				num_layers = trial.params['num_layers']
				learning_rate = trial.params['learning_rate']
				dropout_rate = trial.params.get('dropout_rate')
				batch_size = trial.params['batch_size']

				data_module = MLPDataModule(td=self.td,
																		batch_size=batch_size,
																		bgd = self.bgd,
																		)

				# instantiate the model
				model_chkpt = self.checkpoint_callback.best_model_path
				# not necessarily from best trial
				model = self.ModelType.load_from_checkpoint(model_chkpt)

				test_logger = TensorBoardLogger(self.test_log_pth,
																				name=f"trial_{trial.number}_hidden_{hidden_dim}_layers_{num_layers}_lr_{learning_rate:.4f}".replace(".", "_"),
																				)
				
				test_trainer = pl.Trainer(max_epochs=self.epochs,
																		logger=test_logger,
																		log_every_n_steps=5,
																		enable_progress_bar=self.pbar,
																		enable_checkpointing=False,
																		devices=1,
																		strategy="auto",
																		precision="16-mixed" if str(DEVICE) != 'cpu' else 'bf16-mixed',
																		accelerator="auto",
																		enable_model_summary=True,
																		benchmark=True,
																		inference_mode=False,
																		)

				res = test_trainer.test(model, datamodule=data_module, ckpt_path=model_chkpt)

				return res[0]['test_loss'], model_chkpt
				
		def plainObjective(self, trial: optuna.Trial) -> Any:
				"""	Train with batch gradient descent, manual checkpointing and
												manual tensorboard logging.
				"""

				# suggest hyperparameters
				(hidden_dim, num_layers, learning_rate, dropout_rate, _, weight_decay) = self.suggestHParams(trial)

				# instantiate the model and move to device
				model = self.ModelType(input_dim=self.td.num_features,
															 hidden_dim=hidden_dim,
															 output_dim=self.td.num_targets,
															 num_layers=num_layers,
															 dropout_rate=dropout_rate,
															 ).to(DEVICE)

				# loss criterion and optimizer
				criterion = nn.MSELoss()
				optimizer = optim.Adam(model.parameters(),
															 lr=learning_rate,
															 weight_decay=weight_decay,
															 )

				# TensorBoard logging setup
				run_name = f"trial_{trial.number}_hidden_{hidden_dim}_layers_{num_layers}_lr_{learning_rate:.4f}".replace(
						".", "_")
				writer = SummaryWriter(log_dir=os.path.join(self.log_pth, run_name))
			
				# train
				best_val_loss = np.inf
				for epoch in range(self.epochs):
						model.train()
						optimizer.zero_grad()
						outputs = model(self.td.X_train_tensor)
						loss = criterion(outputs, self.td.y_train_tensor)
						loss.backward()
						optimizer.step()

						# log training loss
						writer.add_scalar('Loss/train', loss.item(), epoch)

						# validation
						model.eval()
						with torch.no_grad():
								val_outputs = model(self.td.X_val_tensor)
								val_loss = criterion(val_outputs, self.td.y_val_tensor).item()
								writer.add_scalar('Loss/val', val_loss, epoch)

								if val_loss < best_val_loss and self.save_chkpts:
										best_val_loss = val_loss
										chkpt = {
												'epoch': epoch,
												'model_state_dict': model.state_dict(),
												'optimizer_state_dict': optimizer.state_dict(),
												'best_val_loss': best_val_loss,
										}
										chkpt_pth = os.path.join(self.chkpt_path, f"{run_name}.pth")
										torch.save(chkpt, chkpt_pth)
										trial.set_user_attr("checkpoint_path", chkpt_pth)

				writer.close()
				return val_loss

		def plainTest(self, trial: FrozenTrial) -> float:
				hidden_dim = trial.params['hidden_dim']
				num_layers = trial.params['num_layers']
				learning_rate = trial.params['learning_rate']
				dropout_rate = trial.params.get('dropout_rate')

				# load model to DEVICE
				model = self.ModelType(input_dim=self.td.num_features,
															 hidden_dim=hidden_dim,
															 output_dim=self.td.num_targets,
															 num_layers=num_layers,
															 dropout_rate=dropout_rate,
															 ).to(DEVICE)

				# load model weights
				best_model_path = trial.user_attrs["checkpoint_path"]
				print("Loading checkpoint from", best_model_path)
				chkpt = torch.load(best_model_path, weights_only=True, )
				model.load_state_dict(chkpt['model_state_dict'],)

				run_name = f"TEST_vloss_{trial.value:.4f}_trial_{trial.number}_hidden_{hidden_dim}_layers_{num_layers}_lr_{learning_rate:.4f}".replace(
						".", "_")
				writer = SummaryWriter(log_dir=os.path.join(self.log_pth, run_name))

				# evaluate on the test set
				model.eval()
				with torch.no_grad():
						test_outputs = model(self.td.X_test_tensor)
						test_loss = nn.MSELoss()(test_outputs, self.td.y_test_tensor).item()
						writer.add_scalar('Loss/test', test_loss, )

				writer.close()
				return test_loss, best_model_path


class Train():
		""" Train all configurations found in the given folder having the specified pattern.

				@param folder_pth
				@type str
				@param pattern
				@type str
				@param optim_trials
				@type int
				@param pruning
				@type bool
				@param bgd
				@type bool
				@param optuna_pbar
				@type bool
				@param lightning_pbar
				@type bool
				@param distribute
				@type bool

		"""

		def __init__(self,
								 folder_pth: Optional[str] = 'config',
								 pattern: Optional[str] = 'mono',
								 test_size: Optional[float] = 0.3,
								 validation_size: Optional[float] = 0.5,
								 random_state: Optional[int] = 42,
								 trans_norm: Optional[Normalization] = Normalization.Z_SCORE,
								 input_norm: Optional[Normalization] = Normalization.Z_SCORE,
								 target_norm: Optional[Normalization] = Normalization.Z_SCORE,
								 epochs: Optional[int] = 50,
								 num_layers: Optional[Tuple] = (1, 10, 1),
								 hidden_dim: Optional[Tuple] = (1, 32, 2),
								 learning_rate: Optional[Tuple] = (1e-4, 1e-2, 1e-3),
								 dropout_rate: Optional[Union[None, Tuple]] = (0, 0.4, 0.01),
								 log_domain: Optional[bool] = False,
								 weight_decay: Optional[Union[None, Tuple]] = (0,0,0),
								 optim_trials: Optional[int] = 100,
								 batch_size: Optional[Tuple] = (18, 32, 64),
								 pruning: Optional[bool] = False,
								 bgd: Optional[bool] = False,
								 optuna_pbar: Optional[bool] = True,
								 lightning_pbar: Optional[bool] = False,
								 distribute: Optional[bool] = False,
								 mdelta_estop: Optional[float] = 0,
								 kfold: Optional[int] = 0,
								 save_chkpts: Optional[bool] = True,
								 ) -> None:

				# load training data per configuration
				pattern = os.path.join(DATA_PTH, 'keypoint/train',
															 folder_pth, f'*{pattern}*')
				self.data_files = glob.glob(pattern, recursive=False)

				self.test_size = test_size
				self.validation_size = validation_size
				self.random_state = random_state
				self.trans_norm = trans_norm
				self.input_norm = input_norm
				self.target_norm = target_norm
				self.epochs = epochs
				self.num_layers = num_layers
				self.hidden_dim = hidden_dim
				self.learning_rate = learning_rate
				self.dropout_rate = dropout_rate
				self.log_domain = log_domain
				self.weight_decay = weight_decay
				self.optim_trials = optim_trials
				self.batch_size = batch_size
				self.pruning = pruning
				self.bgd = bgd
				self.optuna_pbar = optuna_pbar
				self.lightning_pbar = lightning_pbar
				self.distribute = distribute
				self.mdelta_estop = mdelta_estop
				self.kfold = kfold
				self.save_chkpts = save_chkpts
				self.log = {}

		def run(self) -> None:
				for fl in self.data_files:
						if self.distribute:
								self.runPlStudy(fl)
						else:
								self.runPlainStudy(fl)

		def runPlStudy(self, file_pth: str) -> None:
				print(f"\n{50*'#'}")

				td = TrainingData(data_file=file_pth,
													test_size=self.test_size,
													validation_size=self.validation_size,
													random_state=self.random_state,
													trans_norm=self.trans_norm,
													input_norm=self.input_norm,
													target_norm=self.target_norm,
													move_gpu=False,
													kfold=self.kfold,
													)

				trainer = Trainer(td=td,
													epochs=self.epochs,
													model_type=WrappedMLP,
													num_layers=self.num_layers,
													hidden_dim=self.hidden_dim,
													learning_rate=self.learning_rate,
													dropout_rate=self.dropout_rate,
													log_domain=self.log_domain,
													weight_decay=self.weight_decay,
													batch_size=self.batch_size,
													pbar=self.lightning_pbar,
													bgd=self.bgd,
													mdelta_estop=self.mdelta_estop,
													save_chkpts=self.save_chkpts,
													)

				print(f"\n{50*'#'}")
				
				# run optuna optimization
				study = optuna.create_study(direction='minimize',
																		storage=RDBStorage(url="sqlite:///optuna_study.db"),
																		pruner=optuna.pruners.MedianPruner() if self.pruning else optuna.pruners.NopPruner(),
																		)
				study.optimize(trainer.plObjective,
											 n_trials=self.optim_trials, show_progress_bar=self.optuna_pbar, )

				# Retrieve the best trial
				trial = study.best_trial
				print('Best trial:', trial.number)
				print('Value: ', trial.value)
				print('Params: ')
				for key, value in trial.params.items():
						print(f'{key}: {value}')
				print("Best hyperparameters:", study.best_params)
				print("Finished optimization of ", td.name)

				print("\nRunning test...")
				(res, best_model_path) = trainer.plTest(trial)

				self.log.update({td.name: {'best_trial': trial.number, 'val_loss': trial.value, 'test_loss': res,
												'params': trial.params, 'checkpoint': best_model_path, 'scalers': td.scaler_pth}})
				with open(os.path.join(trainer.chkpt_path, "optimization_results.json"), "w") as json_file:
						json.dump(self.log, json_file, indent=4)
				print("Done\n")

		def runPlainStudy(self, file_pth: str) -> None:
				print(f"\n{50*'#'}")

				td = TrainingData(data_file=file_pth,
													test_size=self.test_size,
													validation_size=self.validation_size,
													random_state=self.random_state,
													trans_norm=self.trans_norm,
													input_norm=self.input_norm,
													target_norm=self.target_norm,
													move_gpu=True,
													kfold=0,
													)

				trainer = Trainer(td=td,
													epochs=self.epochs,
													model_type=MLP,
													num_layers=self.num_layers,
													hidden_dim=self.hidden_dim,
													learning_rate=self.learning_rate,
													dropout_rate=self.dropout_rate,
													log_domain=self.log_domain,
													weight_decay=self.weight_decay,
													batch_size=self.batch_size,
													save_chkpts=self.save_chkpts,
													)

				print(f"\n{50*'#'}")

				# run optuna optimization
				study = optuna.create_study(direction='minimize', )
				study.optimize(trainer.plainObjective,
											 n_trials=self.optim_trials, show_progress_bar=self.optuna_pbar, )

				# Retrieve the best trial
				trial = study.best_trial
				print('Best trial:', trial.number)
				print('Value: ', trial.value)
				print('Params: ')
				for key, value in trial.params.items():
						print(f'{key}: {value}')
				print("Best hyperparameters:", study.best_params)
				print("Finished optimization of ", td.name)

				print("Running test...")
				(res, chkpt_pth) = trainer.plainTest(trial)
				print("Result:", res)
				print()

				self.log.update({td.name: {'best_trial': trial.number, 'val_loss': trial.value, 'test_loss': res,
												'params': trial.params, 'checkpoint': chkpt_pth, 'scalers': td.scaler_pth}})
				with open(os.path.join(trainer.chkpt_path, "optimization_results.json"), "a") as json_file:
						json.dump(self.log, json_file, indent=4)

def visData(args) -> None:
	pattern = os.path.join(DATA_PTH, 'keypoint/train', args.folder_pth, f'*{args.pattern}*')
	data_files = glob.glob(pattern, recursive=False)
	assert(len(data_files) == 1) # supports only single file

	td = TrainingData(data_file=data_files[0],
										test_size=args.test_size,
										validation_size=args.val_size,
										random_state=args.random_state,
										trans_norm=args.trans_norm,
										input_norm=args.input_norm,
										target_norm=args.target_norm,
										move_gpu=False,
										kfold=args.kfold,
										)
	
	matplotlib.use('TkAgg') 

	# orig data
	index = td.df.index.to_list()
	for col in td.cols:
		if 'cmd' in col or 'angle' in col or 'dir' in col:
			plotData(index, td.df[col].values, col)
		elif 'quat' in col:
			quats = np.array([np.array(lst) for lst in td.df[col]])
			# quats = smoothMagnQuats(quats)
			plotData(index, quats, col)
		elif 'trans' in col:
			trans = np.array([np.array(lst) for lst in td.df[col]])
			# trans = smoothMagnTrans(trans, s=0)
			plotData(index, trans, col)
	
	if td.use_kfold:
		td.foldData(1)

	# features
	data_vis = np.concatenate((td.X_train_tensor.numpy(), td.X_val_tensor.numpy(), td.X_test_tensor.numpy()), axis=0)
	index = range(len(data_vis))
	for idx, feat in enumerate(td.feature_names):
		if 'cmd' in feat or 'dir'  in feat:
			plotData(index, data_vis[:, idx], f"feature_{feat}")
		elif feat == "x":
			quats = data_vis[:, idx : idx + len(QUAT_COLS)]
			# quats = smoothMagnQuats(quats)
			plotData(index, quats, "feature_quats")	
	
	# targets
	tdata_vis = np.concatenate((td.y_train_tensor.numpy(), td.y_val_tensor.numpy(), td.y_test_tensor.numpy()), axis=0)
	tindex = range(len(tdata_vis))
	for idx, target in enumerate(td.target_names):
		if 'angle' in target:
			plotData(tindex, tdata_vis[:, idx], f"target_{target}")
		elif 'x_' in target:
			trans = tdata_vis[:, idx : idx + len(TRANS_COLS)]
			# t = smoothMagnTrans(t, s=0)
			plotData(tindex, trans, f"target_{target.replace('x_', '')}")

	# denormalized
	if td.trans_norm != Normalization.NONE and td.y_trans and td.y_trans_scalers:
		name = 'trans' if len(td.y_trans.keys()) == 1 else 'trans1'
		trans = tdata_vis[:, : len(TRANS_COLS)]
		trans = td.denormalizeOutputData(td.y_trans_scalers[name], trans)
		plotData(tindex, trans, 'denormalized_trans')

	if td.input_norm != Normalization.NONE and td.X_cmds_scalers:
		name =  'cmd' if len(td.X_cmds.keys()) == 1 else 'cmd1'
		cmd = td.denormalizeOutputData(td.X_cmds_scalers[name], data_vis[:, len(QUAT_COLS)].reshape(-1, 1))
		plotData(index, cmd, 'denormalized_cmd')

	if td.target_norm != Normalization.NONE and td.y_angles_scalers:
		name = 'angle' if len(td.y_angles.keys()) == 1 else 'angle1'
		angle = td.denormalizeOutputData(td.y_angles_scalers[name], tdata_vis[:, len(TRANS_COLS)].reshape(-1, 1))
		plotData(tindex, angle, 'denormalized_angle')

	plt.show()

if __name__ == '__main__':
		parser = argparse.ArgumentParser(description='Training script')
		# cleanup
		parser.add_argument("--clean_log", action='store_true',help='Clean log folder')
		parser.add_argument("--clean_scaler", action='store_true',help='Clean scaler folder')
		parser.add_argument("--clean_checkpoint",action='store_true', help='Clean checkpoints folder')
		parser.add_argument("--clean_all", action='store_true',help='Clean all  folders with saved checkpoints, logs and scalers!')
		# data preparation
		data_group = parser.add_argument_group("Data Settings")
		data_group.add_argument('--folder_pth', type=str, metavar='str',help='Folder path for the data prepared for training.', default="10013")
		data_group.add_argument('--pattern', type=str, metavar='str',help='Search pattern for data files', default=".json")
		data_group.add_argument('--test_size', type=float, metavar='float',help='Percentage of data split for testing (from validation size if holdout method is used).', default=0.3)
		data_group.add_argument('--val_size', type=float, metavar='float',help='Percentage of data split for validation --incl. test size-- (applies only to holdout split).', default=0.5)
		data_group.add_argument('--random_state', type=int, metavar='int',help='Seed value for data randomization.', default=40)
		data_group.add_argument('--trans_norm', type=parseNorm,help=f'Normalization method for translations [{NORMS}]', default=Normalization.Z_SCORE.value)
		data_group.add_argument('--input_norm', type=parseNorm,help=f'Normalization method for translations [{NORMS}]', default=Normalization.Z_SCORE.value)
		data_group.add_argument('--target_norm', type=parseNorm,help=f'Normalization method for translations [{NORMS}]', default=Normalization.Z_SCORE.value)
		data_group.add_argument('--vis_data', action='store_true', help='Plot the input data, the feature and target vectors and terminate program afterwards.')
		# training
		train_group = parser.add_argument_group("Optimization Settings")
		train_group.add_argument('--epochs', type=int, metavar='int', help='Training epochs.', default=100)
		train_group.add_argument('--optim_trials', type=int, metavar='int',help='Number of optimization trials.', default=100)
		train_group.add_argument('--num_layers', type=parseIntTuple,help='Min, max and step value of hidden layers (int), eg. 2,10,2.', default='2,10,2')
		train_group.add_argument('--hidden_dim', type=parseIntTuple,help='Min, max and step value of hidden nodes (int), eg. 2,10,2.', default='2,10,2')
		train_group.add_argument('--batch_size', type=parseIntTuple,help='Choices for mini-batch training (int), eg. 18,32,64,128  (req. --distribute).', default='18,32,64')
		train_group.add_argument('--learning_rate', type=parseFloatTuple,help='Min, max and step value of learning rate (float), eg. 1e-4,1e-2,1e-2.', default='1e-4,1e-2,1e-2')
		train_group.add_argument('--dropout_rate', type=parseFloatTuple,help='Min, max and step value of dropout rate (float). Disable with none, eg. 0.0, 0.4, 0.01 or none.', default='none')
		train_group.add_argument('--weight_decay', type=parseFloatTuple, help='L2 regularization weight decay (float), eg. 1e-1,1e-2,1e-2 or none to disable.', default='none')
		train_group.add_argument('--kfold', type=int, metavar='int', help='Use k-fold c-v and spec num_splits or disable with 0 (req. --distribute). If disabled 3-way holdout split is used', default=0)
		train_group.add_argument('--mdelta_estop', type=float, metavar='float',help='Min req. delta for val loss before early stopping, disable with -1.0 (req. --distribute).', default=-1.0)
		train_group.add_argument('--log_domain', action='store_true',help='Change optimizer params logarithmically.')
		train_group.add_argument('--pruning', action='store_true', help='Turn on pruning during optimization (req. --distribute).')
		train_group.add_argument('--use_bgd', action='store_true', help='Use batch gradient descent training.')
		train_group.add_argument('--distribute', action='store_true', help='Train on multiple devices with lightning setup, else train on plain torch with batch gradient desc.')
		train_group.add_argument('--optuna_pbar', action='store_true', help='Show Optunas trial progessbar (req. --distribute).')
		train_group.add_argument('--lightning_pbar', action='store_true', help='Show Lightnings detailed progessbar (req. --distribute).')
		train_group.add_argument('--save_chkpts', action='store_true', help='Save model checkpoints.')
		train_group.add_argument('--y', action='store_true', help='Discard asking for data cleaning and continue with training after cleanup.')
		args = parser.parse_args()

		# clean opt. data
		clean(args)

		# show training data
		if args.vis_data:
			visData(args)
			exit(0)

		# perform training
		Train(folder_pth=args.folder_pth,
					pattern=args.pattern,
					test_size=args.test_size,
					validation_size=args.val_size,
					random_state=args.random_state,
					trans_norm=args.trans_norm,
					input_norm=args.input_norm,
					target_norm=args.target_norm,
					epochs=args.epochs,
					num_layers=args.num_layers,
					hidden_dim=args.hidden_dim,
					learning_rate=args.learning_rate,
					dropout_rate=args.dropout_rate,
					log_domain=args.log_domain,
					weight_decay=args.weight_decay,
					optim_trials=args.optim_trials,
					batch_size=args.batch_size,
					pruning=args.pruning,
					bgd=args.use_bgd,
					optuna_pbar=args.optuna_pbar,
					lightning_pbar=args.lightning_pbar,
					distribute=args.distribute,
					mdelta_estop=args.mdelta_estop,
					kfold=args.kfold,
					save_chkpts=args.save_chkpts,
					).run()
