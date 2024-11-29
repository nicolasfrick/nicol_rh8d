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
from torch.utils.data import DataLoader, TensorDataset
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from optuna.integration import PyTorchLightningPruningCallback
from util import *

# torch at cuda or cpu
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_DEV = torch.cuda.device_count()
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
						@param norm_quats 
						@type bool
						@param trans_norm 
						@type Normalization
						@param input_norm 
						@type Normalization
						@param target_norm 
						@type Normalization
						@param move_gpu
						@type bool

		Steps for Inference with Normalized Inputs
						Save the normalization parameters: During training, keep track of the mean, standard deviation, minimum, and maximum values used for normalization.
						Normalize the input data: Before feeding the input data into the model, apply the same normalization using the saved parameters.
						Make predictions: Use the normalized input data for the model inference.
						Inverse transform the output (if needed): If your output targets were also normalized, apply the inverse transformation to interpret the results in the original scale.

		"""

		def __init__(self,
								 data_file: str,
								 test_size: Optional[float] = 0.3,
								 validation_size: Optional[float] = 0.5,
								 random_state: Optional[int] = 42,
								 norm_quats: Optional[bool] = False,
								 trans_norm: Optional[Normalization] = Normalization.Z_SCORE,
								 input_norm: Optional[Normalization] = Normalization.Z_SCORE,
								 target_norm: Optional[Normalization] = Normalization.Z_SCORE,
								 move_gpu: Optional[bool] = False,
								 ) -> None:

				# path names
				split = data_file.split('/')
				self.file_name = split[-1]
				self.folder_pth = "/".join(split[: -1])
				self.name = self.file_name.replace('.json', '')
				self.scaler_pth = os.path.join(MLP_SCLRS_PTH, self.name)
				if not os.path.exists(self.scaler_pth):
						os.makedirs(self.scaler_pth, exist_ok=True)

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

				# results
				self.X_train_tensor = None
				self.X_test_tensor = None
				self.X_val_tensor = None
				self.y_train_tensor = None
				self.y_test_tensor = None
				self.y_val_tensor = None

				# load, normalize, split and move data
				self.prepare(df, cols, move_gpu)

				# save fitted scalers
				self.saveScalers()

		def prepare(self, df: pd.DataFrame, cols: list, move_gpu: bool) -> None:
				# load and normalize data
				self.loadData(df, cols)
				print("Added", len(self.X_cmds.keys()), "cmd data frames,",  len(self.X_dirs.keys()), "dir data frames,",
							len(self.y_angles.keys()), "target angle data frames, one input orientation and target translation data frame.\n")

				# stack training data
				(X, y) = self.stackData()

				# split and move data
				self.splitData(X, y, move_gpu)

				print("Loaded features", self.feature_names, "and targets",
							self.target_names, f"to device: {DEVICE}" if move_gpu else "")
				pct_train = 100*(1-self.test_size)
				pct_test = 100*(self.test_size)
				pct_val = self.validation_size * pct_test
				pct_test -= pct_val
				print(
						f"Splitted {self.num_samples} samples into {len(self.X_train_tensor)} training ({pct_train}%),")
				print(
						f"{len(self.X_test_tensor)} test ({pct_test}%) and {len(self.X_val_tensor)} validation ({pct_val}%) samples")

		def stackData(self) -> Tuple[np.ndarray, np.ndarray]:
				"""Stack normalized data to a feature vector X and
												a target vector y.
				"""
				# quaternions first, always present
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

				# translation first if present
				y = None
				if self.y_trans is not None:
						self.target_names = ["x", "y", "z"]
						y = self.y_trans.copy()
				# stack angles
				for name, y_angle in self.y_angles.items():
						if y is None:
								# no trans, angle1 is 1st elmt
								self.target_names = [name]
								y = y_angle.copy()
								y = y.reshape(-1, 1)
						else:
								self.target_names.append(name)
								y = np.hstack([y, y_angle.reshape(-1, 1)])

				self.num_features = len(self.feature_names)
				self.num_targets = len(self.target_names)
				self.num_samples = len(X)

				return X, y

		def splitData(self, X: np.ndarray, y: np.ndarray, move_gpu: bool) -> None:
				"""Split features X and targets y into a  train, test 
												and validation set and move to the present DEVICE.
				"""
				# split off training data
				X_train, X_tmp, y_train, y_tmp = train_test_split(
						X, y, test_size=self.test_size, random_state=self.random_state)
				# split off test and validation data
				X_test, X_val, y_test, y_val = train_test_split(
						X_tmp, y_tmp, test_size=self.validation_size, random_state=self.random_state)

				# map features to tensors and move to device
				self.X_train_tensor = torch.from_numpy(X_train).float()
				self.X_test_tensor = torch.from_numpy(X_test).float()
				self.X_val_tensor = torch.from_numpy(X_val).float()

				# map targets to tensors and move to device
				self.y_train_tensor = torch.from_numpy(y_train).float()
				self.y_test_tensor = torch.from_numpy(y_test).float()
				self.y_val_tensor = torch.from_numpy(y_val).float()

				# move to gpu
				if move_gpu:
						self.X_train_tensor = self.X_train_tensor.to(DEVICE)
						self.X_test_tensor = self.X_test_tensor.to(DEVICE)
						self.X_val_tensor = self.X_val_tensor.to(DEVICE)
						self.y_train_tensor = self.y_train_tensor.to(DEVICE)
						self.y_test_tensor = self.y_test_tensor.to(DEVICE)
						self.y_val_tensor = self.y_val_tensor.to(DEVICE)

		def loadData(self, df: pd.DataFrame, cols: list) -> None:
				"""Fill datastructures with traning data, possibly apply normalization.
				"""
				for col in cols:
						data = df[col].values

						# process input commands
						if 'cmd' in col:
								normalized_data = self.normalizeData(
										data, self.input_norm, self.X_cmds_scalers, col)
								self.X_cmds.update({col: normalized_data.flatten()})
						# process input direction
						elif 'dir' in col:
								self.X_dirs.update({col: np.array(data, dtype=np.int32)})
						# process input orientation
						elif 'quat' in col:
								quats = np.array([list(q) for q in data], dtype=np.float32)
								self.X_quats = self.normQuaternions(
										quats) if self.norm_quats else quats

						# process target angles
						elif 'angle' in col:
								normalized_data = self.normalizeData(
										data, self.target_norm, self.y_angles_scalers, col)
								self.y_angles.update({col: normalized_data.flatten()})
						# process target trans
						elif 'trans' in col:
								self.y_trans = self.normalizeData(np.array(
										[list(t) for t in data], dtype=np.float32), self.trans_norm, self.y_trans_scalers, col, None)
						else:
								raise NotImplementedError

		def normalizeData(self,
											data: np.ndarray,
											norm: Normalization,
											scaler_dict: dict,
											scaler_name: str,
											reshape: Tuple = (-1, 1),
											) -> np.ndarray:

				normalized_data = data.copy()
				if reshape is not None:
						# reshape for single feature
						normalized_data = normalized_data.reshape(reshape[0], reshape[1])

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
						pth = os.path.join(self.scaler_pth, name + ".pkl")
						joblib.dump(scaler, pth)
				for name, scaler in self.y_angles_scalers.items():
						pth = os.path.join(self.scaler_pth, name + ".pkl")
						joblib.dump(scaler, pth)
				for name, scaler in self.y_trans_scalers.items():
						pth = os.path.join(self.scaler_pth, name + ".pkl")
						joblib.dump(scaler, pth)

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

		"""

		def __init__(self,
								 input_dim: int,
								 hidden_dim: int,
								 output_dim: int,
								 num_layers: int,
								 dropout_rate: Union[None, float],
								 ) -> None:

				super().__init__()

				layers: List[nn.Module] = []
				# input
				layers.append(nn.Linear(input_dim, hidden_dim))
				layers.append(nn.ReLU())
				if dropout_rate is not None:
						layers.append(nn.Dropout(p=dropout_rate))
				# hidden
				for _ in range(num_layers - 1):
						layers.append(nn.Linear(hidden_dim, hidden_dim))
						layers.append(nn.ReLU())
						if dropout_rate is not None:
								layers.append(nn.Dropout(p=dropout_rate))
				# output
				layers.append(nn.Linear(hidden_dim, output_dim))

				# model
				self.model = nn.Sequential(*layers)

		# overwrite
		def forward(self, data: torch.Tensor) -> torch.Tensor:
				return self.model(data)

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

		# overwrite
		def forward(self, data: torch.Tensor) -> torch.Tensor:
				return self.model(data)

		# overwrite
		def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> float:
				X, y = batch
				preds = self(X)
				loss = self.loss_fn(preds, y)
				self.log('train_loss', loss, on_step=False,
								 on_epoch=True, prog_bar=True, sync_dist=True, )
				return loss

		# overwrite
		def validation_step(self, batch: List[torch.Tensor], batch_idx: int) -> float:
				X, y = batch
				preds = self(X)
				val_loss = self.loss_fn(preds, y)
				self.log('val_loss', val_loss, on_step=False,
								 on_epoch=True, prog_bar=True, sync_dist=True, )
				self.log("hp_metric", val_loss, on_step=False, on_epoch=True, sync_dist=True)
				return val_loss
		
		def test_step(self, batch: List[torch.Tensor], batch_idx: int) -> float:
				X, y = batch
				preds = self(X)
				test_loss = self.loss_fn(preds, y)
				self.log('test_loss', test_loss, on_step=False,
								 on_epoch=True, prog_bar=True, sync_dist=True, )
				return test_loss

		# def on_test_epoch_end(self, outputs: List[torch.Tensor]) -> dict:
		#     avg_loss = torch.stack([x for x in outputs]).mean()
		#     return {"test_loss": avg_loss}

		# overwrite
		def configure_optimizers(self) -> optim.Optimizer:
				return optim.Adam(self.model.parameters(),
													lr=self.hparams.lr,
													weight_decay=self.hparams.weight_decay,
													)

class MLPDataModule(pl.LightningDataModule):
		""" Load data for lightning trainer.

				@param X_train
				@type torch.Tensor
				@param y_train
				@type torch.Tensor
				@param X_val
				@type torch.Tensor
				@param y_val
				@type torch.Tensor
				@param X_test
				@type torch.Tensor
				@param y_test
				@type torch.Tensor
				@param batch_size
				@type int
				@param bgd
				@type bool

		"""

		def __init__(self,
								 X_train: torch.Tensor,
								 y_train: torch.Tensor,
								 X_val: torch.Tensor,
								 y_val: torch.Tensor,
								 X_test: torch.Tensor,
								 y_test: torch.Tensor,
								 batch_size: int,
								 bgd: bool,
								 ) -> None:

				super().__init__()
				self.X_train = X_train
				self.y_train = y_train
				self.X_val = X_val
				self.y_val = y_val
				self.X_test = X_test
				self.y_test = y_test

				if bgd:
					# batch gradient desc.
					self.train_batch_size = len(X_train)
					self.val_batch_size = len(X_val)
					self.test_batch_size = len(X_test)
				else:
					self.train_batch_size = batch_size
					self.val_batch_size = batch_size
					self.test_batch_size = batch_size

				# overwrite
		def setup(self, stage: Optional[str] = None) -> None:
				self.train_dataset = TensorDataset(self.X_train, self.y_train)
				self.val_dataset = TensorDataset(self.X_val, self.y_val)
				self.test_dataset = TensorDataset(self.X_test, self.y_test)

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

										@param X_train_tensor
										@type torch.Tensor
										@param X_test_tensor
										@type torch.Tensor
										@param X_val_tensor
										@type torch.Tensor
										@param y_train_tensor
										@type torch.Tensor
										@param y_test_tensor
										@type torch.Tensor
										@param y_val_tensor
										@type torch.Tensor
										@param epochs
										@type int
										@param model_type
										@type Any
										@param input_dim
										@type int
										@param output_dim
										@type int
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
										@type float
										@param name
										@type str
										@param pbar
										@type bool
										@param bgd
										@type bool

		"""

		LOW = 0
		HIGH = 1
		STEP = 2

		def __init__(self,
								 X_train_tensor: torch.Tensor,
								 X_test_tensor: torch.Tensor,
								 X_val_tensor: torch.Tensor,
								 y_train_tensor: torch.Tensor,
								 y_test_tensor: torch.Tensor,
								 y_val_tensor: torch.Tensor,
								 epochs: Optional[int] = 50,
								 model_type: Optional[Any] = WrappedMLP,
								 input_dim: Optional[int] = 6,
								 output_dim: Optional[int] = 6,
								 num_layers: Optional[Tuple] = (1, 10, 1),
								 hidden_dim: Optional[Tuple] = (1, 32, 2),
								 batch_size: Optional[Tuple] = (16, 32, 64),
								 learning_rate: Optional[Tuple] = (1e-4, 1e-2, 1e-3),
								 dropout_rate: Optional[Union[None, Tuple]] = (0, 0.4, 0.01),
								 log_domain: Optional[bool] = False,
								 weight_decay: Optional[float] = 0,
								 name: Optional[str] = '',
								 pbar: Optional[bool] = False,
								 bgd: Optional[bool] = False,
								 ) -> None:

				self.epochs = epochs
				self.ModelType = model_type
				self.input_dim = input_dim
				self.output_dim = output_dim
				self.num_layers = num_layers
				self.hidden_dim = hidden_dim
				self.learning_rate = learning_rate
				self.X_train_tensor = X_train_tensor
				self.X_test_tensor = X_test_tensor
				self.X_val_tensor = X_val_tensor
				self.y_train_tensor = y_train_tensor
				self.y_test_tensor = y_test_tensor
				self.y_val_tensor = y_val_tensor
				self.dropout_rate = dropout_rate
				self.log_domain = log_domain
				self.weight_decay = weight_decay
				self.batch_size = batch_size
				self.best_val_loss = np.inf
				self.best_chkpt_pth = None
				self.pbar = pbar
				self.bgd = bgd

				self.chkpt_path = os.path.join(MLP_CHKPT_PTH, name)
				if not os.path.exists(self.chkpt_path):
						os.makedirs(self.chkpt_path, exist_ok=True)
				self.log_pth = os.path.join(MLP_LOG_PTH, name, "train")
				if not os.path.exists(self.log_pth):
						os.makedirs(self.log_pth, exist_ok=True)
				self.test_log_pth = os.path.join(MLP_LOG_PTH, name, "test")
				if not os.path.exists(self.test_log_pth):
						os.makedirs(self.test_log_pth, exist_ok=True)

				self.checkpoint_callback = ModelCheckpoint(monitor='val_loss',
																									mode='min',
																									save_top_k=1,
																									dirpath=self.chkpt_path,
																									filename='',
																									)

				print(f"Initialized Trainer for {name}. \nConsider running:  tensorboard --logdir={MLP_LOG_PTH}\n")
				print("Saving checkpoints in directory", self.chkpt_path)

		def suggestHParams(self, trial: optuna.Trial) -> Tuple[int, int, float, Union[None, float], int]:
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
																		 step=self.dropout_rate[
																				 self.STEP] if not self.log_domain else 1,
																		 log=self.log_domain,
																		 )
				batch_size = trial.suggest_categorical('batch_size', self.batch_size)

				return hidden_dim, num_layers, learning_rate, dropout_rate, batch_size

		def plObjective(self, trial: optuna.trial.Trial) -> float:
				"""	Train with mini-batch gradient descent, automatic checkpointing and 
												tensorboard logging.
				"""

				# suggest hyperparameters
				(hidden_dim, num_layers, learning_rate, dropout_rate, batch_size) = self.suggestHParams(trial)

				# prep data
				data_module = MLPDataModule(X_train=self.X_train_tensor,
																		y_train=self.y_train_tensor,
																		X_val=self.X_val_tensor,
																		y_val=self.y_val_tensor,
																		X_test=self.X_test_tensor,
																		y_test=self.y_test_tensor,
																		batch_size=batch_size,
																		bgd = self.bgd,
																		)
				data_module.prepare_data()
				data_module.setup()

				# instantiate the model
				model = self.ModelType(input_dim=self.input_dim,
															 hidden_dim=hidden_dim,
															 output_dim=self.output_dim,
															 num_layers=num_layers,
															 dropout_rate=dropout_rate,
															 lr=learning_rate,
															 weight_decay=self.weight_decay,
															 )

				# inst. logger
				logger = TensorBoardLogger(self.log_pth,
																	 name=f"trial_{trial.number}_hidden_{hidden_dim}_layers_{num_layers}_lr_{learning_rate:.4f}".replace(".", "_"),)

				# automatic checkpointing
				self.checkpoint_callback.filename = f'best_model_trial{trial.number:02d}_{{epoch:02d}}_{{val_loss:.3f}}'

				# train
				trainer = pl.Trainer(max_epochs=self.epochs,
														 logger=logger,
														 log_every_n_steps=1,
														 callbacks=[self.checkpoint_callback,
																				PyTorchLightningPruningCallback(trial, monitor="val_loss"),
																				],
														 enable_progress_bar=self.pbar,
														 enable_checkpointing=True,
														 devices=NUM_DEV,
														 strategy="ddp_spawn",
														 precision="16-mixed",
														 accelerator="auto",
														 )
				
				trainer.fit(model, datamodule=data_module)

				return trainer.callback_metrics["val_loss"].item()
		
		def plTest(self, trial: optuna.Trial) -> Any:
				hidden_dim = trial.params['hidden_dim']
				num_layers = trial.params['num_layers']
				learning_rate = trial.params['learning_rate']
				dropout_rate = trial.params.get('dropout_rate')
				batch_size = trial.params['batch_size']

				model_chkpt = self.checkpoint_callback.best_model_path
				
				data_module = MLPDataModule(X_train=self.X_train_tensor,
				                            y_train=self.y_train_tensor,
				                            X_val=self.X_val_tensor,
				                            y_val=self.y_val_tensor,
				                            X_test=self.X_test_tensor,
				                            y_test=self.y_test_tensor,
				                            batch_size=batch_size,
																		bgd = self.bgd,
				                            )
				data_module.prepare_data()
				data_module.setup()

				# instantiate the model
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
				                            precision="16-mixed",
				                            accelerator="auto",
				                            )

				test_trainer.test(model, datamodule=data_module, ckpt_path=model_chkpt)

				return test_trainer.callback_metrics["test_loss"].item(), model_chkpt
				
		def plainObjective(self, trial: optuna.Trial) -> Any:
				"""	Train with batch gradient descent, manual checkpointing and
												manual tensorboard logging.
				"""

				# suggest hyperparameters
				(hidden_dim, num_layers, learning_rate,
				 dropout_rate, _) = self.suggestHParams(trial)

				# instantiate the model and move to device
				model = self.ModelType(input_dim=self.input_dim,
															 hidden_dim=hidden_dim,
															 output_dim=self.output_dim,
															 num_layers=num_layers,
															 dropout_rate=dropout_rate,
															 ).to(DEVICE)

				# loss criterion and optimizer
				criterion = nn.MSELoss()
				optimizer = optim.Adam(model.parameters(),
															 lr=learning_rate,
															 weight_decay=self.weight_decay,
															 )

				# TensorBoard logging setup
				run_name = f"trial_{trial.number}_hidden_{hidden_dim}_layers_{num_layers}_lr_{learning_rate:.4f}".replace(
						".", "_")
				writer = SummaryWriter(log_dir=os.path.join(self.log_pth, run_name))

				# train
				model.train()
				for epoch in range(self.epochs):
						optimizer.zero_grad()
						outputs = model(self.X_train_tensor)
						loss = criterion(outputs, self.y_train_tensor)
						loss.backward()
						optimizer.step()

						# log training loss
						writer.add_scalar('Loss/train', loss.item(), epoch)

						# validation
						model.eval()
						with torch.no_grad():
								val_outputs = model(self.X_val_tensor)
								val_loss = criterion(val_outputs, self.y_val_tensor).item()
								writer.add_scalar('Loss/val', val_loss, epoch)

								if val_loss < self.best_val_loss:
										self.best_val_loss = val_loss
										chkpt = {
												'epoch': epoch,
												'model_state_dict': model.state_dict(),
												'optimizer_state_dict': optimizer.state_dict(),
												'best_val_loss': self.best_val_loss,
										}
										self.best_chkpt_pth = os.path.join(
												self.chkpt_path, f"{run_name}.pth")
										torch.save(chkpt, self.best_chkpt_pth)
										# validate
										torch.load(self.best_chkpt_pth, weights_only=True, )

				writer.close()
				return val_loss

		def plainTest(self, trial: FrozenTrial) -> float:
				hidden_dim = trial.params['hidden_dim']
				num_layers = trial.params['num_layers']
				learning_rate = trial.params['learning_rate']
				dropout_rate = trial.params.get('dropout_rate')

				# load model to DEVICE
				model = self.ModelType(input_dim=self.input_dim,
															 hidden_dim=hidden_dim,
															 output_dim=self.output_dim,
															 num_layers=num_layers,
															 dropout_rate=dropout_rate,
															 ).to(DEVICE)

				# load model weights
				print("Loading checkpoint from", self.best_chkpt_pth)
				chkpt = torch.load(self.best_chkpt_pth, weights_only=True, )
				model.load_state_dict(chkpt['model_state_dict'],)

				run_name = f"TEST_vloss_{trial.value:.4f}_trial_{trial.number}_hidden_{hidden_dim}_layers_{num_layers}_lr_{learning_rate:.4f}".replace(
						".", "_")
				writer = SummaryWriter(log_dir=os.path.join(self.log_pth, run_name))

				# evaluate on the test set
				model.eval()
				with torch.no_grad():
						test_outputs = model(self.X_test_tensor)
						test_loss = nn.MSELoss()(test_outputs, self.y_test_tensor).item()
						writer.add_scalar('Loss/test', test_loss, )

				writer.close()
				return test_loss, self.best_chkpt_pth


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
								 norm_quats: Optional[bool] = False,
								 trans_norm: Optional[Normalization] = Normalization.Z_SCORE,
								 input_norm: Optional[Normalization] = Normalization.Z_SCORE,
								 target_norm: Optional[Normalization] = Normalization.Z_SCORE,
								 epochs: Optional[int] = 50,
								 num_layers: Optional[Tuple] = (1, 10, 1),
								 hidden_dim: Optional[Tuple] = (1, 32, 2),
								 learning_rate: Optional[Tuple] = (1e-4, 1e-2, 1e-3),
								 dropout_rate: Optional[Union[None, Tuple]] = (0, 0.4, 0.01),
								 log_domain: Optional[bool] = False,
								 weight_decay: Optional[float] = 0,
								 optim_trials: Optional[int] = 100,
								 batch_size: Optional[Tuple] = (18, 32, 64),
								 pruning: Optional[bool] = False,
								 bgd: Optional[bool] = False,
								 optuna_pbar: Optional[bool] = True,
								 lightning_pbar: Optional[bool] = False,
								 distribute: Optional[bool] = False,
								 ) -> None:

				# load training data per configuration
				pattern = os.path.join(DATA_PTH, 'keypoint/train',
															 folder_pth, f'*{pattern}*')
				self.data_files = glob.glob(pattern, recursive=False)

				self.test_size = test_size
				self.validation_size = validation_size
				self.random_state = random_state
				self.norm_quats = norm_quats
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
													norm_quats=self.norm_quats,
													trans_norm=self.trans_norm,
													input_norm=self.input_norm,
													target_norm=self.target_norm,
													move_gpu=False,
													)

				trainer = Trainer(X_train_tensor=td.X_train_tensor,
													X_test_tensor=td.X_test_tensor,
													X_val_tensor=td.X_val_tensor,
													y_train_tensor=td.y_train_tensor,
													y_test_tensor=td.y_test_tensor,
													y_val_tensor=td.y_val_tensor,
													epochs=self.epochs,
													model_type=WrappedMLP,
													input_dim=td.num_features,
													output_dim=td.num_targets,
													num_layers=self.num_layers,
													hidden_dim=self.hidden_dim,
													learning_rate=self.learning_rate,
													dropout_rate=self.dropout_rate,
													log_domain=self.log_domain,
													weight_decay=self.weight_decay,
													batch_size=self.batch_size,
													name=td.name,
													pbar=self.lightning_pbar,
													bgd=self.bgd,
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
													norm_quats=self.norm_quats,
													trans_norm=self.trans_norm,
													input_norm=self.input_norm,
													target_norm=self.target_norm,
													move_gpu=True,
													)

				trainer = Trainer(X_train_tensor=td.X_train_tensor,
													X_test_tensor=td.X_test_tensor,
													X_val_tensor=td.X_val_tensor,
													y_train_tensor=td.y_train_tensor,
													y_test_tensor=td.y_test_tensor,
													y_val_tensor=td.y_val_tensor,
													epochs=self.epochs,
													model_type=MLP,
													input_dim=td.num_features,
													output_dim=td.num_targets,
													num_layers=self.num_layers,
													hidden_dim=self.hidden_dim,
													learning_rate=self.learning_rate,
													dropout_rate=self.dropout_rate,
													log_domain=self.log_domain,
													weight_decay=self.weight_decay,
													batch_size=self.batch_size,
													name=td.name,
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
		data_group.add_argument('--test_size', type=float, metavar='float',help='Percentage of data split for testing.', default=0.3)
		data_group.add_argument('--val_size', type=float, metavar='float',help='Percentage of data split (from test size) for validation.', default=0.5)
		data_group.add_argument('--random_state', type=int, metavar='int',help='Percentage of data randomization.', default=40)
		data_group.add_argument('--norm_quats', action='store_true', help='Normalize quaternions.')
		data_group.add_argument('--trans_norm', type=parseNorm,help=f'Normalization method for translations [{NORMS}]', default=Normalization.Z_SCORE.value)
		data_group.add_argument('--input_norm', type=parseNorm,help=f'Normalization method for translations [{NORMS}]', default=Normalization.Z_SCORE.value)
		data_group.add_argument('--target_norm', type=parseNorm,help=f'Normalization method for translations [{NORMS}]', default=Normalization.Z_SCORE.value)
		# training
		train_group = parser.add_argument_group("Optimization Settings")
		train_group.add_argument('--epochs', type=int, metavar='int', help='Training epochs.', default=100)
		train_group.add_argument('--optim_trials', type=int, metavar='int',help='Number of optimization trials.', default=100)
		train_group.add_argument('--num_layers', type=parseIntTuple,help='Min, max and step value of hidden layers (int), eg. 2,10,2.', default='2,10,2')
		train_group.add_argument('--hidden_dim', type=parseIntTuple,help='Min, max and step value of hidden nodes (int), eg. 2,10,2.', default='2,10,2')
		train_group.add_argument('--batch_size', type=parseIntTuple,help='Choices for mini-batch training (int), eg. 18,32,64.', default='18,32,64')
		train_group.add_argument('--learning_rate', type=parseFloatTuple,help='Min, max and step value of learning rate (float), eg. 1e-4,1e-2,1e-2.', default='1e-4,1e-2,1e-2')
		train_group.add_argument('--dropout_rate', type=parseFloatTuple,help='Min, max and step value of dropout rate (float). Disable with none, eg. 0.0, 0.4, 0.01 or none.', default='0.0,0.4,0.01')
		train_group.add_argument('--log_domain', action='store_true',help='Change optimizer params logarithmically.')
		train_group.add_argument('--weight_decay', type=float, metavar='float',help='L2 regularization weight decay value, disable with 0.', default=0.01)
		train_group.add_argument('--pruning', action='store_true', help='Turn on pruning during optimization.')
		train_group.add_argument('--use_bgd', action='store_true', help='Use batch gradient descent training.')
		train_group.add_argument('--distribute', action='store_true', help='Train on multiple devices, use pruning, auto logging and checkpointing.')
		train_group.add_argument('--optuna_pbar', action='store_true', help='Show Optunas trial progessbar.')
		train_group.add_argument('--lightning_pbar', action='store_true', help='Show Lightnings detailed progessbar.')
		train_group.add_argument('--y', action='store_true', help='Discard asking for data cleaning and continue with training.')
		args = parser.parse_args()

		# clean data
		clean(args)

		Train(folder_pth=args.folder_pth,
					pattern=args.pattern,
					test_size=args.test_size,
					validation_size=args.val_size,
					random_state=args.random_state,
					norm_quats=args.norm_quats,
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
					).run()
