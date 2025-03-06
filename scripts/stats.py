import os
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from dtaidistance import dtw
import matplotlib.pyplot as plt
from scipy.signal import correlate
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.feature_selection import mutual_info_regression
from util import *

writer = open(os.path.join(DATA_PTH, "keypoint/post_processed/stats.txt"), "w")

def writeStats(txt: str) -> None:
	writer.write(txt)
	writer.write("\n")
	print(txt)

detection_dct = readDetectionDataset(os.path.join(DATA_PTH, "keypoint/post_processed/detection_smoothed.json"))
angles = pd.DataFrame({
	'angle_M1': detection_dct['jointM1']['angle'],
	'angle_M2': detection_dct['jointM2']['angle'],
	'angle_M3': detection_dct['jointM3']['angle']
})

quats = pd.read_json(os.path.join(DATA_PTH, 'keypoint/post_processed/quaternions.json'), orient='index')
quats = pd.DataFrame(quats['quat'].tolist(), index=quats.index, columns=['quat_0', 'quat_1', 'quat_2', 'quat_3'])

eulers = pd.read_json(os.path.join(DATA_PTH, 'keypoint/post_processed/eulers.json'), orient='index')
eulers = pd.DataFrame(eulers['euler'].tolist(), index=eulers.index, columns=['roll', 'pich', 'yaw'])

forces = pd.read_json(os.path.join(DATA_PTH, 'keypoint/post_processed/f_tcp.json'), orient='index')
forces = pd.DataFrame(forces['f_tcp'].tolist(), index=forces.index, columns=['fx', 'fy', 'fz'])

relatives = {'quats':quats, 'eulers':eulers, 'forces':forces}

angles.index = pd.RangeIndex(0, len(angles))
quats.index = pd.RangeIndex(0, len(quats))
eulers.index = pd.RangeIndex(0, len(eulers))
forces.index = pd.RangeIndex(0, len(forces))

def relate() -> None:

	for relative, rel_data in relatives.items():
		writeStats(f"\n\n========={relative}============\n")

		writeStats("=== Cross-Correlation ===")
		for angle in angles.columns:
			for col in rel_data.columns:
				# cross-correlation
				corr = correlate(angles[angle], rel_data[col], mode='full')
				lags = np.arange(-len(angles) + 1, len(angles))
				# normalize by length to get correlation coefficient
				corr = corr / len(angles)
				max_idx = np.argmax(np.abs(corr))
				max_corr = corr[max_idx]
				lag_at_max = lags[max_idx]
				writeStats(f"{angle} vs {col}: Max Corr = {max_corr:.3f} at lag {lag_at_max}")

		writeStats("\n=== Mutual Information ===")
		for angle in angles.columns:
			# compute mi between angle and all quaternion components
			mi = mutual_info_regression(rel_data, angles[angle], discrete_features=False)
			writeStats(f"{angle} vs {relative}: MI = {mi} {rel_data.columns.to_list()}")

		writeStats("\n=== Granger Causality ===")
		data = pd.concat([angles, rel_data], axis=1)
		max_lag = 5
		for angle in angles.columns:
			for col in rel_data.columns:
				try:
					result = grangercausalitytests(data[[angle, col]], maxlag=max_lag)
					min_p = min([result[i+1][0]['ssr_ftest'][1] for i in range(max_lag)])
					writeStats(f"{col} -> {angle}: Min p-value = {min_p:.4f} (across lags 1-{max_lag})")
				except ValueError as e:
					writeStats(f"{col} -> {angle}: Error - {e}")

		writeStats("\n=== Linear Regression ===")
		for angle in angles.columns:
			# fit linear regression model
			reg = LinearRegression()
			reg.fit(rel_data, angles[angle])
			r_squared = reg.score(rel_data, angles[angle])
			coefs = reg.coef_
			writeStats(f"{angle}: R² = {r_squared:.3f}, Coefficients = {coefs} {rel_data.columns.to_list()}")

		writeStats("\n=== Dynamic Time Warping (DTW) ===")
		for angle in angles.columns:
			for col in rel_data.columns:
				# compute DTW distance
				dist = dtw.distance(angles[angle].values, rel_data[col].values)
				# normalize by sequence length for comparability
				normalized_dist = dist / len(angles)
				writeStats(f"{angle} vs {col}: Normalized DTW Distance = {normalized_dist:.3f}")
			
relate()

	# pdf
data = pd.concat([angles, quats], axis=1)

def analyze_distribution(series, name):
	writeStats(f"\n=== {name} Distribution Analysis ===")
	# Basic statistics
	writeStats(f"Mean: {series.mean():.3f}, Std: {series.std():.3f}")
	writeStats(f"Skewness: {stats.skew(series):.3f}, Kurtosis: {stats.kurtosis(series):.3f}")

	# Convert Series to NumPy array to avoid pandas indexing error
	series_array = series.to_numpy()

	# Histogram and KDE
	plt.figure(figsize=(10, 6))
	df = pd.DataFrame(series)
	sns.histplot(data=df, y=df.columns.values[0], x=df.index, bins=50, kde=True, stat='density', label='Data')
	
	# Fit candidate distributions
	dist_names = ['norm', 'gamma', 'beta']
	for dist_name in dist_names:
		if dist_name == 'beta':
			# Beta requires data in (0,1); scale if needed
			scaled = (series_array - series_array.min() + 1e-6) / (series_array.max() - series_array.min() + 2e-6)
			params = stats.beta.fit(scaled)
			dist = stats.beta(*params)
		else:
			params = getattr(stats, dist_name).fit(series_array)
			dist = getattr(stats, dist_name)(*params)
		x = np.linspace(series_array.min(), series_array.max(), 100)
		pdf = dist.pdf(x)
		plt.plot(x, pdf, label=dist_name)

	plt.title(f'{name} Distribution')
	plt.legend()
	plt.show()

	# Goodness-of-fit (Kolmogorov-Smirnov test)
	for dist_name in dist_names:
		if dist_name == 'beta':
			scaled = (series_array - series_array.min() + 1e-6) / (series_array.max() - series_array.min() + 2e-6)
			params = stats.beta.fit(scaled)
			ks_stat, p_val = stats.kstest(scaled, 'beta', params)
		else:
			params = getattr(stats, dist_name).fit(series_array)
			ks_stat, p_val = stats.kstest(series_array, dist_name, params)
		writeStats(f"{dist_name}: KS Stat = {ks_stat:.3f}, p-value = {p_val:.3f}")

def findPdf() -> None:
	# analyze each column
	for col in data.columns:
		analyze_distribution(data[col], col)

	# KDE
	plt.figure(figsize=(12, 8))
	for col in data.columns:
		sns.kdeplot(data=data, y=col, x=data.index, label=col)  
	plt.title('Kernel Density Estimates')
	plt.legend()
	plt.show()

# findPdf()

writer.close()
