import os, sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
np.set_printoptions(threshold=sys.maxsize, suppress=True)

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets/detection/qdec/detection_1.json')
RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets/detection/qdec/results_1.json')

def mpjpe(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    # euclidean distances for each joint
    errors = np.linalg.norm(predicted - ground_truth, axis=-1)
    return np.mean(errors)

def qdecResults():
    df = pd.read_json(DATA, orient='index')

    angle_prox = np.array(df['prox'].apply(lambda x: x['angle']).values)
    angle_med = np.array(df['med'].apply(lambda x: x['angle']).values)
    angle_dist = np.array(df['dist'].apply(lambda x: x['angle']).values)

    qdec_angle_prox = np.array(df['prox'].apply(lambda x: x['qdec_angle']).values)
    qdec_angle_med = np.array(df['med'].apply(lambda x: x['qdec_angle']).values)
    qdec_angle_dist = np.array(df['dist'].apply(lambda x: x['qdec_angle']).values)

    error_prox = np.array(df['prox'].apply(lambda x: x['error']).values)
    error_med = np.array(df['med'].apply(lambda x: x['error']).values)
    error_dist = np.array(df['dist'].apply(lambda x: x['error']).values)

    prox_rmse = root_mean_squared_error(qdec_angle_prox, angle_prox)
    med_rmse = root_mean_squared_error(qdec_angle_med, angle_med)
    dist_rmse = root_mean_squared_error(qdec_angle_dist, angle_dist)

    all_angle = np.concatenate((angle_prox, angle_med, angle_dist))
    all_qdec_angle = np.concatenate((qdec_angle_prox, qdec_angle_med, qdec_angle_dist))
    all_error = np.concatenate((error_prox, error_med, error_dist))

    all_rmse = root_mean_squared_error(all_qdec_angle, all_angle)
    all_mpjpe = mpjpe(all_angle, all_qdec_angle)

    results = {'proximal': {'rmse_rad': prox_rmse, 'mean_err_rad': np.mean(error_prox), 'rmse_deg': np.rad2deg(prox_rmse), 'mean_err_deg': np.rad2deg(np.mean(error_prox)), 'mpjpe': mpjpe(angle_prox, qdec_angle_prox)}, 
                'medial': {'rmse_rad': med_rmse, 'mean_err_rad': np.mean(error_med), 'rmse_deg': np.rad2deg(med_rmse), 'mean_err_deg': np.rad2deg(np.mean(error_med)), 'mpjpe': mpjpe(angle_med, qdec_angle_med)},
                'distal': {'rmse_rad': dist_rmse, 'mean_err_rad': np.mean(error_dist), 'rmse_deg': np.rad2deg(dist_rmse), 'mean_err_deg': np.rad2deg(np.mean(error_dist)), 'mpjpe': mpjpe(angle_dist, qdec_angle_dist)},
                'rmse_rad': all_rmse, 'mean_err_rad': np.mean(all_error), 
                'rmse_deg': np.rad2deg(all_rmse), 'mean_err_deg': np.rad2deg(np.mean(all_error)),
                'mpjpe_rad': all_mpjpe, 'mpjpe_deg': np.rad2deg(all_mpjpe), 
            }
    with open(RES, 'w') as fw:
        json.dump(results, fw, indent=4)

qdecResults()
