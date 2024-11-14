import os, sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
np.set_printoptions(threshold=sys.maxsize, suppress=True)

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets/detection/qdec/detection_1.json')
RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets/detection/qdec/results_1.json')

def mpjpe(ground_truth: np.ndarray, predicted: np.ndarray) -> float:
    """
       1D:  |p-q| = d(p,q) = sqrt((p - q)**2)
        nD: ||p-q|| = d(p,q) = sqrt((p1 - q1)**2  + (p2 - q2)**2 + ...)
        MPJPE: 1/N_F*1/N_J * sum_f,j(||p_f,j - q_f,j) where p in ground_truth and q in predicted
    """
    assert(len(ground_truth.shape) > 1)

    num_frames = ground_truth.shape[1]
    num_joints = ground_truth.shape[0]
    num_scalars = 1 if len(ground_truth.shape) == 2 else ground_truth.shape[2]
    denom = num_frames * num_joints

    pjpe = 0
    for idx in range(num_frames):
        for idy in range(num_joints):
            sqrt_sum = 0
            for idz in range(num_scalars):
                if num_scalars > 1:
                    sqrt_sum += np.square(ground_truth[idy][idx][idz] - predicted[idy][idx][idz])
                else:
                    sqrt_sum += np.square(ground_truth[idy][idx] - predicted[idy][idx])
            pjpe += np.sqrt(sqrt_sum)
    
    mean_pjpe =pjpe / denom
    return mean_pjpe

def qdecResults():
    df = pd.read_json(DATA, orient='index')

    # detections
    angle_prox = np.array(df['prox'].apply(lambda x: x['angle']).values)
    angle_med = np.array(df['med'].apply(lambda x: x['angle']).values)
    angle_dist = np.array(df['dist'].apply(lambda x: x['angle']).values)

    # readings
    qdec_angle_prox = np.array(df['prox'].apply(lambda x: x['qdec_angle']).values)
    qdec_angle_med = np.array(df['med'].apply(lambda x: x['qdec_angle']).values)
    qdec_angle_dist = np.array(df['dist'].apply(lambda x: x['qdec_angle']).values)

    # error recorded
    error_prox = np.array(df['prox'].apply(lambda x: x['error']).values)
    error_med = np.array(df['med'].apply(lambda x: x['error']).values)
    error_dist = np.array(df['dist'].apply(lambda x: x['error']).values)
    # mean
    prox_mean_err = np.mean(error_prox)
    med_mean_err  = np.mean(error_med)
    dist_mean_err  = np.mean(error_dist)

    # rmse
    prox_rmse = root_mean_squared_error(qdec_angle_prox, angle_prox)
    med_rmse = root_mean_squared_error(qdec_angle_med, angle_med)
    dist_rmse = root_mean_squared_error(qdec_angle_dist, angle_dist)

    # concat values
    all_angle = np.concatenate((angle_prox, angle_med, angle_dist))
    all_qdec_angle = np.concatenate((qdec_angle_prox, qdec_angle_med, qdec_angle_dist))
    all_error = np.concatenate((error_prox, error_med, error_dist))

    all_rmse = root_mean_squared_error(all_qdec_angle, all_angle)
    # all_mpjpe = mpjpe(np.vstack((qdec_angle_prox, qdec_angle_med, qdec_angle_dist)), np.vstack((angle_prox, angle_med, angle_dist)))
    all_mpjpe = mpjpe(np.vstack((qdec_angle_prox, qdec_angle_med, qdec_angle_dist)), np.vstack((angle_prox, angle_med, angle_dist)))

    results = {'proximal': {'rmse_rad': prox_rmse, 'rmse_deg': np.rad2deg(prox_rmse), 'mean_err_rad': prox_mean_err, 'mean_err_deg': np.rad2deg(prox_mean_err)}, 
                'medial': {'rmse_rad': med_rmse, 'rmse_deg': np.rad2deg(med_rmse), 'mean_err_rad': med_mean_err, 'mean_err_deg': np.rad2deg(med_mean_err)},
                'distal': {'rmse_rad': dist_rmse, 'rmse_deg': np.rad2deg(dist_rmse), 'mean_err_rad': dist_mean_err, 'mean_err_deg': np.rad2deg(dist_mean_err)},
                'rmse_rad': all_rmse, 'mean_err_rad': np.mean(all_error), 
                'rmse_deg': np.rad2deg(all_rmse), 'mean_err_deg': np.rad2deg(np.mean(all_error)),
                'mpjpe_rad': all_mpjpe, 'mpjpe_deg': np.rad2deg(all_mpjpe), 
            }
    with open(RES, 'w') as fw:
        json.dump(results, fw, indent=4)

qdecResults()
