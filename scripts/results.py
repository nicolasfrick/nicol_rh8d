import os, sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
np.set_printoptions(threshold=sys.maxsize, suppress=True)
from util import *

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets/detection/qdec/detection_2.json')
RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets/detection/qdec/results_2.json')

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

# qdecResults()

def modelResults():
    df_angle_dict_a = readDetectionDataset(os.path.join(DATA_PTH, 'keypoint/detection_validation_vertical.json'))
    df_kpt_dict_a = readDetectionDataset(os.path.join(DATA_PTH, 'keypoint/kpts3D_validation_vertical.json'))

    df_angle_dict_b = readDetectionDataset(os.path.join(DATA_PTH, 'keypoint/detection_validation_palm_pointing_up.json'))
    df_kpt_dict_b = readDetectionDataset(os.path.join(DATA_PTH, 'keypoint/kpts3D_validation_palm_pointing_up.json'))

    df_angle_dict_c = readDetectionDataset(os.path.join(DATA_PTH, 'keypoint/detection_validation_palm_pointing_down.json'))
    df_kpt_dict_c = readDetectionDataset(os.path.join(DATA_PTH, 'keypoint/kpts3D_validation_palm_pointing_down.json'))

    df_angle_dict_d = readDetectionDataset(os.path.join(DATA_PTH, 'keypoint/detection_validation_palm_pointing_left.json.json'))
    df_kpt_dict_d = readDetectionDataset(os.path.join(DATA_PTH, 'keypoint/kpts3D_validation_palm_pointing_left.json'))

    df_angle_dict_e = readDetectionDataset(os.path.join(DATA_PTH, 'keypoint/detection_validation_error.json'))
    df_kpt_dict_e = readDetectionDataset(os.path.join(DATA_PTH, 'keypoint/kpts3D_validation_error.json'))

    results = {'a':{}, 'b':{}, 'c':{}, 'd':{}, 'e':{}}
    combi_mae = 0
    combi_rmse = 0
    combi_rmse_kpt = 0

    # angles a
    for joint, angle_df in df_angle_dict_a.items():
        df = angle_df.dropna(subset=["angle"])
        angles = df["angle"].to_numpy()
        inf_angles = df["inf_angle"].to_numpy()
        # mae
        mae_rad = np.mean(np.abs(inf_angles - angles))
        mae_deg = np.rad2deg(mae_rad)
        # rmse
        rmse_rad = root_mean_squared_error(angles, inf_angles)
        rmse_deg = np.rad2deg(rmse_rad)
        # result
        results['a'].update( {joint: {'mae_rad': mae_rad, 'mae_deg': mae_deg, 'rmse_rad': rmse_rad, 'rmse_deg': rmse_deg}} )
        combi_mae += mae_rad
        combi_rmse += rmse_rad**2
    # keypoints a 
    for joint, kpt_df in df_kpt_dict_a.items():
        df = kpt_df.dropna(subset=["trans"])
        trans = np.vstack(df["trans"].to_numpy()) 
        inf_trans = np.vstack(df["inf_trans"].to_numpy()) 
        rmse = root_mean_squared_error(trans, inf_trans)
        results['a'].update( {joint: {'rmse_m': rmse}} )
        combi_rmse_kpt += rmse**2

    # angles b
    for joint, angle_df in df_angle_dict_b.items():
        df = angle_df.dropna(subset=["angle"])
        angles = df["angle"].to_numpy()
        inf_angles = df["inf_angle"].to_numpy()
        # mae
        mae_rad = np.mean(np.abs(inf_angles - angles))
        mae_deg = np.rad2deg(mae_rad)
        # rmse
        rmse_rad = root_mean_squared_error(angles, inf_angles)
        rmse_deg = np.rad2deg(rmse_rad)
        # result
        results['b'].update( {joint: {'mae_rad': mae_rad, 'mae_deg': mae_deg, 'rmse_rad': rmse_rad, 'rmse_deg': rmse_deg}} )
        combi_mae += mae_rad
        combi_rmse += rmse_rad**2
    # keypoints b 
    for joint, kpt_df in df_kpt_dict_b.items():
        df = kpt_df.dropna(subset=["trans"])
        trans = np.vstack(df["trans"].to_numpy()) 
        inf_trans = np.vstack(df["inf_trans"].to_numpy()) 
        rmse = root_mean_squared_error(trans, inf_trans)
        results['b'].update( {joint: {'rmse_m': rmse}} )
        combi_rmse_kpt += rmse**2

    # angles c
    for joint, angle_df in df_angle_dict_c.items():
        df = angle_df.dropna(subset=["angle"])
        angles = df["angle"].to_numpy()
        inf_angles = df["inf_angle"].to_numpy()
        # mae
        mae_rad = np.mean(np.abs(inf_angles - angles))
        mae_deg = np.rad2deg(mae_rad)
        # rmse
        rmse_rad = root_mean_squared_error(angles, inf_angles)
        rmse_deg = np.rad2deg(rmse_rad)
        # result
        results['c'].update( {joint: {'mae_rad': mae_rad, 'mae_deg': mae_deg, 'rmse_rad': rmse_rad, 'rmse_deg': rmse_deg}} )
        combi_mae += mae_rad
        combi_rmse += rmse_rad**2
    # keypoints c
    for joint, kpt_df in df_kpt_dict_c.items():
        df = kpt_df.dropna(subset=["trans"])
        trans = np.vstack(df["trans"].to_numpy()) 
        inf_trans = np.vstack(df["inf_trans"].to_numpy()) 
        rmse = root_mean_squared_error(trans, inf_trans)
        results['c'].update( {joint: {'rmse_m': rmse}} )
        combi_rmse_kpt += rmse**2

    # angles d
    for joint, angle_df in df_angle_dict_d.items():
        df = angle_df.dropna(subset=["angle"])
        angles = df["angle"].to_numpy()
        inf_angles = df["inf_angle"].to_numpy()
        # mae
        mae_rad = np.mean(np.abs(inf_angles - angles))
        mae_deg = np.rad2deg(mae_rad)
        # rmse
        rmse_rad = root_mean_squared_error(angles, inf_angles)
        rmse_deg = np.rad2deg(rmse_rad)
        # result
        results['d'].update( {joint: {'mae_rad': mae_rad, 'mae_deg': mae_deg, 'rmse_rad': rmse_rad, 'rmse_deg': rmse_deg}} )
        combi_mae += mae_rad
        combi_rmse += rmse_rad**2
    # keypoints d
    for joint, kpt_df in df_kpt_dict_d.items():
        df = kpt_df.dropna(subset=["trans"])
        trans = np.vstack(df["trans"].to_numpy()) 
        inf_trans = np.vstack(df["inf_trans"].to_numpy()) 
        rmse = root_mean_squared_error(trans, inf_trans)
        results['d'].update( {joint: {'rmse_m': rmse}} )
        combi_rmse_kpt += rmse**2

    # angles error test
    err_mae = 0
    err_rmse = 0
    err_rmse_kpt = 0
    for joint, angle_df in df_angle_dict_e.items():
        df = angle_df.dropna(subset=["angle"])
        angles = df["angle"].to_numpy()
        inf_angles = df["inf_angle"].to_numpy()
        # mae
        mae_rad = np.mean(np.abs(inf_angles - angles))
        mae_deg = np.rad2deg(mae_rad)
        # rmse
        rmse_rad = root_mean_squared_error(angles, inf_angles)
        rmse_deg = np.rad2deg(rmse_rad)
        # result
        results['e'].update( {joint: {'mae_rad': mae_rad, 'mae_deg': mae_deg, 'rmse_rad': rmse_rad, 'rmse_deg': rmse_deg}} )
        err_mae += mae_rad
        err_rmse += rmse_rad**2
    # keypoints error test 
    for joint, kpt_df in df_kpt_dict_e.items():
        df = kpt_df.dropna(subset=["trans"])
        trans = np.vstack(df["trans"].to_numpy()) 
        inf_trans = np.vstack(df["inf_trans"].to_numpy()) 
        rmse = root_mean_squared_error(trans, inf_trans)
        results['e'].update( {joint: {'rmse_m': rmse}} )
        err_rmse_kpt += rmse**2

    combined_mae_rad = combi_mae/4
    combined_mae_deg = np.rad2deg(combined_mae_rad)
    combined_rmse_rad = np.sqrt(combi_rmse/4)
    combined_rmse_deg = np.rad2deg(combined_rmse_rad)
    combined_err_mae_rad = err_mae/4
    combined_err_mae_deg = np.rad2deg(combined_err_mae_rad)
    combined_err_rmse_rad = np.sqrt(err_rmse/4)
    combined_err_rmse_deg = np.rad2deg(combined_err_rmse_rad)
    results.update( {"combined_mae_rad":combined_mae_rad,
                                    "combined_mae_deg":combined_mae_deg,
                                    "combined_rmse_rad":combined_rmse_rad,
                                    "combined_rmse_deg":combined_rmse_deg, 
                                    "combined_rmse_kpt": np.sqrt(combi_rmse_kpt/4),
                                    "combined_err_mae_rad": combined_err_mae_rad, 
                                    "combined_err_mae_deg": combined_err_mae_deg, 
                                    "combined_err_rmse_rad": combined_err_rmse_rad, 
                                    "combined_err_rmse_deg": combined_err_rmse_deg, 
                                    "combined_err_rmse_kpt": np.sqrt(err_rmse_kpt)} )

    with open(os.path.join(DATA_PTH, 'keypoint/results.json'), 'w') as fw:
        json.dump(results, fw, indent=4)

modelResults()


