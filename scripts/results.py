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
    num_joints = len(df_angle_dict_a.keys())
    num_kpts = len(df_kpt_dict_a.keys())

    # angles a
    combi_mae_a = 0
    combi_rmse_a = 0
    combi_rmse_kpt_a = 0
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
        combi_mae_a += mae_rad
        combi_rmse_a += rmse_rad**2
    # keypoints a 
    for joint, kpt_df in df_kpt_dict_a.items():
        df = kpt_df.dropna(subset=["trans"])
        trans = np.vstack(df["trans"].to_numpy()) 
        inf_trans = np.vstack(df["inf_trans"].to_numpy()) 
        rmse = root_mean_squared_error(trans, inf_trans)
        results['a'].update( {joint: {'rmse_m': rmse}} )
        combi_rmse_kpt_a += rmse**2
    combi_mae_a /= num_joints
    combi_rmse_a /= num_joints
    combi_rmse_kpt_a /= num_kpts

    # angles b
    combi_mae_b = 0
    combi_rmse_b = 0
    combi_rmse_kpt_b = 0
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
        combi_mae_b += mae_rad
        combi_rmse_b += rmse_rad**2
    # keypoints b 
    for joint, kpt_df in df_kpt_dict_b.items():
        df = kpt_df.dropna(subset=["trans"])
        trans = np.vstack(df["trans"].to_numpy()) 
        inf_trans = np.vstack(df["inf_trans"].to_numpy()) 
        rmse = root_mean_squared_error(trans, inf_trans)
        results['b'].update( {joint: {'rmse_m': rmse}} )
        combi_rmse_kpt_b += rmse**2
    combi_mae_b /= num_joints
    combi_rmse_b /= num_joints
    combi_rmse_kpt_b /= num_kpts

    # angles c
    combi_mae_c = 0
    combi_rmse_c = 0
    combi_rmse_kpt_c = 0
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
        combi_mae_c += mae_rad
        combi_rmse_c += rmse_rad**2
    # keypoints c
    for joint, kpt_df in df_kpt_dict_c.items():
        df = kpt_df.dropna(subset=["trans"])
        trans = np.vstack(df["trans"].to_numpy()) 
        inf_trans = np.vstack(df["inf_trans"].to_numpy()) 
        rmse = root_mean_squared_error(trans, inf_trans)
        results['c'].update( {joint: {'rmse_m': rmse}} )
        combi_rmse_kpt_c += rmse**2
    combi_mae_c /= num_joints
    combi_rmse_c /= num_joints
    combi_rmse_kpt_c /= num_kpts

    # angles d
    combi_mae_d = 0
    combi_rmse_d = 0
    combi_rmse_kpt_d = 0
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
        combi_mae_d += mae_rad
        combi_rmse_d += rmse_rad**2
    # keypoints d
    for joint, kpt_df in df_kpt_dict_d.items():
        df = kpt_df.dropna(subset=["trans"])
        trans = np.vstack(df["trans"].to_numpy()) 
        inf_trans = np.vstack(df["inf_trans"].to_numpy()) 
        rmse = root_mean_squared_error(trans, inf_trans)
        results['d'].update( {joint: {'rmse_m': rmse}} )
        combi_rmse_kpt_d += rmse**2
    combi_mae_d /= num_joints
    combi_rmse_d /= num_joints
    combi_rmse_kpt_d /= num_kpts

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
    err_mae /= num_joints
    err_rmse /= num_joints
    err_rmse_kpt /= num_kpts

    # jointwise combined mae degree
    combi_mae_a_deg = np.rad2deg(combi_mae_a)
    combi_rmse_a_deg = np.rad2deg(combi_rmse_a)
    combi_mae_b_deg = np.rad2deg(combi_mae_b)
    combi_rmse_b_deg = np.rad2deg(combi_rmse_b)
    combi_mae_c_deg = np.rad2deg(combi_mae_c)
    combi_rmse_c_deg = np.rad2deg(combi_rmse_c)
    combi_mae_d_deg = np.rad2deg(combi_mae_d)
    combi_rmse_d_deg = np.rad2deg(combi_rmse_d)

    # loopwise mae/ rmse
    combined_mae_rad = (combi_mae_a + combi_mae_b + combi_mae_c + combi_mae_d) / 4
    combined_mae_deg = np.rad2deg(combined_mae_rad)
    combined_rmse_rad = np.sqrt((combi_rmse_a + combi_rmse_b + combi_rmse_c + combi_rmse_d)/4)
    combined_rmse_deg = np.rad2deg(combined_rmse_rad)
    combined_err_mae_rad = err_mae
    combined_err_mae_deg = np.rad2deg(combined_err_mae_rad)
    combined_err_rmse_rad = np.sqrt(err_rmse)
    combined_err_rmse_deg = np.rad2deg(combined_err_rmse_rad)
    results.update( {"combi_mae_a_rad": combi_mae_a,
                                    "combi_mae_a_deg": combi_mae_a_deg,
                                    "combi_rmse_a_rad": combi_rmse_a,
                                    "combi_rmse_a_deg": combi_rmse_a_deg,
                                    "combi_rmse_kpt_a_rad": combi_rmse_kpt_a,
                                    "combi_mae_b_rad": combi_mae_b,
                                    "combi_mae_b_deg": combi_mae_b_deg,
                                    "combi_rmse_b_rad": combi_rmse_b,
                                    "combi_rmse_b_deg": combi_rmse_b_deg,
                                    "combi_rmse_kpt_b_rad": combi_rmse_kpt_b,
                                    "combi_mae_c_rad": combi_mae_c,
                                    "combi_mae_c_deg": combi_mae_c_deg,
                                    "combi_rmse_c_rad": combi_rmse_c,
                                    "combi_rmse_c_deg": combi_rmse_c_deg,
                                    "combi_rmse_kpt_c_rad": combi_rmse_kpt_c,
                                    "combi_mae_d_rad": combi_mae_d,
                                    "combi_mae_d_deg": combi_mae_d_deg,
                                    "combi_rmse_d_rad": combi_rmse_d,
                                    "combi_rmse_d_deg": combi_rmse_d_deg,
                                    "combi_rmse_kpt_d_rad": combi_rmse_kpt_d,
                                    "combined_mae_rad":combined_mae_rad,
                                    "combined_mae_deg":combined_mae_deg,
                                    "combined_rmse_rad":combined_rmse_rad,
                                    "combined_rmse_deg":combined_rmse_deg, 
                                    "combined_rmse_kpt": np.sqrt((combi_rmse_kpt_a + combi_rmse_kpt_b + combi_rmse_kpt_c + combi_rmse_kpt_d)/4),
                                    "combined_err_mae_rad": combined_err_mae_rad, 
                                    "combined_err_mae_deg": combined_err_mae_deg, 
                                    "combined_err_rmse_rad": combined_err_rmse_rad, 
                                    "combined_err_rmse_deg": combined_err_rmse_deg, 
                                    "combined_err_rmse_kpt": np.sqrt(err_rmse_kpt)} )

    with open(os.path.join(DATA_PTH, 'keypoint/results.json'), 'w') as fw:
        json.dump(results, fw, indent=4)

modelResults()


