#!/usr/bin/env python3

import os, cv2
import ppyutil.image_plot
from pose_detector import PoseDetector
import warnings
warnings.filterwarnings('ignore')

HOME = os.path.expanduser("~")
CONDA = os.environ['CONDA_PREFIX']
MMDETDIR = os.path.join(CONDA,'lib/python3.9/site-packages/mmdet/.mim')
MMPOSEDIR = os.path.join(CONDA,'lib/python3.9/site-packages/mmpose/.mim')

def plot_images(*images):
    ppyutil.image_plot.plot_images(tuple(image[:,:,::-1] for image in images))

with PoseDetector(
        det=(os.path.join(MMPOSEDIR, 'demo/mmdetection_cfg/cascade_rcnn_x101_64x4d_fpn_1class.py'), 'https://download.openmmlab.com/mmpose/mmdet_pretrained/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k-dac19597_20201030.pth'),
        # dd=(os.path.join(MMPOSEDIR, 'configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/onehand10k/res50_onehand10k_256x256.py'), 'https://download.openmmlab.com/mmpose/top_down/resnet/res50_onehand10k_256x256-e67998f6_20200813.pth'),
        dd=(os.path.join(MMPOSEDIR, 'configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/onehand10k/hrnetv2_w18_onehand10k_256x256_dark.py'), 'https://download.openmmlab.com/mmpose/top_down/resnet/res50_onehand10k_256x256-e67998f6_20200813.pth'),
    ) as detector:
    
    while input("Press key to scan, q to escape") != 'q':
        demo_hands = cv2.imread(os.path.join(HOME, 'nicol_rh8d/datasets/OpenMM/data/5.jpeg'))
        # demo_hands = cv2.resize(demo_hands, (256,256))
        result, _ = detector.process_image(demo_hands, output_vis=True)
        plot_images(result.vis_image.det, result.vis_image.dd)