#!/usr/bin/env python3

import os, sys, cv2, json, numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../mmscripts'))
import ppyutil.image_plot
from pose_detector import PoseDetector

HOME = os.path.expanduser("~")
CONDA = os.environ['CONDA_PREFIX']
MMDETDIR = os.path.join(CONDA,'lib/python3.9/site-packages/mmdet/.mim')
MMPOSEDIR = os.path.join(CONDA,'lib/python3.9/site-packages/mmpose/.mim')

def plot_images(*images):
    ppyutil.image_plot.plot_images(tuple(image[:,:,::-1] for image in images))

def detect():
    with PoseDetector(
            det=(os.path.join(MMPOSEDIR, 'demo/mmdetection_cfg/cascade_rcnn_x101_64x4d_fpn_1class.py'), 'https://download.openmmlab.com/mmpose/mmdet_pretrained/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k-dac19597_20201030.pth'),
            # dd=(os.path.join(MMPOSEDIR, 'configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/onehand10k/res50_onehand10k_256x256.py'), 'https://download.openmmlab.com/mmpose/top_down/resnet/res50_onehand10k_256x256-e67998f6_20200813.pth'),
            dd=(os.path.join(MMPOSEDIR, 'configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/onehand10k/hrnetv2_w18_onehand10k_256x256_dark.py'), 'https://download.openmmlab.com/mmpose/hand/dark/hrnetv2_w18_onehand10k_256x256_dark-a2f80c64_20210330.pth'),
        ) as detector:
        
        demo_hands = cv2.imread(os.path.join(HOME, 'nicol_rh8d/datasets/OpenMM/data/color_img3.jpg'))
        demo_hands = cv2.resize(demo_hands, (630,900))
        result, _ = detector.process_image(demo_hands, output_vis=True)
        plot_images(result.vis_image.det, result.vis_image.dd)


def convertColors():
    demo_hands = cv2.imread(os.path.join(HOME, 'nicol_rh8d/datasets/OpenMM/data/2.jpeg'))

    # with open(os.path.join(HOME, 'nicol_rh8d/datasets/OpenMM/data/out.txt'), 'w') as f:
    #     json.dump(demo_hands.tolist(), f, indent=1)

    # convert
    # ong_rgba = (172,219,255)
    # thresh_rgba = (100, 100, 100)
    # height, width, _ = demo_hands.shape
    # for x in range(0,width):
    #     for y in range(0,height):
    #         pixel = demo_hands[y,x]
    #         if all(pixel <= thresh_rgba):
    #             demo_hands[y,x] = ong_rgba

    # invert
    demo_hands = ~demo_hands

    cv2.imwrite(os.path.join(HOME, 'nicol_rh8d/datasets/OpenMM/data/color_img3.jpg'), demo_hands)
    # cv2.imshow("image", demo_hands)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# convertColors()
detect()


