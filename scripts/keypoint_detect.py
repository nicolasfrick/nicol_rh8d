

import cv2
import ppyutil.image_plot
from pose_detector import PoseDetector
import warnings
warnings.filterwarnings('ignore')

MMDETDIR = '/home/allgeuer/Programs/DeepStack/envs/mmdev/mmdetection/'
MMPOSEDIR = '/home/allgeuer/Programs/DeepStack/envs/mmdev/mmpose/'

def plot_images(*images):
    ppyutil.image_plot.plot_images(tuple(image[:,:,::-1] for image in images))