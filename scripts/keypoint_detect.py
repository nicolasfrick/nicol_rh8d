#!/usr/bin/env python3

import os, sys, cv2, json, rospy, numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../mmscripts'))
import ppyutil.image_plot
from pose_detector import PoseDetector

HOME = os.path.expanduser("~")
CONDA = os.environ['CONDA_PREFIX']
MMDETDIR = os.path.join(CONDA,'lib/python3.9/site-packages/mmdet/.mim')
MMPOSEDIR = os.path.join(CONDA,'lib/python3.9/site-packages/mmpose/.mim')

def plot_images(self, *images):
    ppyutil.image_plot.plot_images(tuple(image[:,:,::-1] for image in images))

class KeypointDetector:

    def __init__(self, 
                 vis:bool=True, 
                 human:bool=False, 
                 color_topic:str='out', 
                 detect_topic:str='img_detect'
                 ) -> None:
        self.human = human
        self.vis = vis

        self.detector = PoseDetector(
                det=(os.path.join(MMPOSEDIR, 'demo/mmdetection_cfg/cascade_rcnn_x101_64x4d_fpn_1class.py'), 'https://download.openmmlab.com/mmpose/mmdet_pretrained/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k-dac19597_20201030.pth'),
                # dd=(os.path.join(MMPOSEDIR, 'configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/onehand10k/res50_onehand10k_256x256.py'), 'https://download.openmmlab.com/mmpose/top_down/resnet/res50_onehand10k_256x256-e67998f6_20200813.pth'),
                dd=(os.path.join(MMPOSEDIR, 'configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/onehand10k/hrnetv2_w18_onehand10k_256x256_dark.py'), 'https://download.openmmlab.com/mmpose/hand/dark/hrnetv2_w18_onehand10k_256x256_dark-a2f80c64_20210330.pth'),
                )
        
        self.bridge = CvBridge()
        self.img_sub = rospy.Subscriber(color_topic, Image, self.camCB, queue_size=100)
        self.img_pub = rospy.Publisher(detect_topic, Image, queue_size=10)

    def detect(self, cv_image):
        # with PoseDetector(
        #         det=(os.path.join(MMPOSEDIR, 'demo/mmdetection_cfg/cascade_rcnn_x101_64x4d_fpn_1class.py'), 'https://download.openmmlab.com/mmpose/mmdet_pretrained/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k-dac19597_20201030.pth'),
        #         # dd=(os.path.join(MMPOSEDIR, 'configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/onehand10k/res50_onehand10k_256x256.py'), 'https://download.openmmlab.com/mmpose/top_down/resnet/res50_onehand10k_256x256-e67998f6_20200813.pth'),
        #         dd=(os.path.join(MMPOSEDIR, 'configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/onehand10k/hrnetv2_w18_onehand10k_256x256_dark.py'), 'https://download.openmmlab.com/mmpose/hand/dark/hrnetv2_w18_onehand10k_256x256_dark-a2f80c64_20210330.pth'),
        #     ) as detector:
        with self.detector as detector:    
            # demo_hands = cv2.imread(os.path.join(HOME, 'nicol_rh8d/datasets/OpenMM/data/color_img3.jpg'))
            # demo_hands = cv2.resize(demo_hands, (630,900))
            result, _ = detector.process_image(cv_image, output_vis=True)
            # plot_images(result.vis_image.det, result.vis_image.dd)
            return result

    def camCB(self, msg: Image):
        try:
            # cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        except CvBridgeError as e:
            print(e)

        if not self.human:
            cv_image = ~cv_image
        
        detection = self.detect(cv_image)

        if self.vis:
            cv2.imshow("Image window", detection)
            cv2.waitKey(3)

        try:
            self.img_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "rgb8"))
        except CvBridgeError as e:
            print(e)

def main():
    rospy.init_node('detector_node', anonymous=True)
    
    kd = KeypointDetector(rospy.get_param('~vis'),
                          rospy.get_param('~human'),
                          rospy.get_param('~color_topic'),
                          rospy.get_param('~detect_topic'))
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

def detect():
    with PoseDetector(
        det=(os.path.join(MMPOSEDIR, 'demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py'), 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'),
        dd=(os.path.join(MMPOSEDIR, 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py'), 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'),
        ddd=(os.path.join(MMPOSEDIR, 'configs/body/3d_kpt_sview_rgb_vid/video_pose_lift/h36m/videopose3d_h36m_243frames_fullconv_supervised_cpn_ft.py'), 'https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_243frames_fullconv_supervised_cpn_ft-88f5abbb_20210527.pth'),
        ddd_causal=True,
    ) as detector:
    
        detector.start_sequence(image_size=(640, 480), nominal_fps=24, output_vis=True)
        # cap = cv2.VideoCapture(4)
        cap = cv2.VideoCapture('/export/home/9frick/catkin_ws/src/nicol_rh8d/mmscripts/demo/demo.mp4')
        for i in range(240):
            print("read")
            retval, frame = cap.read()
            if not retval:
                break
            result, _ = detector.process_frame(frame)
            # if i % 60 == 30:
            #     plot_images(result.vis_image.dd, result.vis_image.ddd, result.vis_image.ddd_alt)
        cap.release()

if __name__ == '__main__':
    # main()
    # detect()


    # define a video capture object 
    vid = cv2.VideoCapture(6) 
    print(vid.isOpened())

    while vid.isOpened(): 
        
        # Capture the video frame 
        # by frame 
        ret, frame = vid.read() 
        print(vid.get(cv2.CAP_PROP_FPS), vid.get(cv2.CAP_PROP_FRAME_WIDTH), vid.get(cv2.CAP_PROP_FRAME_HEIGHT), vid.get(cv2.CAP_PROP_CODEC_PIXEL_FORMAT))

        # Display the resulting frame 
        cv2.imshow('frame', frame) 
        
        # the 'q' button is set as the 
        # quitting button you may use any 
        # desired button of your choice 
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    
    # After the loop release the cap object 
    vid.release() 

    # Destroy all the windows 
    cv2.destroyAllWindows() 