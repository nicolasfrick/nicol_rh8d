# Installation

## Install miniconda3
$ mkdir -p ~/miniconda3
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
$ bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
$ rm -rf ~/miniconda3/miniconda.sh
$ ~/miniconda3/bin/conda init bash

## Install env

Either 

$ conda env create --name nicol_rh8d --file nicol_rh8d/environment.yml

Or following nicol_demos/commands_env.txt NICOL Frozen Environment

$ ENV=nicol_rh8d
$ PYTHON=3.9
$ DATASETS_DIR=~/nicol_rh8d/datasets/OpenMM/data
$ NICOL_RH8D=~/nicol_rh8d
$ conda create -n $ENV python=$PYTHON
$ conda activate $ENV
$ conda install -c pytorch numpy=1.22 cudatoolkit=10.2 pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 torchtext==0.13.1 'libtiff<4.5'

## OpenMM
$ pip install --upgrade openmim
$ mim install 'mmtrack<1' 'mmpose<1' 'mmdet<3' mmcls 'mmcv-full<1.7'
$ mim list
$ pip uninstall xtcocotools  # <-- Otherwise import mmpose.apis fails...
$ pip install --no-binary xtcocotools xtcocotools
$ pip check

## Config
$ ln -s "$DATASETS_DIR" "$CONDA_PREFIX"/lib/python$PYTHON/site-packages/mmdet/.mim/data
$ ln -s "$DATASETS_DIR" "$CONDA_PREFIX"/lib/python$PYTHON/site-packages/mmpose/.mim/data
$ ln -s "$CONDA_PREFIX"/lib/python$PYTHON/site-packages/tests "$CONDA_PREFIX"/lib/python$PYTHON/site-packages/mmpose/.mim/tests

## Cam
Check video devices:
$ sudo apt install v4l-utils
$ v4l2-ctl --list-devices
$ sudo snap install ffmpeg # to fix unknonwn libx264 error compile from source: https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu  
$ ffplay /dev/video0

Create virtual video device:
$ sudo apt-get install v4l2loopback-dkms v4l2loopback-utils gstreamer1.0-tools gstreamer1.0-plugins-good libopencv-dev build-essential vlc 
$ v4l2loopback-ctl --version # version
$ sudo modprobe v4l2loopback devices=1 exclusive_caps=1 video_nr=6 card_label="OpenCV Camera" # add device
$ v4l2-ctl --list-devices -d5
$ sudo modprobe -r v4l2loopback # remove device

(v4l2loopback-ctl add and remove are n/a for older versions)

Test image:
$ gst-launch-1.0 videotestsrc ! v4l2sink device=/dev/video6
Map to virtual device:
$ ffmpeg -f v4l2 -i /dev/video4 -f v4l2 /dev/video6
Check mapping:
$ ffplay /dev/video6
Negate colors:
$ ffmpeg -f v4l2 -i /dev/video4 -vf negate -f v4l2 /dev/video6
Probe frames:
$ ffprobe -v 0 -select_streams v -show_frames /dev/video4
Trim:
ffmpeg -f v4l2 -i /dev/video4 -vf "scale=w=640:h=480:force_original_aspect_ratio=1,pad=640:480:(ow-iw)/2:(oh-ih)/2,negate" -f v4l2 /dev/video6
List device capabilities:
ffmpeg -f v4l2 -list_formats all -i /dev/video4
Codecs and formats avl.
ffmpeg -codecs
ffmpeg -formats
Adjusting camera functions:
v4l2-ctl -L
v4l2-ctl -c <option>=<value>
Encoding to mkv:
ffmpeg -f v4l2 -framerate 25 -video_size 640x480 -i /dev/video4 output.mkv
Encoding to mp4:
ffmpeg -i input.mov -preset slow -codec:a libfdk_aac -b:a 128k -codec:v libx264 -pix_fmt yuv420p -b:v 2500k -minrate 1500k -maxrate 4000k -bufsize 5000k -vf scale=-1:720 output.mp4
| Flag | Options | Description |
| ---- | ------- | ----------- |
| `-codec:a` | libfaac, libfdk_aac, libvorbis | Audio Codec |
| `-quality` | best, good, realtime | Video Quality |
| `-b:a` | 128k, 192k, 256k, 320k | Audio Bitrate |
| `-codec:v` | mpeg4, libx264, libvpx-vp9 | Video Codec |
| `-b:v` | 1000, 2500, 5000, 8000 | Video Bitrate |
| `-vf scale` | -1:X | Resize Video (X is height) |
| `-qmin 10 -qmax 42` | Quantization | https://gist.github.com/dvlden/b9d923cb31775f92fa54eb8c39ccd5a9#gistcomment-2972745 |
|       --preset <string>  |      Use a preset to select encoding settings [medium]
                                  Overridden by user settings.
                                  - ultrafast,superfast,veryfast,faster,fast
                                  - medium,slow,slower,veryslow,placebo |

OpenCV cannot open h264 encoding by default, use other format like rawvideo/yuyv422 (as relasense):
***ffmpeg -f v4l2 -i /dev/video4 -preset ultrafast -c:v rawvideo -pix_fmt yuyv422 -b:v 2500k -minrate 1500k -maxrate 4000k -bufsize 5000k -vf "scale=w=640:h=480:force_original_aspect_ratio=1,pad=640:480:(ow-iw)/2:(oh-ih)/2,negate" -f v4l2 /dev/video6***

Modify RGB cam setting online via 
$ realsense-viewer
restore settings by unplugging the camera

Note: For Realsense, video4 is the rgb camera dev

## Test
$ conda activate nicol_rh8d
$ python mmscripts/pose_video.py --video "/dev/video6" \
    --det \
    "${CONDA_PREFIX}/lib/python3.9/site-packages/mmpose/.mim/demo/mmdetection_cfg/cascade_rcnn_x101_64x4d_fpn_1class.py" \
    "https://download.openmmlab.com/mmpose/mmdet_pretrained/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k-dac19597_20201030.pth" \
    --2d \
    "${CONDA_PREFIX}/lib/python3.9/site-packages/mmpose/.mim/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/onehand10k/ hrnetv2_w18_onehand10k_256x256_dark.py" \
    "https://download.openmmlab.com/mmpose/hand/dark/hrnetv2_w18_onehand10k_256x256_dark-a2f80c64_20210330.pth" \
    --vis 

### 3D lifting
python mmscripts/pose_video.py --video '/dev/video6' \
    --det "${CONDA_PREFIX}/lib/python3.9/site-packages/mmpose/.mim/demo/mmdetection_cfg/cascade_rcnn_x101_64x4d_fpn_1class.py" \
    "https://download.openmmlab.com/mmpose/mmdet_pretrained/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k-dac19597_20201030.pth" \
    --2d "${CONDA_PREFIX}/lib/python3.9/site-packages/mmpose/.mim/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/onehand10k/hrnetv2_w18_onehand10k_256x256_dark.py" \
    "https://download.openmmlab.com/mmpose/hand/dark/hrnetv2_w18_onehand10k_256x256_dark-a2f80c64_20210330.pth" \
    --3d "${CONDA_PREFIX}/lib/python3.9/site-packages/mmpose/.mim/configs/hand/3d_kpt_sview_rgb_img/internet/interhand3d/res50_interhand3d_all_256x256.py" \
    "https://download.openmmlab.com/mmpose/hand3d/internet/res50_intehand3d_all_256x256-b9c1cf4c_20210506.pth" \
    --vis 

-> 3d demo config: 
miniconda3/envs/nicol_rh8d/lib/python3.9/site-packages/mmpose/.mim/configs/body/3d_kpt_sview_rgb_vid/video_pose_lift/h36m/videopose3d_h36m_243frames_fullconv_supervised_cpn_ft.py
<class 'mmpose.models.detectors.pose_lifter.PoseLifter'>
-> 3d hand config:
/export/home/9frick/miniconda3/envs/nicol_rh8d/lib/python3.9/site-packages/mmpose/.mim/configs/hand/3d_kpt_sview_rgb_img/internet/interhand3d/res50_interhand3d_all_256x256.py
<class 'mmpose.models.detectors.interhand_3d.Interhand3D'>

#### Test with whole body pose (long video latency)
    python mmscripts/pose_video.py --vis --video '/dev/video6' \
    --det "${CONDA_PREFIX}/lib/python3.9/site-packages/mmpose/.mim/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py" \
    "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth" \
    --2d "${CONDA_PREFIX}/lib/python3.9/site-packages/mmpose/.mim/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py" \
    "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth" \
    --3d "${CONDA_PREFIX}/lib/python3.9/site-packages/mmpose/.mim/configs/body/3d_kpt_sview_rgb_vid/video_pose_lift/h36m/videopose3d_h36m_243frames_fullconv_supervised_cpn_ft.py" \
    "https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_243frames_fullconv_supervised_cpn_ft-88f5abbb_20210527.pth" 


## MMPose v1.x
$ conda create -n mmpose python=3.9
$ conda activate mmpose
$ conda install pytorch torchvision -c pytorch
$ pip install -U openmim
$ pip install ffspec
$ mim install mmengine
$ mim install "mmcv>=2.0.1"
$ mim install "mmdet>=3.1.0"
$ mim install "mmpose>=1.1.0"
$ conda install pytorch torchvision -c pytorch

### Inferencer 
python $CONDA_PREFIX/lib/python3.9/site-packages/mmpose/.mim/demo/inferencer_demo.py ~/catkin_ws/src/nicol_rh8d/datasets/OpenMM/data/image29590.jpg --pose2d hand2d --pose3d hand3d --vis-out-dir vis_results/hand3d
    