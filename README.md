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
$ sudo modprobe v4l2loopback devices=2 exclusive_caps=1,1 video_nr=6,7 card_label="VideoTest","OpenCV Camera" # add device
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
<!-- $ ffmpeg -y -nostdin -i INPUT.mkv -ss 8 -to 68 -map 0:v:0 -vsync 0 -enc_time_base -1 -vf "scale=w=640:h=480:force_original_aspect_ratio=1,pad=640:480:(ow-iw)/2:(oh-ih)/2,negate" -c:v libx264 -crf 12 -bf 0 OUTPUT.mp4 -->

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

Line 46 in needs an integer for accessing a video device -> change args.video to int(args.video) in order to use a device like video0
