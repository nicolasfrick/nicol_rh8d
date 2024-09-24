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

## OpenCV
as described in https://pypi.org/project/opencv-python/: To use the new manylinux2014 pre-built wheels (or to build from source), your pip version must be >= 19.3 (otherwise imports of cv static libraries coming with opencv-python may fail)
$ pip install --upgrade pip
$ pip install opencv-contrib-python==4.10.0.84

## OpenMM
$ pip install --upgrade openmim
$ mim install 'mmtrack<1' 'mmpose<1' 'mmdet<3' mmcls 'mmcv-full<1.7'
$ mim list
$ pip uninstall xtcocotools  # <-- Otherwise import mmpose.apis fails...
$ pip install --no-binary xtcocotools xtcocotools
$ pip check

## PyKDL
$ sudo apt install libeigen3-dev libcppunit-dev
$ conda install future pybind11
$ ENV=nicol_rh8d; PYTHON=3.9
$ conda activate $ENV; OLDDIR="$(pwd)"; cd /tmp; rm -rf /tmp/orocos_kinematics_dynamics; git clone --recursive https://github.com/orocos/orocos_kinematics_dynamics.git; cd /tmp/orocos_kinematics_dynamics; git checkout 8dbdda7; git checkout --recurse-submodules 8dbdda7; git submodule sync; git submodule update --init --recursive; git submodule status; cd /tmp/orocos_kinematics_dynamics/orocos_kdl; mkdir build; cd build; cmake -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" ..; make -j8; make install; cd /tmp/orocos_kinematics_dynamics/python_orocos_kdl; mkdir build; cd build; cmake -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" -DROS_PYTHON_VERSION=3 ..; make -j8; make install; ln -s "$CONDA_PREFIX/lib/python3/dist-packages/PyKDL.so" "$CONDA_PREFIX/lib/python$PYTHON/lib-dynload/PyKDL.so"; cd "$OLDDIR"; rm -rf /tmp/orocos_kinematics_dynamics


## Config
$ ln -s "$DATASETS_DIR" "$CONDA_PREFIX"/lib/python$PYTHON/site-packages/mmdet/.mim/data
$ ln -s "$DATASETS_DIR" "$CONDA_PREFIX"/lib/python$PYTHON/site-packages/mmpose/.mim/data
# $ ln -s "$CONDA_PREFIX"/lib/python$PYTHON/site-packages/tests "$CONDA_PREFIX"/lib/python$PYTHON/site-packages/mmpose/.mim/tests
cd ~ &&  git clone https://github.com/open-mmlab/mmpose.git
ln -s ~/mmpose/tests "$CONDA_PREFIX"/lib/python3.9/site-packages/mmpose/.mim/tests



## Cam
Install realsense-viewer:
 sudo mkdir -p /etc/apt/keyrings
 curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null
 sudo apt-get install apt-transport-https
echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo `lsb_release -cs` main" | \
sudo tee /etc/apt/sources.list.d/librealsense.list
sudo apt-get update
sudo apt-get install librealsense2-dkms
sudo apt-get install librealsense2-utils
realsense-viewer # test

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
***ffmpeg -f v4l2 -i /dev/video4 -preset ultrafast -c:v rawvideo -pix_fmt yuyv422 -b:v 2500k -minrate 1500k -maxrate 4000k -bufsize 5000k -vf "fps=1,scale=w=640:h=480:force_original_aspect_ratio=1,pad=640:480:(ow-iw)/2:(oh-ih)/2,negate" -f v4l2 /dev/video6***

DO NOT USE ffmpeg from conda!

Modify RGB cam setting online via 
$ realsense-viewer
restore settings by unplugging the camera

Note: For Realsense, video4 is the rgb camera dev

### Test 2D
$ conda activate nicol_rh8d
$ python mmscripts/pose_video.py --video "/dev/video6" \
    --det \
    "${CONDA_PREFIX}/lib/python3.9/site-packages/mmpose/.mim/demo/mmdetection_cfg/cascade_rcnn_x101_64x4d_fpn_1class.py" \
    "https://download.openmmlab.com/mmpose/mmdet_pretrained/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k-dac19597_20201030.pth" \
    --2d \
    "${CONDA_PREFIX}/lib/python3.9/site-packages/mmpose/.mim/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/onehand10k/ hrnetv2_w18_onehand10k_256x256_dark.py" \
    "https://download.openmmlab.com/mmpose/hand/dark/hrnetv2_w18_onehand10k_256x256_dark-a2f80c64_20210330.pth" \
    --vis --vis_autorun 

### Test 2D Interhand
python mmscripts/pose_video.py --video "/dev/video6" \
    --det \
    "${CONDA_PREFIX}/lib/python3.9/site-packages/mmpose/.mim/demo/mmdetection_cfg/cascade_rcnn_x101_64x4d_fpn_1class.py" \
    "https://download.openmmlab.com/mmpose/mmdet_pretrained/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k-dac19597_20201030.pth" \
    --2d \
    "${CONDA_PREFIX}/lib/python3.9/site-packages/mmpose/.mim/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/interhand2d/res50_interhand2d_all_256x256.py" \
    "https://download.openmmlab.com/mmpose/hand/resnet/res50_interhand2d_256x256_all-78cc95d4_20201102.pth" \
    --vis --vis_autorun 

weights: ${CONDA_PREFIX}/lib/python3.9/site-packages/mmpose/.mim/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/interhand2d/resnet_interhand2d.yml

### Hand 3D lifting not working for pose_video.py 
python mmscripts/pose_video.py --video '/dev/video6' \
    --det "${CONDA_PREFIX}/lib/python3.9/site-packages/mmpose/.mim/demo/mmdetection_cfg/cascade_rcnn_x101_64x4d_fpn_1class.py" \
    "https://download.openmmlab.com/mmpose/mmdet_pretrained/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k-dac19597_20201030.pth" \
    --2d "${CONDA_PREFIX}/lib/python3.9/site-packages/mmpose/.mim/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/onehand10k/hrnetv2_w18_onehand10k_256x256_dark.py" \
    "https://download.openmmlab.com/mmpose/hand/dark/hrnetv2_w18_onehand10k_256x256_dark-a2f80c64_20210330.pth" \
    --3d "${CONDA_PREFIX}/lib/python3.9/site-packages/mmpose/.mim/configs/hand/3d_kpt_sview_rgb_img/internet/interhand3d/res50_interhand3d_all_256x256.py" \
    "https://download.openmmlab.com/mmpose/hand3d/internet/res50_intehand3d_all_256x256-b9c1cf4c_20210506.pth" \
    --vis --vis_autorun 

-> 3d hand config:
/export/home/9frick/miniconda3/envs/nicol_rh8d/lib/python3.9/site-packages/mmpose/.mim/configs/hand/3d_kpt_sview_rgb_img/internet/interhand3d/res50_interhand3d_all_256x256.py
<class 'mmpose.models.detectors.interhand_3d.Interhand3D'>

#### Test with whole body pose (long video latency)
    python mmscripts/pose_video.py --vis --vis_autorun --video '/dev/video6' \
    --det "${CONDA_PREFIX}/lib/python3.9/site-packages/mmpose/.mim/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py" \
    "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth" \
    --2d "${CONDA_PREFIX}/lib/python3.9/site-packages/mmpose/.mim/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py" \
    "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth" \
    --3d "${CONDA_PREFIX}/lib/python3.9/site-packages/mmpose/.mim/configs/body/3d_kpt_sview_rgb_vid/video_pose_lift/h36m/videopose3d_h36m_243frames_fullconv_supervised_cpn_ft.py" \
    "https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_243frames_fullconv_supervised_cpn_ft-88f5abbb_20210527.pth" 

-> 3d demo config: 
miniconda3/envs/nicol_rh8d/lib/python3.9/site-packages/mmpose/.mim/configs/body/3d_kpt_sview_rgb_vid/video_pose_lift/h36m/videopose3d_h36m_243frames_fullconv_supervised_cpn_ft.py
<class 'mmpose.models.detectors.pose_lifter.PoseLifter'>

#### Test 3D Hand Image Demo 

python demo/interhand3d_img_demo.py \
    ${MMPOSE_CONFIG_FILE} ${MMPOSE_CHECKPOINT_FILE} \
    --json-file ${JSON_FILE} \
    --img-root ${IMG_ROOT} \
    [--camera-param-file ${CAMERA_PARAM_FILE}] \
    [--gt-joints-file ${GT_JOINTS_FILE}]\
    [--show] \
    [--device ${GPU_ID or CPU}] \
    [--out-img-root ${OUTPUT_DIR}] \
    [--rebase-keypoint-height] \
    [--show-ground-truth]

Example with gt keypoints and camera parameters:

python ${CONDA_PREFIX}/lib/python3.9/site-packages/mmpose/.mim/demo/interhand3d_img_demo.py \
    ${CONDA_PREFIX}/lib/python3.9/site-packages/mmpose/.mim/configs/hand/3d_kpt_sview_rgb_img/internet/interhand3d/res50_interhand3d_all_256x256.py \
    https://download.openmmlab.com/mmpose/hand3d/internet/res50_intehand3d_all_256x256-b9c1cf4c_20210506.pth \
    --json-file ${CONDA_PREFIX}/lib/python3.9/site-packages/mmpose/.mim/tests/data/interhand2.6m/test_interhand2.6m_data.json \
    --img-root ${CONDA_PREFIX}/lib/python3.9/site-packages/mmpose/.mim/tests/data/interhand2.6m/ \
    --camera-param-file ${CONDA_PREFIX}/lib/python3.9/site-packages/mmpose/.mim/tests/data/interhand2.6m/test_interhand2.6m_camera.json \
    --gt-joints-file ${CONDA_PREFIX}/lib/python3.9/site-packages/mmpose/.mim/tests/data/interhand2.6m/test_interhand2.6m_joint_3d.json \
    --out-img-root vis_results \
    --rebase-keypoint-height \
    --show-ground-truth

Example without gt keypoints and camera parameters:

python ${CONDA_PREFIX}/lib/python3.9/site-packages/mmpose/.mim/demo/interhand3d_img_demo.py \
    ${CONDA_PREFIX}/lib/python3.9/site-packages/mmpose/.mim/configs/hand/3d_kpt_sview_rgb_img/internet/interhand3d/res50_interhand3d_all_256x256.py \
    https://download.openmmlab.com/mmpose/hand3d/internet/res50_intehand3d_all_256x256-b9c1cf4c_20210506.pth \
    --json-file ${CONDA_PREFIX}/lib/python3.9/site-packages/mmpose/.mim/tests/data/interhand2.6m/test_interhand2.6m_data.json \
    --img-root ${CONDA_PREFIX}/lib/python3.9/site-packages/mmpose/.mim/tests/data/interhand2.6m/ \
    --out-img-root vis_results \
    --rebase-keypoint-height

#### All datasets (lib/python3.9/site-packages/mmpose/datasets/__init__.py):
__all__ = [
    'TopDownCocoDataset', 'BottomUpCocoDataset', 'BottomUpMhpDataset',
    'BottomUpAicDataset', 'BottomUpCocoWholeBodyDataset', 'TopDownMpiiDataset',
    'TopDownMpiiTrbDataset', 'OneHand10KDataset', 'PanopticDataset',
    'HandCocoWholeBodyDataset', 'FreiHandDataset', 'InterHand2DDataset',
    'InterHand3DDataset', 'TopDownOCHumanDataset', 'TopDownAicDataset',
    'TopDownCocoWholeBodyDataset', 'MeshH36MDataset', 'MeshMixDataset',
    'MoshDataset', 'MeshAdversarialDataset', 'TopDownCrowdPoseDataset',
    'BottomUpCrowdPoseDataset', 'TopDownFreiHandDataset',
    'TopDownOneHand10KDataset', 'TopDownPanopticDataset',
    'TopDownPoseTrack18Dataset', 'TopDownJhmdbDataset', 'TopDownMhpDataset',
    'DeepFashionDataset', 'Face300WDataset', 'FaceAFLWDataset',
    'FaceWFLWDataset', 'FaceCOFWDataset', 'FaceCocoWholeBodyDataset',
    'Body3DH36MDataset', 'AnimalHorse10Dataset', 'AnimalMacaqueDataset',
    'AnimalFlyDataset', 'AnimalLocustDataset', 'AnimalZebraDataset',
    'AnimalATRWDataset', 'AnimalPoseDataset', 'TopDownH36MDataset',
    'TopDownPoseTrack18VideoDataset', 'build_dataloader', 'build_dataset',
    'Compose', 'DistributedSampler', 'DATASETS', 'PIPELINES', 'DatasetInfo',
    'Body3DMviewDirectPanopticDataset', 'Body3DMviewDirectShelfDataset',
    'Body3DMviewDirectCampusDataset', 'NVGestureDataset'
]

#### Hand datasets (lib/python3.9/site-packages/mmpose/datasets/datasets/hand/__init__.py):
__all__ = [
    'FreiHandDataset', 'InterHand2DDataset', 'InterHand3DDataset',
    'OneHand10KDataset', 'PanopticDataset', 'Rhd2DDataset',
    'HandCocoWholeBodyDataset'
]

#### Datasets: type | base | path | config
##### Hand 2D:
InterHand2DDataset Kpt2dSviewRgbImgTopDownDataset lib/python3.9/site-packages/mmpose/datasets/datasets/hand/interhand2d_dataset.py
lib/python3.9/site-packages/mmpose/.mim/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/interhand2d/res50_interhand2d_all_256x256.py
lib/python3.9/site-packages/mmpose/.mim/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/interhand2d/res50_interhand2d_human_256x256.py
lib/python3.9/site-packages/mmpose/.mim/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/interhand2d/res50_interhand2d_machine_256x256.py

OneHand10KDataset Kpt2dSviewRgbImgTopDownDataset lib/python3.9/site-packages/mmpose/datasets/datasets/hand/onehand10k_dataset.py
lib/python3.9/site-packages/mmpose/.mim/configs/hand/2d_kpt_sview_rgb_img/deeppose/onehand10k/res50_onehand10k_256x256.py

FreiHandDataset Kpt2dSviewRgbImgTopDownDataset lib/python3.9/site-packages/mmpose/datasets/datasets/hand/freihand_dataset.py
lib/python3.9/site-packages/mmpose/.mim/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/freihand2d/hrnetv2_w18_freihand2d_256x256.py

HandCocoWholeBodyDataset Kpt2dSviewRgbImgTopDownDataset lib/python3.9/site-packages/mmpose/datasets/datasets/hand/hand_coco_wholebody_dataset.py
lib/python3.9/site-packages/mmpose/.mim/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/coco_wholebody_hand/hourglass52_coco_wholebody_hand_256x256.py
lib/python3.9/site-packages/mmpose/.mim/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/coco_wholebody_hand/hrnetv2_w18_coco_wholebody_hand_256x256.py
lib/python3.9/site-packages/mmpose/.mim/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/coco_wholebody_hand/res50_coco_wholebody_hand_256x256.py
...
##### Hand 3D:
InterHand3DDataset Kpt3dSviewRgbImgTopDownDataset lib/python3.9/site-packages/mmpose/datasets/datasets/hand/interhand3d_dataset.py
lib/python3.9/site-packages/mmpose/.mim/configs/hand/3d_kpt_sview_rgb_img/internet/interhand3d/res50_interhand3d_all_256x256.py

##### 2D:
TopDownCocoDataset Kpt2dSviewRgbImgTopDownDataset lib/python3.9/site-packages/mmpose/datasets/datasets/top_down/topdown_coco_dataset.py
lib/python3.9/site-packages/mmpose/.mim/configs/body/2d_kpt_sview_rgb_img/deeppose/coco/res50_coco_256x192_rle.py
...

##### 3D:
lib/python3.9/site-packages/mmpose/.mim/configs/body/3d_kpt_sview_rgb_vid/video_pose_lift/h36m/videopose3d_h36m_243frames_fullconv_supervised_cpn_ft.py

##### Detector (CocoDataset):
lib/python3.9/site-packages/mmpose/.mim/demo/mmdetection_cfg/cascade_rcnn_x101_64x4d_fpn_1class.py
lib/python3.9/site-packages/mmpose/.mim/demo/mmdetection_cfg/ssdlite_mobilenetv2_scratch_600e_onehand.py
lib/python3.9/site-packages/mmpose/.mim/demo/mmdetection_cfg/ssdlite_mobilenetv2_scratch_600e_coco.py 
lib/python3.9/site-packages/mmpose/.mim/demo/mmdetection_cfg/yolov3_d53_320_273e_coco.py
...

## MMPose v1.x
$ conda create -n mmpose python=3.9
$ conda activate mmpose
$ conda install pytorch torchvision -c pytorch # cuda >= 11
$ conda install -c pytorch numpy=1.22 cudatoolkit=10.2 pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 torchtext==0.13.1 # cuda < 11
$ pip install fsspec
$ pip install -U openmim
$ mim install mmengine
$ mim install "mmcv>=2.0.1" "mmdet>=3.1.0" "mmpose>=1.1.0"
$ pip uninstall xtcocotools 
$ pip install --no-binary xtcocotools xtcocotools
$ pip install "numpy<2"

## Hand Keypoint Estimation
### 2D Hand Image Demo

python $CONDA_PREFIX/lib/python3.9/site-packages/mmpose/.mim/demo/topdown_demo_with_mmdet.py \
    $CONDA_PREFIX/lib/python3.9/site-packages/mmpose/.mim/demo/mmdetection_cfg/rtmdet_nano_320-8xb32_hand.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmdet_nano_8xb32-300e_hand-267f9c8f.pth \
    $CONDA_PREFIX/lib/python3.9/site-packages/mmpose/.mim/configs/hand_2d_keypoint/rtmpose/hand5/rtmpose-m_8xb256-210e_hand5-256x256.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth \
    --input $CONDA_PREFIX/lib/python3.9/site-packages/mmpose/.mim/tests/data/onehand10k/9.jpg \
    --show --draw-heatmap

### 2D Hand Keypoints Video Demo (@ video0)
python $CONDA_PREFIX/lib/python3.9/site-packages/mmpose/.mim/demo/topdown_demo_with_mmdet.py \
$CONDA_PREFIX/lib/python3.9/site-packages/mmpose/.mim/demo/mmdetection_cfg/rtmdet_nano_320-8xb32_hand.py \
https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmdet_nano_8xb32-300e_hand-267f9c8f.pth \
$CONDA_PREFIX/lib/python3.9/site-packages/mmpose/.mim/configs/hand_2d_keypoint/rtmpose/hand5/rtmpose-m_8xb256-210e_hand5-256x256.py \
https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth \
--input webcam --show --kpt-thr 0.1 --draw-heatmap --draw-bbox # --save-predictions  --output-root /home/nic/catkin_ws/src/nicol_rh8d/datasets/OpenMM/out

press 100x 'esc' to exit display
use pose anything for 3D lifting?

### Interhand 3D Hand Pose Estimation Demo
python demo/hand3d_internet_demo.py \
    ${MMPOSE_CONFIG_FILE} ${MMPOSE_CHECKPOINT_FILE} \
    --input ${INPUT_FILE} \
    --output-root ${OUTPUT_ROOT} \
    [--save-predictions] \
    [--gt-joints-file ${GT_JOINTS_FILE}]\
    [--disable-rebase-keypoint] \
    [--show] \
    [--device ${GPU_ID or CPU}] \
    [--kpt-thr ${KPT_THR}] \
    [--show-kpt-idx] \
    [--show-interval] \
    [--radius ${RADIUS}] \
    [--thickness ${THICKNESS}]

python $CONDA_PREFIX/lib/python3.9/site-packages/mmpose/.mim/demo/hand3d_internet_demo.py \
    $CONDA_PREFIX/lib/python3.9/site-packages/mmpose/.mim/configs/hand_3d_keypoint/internet/interhand3d/internet_res50_4xb16-20e_interhand3d-256x256.py \
    https://download.openmmlab.com/mmpose/hand3d/internet/res50_intehand3dv1.0_all_256x256-42b7f2ac_20210702.pth \
    --input webcam --show --kpt-thr 0.1 --radius 4 --thickness 2 --show-kpt-idx \
    --save-predictions --output-root  /home/nic/catkin_ws/src/nicol_rh8d/datasets/OpenMM/out 

-> apis/inferencers/hand3d_inferencer.py # disables --hand2d, inference only by image

### 2D Hand Keypoints Demo with Inferencer
python $CONDA_PREFIX/lib/python3.9/site-packages/mmpose/.mim/demo/inferencer_demo.py tests/data/onehand10k \
    --pose2d hand --vis-out-dir vis_results/onehand10k \
    --bbox-thr 0.5 --kpt-thr 0.05

### 3D Hand Pose Estimation with Inferencer 
python $CONDA_PREFIX/lib/python3.9/site-packages/mmpose/.mim/demo/inferencer_demo.py ~/catkin_ws/src/nicol_rh8d/datasets/OpenMM/data/image29590.jpg --pose2d hand2d --pose3d hand3d --vis-out-dir vis_results/hand3d 

--show-alias
ALIAS      MODEL_NAME
animal     rtmpose-m_8xb64-210e_ap10k-256x256
body       rtmpose-m_8xb256-420e_body8-256x192
body17     rtmpose-m_8xb256-420e_body8-256x192
body26     rtmpose-m_8xb512-700e_body8-halpe26-256x192
edpose     edpose_res50_8xb2-50e_coco-800x1333
face       rtmpose-m_8xb64-120e_lapa-256x256
hand       rtmpose-m_8xb256-210e_hand5-256x256
hand3d     internet_res50_4xb16-20e_interhand3d-256x256     -> hand2d flag ignored
human      rtmpose-m_8xb256-420e_body8-256x192
human3d    motionbert_dstformer-ft-243frm_8xb32-120e_h36m
rtmo       rtmo-l_16xb16-600e_body7-640x640
rtmpose-l  rtmpose-l_8xb256-420e_body8-384x288
vitpose    td-hm_ViTPose-base-simple_8xb64-210e_coco-256x192
vitpose-b  td-hm_ViTPose-base-simple_8xb64-210e_coco-256x192
vitpose-h  td-hm_ViTPose-huge-simple_8xb64-210e_coco-256x192
vitpose-l  td-hm_ViTPose-large-simple_8xb64-210e_coco-256x192
vitpose-s  td-hm_ViTPose-small-simple_8xb64-210e_coco-256x192
wholebody  rtmw-m_8xb1024-270e_cocktail14-256x192

python $CONDA_PREFIX/lib/python3.9/site-packages/mmpose/.mim/demo/inferencer_demo.py tests/data/onehand10k \
    --pose2d \
    --pose2d-weights --show --bbox-thr 0.5 --kpt-thr 0.05


# Keypoint Transformer
## Install
sudo apt-get install qt5-default
conda create --name kypt_trans python==3.8.11
conda activate kypt_trans
conda install numpy==1.23.1 matplotlib==3.5.2 mkl_fft==1.3.1 mkl_random==1.2.2 mkl-service==2.4.0 open3d==0.15.2 pycocotools==2.0.4 pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 kiwisolver==1.4.2 freetype-py==2.3.0 argon2-cffi==21.3.0 argon2-cffi-bindings==21.2.0 -c pytorch -c conda-forge -c intel -c open3d-admin
cd kypt_transformer
pip install -r requirements.txt
pip uninstall PyQt5 # resolve xcb plugin import error

## Config
In kypt_transformer/main/config.py set lines
31: smplx_path = '/home/<uname>/datasets/mano_v1_2/models/' # Path to MANO model files
36: interhand_anno_dir = '/home/<uname>/datasets/interhand/annotations' # Path to interghand annotations
37: interhand_images_path = '/home/<uname>/datasets/interhand/InterHand2.6M_5fps_batch1/images' # Path to interghand imgs

Download https://download.is.tue.mpg.de/download.php?domain=mano&resume=1&sfile=mano_v1_2.zip and unzip to smplx_path, mv folder mano_v1_2/models -> mano_v1_2/models/mano

Run scripts/interhand_download_script.py and unzip images via scripts/interhand_unzip.sh to interhand_images_path (80GB!)

Download annotations https://1drv.ms/f/s!All7gdNh7XE5k2k8pAXys2xIRpNn?e=rVFzlj and unzip to interhand_anno_dir, mkdir interhand_anno_dir/all; cp interhand_anno_dir/test/* interhand_anno_dir/all; cp interhand_anno_dir/train/* interhand_anno_dir/all; mv interhand_anno_dir/all/InterHand2.6M_train_MANO_NeuralAnnot.json interhand_anno_dir/all/InterHand2.6M_train_MANO.json; mv interhand_anno_dir/all/InterHand2.6M_test_MANO_NeuralAnnot.json interhand_anno_dir/all/InterHand2.6M_test_MANO.json

Download checkpoint from https://1drv.ms/u/s!AsG9HA3ULXQRgskO89bIy8j3h-HTXQ?e=J6FHVu to a <checkpoints_dir>

## Run
cd kypt_transformer/main

In demo.py add:
```
#!/usr/bin/env python3
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

python demo.py --ckpt_path checkpoints_dir--use_big_decoder --dec_layers 6