# Installation

Follows nicol_demos/commands_env.txt NICOL Frozen Environment

ENV=nicol_rh8d
PYTHON=3.10
DATASETS_DIR=/home/nicol/repositories/nicol_rh8d/datasets/OpenMM/data
NICOL_RH8D=/home/nicol/repositories/nicol_rh8d
conda create -n $ENV python=$PYTHON
conda activate $ENV
conda install -c pytorch -c nvidia numpy=1.22 pytorch-cuda=11.7 pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 'libtiff<4.5'
conda install future pybind11
# OpenMM
pip install --upgrade openmim
mim install 'mmtrack<1' 'mmpose<1' 'mmdet<3' mmcls 'mmcv-full<1.7'
