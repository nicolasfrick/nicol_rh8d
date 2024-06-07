# Installation

## Install miniconda3
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash

## Install env
Follows nicol_demos/commands_env.txt NICOL Frozen Environment

ENV=nicol_rh8d
PYTHON=3.9
DATASETS_DIR=~/nicol_rh8d/datasets/OpenMM/data
NICOL_RH8D=~/nicol_rh8d
conda create -n $ENV python=$PYTHON
conda activate $ENV

## PyTorch
conda install -c pytorch numpy=1.22 cudatoolkit=10.2 pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 torchtext==0.13.1 'libtiff<4.5'

## OpenMM
pip install --upgrade openmim
mim install 'mmtrack<1' 'mmpose<1' 'mmdet<3' mmcls 'mmcv-full<1.7'
mim list
pip uninstall xtcocotools  # <-- Otherwise import mmpose.apis fails...
pip install --no-binary xtcocotools xtcocotools
pip check

## Config
ln -s "$DATASETS_DIR" "$CONDA_PREFIX"/lib/python$PYTHON/site-packages/mmdet/.mim/data
ln -s "$DATASETS_DIR" "$CONDA_PREFIX"/lib/python$PYTHON/site-packages/mmpose/.mim/data
ln -s "$CONDA_PREFIX"/lib/python$PYTHON/site-packages/tests "$CONDA_PREFIX"/lib/python$PYTHON/site-packages/mmpose/.mim/tests



