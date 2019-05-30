#!/bin/bash

source $(conda info --root)/etc/profile.d/conda.sh
conda activate xray_unet
cd code/unet

TIME=`date +%Y.%m.%d.%H-%M-%S`
TRAIN_LOG_DIR="/data/train_${TIME}"
DATASET_ARGS='--data-root=/mnt/jinnan2_round2_train_20190401 --data-fold 0 --data-normal 0.3'
TRAIN_CFG='--jobs 4 --apex-opt-level O2 --lr-scheduler=noam --optim sgd --learning-rate 1e-2 --max-epochs 1000 --batch-size 16 --gradient-accumulation 1 --version unet2_e_simple --basenet senet154 --image-size 128'

# for single GPU
python train.py --log-dir $TRAIN_LOG_DIR $DATASET_ARGS $TRAIN_CFG
# for 2 GPU
#python -m torch.distributed.launch --nproc_per_node 2 train.py --log-dir $TRAIN_LOG_DIR $DATASET_ARGS $TRAIN_CFG

python swa.py $DATASET_ARGS -i $TRAIN_LOG_DIR/checkpoints -o $TRAIN_LOG_DIR/swa.model.pth --batch-size 4

