#!/bin/bash

# handle optional download dir
if [ -z "$1" ]
  then
    # default
    echo "please give test images folder"
    exit -1
  else
    # check if specified dir is valid
    if [ ! -d $1 ]; then
        echo $1 " is not a valid directory"
        exit -1
    fi
    TEST_DATASET=$1
fi

if [ -z "$2" ]
  then
    # default
    SUBMISSION_BASE_NAME='submission'
  else
    SUBMISSION_BASE_NAME=$2
fi
TIME=`date +%Y.%m.%d.%H-%M-%S`
SUBMISSION_NAME="${SUBMISSION_BASE_NAME}_${TIME}.json"

echo "test dataset $TEST_DATASET"

MODEL_FOLDER='../../data/models'
TEST_MODEL0='m0.4384.model.pth'
TEST_MODEL1='m1.4388.model.pth'
TEST_ARGS='--image-size 0 --tta -1'
PRED_FOLDER='/data'
SUBMISSION="../../submit/$SUBMISSION_NAME"

# activate conda
source $(conda info --root)/etc/profile.d/conda.sh
conda activate xray_unet

python test.py -i $TEST_DATASET -m $MODEL_FOLDER/$TEST_MODEL0 $TEST_ARGS -o $PRED_FOLDER/${SUBMISSION_BASE_NAME}_m0
python test.py -i $TEST_DATASET -m $MODEL_FOLDER/$TEST_MODEL1 $TEST_ARGS -o $PRED_FOLDER/${SUBMISSION_BASE_NAME}_m1

rm -rf $PRED_FOLDER/$SUBMISSION_BASE_NAME
python ensemble.py $PRED_FOLDER/${SUBMISSION_BASE_NAME}_m0 $PRED_FOLDER/${SUBMISSION_BASE_NAME}_m1 -o $PRED_FOLDER/${SUBMISSION_BASE_NAME}
python export.py $PRED_FOLDER/${SUBMISSION_BASE_NAME}/detections.pkl -o $SUBMISSION


