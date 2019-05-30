#!/bin/bash

TEST_DATASET='/mnt/jinnan2_round2_test_b_20190424/'

cd code/unet
./test.sh $TEST_DATASET test_b

