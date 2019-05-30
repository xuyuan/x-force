#!/bin/bash

TEST_DATASET='/mnt/jinnan2_round2_test_c/'

cd code/unet
./test.sh $TEST_DATASET test_c

