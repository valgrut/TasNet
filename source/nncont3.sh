#!/usr/bin/env bash

training_dir="2020-03-18_00:11_X1_R1"

python3 train.py \
    --epochs 5 --X 1 --R 1 \
    --load-checkpoint="$HOME/TEST/"$training_dir"/tasnet_model_checkpoint_2020-03-18_X1_R1_e4_a24.tar"\
    --basepath="$HOME/Documents/full/min/" \
    --dst-dir="$HOME/TEST/"$training_dir"/" \
    --minibatch-size 4

exit 0

