#! /usr/bin/env bash

python3 histogram.py \
    --round=1 \
    --sdr-file-path=/root/TEST/2020-07-01_23\:03_X1_R1/tasnet_model_checkpoint_2020-07-01_X1_R1_e2.tar.sdr \
    --men-id-path=/root/Documents/Projects/TasNet/source/men_speaker_id.txt \
    --women-id-path=/root/Documents/Projects/TasNet/source/women_speaker_id.txt
