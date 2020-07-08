#! /usr/bin/env bash

# tasnet_model_checkpoint_2020-05-06_X4_R4_e60.tar.sdr
    # --sdr-file-path=/root/TEST/2020-07-01_23\:03_X1_R1/tasnet_model_checkpoint_2020-07-01_X1_R1_e2.tar.sdr \

python3 histogram.py \
    --round=1 \
    --sdr-file-path=/root/TEST/2020-07-01_23\:03_X1_R1/tasnet_model_checkpoint_2020-05-06_X4_R4_e60.tar.sdr \
    --men-id-path=/root/Documents/Projects/TasNet/source/men_prefix.txt \
    --women-id-path=/root/Documents/Projects/TasNet/source/women_prefix.txt
