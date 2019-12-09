#!/bin/bash

## Toto v ruznych podobach bude v tom notebooku - jeden task bude jeden setup

# test network - si-snr
# python3 main.py --test --R 2 --X 8 --basepath="$HOME/Documents/full/min/" --load-checkpoint="$HOME/Downloads/tasnet_model_checkpoint_2019-11-20_20_46_X7_R3_e5_a19999.tar"
# exit 0

# train network on PC
python3 main.py --train --epochs 5 --X 7 --R 3 --basepath="$HOME/Documents/full/min/" --dst-dir="$HOME/TEST/"
exit 0

# train network in Colab
python3 main.py --train --epochs 5 --X 7 --R 3 --basepath="/gdrive/My Drive/FIT/dataset/" --dst-dir="/gdrive/My Drive/FIT/reconstruction/"

# load checkpoint for inference and process single audio.wav
python3 main.py --inference --load-checkpoint="/gdrive/My Drive/FIT/somefile.tar" --basepath="/gdrive/My Drive/FIT/" --dst-dir="/gdrive/My Drive/FIT/reconstructions/" --input-mixture="/gdrive/My Drive/FIT/some_audio.wav"
