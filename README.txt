# TasNet - Time-Domain Audio Separation Network
Bakalarska prace (Bachelor thesis)


## Adresarova struktura
- README.md
- README.txt
- experiments/
    - fragmenty z prubehu implementace
- colabntbs/
    - soubory pro spousteni site na Google colab
- source/
    - requirements.txt
    - train.py
    - test.py
    - inference.py
    - SegmentDataset.py
    - ResBlock.py
    - TasNet.py
    - AudioDataset.py
    - SegmentDataset.py
    - util.py
    - tools.py
    - snr.py
    - *.sh helper scripts
    - soubory s prefixy podle pohlavi na nahravkach
    - ...
- text/
    - obrazky-figures/
        - obrazky a grafy k textu
    - skripty pro odesilani textu na server pro preklad
    - *.bib a *.tex soubory s textem
    - Makefile
    - zadani.pdf
    - bp.pdf
- trained/
    - x.y.2020/
        - .log
        - .loss
        - checkpoints
        - inference


## Pouziti
usage: train.py [-h] [--epochs EPOCHS] [--segment-length SEGMENT_LENGTH] [--padding PADDING]
                [--stride STRIDE] [--minibatch-size MINIBATCH_SIZE] [--lr LEARNING_RATE] [--print-loss]
                [--load-checkpoint CHECKPOINT_FILE] [--disable-validation] [--disable-training]
                [--debug] [--X X] [--R R] [--basepath BASE_DATA_PATH] [--dst-dir DST_DIR]

### Setup and init neural network

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       number of epochs for training
  --segment-length SEGMENT_LENGTH
                        length of segments, default is 32k (4s)
  --padding PADDING     padding
  --stride STRIDE       stride
  --minibatch-size MINIBATCH_SIZE
                        size of mini-batches
  --lr LEARNING_RATE    set learning rate
  --print-loss          if option set, loss is printed every num of processed audios, where num is
                        given by parameter.
  --load-checkpoint CHECKPOINT_FILE
                        path to checkpoint file with .tar extension
  --disable-validation  disables validation after epoch
  --disable-training    disables backpropagation operation in training
  --debug               enable debug print
  --X X                 number of ConvBlocks in one Repeat in training
  --R R                 number of Repeats in training
  --basepath BASE_DATA_PATH
                        path where related files for training will be saved (checkpoints, graphs,
                        reconstructions..).
  --dst-dir DST_DIR     path to directory where separated mixtures will be saved.


## Priklady
python3 train.py \
    --epochs 60 \
    --X 8 \
    --R 4 \
    --basepath="$HOME/Documents/full/min/" \
    --dst-dir="$HOME/TEST/" \
    --minibatch-size 50


infere="$HOME/Desktop/speech_e2_a15000_mix.wav"
python3 inference.py --R 3 --X 7 \
    --load-checkpoint="$HOME/TEST/tasnet_model_checkpoint_X7_R3.tar" \
    --basepath="$(dirname $infere)" \
    --input-mixture="$(basename $infere)" \
    --dst-dir="$HOME/TEST/"

python3 test.py --R 1 --X 1 \
    --basepath="$HOME/Documents/full/min/" \
    --load-checkpoint="$HOME/TEST/2020-07-01_X1_R1/checkpoint_X1_R1_e2.tar" \
    --dst-dir="$HOME/TEST/2020-07-01_X1_R1/" \
    --minibatch-size 1


