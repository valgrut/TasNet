# TasNet - Time-Domain Audio Separation Network
Bakalarska prace (Bachelor thesis)



## Adresarova struktura
- README.txt
- BP.pdf
- BP_tisk.pdf
- colabntbs/
    - soubory pro spousteni site na Google colab
- source/
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
    - nn.*.sh pomocne skripty pro jednoduche spousteni site s ruznym nastavenim

- scripts/
    - skripty pro odesilani dat na google drive, merlina a pod.
    - getFromMerlin.sh
    - sendToMerlin.sh
    - gupload.sh
    - gdownload.sh
    - skripty pro vykresleni nekterych grafu
    - soubory s prefixy podle pohlavi na nahravkach

- text/
    - bib-styles/
    - template-fig/
    - obrazky-figures/
        - obrazky a grafy k textu
    - *.bib a *.tex soubory s textem
    - fitthesis.cls
    - Makefile
    - zadani.pdf

- examples/
    - mix.wav
	- reconstructed_s1.wav
    	- reconstructed_s2.wav
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

usage: test.py [-h] [--padding PADDING] [--stride STRIDE] [--minibatch-size MINIBATCH_SIZE] [--lr LEARNING_RATE] [--load-checkpoint CHECKPOINT_FILE] [--debug] [--X X] [--R R] [--basepath BASE_DATA_PATH]
               [--dst-dir DST_DIR]

usage: inference.py [-h] [--epochs EPOCHS] [--padding PADDING] [--stride STRIDE] [--minibatch-size MINIBATCH_SIZE] [--lr LEARNING_RATE] [--load-checkpoint CHECKPOINT_FILE] [--debug] [--X X] [--R R]
                    [--basepath BASE_DATA_PATH] [--dst-dir DST_DIR] [--input-mixture INPUT_MIXTURE]


## Setup and init neural network
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
### spusteni trenovani site
python3 train.py \
    --epochs 60 \
    --X 8 \
    --R 4 \
    --basepath="$HOME/Documents/full/min/" \
    --dst-dir="$HOME/TEST/" \
    --minibatch-size 50

### inference - preda se smes mluvcich a vystupem budou dve nahravky se separovanymi mluvcimi
infere="$HOME/Desktop/speech_e2_a15000_mix.wav"
python3 inference.py --R 3 --X 7 \
    --load-checkpoint="$HOME/TEST/tasnet_model_checkpoint_X7_R3.tar" \
    --basepath="$(dirname $infere)" \
    --input-mixture="$(basename $infere)" \
    --dst-dir="$HOME/TEST/"

### Testovani modelu
python3 test.py --R 1 --X 1 \
    --basepath="$HOME/Documents/full/min/" \
    --load-checkpoint="$HOME/TEST/2020-07-01_X1_R1/checkpoint_X1_R1_e2.tar" \
    --dst-dir="$HOME/TEST/2020-07-01_X1_R1/" \
    --minibatch-size 1

### Nacteni checkpointu a pokracovani trenovani
python3 train.py \
    --epochs 5 --X 1 --R 1 \
    --load-checkpoint="$HOME/TEST/"$training_dir"/tasnet_model_checkpoint_2020-03-22_X1_R1_e5.tar" \
    --basepath="$HOME/Documents/full/min/" \
    --dst-dir="$HOME/TEST/"$training_dir"/" \
    --minibatch-size 4

### Dalsi moznosti
Nebo lze pouzit skripty jako nntrain.sh a pod, ktere obsahuji prednastavene parametry a zjednodusuji tak spousteni. Je ale nutne modifikovat patricne parametry argumentu pro aktualni prostredi.




## Nektere pozadovane knihovny
- pystoi
    - https://github.com/mpariente/pystoi
    - pip install pystoi
    - or for python3:
    - pip3 install pystoi

- pesq
    - https://github.com/ludlows/python-pesq
    - pip3 install https://github.com/ludlows/python-pesq/archive/master.zip

- si-sdr
    - https://github.com/sigsep/bsseval
    - pip install bsseval

- PyTorch
- python3

