# train network on PC
python3 train.py \
    --epochs 4 --X 1 --R 1 \
    --basepath="$HOME/Documents/full/min/" \
    --dst-dir="$HOME/TEST/" \
    --disable-validation \
    --disable-training \
    --minibatch-size 3 
    # --debug

exit 0

