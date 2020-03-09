# train network on PC
python3 train.py \
    --epochs 5 --X 1 --R 1 \
    --basepath="$HOME/Documents/full/min/" \
    --dst-dir="$HOME/TEST/" \
    --disable-validation \
    --debug \
    --minibatch-size 4 

exit 0

