# train network on PC
timeout --kill-after=12m 10m python3 main.py --train --epochs 5 --X 4 --R 2 --basepath="$HOME/Documents/full/min/" --dst-dir="$HOME/TEST/" --minibatch-size 3
exit 0

