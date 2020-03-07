# test network - si-snr

###
# X a R musi sedet. Lze predat parametrama.
###

# --load-checkpoint="$HOME/TEST/tasnet_model_checkpoint_2020-02-12_11:03_X2_R2_e0_a6.tar" \

python3 test.py --R 3 --X 7 \
    --basepath="$HOME/Documents/full/min/" \
    --load-checkpoint="$HOME/TEST/tasnet_model_checkpoint_X7_R3_new.tar" \
    --dst-dir="$HOME/TEST/2020-03-03_23:49_X7_R3/" \
    --minibatch-size 1

# python3 main.py --test --R 2 --X 8 --basepath="$HOME/Documents/full/min/" --load-checkpoint="$HOME/Downloads/tasnet_model_checkpoint_2019-11-20_20_46_X7_R3_e5_a19999.tar"

exit 0

