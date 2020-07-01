# test network - si-snr

###
# X a R musi sedet. Lze predat parametrama.
###

# --load-checkpoint="$HOME/TEST/tasnet_model_checkpoint_2020-02-12_11:03_X2_R2_e0_a6.tar" \

python3 test.py --R 1 --X 1 \
    --basepath="$HOME/Documents/full/min/" \
    --load-checkpoint="$HOME/TEST/2020-07-01_23:03_X1_R1/tasnet_model_checkpoint_2020-07-01_X1_R1_e2.tar" \
    --dst-dir="$HOME/TEST/2020-07-01_23:03_X1_R1/" \
    --minibatch-size 1

# python3 main.py --test --R 2 --X 8 --basepath="$HOME/Documents/full/min/" --load-checkpoint="$HOME/Downloads/tasnet_model_checkpoint_2019-11-20_20_46_X7_R3_e5_a19999.tar"

exit 0

