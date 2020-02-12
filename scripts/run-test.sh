# test network - si-snr

###
# X a R musi sedet. Lze predat parametrama.
###

python3 main.py --test --R 3 --X 7 --basepath="$HOME/Documents/full/min/" --load-checkpoint="$HOME/Downloads/old_reconstructions/tasnet_model_checkpoint_2019-11-20_20_46_X7_R3_e5_a19999.tar"

# python3 main.py --test --R 2 --X 8 --basepath="$HOME/Documents/full/min/" --load-checkpoint="$HOME/Downloads/tasnet_model_checkpoint_2019-11-20_20_46_X7_R3_e5_a19999.tar"

exit 0

