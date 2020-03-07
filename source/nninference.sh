# inference network - separuj vstupni smes

# POZOR: X a R musi sedet. Lze predat parametrama.

infere="$HOME/Desktop/speech_e2_a15000_mix.wav"
# dirname $infere
# basename $infere

# --load-checkpoint="$HOME/TEST/tasnet_model_checkpoint_2020-02-12_11:03_X2_R2_e0_a6.tar" \
# python3 main.py --inference --R 2 --X 2 \

python3 inference.py --R 3 --X 7 \
    --load-checkpoint="$HOME/TEST/tasnet_model_checkpoint_X7_R3_new.tar" \
    --basepath="$(dirname $infere)" \
    --input-mixture="$(basename $infere)" \
    --dst-dir="$HOME/TEST/"

exit 0

