#!/usr/bin/env bash

# Description: Script stahne z Google Drive nejnovejsi slozku obsahujici data z posledniho trenovani. Slozka je stazena jen pokud neni na serveru prazdna.

# Pouziti: gdownload.sh [training-dir-name]

if [ "$#" -eq 1 ]; then
    training_dir="$1"
    echo Stahuje se $training_dir

    # Zkopiruje obsah adresare na local
    mkdir $HOME/TEST/$training_dir/
    rclone copy --progress --no-traverse gdrive:FIT/reconstruction/$training_dir/ $HOME/TEST/$training_dir/
else
    # Najde nejnoveji vytvoreny adresar
    recent_training_dir="$(rclone lsd gdrive:FIT/reconstruction/ | awk '{print $5}' | sort -r | head -1)"
    echo Stahuje se $recent_training_dir

    # Zkopiruje obsah adresare na local
    mkdir $HOME/TEST/$recent_training_dir/
    rclone copy --progress --no-traverse gdrive:FIT/reconstruction/$recent_training_dir/ $HOME/TEST/$recent_training_dir/
fi
