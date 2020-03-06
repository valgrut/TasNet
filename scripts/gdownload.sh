#!/usr/bin/env bash

# Description: Script stahne z Google Drive nejnovejsi slozku obsahujici data z posledniho trenovani.


# Najde nejnoveji vytvoreny adresar
recent_training_dir="$(rclone lsd gdrive:FIT/reconstruction/ | awk '{print $5}' | sort)"

# Zkopiruje obsah adresare na local
rclone copy --progress --no-traverse gdrive:FIT/reconstruction/$recent_training_dir/ $HOME/TEST/$recent_training_dir/
