#!/usr/bin/env bash

# Description: Zkopiruje vsechny potrebne scripty na Google Drive

# Seznam souboru ke zkopirovani na Disk
declare -a filelist=("inference.py" "snr.py" "test.py" "tools.py" "train.py" "util.py" "AudioDataset.py" "ResBlock.py" "SegmentDataset.py" "TasNet.py")

# Kopirovani souboru na listu
for script in "${filelist[@]}"; do
    echo "Kopirovani souboru $script na google drive"
    rclone copy --max-age 24h --progress --no-traverse "./$script" gdrive:TasNet/
done
