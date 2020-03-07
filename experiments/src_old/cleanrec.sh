#! /usr/bin/bash

#rm /home/valgrut/Documents/reconstruction/*.wav;
find /home/valgrut/Documents/reconstruction/ -maxdepth 1 -name '*.wav' -delete
find /home/valgrut/Documents/testdata_recon/ -maxdepth 1 -name '*.wav' -delete
