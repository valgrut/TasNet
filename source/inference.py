import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data_utils
from torch.autograd import Variable
import torchvision.transforms as transforms
import os
from datetime import datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt

from AudioDataset import AudioDataset
from SegmentDataset import SegmentDataset
from TasNet import Net
from ResBlock import ResBlock
from tools import *
from snr import *

if __name__== "__main__":
    print("Version 07")
    parser = argparse.ArgumentParser(description='Setup and init neural network')

    parser.add_argument('--epochs',
            dest='epochs',
            type=int,
            help='number of epochs for training')

    parser.add_argument('--padding',
            dest='padding',
            default=10,
            type=int,
            help='padding')

    parser.add_argument('--stride',
            dest='stride',
            default=20,
            type=int,
            help='stride')

    parser.add_argument('--minibatch-size',
            dest='minibatch_size',
            default=1,
            type=int,
            help='size of mini-batches')

    parser.add_argument('--lr',
            dest='learning_rate',
            default='0.0001',
            type=float,
            help='set learning rate')

    # checkpoint or inference file
    parser.add_argument('--load-checkpoint',
            dest='checkpoint_file',
            type=str,
            help='path to checkpoint file with .tar extension')

    parser.add_argument('--debug',
            dest='DEBUG',
            default=False,
            action='store_true',
            help='enable debug print')

    parser.add_argument('--X',
            type=int,
            dest='X',
            help='number of ConvBlocks in one Repeat in training')

    parser.add_argument('--R',
            type=int,
            dest='R',
            help='number of Repeats in training')

    parser.add_argument('--basepath',
            dest='BASE_DATA_PATH',
            type=str,
            help='path where related files for training will be saved (checkpoints, graphs, reconstructions..).')

    parser.add_argument('--dst-dir',
            dest="dst_dir",
            type=str,
            help='path to directory where separated mixtures will be saved.')

    parser.add_argument('--input-mixture',
            type=str,
            help='mixture that will be separated into source speakers')

    args = parser.parse_args()
    print(args)

####################################################################################################################################################################################

    ### hyperparameters and paths from parsed arguments
    DEBUG = args.DEBUG

    #BASE_DATA_PATH = r"/gdrive/My Drive/FIT/"
    BASE_DATA_PATH = args.BASE_DATA_PATH

    MINIBATCH_SIZE = args.minibatch_size
    R = args.R #number of repeats of ConvBlocks
    X = args.X #num of ConvBlocks in one repeat

    # Adam
    learning_rate   = args.learning_rate
    opt_decay       = 0       # 0.0001

    bias_enabled    = False
    padd            = args.padding
    nn_stride       = args.stride

    use_cuda        = True
    epochs          = args.epochs
    # audios_in_epoch = 20000 # kolik zpracovat nahravek v jedne epose

####################################################################################################################################################################################

    # create TasNet class
    tasnet = Net(X=X, R=R, nn_stride=nn_stride, padd=padd, batch_size=MINIBATCH_SIZE, DEBUG=DEBUG)

    # Check if cuda is available
    if use_cuda and torch.cuda.is_available():
        print("Cuda is available!")
        tasnet.cuda()
    else:
        print("Cuda is NOT available")

    # Optimizer
    optimizer = optim.Adam(tasnet.parameters(), lr = learning_rate, weight_decay=opt_decay)

####################################################################################################################################################################################
    # load NN from checkpoint and continue training
    if args.checkpoint_file:
        checkpoint = None
        if use_cuda and torch.cuda.is_available():
            checkpoint = torch.load(args.checkpoint_file)
        else:
            checkpoint = torch.load(args.checkpoint_file, map_location=torch.device('cpu'))

        tasnet.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        # tasnet.train() # For inference
        tasnet.eval() # For inference

        print("Nactena epocha a loss: ", str(epoch), str(loss))

####################################################################################################################################################################################
####################################################################################################################################################################################
    print('Prepared for inference, load your audio.')

    # mix_name = "smes2_resampled3.wav"
    mix_name = args.input_mixture
    data_path = args.BASE_DATA_PATH + "/" + args.input_mixture

    mixture = getAudioSamples(data_path)
    mixture.unsqueeze_(0)
    mixture.unsqueeze_(0)

    input_mixture = mixture
    print(input_mixture.size())

    if use_cuda and torch.cuda.is_available():
        input_mixture = input_mixture.cuda()

    separated_sources = tasnet(input_mixture)

    # === Save audio ===
    mixture_prep = 0
    source1_prep = 0
    source2_prep = 0

    if use_cuda and torch.cuda.is_available():
        mixture_prep = input_mixture.cpu().detach().numpy()
        source1_prep = separated_sources[0][0].cpu().detach().numpy()
        source2_prep = separated_sources[0][1].cpu().detach().numpy()
    else:
        mixture_prep = input_mixture.detach().numpy()
        source1_prep = separated_sources[0][0].detach().numpy()
        source2_prep = separated_sources[0][1].detach().numpy()

    wav.write(args.dst_dir + "s1-" + mix_name, 8000, source1_prep)
    wav.write(args.dst_dir + "s2-" + mix_name, 8000, source2_prep)

    print("Inference done, separated speakers saved into " + args.dst_dir)

