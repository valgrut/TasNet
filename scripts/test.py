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
from TasNet import Net
from ResBlock import ResBlock
from tools import *
from snr import *

if __name__== "__main__":
    print("Version 07")
    parser = argparse.ArgumentParser(description='Setup and init neural network')

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

        tasnet.eval() # For inference and testing


    learning_started_date = datetime.now().strftime('%Y-%m-%d_%H:%M')

    # Load Test dataset
    test_data_path = BASE_DATA_PATH+"tt/"
    testset        = AudioDataset(test_data_path)
    testloader     = data_utils.DataLoader(testset, batch_size = MINIBATCH_SIZE, shuffle=False)

    # Start Testing
    sdr_sum = 0
    global_segment_cnt = 0
    running_loss = 0.0
    test_segment_cnt = 0
    current_testing_result = 0

    # kopie  z validace - UPRAVIT!!!
    with torch.no_grad():
        for audio_cnt, data in enumerate(testloader, 0):
            global_segment_cnt += 1
            # test_audio_cnt += 1
            # torch.autograd.set_detect_anomaly(True)

            if audio_cnt % 500 == 0:
                print("") # Kvuli Google Colab je nutne minimalizovat vypisovani na OUT
                print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch, audio_cnt)

            input_mixture  = data[0]
            target_source1 = data[1]
            target_source2 = data[2]

            if use_cuda and torch.cuda.is_available():
                input_mixture = input_mixture.cuda()
                target_source1 = target_source1.cuda()
                target_source2 = target_source2.cuda()

            separated_sources = tasnet(input_mixture)

            separated_sources = separated_sources.transpose(1,0)

            s1 = separated_sources[0].unsqueeze(1)
            s2 = separated_sources[1].unsqueeze(1)

            if(s1.shape[2] != target_source1.shape[2]):
                smallest = min(input_mixture.shape[2], s1.shape[2], s2.shape[2], target_source1.shape[2], target_source2.shape[2])
                s1 = s1.narrow(2, 0, smallest)
                s2 = s2.narrow(2, 0, smallest)
                target_source1 = target_source1.narrow(2, 0, smallest)
                target_source2 = target_source2.narrow(2, 0, smallest)
                input_mixture = input_mixture.narrow(2, 0, smallest)


            # Vypocet metrik
            # compute SI-SNR / TODO zde asi taky bude potreba udelat to samo jako pro pocitani loss
            # protoze taky nemuzu vedet, ze mix[0] odpovida s1, nebo jestli odpovida s2

            print(target_source1.shape, target_source2.shape)
            print(s1.shape, s2.shape)

            batch_loss1 = np.add(np.negative(siSNRloss(s1, target_source1)), np.negative(siSNRloss(s2, target_source2)))
            batch_loss2 = np.add(np.negative(siSNRloss(s1, target_source2)), np.negative(siSNRloss(s2, target_source1)))

            # calculate MIN for each col (batch pair) of batches in range(0,batch_size-1)
            loss = 0
            for batch_id in range(MINIBATCH_SIZE):
                loss = loss + min(batch_loss1[batch_id], batch_loss2[batch_id])

            # calculate average loss
            running_loss += loss.item()
            current_testing_result += loss.item()

            (sdr, sir, sarn, perm) = bss_eval_sources(ref_sources_prep, estimated_sources ,compute_permutation=True)
            # (sdr, sir, sarn, perm) = bss_eval_sources(ref_sources_prep, estimated_sources_prep, compute_permutation=True)
            print(sdr)

            sdr_sum += sdr


    # pridat zde do testovani SI_SNR  a pro kazdou nahravku a vysledky zprumerovat a vyhodnotit (GIT)
    # with torch.no_grad():
    #     for audio_cnt, data in enumerate(testloader, 0):
    #         global_segment_cnt += 1

    #         input_mixture  = data[0]
    #         target_source1 = data[1]
    #         target_source2 = data[2]

    #         if use_cuda and torch.cuda.is_available():
    #             input_mixture = input_mixture.cuda()
    #             target_source1 = target_source1.cuda()
    #             target_source2 = target_source2.cuda()

    #         # separation
    #         separated_sources = tasnet(input_mixture)

    #         # unitize length of audio
    #         smallest = min(input_mixture.shape[2], target_source1.shape[2], target_source2.shape[2], separated_sources.shape[2])
    #         input_mixture = input_mixture.narrow(2, 0, smallest)
    #         target_source1 = target_source1.narrow(2, 0, smallest)
    #         target_source2 = target_source2.narrow(2, 0, smallest)
    #         separated_sources = separated_sources.narrow(2, 0, smallest)

    #         # spojeni ref sources do jedne matice
    #         target_sources = torch.cat((target_source1, target_source2), 1)

    #         # prepare tensor for SI-SNR
    #         estimated_sources_prep = 0
    #         ref_sources_prep = 0

    #         if use_cuda and torch.cuda.is_available():
    #             estimated_sources_prep = separated_sources[0].cpu().detach().numpy()
    #             ref_sources_prep = target_sources[0].cpu().detach().numpy()
    #         else:
    #             estimated_sources_prep = separated_sources[0].detach().numpy()
    #             ref_sources_prep = target_sources[0].detach().numpy()

    #         # compute SI-SNR
    #         (sdr, sir, sarn, perm) = bss_eval_sources(ref_sources_prep, estimated_sources_prep, compute_permutation=True)
    #         print(sdr)

    #         sdr_sum += sdr

    print("Final SDR: " + str(sdr_sum/global_segment_cnt))
    print('Finished Testing')
