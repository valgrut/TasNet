import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data_utils
from torch.autograd import Variable
import torchvision.transforms as transforms
import os
from datetime import datetime
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pystoi import stoi
from pesq import pesq

from AudioDataset import AudioDataset
from TasNet import Net
from ResBlock import ResBlock
from tools import *
from snr import *

if __name__== "__main__":
    print("Version 14")
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
    else:
        print("Error: Checkpoint is required for evaluation.")
        exit(6)

    training_dir = args.dst_dir
    with open(training_dir + "testing.log", "a") as testlog:
        testlog.write("Loaded Checkpoint: " + args.checkpoint_file + "\n")

    learning_started_date = datetime.now().strftime('%Y-%m-%d_%H:%M')

    # Load Test dataset
    test_data_path = BASE_DATA_PATH+"tt/"
    testset        = AudioDataset(test_data_path)
    testloader     = data_utils.DataLoader(testset, batch_size=MINIBATCH_SIZE, shuffle=False)

    # Start Testing
    sdr_sum = 0
    sir_sum = 0
    sarn_sum = 0
    stoi_sum = 0
    pesq_sum = 0
    perm_sum = 0

    global_audio_cnt = 0
    running_loss = 0.0
    current_testing_result = 0

    with torch.no_grad():
        for audio_cnt, data in enumerate(testloader, 0):
            global_audio_cnt += 1

            if (global_audio_cnt) % 500 == 0.0:
                print("")
                print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch, audio_cnt)
                # with open(training_dir + "testing.log", "a") as testlog:
                    # Note: value of sdr is decreasing, because global_audio_cnt is increasing and sdr is divided by that value in this log
                    # testlog.write("Audio_cnt: " + str(audio_cnt) + " SDR: " + str(sdr_sum/global_audio_cnt) + "\n")

            input_mixture  = data[0]
            target_source1 = data[1]
            target_source2 = data[2]

            if use_cuda and torch.cuda.is_available():
                input_mixture = input_mixture.cuda()
                target_source1 = target_source1.cuda()
                target_source2 = target_source2.cuda()

            # separation
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

            # Calculation of metrics
            s1.squeeze_(0)
            s2.squeeze_(0)
            target_source1.squeeze_(0)
            target_source2.squeeze_(0)

            # prepare tensor for SI-SNR
            estimated_sources_prep = 0
            ref_sources_prep = 0

            # Create numpy array from tensors
            if use_cuda and torch.cuda.is_available():
                estimated_source1_prep = s1.cpu().detach().numpy()
                estimated_source2_prep = s2.cpu().detach().numpy()
                ref_source1_prep = target_source1.cpu().detach().numpy()
                ref_source2_prep = target_source2.cpu().detach().numpy()
            else:
                estimated_source1_prep = s1.detach().numpy()
                estimated_source2_prep = s2.detach().numpy()
                ref_source1_prep = target_source1.detach().numpy()
                ref_source2_prep = target_source2.detach().numpy()

            # Create np array of shape (NumOfSpeakers, NumOfSamples)
            estimated_sources = np.concatenate((estimated_source1_prep, estimated_source2_prep))
            ref_sources = np.concatenate((ref_source1_prep, ref_source2_prep))

            # calculation of metrics
            ###################################################################
            # bss_eval_sources calculation taken from https://github.com/sigsep/bsseval
            (sdr, sir, sarn, perm) = bss_eval_sources(ref_sources, estimated_sources, compute_permutation=True)
            # print(sdr, sir, sarn, perm)
            
            with open(args.checkpoint_file + ".sdr", "a") as testsdr:
                testsdr.write("mixtureXYZTODO" + " " + str(np.round(max(sdr), 12)) + "\n")

            # print(type(np.round(max(sdr))))
            
            ###################################################################
            # stoi function taken from https://github.com/mpariente/pystoi
            stoi1 = stoi(ref_sources[0], estimated_sources[0], 8000, extended=False)
            stoi2 = stoi(ref_sources[0], estimated_sources[1], 8000, extended=False)
            stoi3 = stoi(ref_sources[1], estimated_sources[0], 8000, extended=False)
            stoi4 = stoi(ref_sources[1], estimated_sources[1], 8000, extended=False)
            # print(stoi1, stoi2, stoi3, stoi4)
            stoi_max1 = max(stoi1, stoi2)
            stoi_max2 = max(stoi3, stoi4)
            # print(stoi_max1, stoi_max2)

            # Short Term Objective Intelligibility (STOI)
            stoi_sum += max(stoi_max1, stoi_max2)


            ###################################################################
            # PESQ function taken from https://github.com/ludlows/python-pesq
            pesq1 = pesq(8000, ref_sources[0], estimated_sources[0], 'nb')
            pesq2 = pesq(8000, ref_sources[0], estimated_sources[1], 'nb')
            pesq3 = pesq(8000, ref_sources[1], estimated_sources[0], 'nb')
            pesq4 = pesq(8000, ref_sources[1], estimated_sources[1], 'nb')
            # print(pesq1, pesq2, pesq3, pesq4)
            pesq_max1 = max(pesq1, pesq2)
            pesq_max2 = max(pesq3, pesq4)
            # print(pesq_max1, pesq_max2)

            # Short Term Objective Intelligibility (STOI)
            pesq_sum += max(pesq_max1, pesq_max2)


            # TODO MOZNA zde asi taky bude potreba udelat to samo jako pro pocitani loss
            # protoze taky nemuzu vedet, ze mix[0] odpovida s1, nebo jestli odpovida s2

            # Signal to Distortion Ratios (SDR)
            sdr_sum += max(sdr) #add larger of two values

            # Source to Interference Ratios (SIR)
            sir_sum += max(sir)

            # Sources to Artifacts Ratios (SAR)
            sarn_sum += max(sarn)

            # vector containing the best ordering of estimated sources in the mean SIR sense
            # Just permutation - tells position of the best value when comparing cross sources. (s1-t1, s1-t2, s2-t1, s2-t2)
            # perm_sum += perm

    print("Final SDR:  " + str(sdr_sum/global_audio_cnt))
    print("Final SIR:  " + str(sir_sum/global_audio_cnt))
    print("Final SAR:  " + str(sarn_sum/global_audio_cnt))
    print("Final STOI: " + str(stoi_sum/global_audio_cnt))
    print("Final PESQ: " + str(pesq_sum/global_audio_cnt))

    # Save results into the
    with open(training_dir + "testing.log", "a") as testlog:
        testlog.write("Final SDR:  " + str(sdr_sum/global_audio_cnt) + "\n")
        testlog.write("Final SIR:  " + str(sir_sum/global_audio_cnt) + "\n")
        testlog.write("Final SAR:  " + str(sarn_sum/global_audio_cnt) + "\n")
        testlog.write("Final STOI: " + str(stoi_sum/global_audio_cnt) + "\n")
        testlog.write("Final PESQ: " + str(pesq_sum/global_audio_cnt) + "\n")

    print('Finished Testing')
