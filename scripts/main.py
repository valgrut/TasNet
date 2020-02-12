import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data_utils
from torch.autograd import Variable
import torchvision.transforms as transforms
from datetime import datetime
import argparse
import numpy as np

from Dataset import AudioDataset
from TrainDataset import TrainDataset
from TasNet import Net
from ResBlock import ResBlock
from tools import *
from snr import *

if __name__== "__main__":
    parser = argparse.ArgumentParser(description='Setup and init neural network')

    parser.add_argument('--train',
            dest='TRAIN',
            action='store_true',
            help='if option set, loss is printed every num of processed audios, where num is given by parameter.')

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

    parser.add_argument('--print-loss',
            action='store_true',
            help='if option set, loss is printed every num of processed audios, where num is given by parameter.')

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

    # parser.add_argument('--disable-validation',
            # action='store_true',
            # help='disable validation after epoch.')

    parser.add_argument('--dst-dir',
            dest="dst_dir",
            type=str,
            help='path to directory where separated mixtures will be saved.')

    parser.add_argument('--test',
            dest='TEST',
            action='store_true',
            help='start calculate SI-SNR on testing set') #require 1.inf file, 2.dstdir, 3.datapath.

    parser.add_argument('--inference',
            dest='INFERENCE',
            action='store_true',
            help='create Network and load checkpoint for inference.') #req: 1. input-mixture, 2. inf file, 3. dst-dir

    parser.add_argument('--input-mixture',
            type=str,
            help='mixture that will be separated into source speakers')

    args = parser.parse_args()
    print(args)
####################################################################################################################################################################################
####################################################################################################################################################################################
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

    audio_save_frequency = 5000
    print_loss_frequency = 5000 # za kolik segmentu (minibatchu) vypisovat loss
    print_valid_loss_frequency = 6000
    #log_loss_frequency = 5000
    create_checkpoint_frequency = 6000

####################################################################################################################################################################################
####################################################################################################################################################################################
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
####################################################################################################################################################################################
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

        if args.TRAIN:
            tasnet.train()
        else:
            tasnet.eval() # For inference

        print("Nactena epocha a loss: ", str(epoch), str(loss))

####################################################################################################################################################################################
####################################################################################################################################################################################

    learning_started_date = datetime.now().strftime('%Y-%m-%d %H:%M')

    if args.TRAIN:
        train_data_path = BASE_DATA_PATH+"tr/"
        valid_data_path = BASE_DATA_PATH+"cv/"

        # trainset = AudioDataset(train_data_path)
        trainset = TrainDataset(train_data_path)
        validset = AudioDataset(valid_data_path)

        # Note: We shuffle the loading process of train_dataset to make the learning process
        # independent of data order, but the order of test_loader
        # remains so as to examine whether we can handle unspecified bias order of inputs.
        # trainloader = data_utils.DataLoader(trainset, batch_size = MINIBATCH_SIZE, shuffle=True)
        # trainloader = data_utils.DataLoader(trainset, batch_size = MINIBATCH_SIZE, shuffle=False)
        # trainloader = data_utils.DataLoader(trainset, batch_size = MINIBATCH_SIZE, shuffle=True, collate_fn = audio_collate)
        trainloader = data_utils.DataLoader(trainset, batch_size = MINIBATCH_SIZE, shuffle=True, collate_fn = train_collate, drop_last = True)
        validloader = data_utils.DataLoader(validset, batch_size = MINIBATCH_SIZE, shuffle=False, collate_fn = audio_collate)

        # from torch.utils.data import *
        # print(list(BatchSampler(SequentialSampler(trainloader), batch_size=5, drop_last=False)))

        # TESTING of Dataloading
        # itr = iter(trainloader)
        # for audio_cnt, data in enumerate(trainloader, 0):
        #     # test collate_fn:
        #     print("cnt: ", audio_cnt)
        #     print("ITER.next: ", itr.next())
        #     print(itr.next())
        # print("konec")
        # exit(1)

        best_validation_result = 42   #initial value
        graph_x = []
        graph_y = []

        ### Uz to neni audio_cnt ale spis segment_cnt
        global_segment_cnt = 0
        cont_epoch = 0
        for (epoch) in range(epochs):
            running_loss = 0.0
            segment_cnt = 0

            epoch = epoch + cont_epoch

            for audio_cnt, data in enumerate(trainloader, 0):
                global_segment_cnt += MINIBATCH_SIZE
                segment_cnt += MINIBATCH_SIZE

                torch.autograd.set_detect_anomaly(True)

                if segment_cnt % (MINIBATCH_SIZE*100) == 0:
                    print("") # Kvuli Google Colab je nutne minimalizovat vypisovani na OUT
                    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch, segment_cnt)

                input_mixture  = data[0]
                target_source1 = data[1]
                target_source2 = data[2]

                if use_cuda and torch.cuda.is_available():
                    input_mixture = input_mixture.cuda()
                    target_source1 = target_source1.cuda()
                    target_source2 = target_source2.cuda()

                optimizer.zero_grad()
                separated_sources = tasnet(input_mixture)

                separated_sources = separated_sources.transpose(1,0)
                # print("separated sources shape: ", separated_sources.shape)

                s1 = separated_sources[0].unsqueeze(1)
                s2 = separated_sources[1].unsqueeze(1)

                # print("sepsrc1 shape: ", s1.shape)
                # print("sepsrc2 shape: ", s2.shape)
                # print("target_source1 shape: ", target_source1.shape)
                # print("target_source2 shape: ", target_source2.shape)

                if(s1.shape[2] != target_source1.shape[2]):
                    smallest = min(input_mixture.shape[2], s1.shape[2], s2.shape[2], target_source1.shape[2], target_source2.shape[2])
                    s1 = s1.narrow(2, 0, smallest)
                    s2 = s2.narrow(2, 0, smallest)
                    target_source1 = target_source1.narrow(2, 0, smallest)
                    target_source2 = target_source2.narrow(2, 0, smallest)

                batch_loss1 = np.add(np.negative(siSNRloss(s1, target_source1)), np.negative(siSNRloss(s2, target_source2)))
                batch_loss2 = np.add(np.negative(siSNRloss(s1, target_source2)), np.negative(siSNRloss(s2, target_source1)))

                # calculate MIN for each col (batch pair) of batches in range(0,batch_size-1)
                loss = 0
                for batch_id in range(MINIBATCH_SIZE):
                    loss = min(batch_loss1[batch_id], batch_loss2[batch_id])
                    loss.backward(retain_graph=True)

                optimizer.step()

                # calculate average loss
                running_loss += loss.item()

                # === print loss ===
                # if (segment_cnt/MINIBATCH_SIZE) % (print_loss_frequency) == print_loss_frequency - 1:
                if (segment_cnt/MINIBATCH_SIZE) % (print_loss_frequency) == 0:
                    print('[%d, %5d] loss: %.5f' % (epoch, segment_cnt, running_loss/print_loss_frequency))
                    graph_x.append(global_segment_cnt)
                    graph_y.append(running_loss/print_loss_frequency)

                    # Write loss to file
                    with open(args.dst_dir + "loss_"+ learning_started_date + "_X"+str(X) + "_R" + str(R) + ".log", "a") as logloss:
                        logloss.write(str(global_segment_cnt)+","+str(running_loss/print_loss_frequency)+"\n")

                    running_loss = 0.0

                # === Create checkpoint ===
                if (segment_cnt/MINIBATCH_SIZE) % (create_checkpoint_frequency) == create_checkpoint_frequency - 1:
                    # Create snapshot - checkpoint
                    torch.save({
                      'epoch': epoch,
                      'audio_cnt': segment_cnt,
                      'model_state_dict': tasnet.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'loss': loss,
                    }, args.dst_dir+'tasnet_model_checkpoint_'+str(datetime.now().strftime('%Y-%m-%d_%H:%M'))+'_X'+str(X)+'_R'+str(R)+'_e'+str(epoch)+'_a'+str(segment_cnt)+'.tar')

                # === Save reconstruction ===
                if (segment_cnt/MINIBATCH_SIZE) % (audio_save_frequency) == 0:
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

                    wav.write(args.dst_dir+"speech_e"+str(epoch)+"_a"+str(segment_cnt)+"_s1.wav", 8000, source1_prep)
                    wav.write(args.dst_dir+"speech_e"+str(epoch)+"_a"+str(segment_cnt)+"_s2.wav", 8000, source2_prep)
                    wav.write(args.dst_dir+"speech_e"+str(epoch)+"_a"+str(segment_cnt)+"_mix.wav", 8000, mixture_prep)
            # ==== End Of Epoch of training ====

            # === VALIDACE na konci epochy ===
            print("")
            print("Validace")
            valid_audio_cnt = 0
            running_loss = 0.0
            current_validation_result = 0

            with torch.no_grad():
                for audio_cnt, data in enumerate(validloader, 0):
                    valid_audio_cnt += 1

                    torch.autograd.set_detect_anomaly(True)

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

                    batch_loss1 = np.add(np.negative(siSNRloss(s1, target_source1)), np.negative(siSNRloss(s2, target_source2)))
                    batch_loss2 = np.add(np.negative(siSNRloss(s1, target_source2)), np.negative(siSNRloss(s2, target_source1)))

                    # calculate MIN for each col (batch pair) of batches in range(0,batch_size-1)
                    loss = 0
                    for batch_id in range(MINIBATCH_SIZE):
                        loss = loss + min(batch_loss1[batch_id], batch_loss2[batch_id])

                    # calculate average loss
                    running_loss += loss.item()
                    current_validation_result += loss.item()

                    # === print loss ===
                    if audio_cnt % print_valid_loss_frequency == print_valid_loss_frequency - 1:
                        print('[%d, %5d] loss: %.5f' % (epoch, audio_cnt, running_loss/print_valid_loss_frequency))
                        graph_x.append(valid_audio_cnt)
                        graph_y.append(running_loss/print_valid_loss_frequency)

                        # Write loss to file
                        with open(args.dst_dir + "validloss_"+ learning_started_date + "_X"+str(X) + "_R" + str(R) + ".log", "a") as logloss:
                            logloss.write(str(valid_audio_cnt)+","+str(running_loss/print_valid_loss_frequency)+"\n")

                        running_loss = 0.0

                # == Validacni dataset je zpracovan, Vyhodnoceni validace ==
                # TODO vykreslit i tuto loss, ukladat a upravit funkci aby vykreslila obe dve z trenovani i validacni a jinou barvou rpes sebe. (GIT)
                current_validation_result /= valid_audio_cnt # prumer
                print(current_validation_result, " ", best_validation_result)
                if current_validation_result >= best_validation_result:
                    learning_rate /= 2 #TODO zjistit kdy se to ma delit
                else:
                    best_validation_result = current_validation_result
                print('Finished Validating')
                print('')


        # Save Network For Inference in the end of training
        torch.save(tasnet.state_dict(), args.dst_dir+'tasnet_model_inference'+'_X'+str(X)+'_R'+str(R)+'_e'+str(epoch)+'_ga'+str(global_segment_cnt)+'.pkl')
        print('Finished Training')

        plt.plot(graph_x, graph_y)
        plt.show()


####################################################################################################################################################################################
####################################################################################################################################################################################

    if args.TEST:
        # Load Test dataset
        test_data_path = BASE_DATA_PATH+"tt/"
        testset        = AudioDataset(test_data_path)
        testloader     = data_utils.DataLoader(testset, batch_size = MINIBATCH_SIZE, shuffle=False)

        # Start Testing
        sdr_sum = 0
        global_segment_cnt = 0
        # pridat zde do testovani SI_SNR  a pro kazdou nahravku a vysledky zprumerovat a vyhodnotit (GIT)
        with torch.no_grad():
            for audio_cnt, data in enumerate(testloader, 0):
                global_segment_cnt += 1

                input_mixture  = data[0]
                target_source1 = data[1]
                target_source2 = data[2]

                if use_cuda and torch.cuda.is_available():
                    input_mixture = input_mixture.cuda()
                    target_source1 = target_source1.cuda()
                    target_source2 = target_source2.cuda()

                # separation
                separated_sources = tasnet(input_mixture)

                # unitize length of audio
                smallest = min(input_mixture.shape[2], target_source1.shape[2], target_source2.shape[2], separated_sources.shape[2])
                input_mixture = input_mixture.narrow(2, 0, smallest)
                target_source1 = target_source1.narrow(2, 0, smallest)
                target_source2 = target_source2.narrow(2, 0, smallest)
                separated_sources = separated_sources.narrow(2, 0, smallest)

                # spojeni ref sources do jedne matice
                target_sources = torch.cat((target_source1, target_source2), 1)

                # prepare tensor for SI-SNR
                estimated_sources_prep = 0
                ref_sources_prep = 0

                if use_cuda and torch.cuda.is_available():
                    estimated_sources_prep = separated_sources[0].cpu().detach().numpy()
                    ref_sources_prep = target_sources[0].cpu().detach().numpy()
                else:
                    estimated_sources_prep = separated_sources[0].detach().numpy()
                    ref_sources_prep = target_sources[0].detach().numpy()

                # compute SI-SNR
                (sdr, sir, sarn, perm) = bss_eval_sources(ref_sources_prep, estimated_sources_prep, compute_permutation=True)
                print(sdr)

                sdr_sum += sdr

        print("Final SDR: " + str(sdr_sum/global_segment_cnt))
        print('Finished Testing')

####################################################################################################################################################################################
####################################################################################################################################################################################
    if args.INFERENCE:
        print('Prepared for inference, load your audio.')

        mix_name = "smes2_resampled3.wav"
        mix_name = args.input_mixture

        data_path = args.input_mixture
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

        print("Inference done, separated speakers saved into " + args.dst_dir + "inferenced/")


