import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn as nn
import torch.utils.data as data_utils
from torch.autograd import Variable
import torchvision.transforms as transforms
import os
from datetime import datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt

from SegmentDataset import SegmentDataset
from TasNet import Net
from ResBlock import ResBlock
from tools import *
from snr import *


def log(info):
    """
    Write string from parameter 'info' into the file.
    """
    if os.path.exists(training_dir):
        with open(training_dir + "training.log", "a") as trainlog:
            trainlog.write(str(info) + "\n")
    else:
        print("Error: Cant write log into the file because directory does not exist.")


if __name__== "__main__":
    print("Version 20")

    parser = argparse.ArgumentParser(description='Setup and init neural network')

    parser.add_argument('--epochs',
            dest='epochs',
            type=int,
            help='number of epochs for training')

    parser.add_argument('--segment-length',
            dest='segment_length',
            default=32000,
            type=int,
            help='length of segments, default is 32k (4s)')

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
            default='0.001',
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

    parser.add_argument('--disable-validation',
            dest='disable_validation',
            default=False,
            action='store_true',
            help='disables validation after epoch')

    parser.add_argument('--disable-training',
            dest='disable_training',
            default=False,
            action='store_true',
            help='disables backpropagation operation in training')

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
    segment_length  = args.segment_length

    use_cuda        = True
    epochs          = args.epochs

    # hodnota je rovna poctu zpracovanych batchu
    # (pocet_segmentu = pocet_batchu * velikost_batche)
    print_controll_check = 50
    print_loss_frequency = 100 # za kolik segmentu (minibatchu) vypisovat loss
    print_valid_loss_frequency = 100 #100
    #log_loss_frequency = 5000
    #create_checkpoint_frequency = 800

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
    scheduler = lr_sched.ReduceLROnPlateau(optimizer, patience=3, mode='min', factor=0.5, verbose=True)
####################################################################################################################################################################################

    # load NN from checkpoint and continue training
    loaded_epoch = 0
    loaded_segments = 0
    best_validation_result = 50 #init value
    if args.checkpoint_file:
        checkpoint = None
        if use_cuda and torch.cuda.is_available():
            checkpoint = torch.load(args.checkpoint_file)
        else:
            checkpoint = torch.load(args.checkpoint_file, map_location=torch.device('cpu'))

        tasnet.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loaded_epoch = checkpoint['epoch']
        loaded_loss = checkpoint['loss']
        if 'glob_seg_cnt' in checkpoint.keys():
            loaded_segments = checkpoint['glob_seg_cnt']
        if 'best_validation_result' in checkpoint.keys():
            best_validation_result = checkpoint['best_validation_result']
        # if 'learning_rate' in checkpoint.keys():
            # learning_rate = checkpoint['learning_rate']
        if 'scheduler_state_dict' in checkpoint.keys():
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        tasnet.train()

        print("Z checkpointu nactena epocha, loss a pocet segmentu: ", str(loaded_epoch), str(loaded_loss), str(loaded_segments))

####################################################################################################################################################################################

    learning_started_date = datetime.now().strftime('%Y-%m-%d_%H:%M')

    train_data_path = BASE_DATA_PATH+"tr/"
    valid_data_path = BASE_DATA_PATH+"cv/"

    trainset = SegmentDataset(train_data_path, segment_length)
    validset = SegmentDataset(valid_data_path, segment_length)

    # Note: We shuffle the loading process of train_dataset to make the learning process
    # independent of data order, but the order of test_loader
    # remains so as to examine whether we can handle unspecified bias order of inputs.
    trainloader = data_utils.DataLoader(dataset=trainset,
            batch_size = MINIBATCH_SIZE,
            shuffle=False,
            collate_fn = train_collate,
            drop_last = False)

    validloader = data_utils.DataLoader(dataset=validset,
            batch_size = MINIBATCH_SIZE,
            shuffle=False,
            collate_fn = train_collate,
            drop_last = False)

    # Create directory for loss file, reconstructions and checkpoints
    training_dir = ""
    if not args.checkpoint_file:
        # Start training
        training_dir = args.dst_dir + learning_started_date + "_X"+str(X) + "_R" + str(R) + "/"
        print("Trainign directory: ", training_dir)
        if not os.path.exists(training_dir):
            os.makedirs(training_dir)
            os.makedirs(training_dir+"reconstruction")
            os.makedirs(training_dir+"inference")

        start_epoch = 1
    else:
        # Training will continue from given checkpoint
        training_dir = args.dst_dir
        if not os.path.exists(training_dir):
            print("Error: Training cant continue, because given directory does not exist.")
            exit(6)
        print("Continue trainign in directory: ", training_dir)
        log("Training continues from checkpoint "+args.checkpoint_file)

        start_epoch = loaded_epoch + 1

    epochs = loaded_epoch + epochs

    log(str(datetime.now()))
    log(args)
    log("numpy version: " + np.__version__)
    log("pytorch version: " + torch.__version__)
    log("Creating Trainign directory: " + training_dir)

    global_segment_cnt = 0 + loaded_segments

    log("##### Training started #####")
    for (epoch) in range(start_epoch, epochs + 1):
        epoch_start = datetime.now()
        print("Epoch ", epoch, "/",epochs," started at ", epoch_start)
        log("## Epoch " + str(epoch) + "/" + str(epochs) + " started at " + str(epoch_start))

        loss = 0
        running_loss = 0.0
        segment_cnt = 0
        valid_segment_cnt = 0
        batch_cnt = 0

        for batch_cnt, data in enumerate(trainloader, 1):
            # print("batch_cnt: ", (batch_cnt))

            actual_batch_size = len(data[0])
            global_segment_cnt += actual_batch_size
            segment_cnt += actual_batch_size

            # torch.autograd.set_detect_anomaly(True)

            if (segment_cnt/MINIBATCH_SIZE) % (print_controll_check) == 0.0:
                # print("") # Kvuli Google Colab je nutne minimalizovat vypisovani na OUT
                # print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch, segment_cnt)
                with open(training_dir + "controll_check.log", "a") as logloss:
                    logloss.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " epoch: " + str(epoch) + " segment_cnt: " + str(segment_cnt) + "\n")

            input_mixture  = data[0] # tensor(N, 1, 32000) # N of mix segments
            target_source1 = data[1] # tensor(N, 1, 32000) # N of s1 segments
            target_source2 = data[2] # tensor(N, 1, 32000) # N of s2 segments

            if use_cuda and torch.cuda.is_available():
                input_mixture = input_mixture.cuda()
                target_source1 = target_source1.cuda()
                target_source2 = target_source2.cuda()

            separated_sources = tasnet(input_mixture)
            # Outputs: (N, 2, 32000)

            separated_sources = separated_sources.transpose(1,0)
            # Outputs: (2, N, 32000)

            s1 = separated_sources[0].unsqueeze(1)
            s2 = separated_sources[1].unsqueeze(1)
            # Outputs: eval. s1, s2 = torch.size([N, 1, 32000])
            #          ref.  s1, s2 = torch.size([N, 1, 32000])

            # Loss calculation
            batch_loss1 = np.add(np.negative(siSNRloss(s1, target_source1)), np.negative(siSNRloss(s2, target_source2)))
            batch_loss2 = np.add(np.negative(siSNRloss(s1, target_source2)), np.negative(siSNRloss(s2, target_source1)))

            # calculate MIN for each col (batch pair) of batches in range(0,batch_size-1)
            optimizer.zero_grad()
            loss = 0
            for batch_id in range(actual_batch_size):
                loss += min(batch_loss1[batch_id], batch_loss2[batch_id])

            if not args.disable_training:
                loss.backward()
                optimizer.step()

            # calculate average loss
            running_loss += loss.item()

            # === print loss ===
            if (segment_cnt/MINIBATCH_SIZE) % (print_loss_frequency) == 0.0:
                # print('[%d, %5d] loss: %.5f' % (epoch, segment_cnt, running_loss/print_loss_frequency))
                # Write loss to file
                with open(training_dir + "training_loss.log", "a") as logloss:
                    logloss.write(str(global_segment_cnt)+","+str(running_loss/print_loss_frequency)+"\n")
                running_loss = 0.0

        ### End of epoch ###
        epoch_end = datetime.now()

        # print("batch_cnt: ", batch_cnt, " segment-cnt: ", segment_cnt)

        print("Epoch ", epoch, "/",epochs," finished - processed in ", (epoch_end - epoch_start),"\n")
        log("## Epoch " + str(epoch) + "/" + str(epochs) + " finished - processed in " + str((epoch_end - epoch_start))+ "\n")
        # ====== End Of Epoch ======


        # ====== VALIDACE na konci epochy ======
        if not args.disable_validation:
            print("Validace")
            log("## Validation started")
            validation_start = datetime.now()

            valid_segment_cnt = 1
            running_loss = 0.0
            current_validation_result = 0
            with torch.no_grad():
                for batch_cnt, data in enumerate(validloader, 1):
                    # valid_segment_cnt += MINIBATCH_SIZE

                    actual_batch_size = len(data[0])
                    valid_segment_cnt += actual_batch_size

                    # torch.autograd.set_detect_anomaly(True)

                    if valid_segment_cnt % 500 == 0:
                        print("") # Kvuli Google Colab je nutne minimalizovat vypisovani na OUT
                        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch, valid_segment_cnt)

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

                    # loss calculation
                    batch_loss1 = np.add(np.negative(siSNRloss(s1, target_source1)), np.negative(siSNRloss(s2, target_source2)))
                    batch_loss2 = np.add(np.negative(siSNRloss(s1, target_source2)), np.negative(siSNRloss(s2, target_source1)))

                    # calculate MIN for each col (batch pair) of batches in range(0,batch_size-1)
                    loss = 0
                    for batch_id in range(actual_batch_size):
                        loss += min(batch_loss1[batch_id], batch_loss2[batch_id])

                    # calculate average loss
                    running_loss += loss.item()
                    current_validation_result += loss.item()

                # Modify learning rate if loss not improved in 3 consecutive epochs
                current_validation_result /= valid_segment_cnt # prumer
                print("new: ", current_validation_result, " old: ", best_validation_result)
                scheduler.step(current_validation_result)
                if current_validation_result < best_validation_result:
                    best_validation_result = current_validation_result

                # Write current validation loss to file
                with open(training_dir + "validation_loss.log", "a") as logloss:
                    logloss.write(str(epoch)+","+str(best_validation_result)+"\n")

                # == Validacni dataset je zpracovan, Vyhodnoceni validace ==
                validation_end = datetime.now()
                print('Validation Finished in ', (validation_end - validation_start))
                log('## Validation Finished in ' + str((validation_end - validation_start)))
                print('')

        # ===== Validation skipped
        else:
            print('Warning: Validation skipped\n')
            log('Warning: Validation skipped\n')


        # Create checkpoint at the end of epoch
        torch.save({
          'epoch': epoch,
          'audio_cnt': segment_cnt,
          'glob_seg_cnt': global_segment_cnt,
          'model_state_dict': tasnet.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'scheduler_state_dict': scheduler.state_dict(),
          'loss': loss,
          'best_validation_result': best_validation_result,
        }, training_dir + 'tasnet_model_checkpoint_'+str(datetime.now().strftime('%Y-%m-%d'))+'_X'+str(X)+'_R'+str(R)+'_e'+str(epoch)+'.tar')

        # Print log message
        print("Checkpoint has been created after epoch.\n")
        log("Checkpoint created after epoch: "+training_dir + 'tasnet_model_checkpoint_'+str(datetime.now().strftime('%Y-%m-%d'))+'_X'+str(X)+'_R'+str(R)+'_e'+str(epoch)+'_a'+str(segment_cnt)+'.tar')
        # ====== END OF EPOCH =====


    # Save Network For Inference in the end of training
    torch.save(tasnet.state_dict(), training_dir+'tasnet_model_inference'+'_X'+str(X)+'_R'+str(R)+'_e'+str(epoch)+'_a'+str(global_segment_cnt)+'.pkl')
    log("Created Inference checkpoint: " + training_dir+'tasnet_model_inference'+'_X'+str(X)+'_R'+str(R)+'_e'+str(epoch)+'_a'+str(global_segment_cnt)+'.pkl')
    print('Finished Training')
    log("##### Training Finished #####")

