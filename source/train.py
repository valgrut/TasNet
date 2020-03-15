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

from SegmentDataset import SegmentDataset
from TasNet import Net
from ResBlock import ResBlock
from tools import *
from snr import *

if __name__== "__main__":
    print("Version 15")

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

    use_cuda        = True
    epochs          = args.epochs

    # hodnota je rovna poctu zpracovanych batchu
    # (pocet_segmentu = pocet_batchu * velikost_batche)
    print_controll_check = 50
    print_loss_frequency = 100 # za kolik segmentu (minibatchu) vypisovat loss
    print_valid_loss_frequency = 100
    #log_loss_frequency = 5000
    create_checkpoint_frequency = 800

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

        tasnet.train()

        print("Z checkpointu nactena epocha a loss: ", str(epoch), str(loss))

####################################################################################################################################################################################

    learning_started_date = datetime.now().strftime('%Y-%m-%d_%H:%M')

    train_data_path = BASE_DATA_PATH+"tr/"
    valid_data_path = BASE_DATA_PATH+"cv/"

    trainset = SegmentDataset(train_data_path)
    validset = SegmentDataset(valid_data_path)

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
    training_dir = args.dst_dir + learning_started_date + "_X"+str(X) + "_R" + str(R) + "/"
    print("Trainign directory: ", training_dir)
    if not os.path.exists(training_dir):
        os.makedirs(training_dir)
        os.makedirs(training_dir+"reconstruction")
        os.makedirs(training_dir+"inference")

    def log(info):
        with open(training_dir + "training.log", "a") as trainlog:
            trainlog.write(str(info) + "\n")

    log(str(datetime.now()))
    log(args)
    log("numpy version: " + np.__version__)
    log("pytorch version: " + torch.__version__)
    log("Creating Trainign directory: " + training_dir)


    # TESTING of Dataloading
    # itr = iter(trainloader)
    # for audio_cnt, data in enumerate(trainloader, 0):
    # #     # test collate_fn:
    #     print("cnt: ", audio_cnt)
    #     bat = itr.next()
    #     print("ITER.next: ", len(bat))
    #     # print(itr.next())
    #     input("Press Enter to continue...\n\n")
    # print("konec")
    # exit(1)


    best_validation_result = 42   #initial value

    global_segment_cnt = 0
    cont_epoch = 0

    log("##### Training started #####")
    for (epoch) in range(1,epochs+1):
        epoch_start = datetime.now()
        print("Epoch ", epoch, "/",epochs," started at ", epoch_start)
        log("## Epoch " + str(epoch) + "/" + str(epochs) + " started at " + str(epoch_start))

        loss = 0
        running_loss = 0.0
        segment_cnt = 0
        valid_segment_cnt = 0
        batch_cnt = 0

        # epoch = epoch + cont_epoch
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

            # TODO pozn neni potreba, protoze segmenty jsou upravovany v collate_fn a dale.
            # A proc to furt potreba je... nekde se to z nejakyho duvodu nici
            # if(s1.shape[2] != target_source1.shape[2] != s2.shape[2] != target_source2.shape[2]):
                # smallest = min(input_mixture.shape[2], s1.shape[2], s2.shape[2], target_source1.shape[2], target_source2.shape[2])
                # s1 = s1.narrow(2, 0, smallest)
                # s2 = s2.narrow(2, 0, smallest)
                # target_source1 = target_source1.narrow(2, 0, smallest)
                # target_source2 = target_source2.narrow(2, 0, smallest)


            # Loss calculation
            batch_loss1 = np.add(np.negative(siSNRloss(s1, target_source1)), np.negative(siSNRloss(s2, target_source2)))
            batch_loss2 = np.add(np.negative(siSNRloss(s1, target_source2)), np.negative(siSNRloss(s2, target_source1)))

            # calculate MIN for each col (batch pair) of batches in range(0,batch_size-1)
            optimizer.zero_grad()
            loss = 0
            for batch_id in range(actual_batch_size):
                loss += min(batch_loss1[batch_id], batch_loss2[batch_id])
                # loss = min(batch_loss1[batch_id], batch_loss2[batch_id])
                # loss.backward(retain_graph=True)

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


            # === Create checkpoint ===
            # if (segment_cnt/MINIBATCH_SIZE) % (create_checkpoint_frequency) == 0.0:
            #     # Create snapshot - checkpoint
            #     torch.save({
            #       'epoch': epoch,
            #       'audio_cnt': segment_cnt,
            #       'model_state_dict': tasnet.state_dict(),
            #       'optimizer_state_dict': optimizer.state_dict(),
            #       'loss': loss,
            #     }, training_dir + 'tasnet_model_checkpoint_'+str(datetime.now().strftime('%Y-%m-%d'))+'_X'+str(X)+'_R'+str(R)+'_e'+str(epoch)+'_a'+str(segment_cnt)+'.tar')
            #     print("Checkpoint has been created.")
            #     log("Checkpoint created: "+training_dir + 'tasnet_model_checkpoint_'+str(datetime.now().strftime('%Y-%m-%d'))+'_X'+str(X)+'_R'+str(R)+'_e'+str(epoch)+'_a'+str(segment_cnt)+'.tar')

        # ### End of epoch ###
        epoch_end = datetime.now()
        print(">>Epoch ends. Post epoch operations:")

        # Create checkpoint at the end of epoch
        torch.save({
          'epoch': epoch,
          'audio_cnt': segment_cnt,
          'model_state_dict': tasnet.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'loss': loss,
        }, training_dir + 'tasnet_model_checkpoint_'+str(datetime.now().strftime('%Y-%m-%d'))+'_X'+str(X)+'_R'+str(R)+'_e'+str(epoch)+'_a'+str(segment_cnt)+'.tar')

        print("Checkpoint has been created after epoch.")
        log("Checkpoint created after epoch: "+training_dir + 'tasnet_model_checkpoint_'+str(datetime.now().strftime('%Y-%m-%d'))+'_X'+str(X)+'_R'+str(R)+'_e'+str(epoch)+'_a'+str(segment_cnt)+'.tar')

        print("batch_cnt: ", batch_cnt, " segment-cnt: ", segment_cnt)

        print("Epoch ", epoch, " finished - processed in ", (epoch_end - epoch_start), "\n")
        log("## Epoch " + str(epoch) + " finished - processed in " + str((epoch_end - epoch_start))+ "\n")
        # ====== End Of Epoch ======


        # ====== VALIDACE na konci epochy ======
        if not args.disable_validation:
            print("")
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

                    if(s1.shape[2] != target_source1.shape[2]):
                        smallest = min(input_mixture.shape[2], s1.shape[2], s2.shape[2], target_source1.shape[2], target_source2.shape[2])
                        s1 = s1.narrow(2, 0, smallest)
                        s2 = s2.narrow(2, 0, smallest)
                        target_source1 = target_source1.narrow(2, 0, smallest)
                        target_source2 = target_source2.narrow(2, 0, smallest)

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

                    # === print loss ===
                    if valid_segment_cnt % print_valid_loss_frequency == print_valid_loss_frequency - 1:
                        print('[%d, %5d] loss: %.5f' % (epoch, valid_segment_cnt, running_loss/print_valid_loss_frequency))

                        # Write loss to file
                        with open(training_dir + "validation_loss.log", "a") as logloss:
                            logloss.write(str(valid_segment_cnt)+","+str(running_loss/print_valid_loss_frequency)+"\n")

                        running_loss = 0.0

                # == Validacni dataset je zpracovan, Vyhodnoceni validace ==
                validation_end = datetime.now()
                print('Validation Finished in ', (validation_end - validation_start))
                log('## Validation Finished in ' + str((validation_end - validation_start)))
                print('')

                # TODO vykreslit i tuto loss, ukladat a upravit funkci aby vykreslila obe dve z trenovani i validacni a jinou barvou rpes sebe. (GIT)
                current_validation_result /= valid_segment_cnt # prumer
                print(current_validation_result, " ", best_validation_result)
                if current_validation_result >= best_validation_result:
                    learning_rate /= 2 #TODO zjistit kdy se to ma delit
                else:
                    best_validation_result = current_validation_result

        # ===== Validation skipped
        else:
            print('Warning: Validation skipped\n')
            log('Warning: Validation skipped\n')

    # Save Network For Inference in the end of training
    torch.save(tasnet.state_dict(), training_dir+'tasnet_model_inference'+'_X'+str(X)+'_R'+str(R)+'_e'+str(epoch)+'_a'+str(global_segment_cnt)+'.pkl')
    log("Created Inference checkpoint: " + training_dir+'tasnet_model_inference'+'_X'+str(X)+'_R'+str(R)+'_e'+str(epoch)+'_a'+str(global_segment_cnt)+'.pkl')
    print('Finished Training')
    log("##### Training Finished #####")

