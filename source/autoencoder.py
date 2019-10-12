from scipy.io import wavfile as wav
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torchvision.transforms as transforms
import IPython.display as ipd
import numpy as np
import torch
import torch.utils.data as data_utils
from torch._six import int_classes as _int_classes
import signal
import sys
from os import listdir
from os.path import isfile, join
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as fc

######################################################################

def signal_handler(sig, frame):
    plt.plot(graph_x, graph_y)        
    plt.show()
    exit(0)
    
def signal_plot(sig, frame):
    plt.plot(audio_x, audio_y)
    plt.show()

#signal.signal(signal.SIGINT, signal_handler)
#signal.signal(signal.SIGUSR1, signal_plot)

#######################################################################
# SETTING and PARAMETERS

MINIBATCH_SIZE  = 1

optim_SGD       = False   # Adam / SGD
opt_lr          = 0.01    # 0.0001 pro adama

bias_enabled    = True

padd = 10  # 20 nebo 10?      (parametry nahovno: 20, lr=0,0001)

epochs          = 5
audios_in_epoch = 500 # kolik zpracovat nahravek v jedne epose
print_frequency = 50 # za kolik segmentu (minibatchu) vypisovat loss

######################################################################
class ResBlock(nn.Module):
    def __init__(self, in_channels, dilation):
        super(ResBlock, self).__init__()
        self.dilation = dilation
        
        self.conv1 = nn.Conv1d(256, 512, kernel_size=1)
        self.D_conv = nn.Conv1d(512, 512, kernel_size=3, padding=self.dilation, groups=512, dilation=self.dilation)
        self.conv2 = nn.Conv1d(512, 256, kernel_size=1)
        
        self.batch1 = nn.BatchNorm1d(512)
        self.batch2 = nn.BatchNorm1d(512)
     
        self.prelu1 = nn.PReLU(512)
        self.prelu2 = nn.PReLU(512)
    
    def forward(self, input_data):
        #print("shape start:", input_data.shape)
        x = self.conv1(input_data)
        x = self.prelu1(x)
        x = self.batch1(x)
        x = self.D_conv(x)
        #print("shape middle :", x.shape)
        #x = torch.reshape(x, (1, -1,))
        #x = torch.reshape(x, (-1,))
        #print("po concatenaci:", x.shape)
        x = self.prelu2(x)
        x = self.batch2(x)
        x = self.conv2(x)
        #print(x.shape)
        #print(input_data.shape)
        return torch.add(x, input_data)

######################################################################
# Separation - base
#
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 256, 20, bias=bias_enabled, stride=20, padding=padd)
        #self.deconv = nn.ConvTranspose1d(256, 2, 20, padding=padd, bias=bias_enabled, stride=20)
        self.deconv = nn.ConvTranspose1d(512, 2, 20, padding=padd, bias=bias_enabled, stride=20, groups=2)
        
        #self.layer_norm = nn.LayerNorm(256)
        self.bottleneck1 = nn.Conv1d(256, 256, 1) #TODO padding, stride???
        self.bottleneck2 = nn.Conv1d(256, 512, 1) #TODO 512 = NxC
        self.softmax = nn.Softmax(2)

        self.resblock1 = ResBlock(256, 1)
        self.resblock2 = ResBlock(256, 2)
        self.resblock3 = ResBlock(256, 4)
        self.resblock4 = ResBlock(256, 8)
        self.resblock5 = ResBlock(256, 16)

        self.resblock11 = ResBlock(256, 1)
        self.resblock12 = ResBlock(256, 2)
        self.resblock13 = ResBlock(256, 4)
        self.resblock14 = ResBlock(256, 8)
        self.resblock15 = ResBlock(256, 16)

        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.deconv.weight)
        

    def forward(self, input_data):
        input_data = self.conv1(input_data)
        input_data = fc.relu(input_data)

        data = self.bottleneck1(input_data)
 
        data = self.resblock1(data)
        data = self.resblock2(data)
        data = self.resblock3(data)
        data = self.resblock4(data)
        data = self.resblock5(data)
        
        data = self.resblock11(data)
        data = self.resblock12(data)
        data = self.resblock13(data)
        data = self.resblock14(data)
        data = self.resblock15(data)

        data = self.bottleneck2(data)
        data = torch.reshape(data, (1, 256, 2, -1,))
        data = self.softmax(data)
        separate_data = torch.mul(input_data[:,:,None,:], data)
        separate_data = torch.reshape(separate_data, (1, 512, -1))
        separate_data = self.deconv(separate_data)
        return separate_data

###################################################################
"""
AudioDataset
"""
class AudioDataset(data_utils.Dataset):
    """
    Dataset of speech mixtures for speech separation. 
    """
    def __init__(self, path, transform=None):
        super(AudioDataset, self).__init__()
        self.path = path
        self.mixtures_path = self.path + "mix/"
        #self.mixtures_path = self.path + "s1/"
        self.sources1_path  = self.path + "s1/"
        self.sources2_path  = self.path + "s2/"
        # self.mixtures je vektor, kde jsou ulozeny nazvy vsech audio nahravek urcenych k uceni site.
        self.mixtures = [mix for mix in listdir(self.mixtures_path) if isfile(join(self.mixtures_path, mix))]
        self.sources1 = [s1 for s1 in listdir(self.sources1_path) if isfile(join(self.sources1_path, s1))]
        self.sources2 = [s2 for s2 in listdir(self.sources2_path) if isfile(join(self.sources2_path, s2))]

    def __len__(self):
        """
        Vraci pocet celkovy dat, ktere jsou zpracovavane
        """
        return len(self.mixtures)


    def __getitem__(self, index):
        """
        v2: transformovane a nachystane audio, ale pouze jeden segment v podobe tensoru
        """
        mixture = self.getAudioSamples(self.mixtures_path + self.mixtures[index])
        source1 = self.getAudioSamples(self.sources1_path + self.sources1[index])
        source2 = self.getAudioSamples(self.sources2_path + self.sources2[index])
        mixture.unsqueeze_(0)
        source1.unsqueeze_(0)
        source2.unsqueeze_(0)
        return mixture, source1, source2


    def getAudioSamples(self, audio_file_path):
        """
        Precte a vrati vsechny vzorky zadaneho audio souboru
        """
        rate, samples = wav.read(audio_file_path)
        return self.prepare(samples)

    def prepare(self, samples):
        """
        Funkce prevede vstupni vzorky na numpy array a nasledne na tensor.
        """
        # normalisation - zero mean & jednotkova variance (unit variation)
        numpy = np.array(samples)
        #normalizace
        numpy = numpy / 2**15
        tensor = torch.as_tensor(numpy)
        tensor_float32 = torch.tensor(tensor, dtype=torch.float32)
        return tensor_float32

####################################################

# Vytvoreni instance neuronove site
autoencoderNN = Net()

train_data_path = "/home/valgrut/Documents/full/min/tr/"
test_data_path  = "/home/valgrut/Documents/full/min/tt/"
valid_data_path = "/home/valgrut/Documents/full/min/cv/"

trainset = AudioDataset(train_data_path) 
testset  = AudioDataset(test_data_path) 
validset = AudioDataset(valid_data_path) 

trainloader = data_utils.DataLoader(trainset, batch_size = MINIBATCH_SIZE, shuffle=False) 
testloader  = data_utils.DataLoader(testset, batch_size = MINIBATCH_SIZE, shuffle=False) 
validloader = data_utils.DataLoader(validset, batch_size = MINIBATCH_SIZE, shuffle=False) 

### Criterion and optimizer ###
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoderNN.parameters(), lr = opt_lr)

best_validation_result = 42
action = ""
while action not in ["quit", "q"]:
    action = input("Choose an action (train/tr, test/te, quit/q): ")
    
    if action in ["train", "tr"]:
        graph_x = []
        graph_y = []
        global_audio_cnt = 0
        for epoch in range(epochs):
            running_loss = 0.0
            for audio_cnt, data in enumerate(trainloader, 0):
                if audio_cnt > audios_in_epoch:
                    break #TODO pak oddelat
                
                global_audio_cnt += 1
                #if audio_cnt % 100 == 0:
                print(epoch, audio_cnt)

                input_mixture  = data[0]
                target_source1 = data[1]
                target_source2 = data[2]

                optimizer.zero_grad()
                separated_sources = autoencoderNN(input_mixture)

                #print(outputs.shape, target.shape) 
                #if target.shape[2] != outputs.shape[2]:
                #    target = target.narrow(2, 0, outputs.shape[2])

                smallest = min(input_mixture.shape[2], target_source1.shape[2], target_source2.shape[2], separated_sources.shape[2])
                input_mixture = input_mixture.narrow(2, 0, smallest)
                target_source1 = target_source1.narrow(2, 0, smallest)
                target_source2 = target_source2.narrow(2, 0, smallest)
                separated_sources = separated_sources.narrow(2, 0, smallest)

                # spojeni sources do jedne matice
                target_sources = torch.cat((target_source1, target_source2), 1)

                loss = criterion(separated_sources, target_sources)
                loss.backward()
                optimizer.step()

                # average
                running_loss += loss.item()
                if audio_cnt % print_frequency == print_frequency-1:
                    print('[%d, %5d] loss: %.5f' % (epoch, audio_cnt, running_loss/print_frequency))
                    #graph_x.append(epoch) #TODO
                    graph_x.append(print_frequency)
                    graph_y.append(running_loss/print_frequency)
                    running_loss = 0.0
                        
                # ulozeni pouze prvni nahravky pro porovnani epoch
                #if audio_cnt == 0: 
                if audio_cnt % 10 == 0: 
                    mixture_prep = input_mixture.detach().numpy()
                    source1_prep = separated_sources[0][0].detach().numpy()
                    source2_prep = separated_sources[0][1].detach().numpy()
                    wav.write("/home/valgrut/Documents/reconstruction/speech_e"+str(epoch)+"_a"+str(audio_cnt)+"_s1.wav", 8000, source1_prep)
                    wav.write("/home/valgrut/Documents/reconstruction/speech_e"+str(epoch)+"_a"+str(audio_cnt)+"_s2.wav", 8000, source2_prep)
                    wav.write("/home/valgrut/Documents/reconstruction/speech_e"+str(epoch)+"_a"+str(audio_cnt)+"_mix.wav", 8000, mixture_prep)

            # === validation na konci epochy ===
            print("")
            print("Validace")
            valid_audio_cnt = 0
            running_loss = 0.0
            current_validation_result = 0

            for audio_cnt, data in enumerate(validloader, 0):
                if valid_audio_cnt > 500:
                    break #TODO pak oddelat
                valid_audio_cnt += 1

                input_mixture  = data[0]
                target_source1 = data[1]
                target_source2 = data[2]

                optimizer.zero_grad()
                separated_sources = autoencoderNN(input_mixture)

                smallest = min(input_mixture.shape[2], target_source1.shape[2], target_source2.shape[2], separated_sources.shape[2])
                input_mixture = input_mixture.narrow(2, 0, smallest)
                target_source1 = target_source1.narrow(2, 0, smallest)
                target_source2 = target_source2.narrow(2, 0, smallest)
                separated_sources = separated_sources.narrow(2, 0, smallest)

                # spojeni sources do jedne matice
                target_sources = torch.cat((target_source1, target_source2), 1)

                loss = criterion(separated_sources, target_sources)

                current_validation_result += loss.item()
                running_loss += loss.item()
                if audio_cnt % print_frequency == print_frequency-1:
                    print('[%5d] loss: %.4f' % (audio_cnt+1, running_loss/print_frequency))
                    running_loss = 0.0
            

            # vyhodnoceni validace
            current_validation_result /= valid_audio_cnt # prumer 
            print(current_validation_result, " ", best_validation_result)
            if current_validation_result >= best_validation_result:
                opt_lr /= 2
            else:
                best_validation_result = current_validation_result
            print('Finished Validating')
            print('')


        print('Finished Training')

        plt.plot(graph_x, graph_y)        
        plt.show()
    

    elif action in ["test","te"]:
        global_audio_cnt = 0
        #running_loss = 0.0




        for audio_cnt, source1 in enumerate(testloader, 0):
            global_audio_cnt += 1
            inputs = source1
            target = inputs.clone()

            optimizer.zero_grad()
            outputs = autoencoderNN(inputs)

            if target.shape[2] != outputs.shape[2]:
                target = target.narrow(2, 0, outputs.shape[2])
                #print("Reshaped:", outputs.shape, target.shape) 

            loss = criterion(outputs, target)

            running_loss += loss.item()
            if audio_cnt % print_frequency == print_frequency-1:
                print('[%5d] loss: %.4f' % (audio_cnt+1, running_loss/print_frequency))
                running_loss = 0.0
                    
            speech_prep = outputs.detach().numpy()
            wav.write("/home/valgrut/Documents/testdata_recon/speech_a"+str(audio_cnt)+".wav", 8000, speech_prep)
     
        print('Finished Testing')


#TODO ulozeni site a vah, na zacatku moznost nacist naucene vahy
print("quit")

exit(0)

