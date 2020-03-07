from scipy.io import wavfile as wav
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torchvision.transforms as transforms
import IPython.display as ipd
import numpy as np
import torch
import torch.utils.data as data_utils
from torch._six import int_classes as _int_classes
from os import listdir
from os.path import isfile, join
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as fc

# #####################################################################
# Separation - base
#
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 256, 20)
        self.deconv = nn.ConvTranspose1d(256, 1, 20)

    def forward(self, input_data):
        input_data = self.conv1(input_data);
        input_data = fc.relu(input_data);

        input_data = self.deconv(input_data)
        return input_data

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
        self.source1_path  = self.path + "s1/"
        self.source2_path  = self.path + "s2/"
        self.mixtures = [f for f in listdir(self.mixtures_path) if isfile(join(self.mixtures_path, f))]
        self.transform = transform
        self.generator = self.generate_segment()

        self.current_mixture = ""
        self.audioindex = 0
        self.loadNextAudio()

    def __len__(self):
        return len(self.current_mixture)


    def __getitem__(self, index):
        """
        v2: transformovane a nachystane audio, ale pouze jeden segment v podobe tensoru
        """
        #print("f: __getitem__")
        segment = self.getSegment(self.current_mixture) # tato bude, misto return, mit yield
        return segment

    def loadNextAudio(self):
        #print("f: load_next : " + self.mixtures[self.audioindex])
        print(str(self.audioindex) + " <?= " + str(len(self.mixtures)))
        self.current_mixture = self.getAudioSamples(self.mixtures_path + self.mixtures[self.audioindex])
        #normalizace na 0 mean a unit variance
        #self.current_mixture = norm(self.current_mixture)
        self.audioindex += 1
        self.generator = self.generate_segment() # reinitialisation of generator
        #self.i = 0 # segment step counter
        if self.audioindex > len(self.mixtures):
            print(str(self.audioindex) + " > " + str(len(self.mixtures)))
            return

    def getAudioSamples(self, audio_file_path):
        """
        Vrati vzorky zadaneho audio souboru
        """
        rate, samples = wav.read(audio_file_path)
        #print("f: get_audio_samples")
        #print(samples[0:19])
        #print(samples[20:39])
        return samples 

    def prepare(self, samples):
        # normalisation - zero mean & jednotkova variance (unit variation)
        numpy = np.array(samples)
        tensor = torch.as_tensor(numpy)
        return torch.tensor(tensor, dtype=torch.float32)
    #jeste normalizace TODO !!!

    def getSegment(self, path):
        #print("f: get_segment")
        next_segment = next(self.generator)
        return self.prepare(next_segment)

    def generate_segment(self):
        #print("f: generate_segments")
        samples = self.current_mixture
        segment = []
    
        batch_size = 40

        for self.i in range(0, len(samples), batch_size):
            if self.i+batch_size > len(samples):
                break # kontrola, jestli i neni mimo pole
            #print("index i: " + str(self.i))
            segment = [samples[self.i] for self.i in range(self.i, self.i+batch_size)]
            yield [segment]

        self.loadNextAudio() # uz neni co nacitat

####################################################

def normalize(x):
    x_normed = x / x.max(0, keepdim=True)[0]
    return x_normed

def norm(input_data):
    # v1
    #if self.transform:
    #    self.current_mixture = self.transform(self.current_mixture)  
    
    #v2
    #input_data = (-1.0 - 1.0) * input_data + 1.0
    #return input_data
    
    #v3
    #tmp=np.array(input_data)
    #d=tmp/sum(input_data)
    #return d
    
    #v4
    #print(input_data)
    #s = sum(input_data)
    #normalized = [float(i)/s for i in input_data]
    #return normalized

    #v5
    return normalize(input_data)


#######################################################################

autoencoderNN = Net()
print(autoencoderNN)

#######################################################################
# --- testing ---
#input_data = torch.randn(10, 1, 20)
#output = autoencoderNN(input_data) 
#print(output)

#target = torch.randn(10, 1, 20)  # a dummy target, for example
#criterion = nn.MSELoss()
#loss = criterion(output, target)
#print(loss)

##########################################################################
#transform = transforms.Compose([
#    transforms.ToTensor(),
#    transforms.Normalize(mean=[0.456],
#                         std=[0.229])
#])

train_data_path = "/home/valgrut/Documents/full/min/tr/"
trainset = AudioDataset(train_data_path)
#print(len(trainset)) 

dataloader = data_utils.DataLoader(trainset, batch_size = 10, shuffle=False)
#####################################
#iterator = iter(dataloader)
#print("Cyklus:")
#for i in range (1, 10):
#minibatch = iterator.next()
#print(minibatch)

#########################################################################################
epochs_number = 1
audios_in_epoch = 100
opt_lr = 0.000001    # 0.000 001
opt_momentum = 0.3   # 0.9
print_frequency = 300
#########################################################################################
#criterion = nn.L1Loss()
#criterion = nn.KLDivLoss()

criterion = nn.MSELoss()
#criterion = nn.SmoothL1Loss()

optimizer = optim.SGD(autoencoderNN.parameters(), lr = opt_lr, momentum = opt_momentum)

graph_x = []
graph_y = []
counter = 0

debug = False

speech = []
for epoch_outer in range(epochs_number):
    for epoch in range(audios_in_epoch):  # loop over the dataset multiple times
        running_loss = 0.0
        for cnt, data in enumerate(dataloader, 0):
            #print("training " + str(data.shape))
            ### nacteni vstupu
            #print(str(cnt) + " epoch: " + str(epoch))
            
            inputs = data
            targets = data.clone()

            ### Prevod na float
            #inputs  = Variable(inputs.float())
            #targets = Variable(targets.float())
            #targets = targets.long()

            ### zero the parameter gradients
            optimizer.zero_grad()

            ### Normalizace vstupu a referencniho vystupu
            norm_inputs = norm(inputs)
            norm_targets = norm(targets)
            
            if debug:
                print("inputs")
                print(inputs.shape)
                print(inputs)
                print(norm_inputs.shape)
                print(norm_inputs)
 
            ### Prizeneme data siti
            outputs = autoencoderNN(norm_inputs)
            
            if debug:
                print("outputs")
                print(outputs)
                print(norm(outputs))

            
            #outputs = norm(outputs)

            lists = outputs.tolist()
            for elm in lists:
                speech += elm[0]

            #loss = criterion(norm_outputs, norm_targets)
            loss = criterion(outputs, norm_targets)
            #loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()

            # print statistics

            running_loss += loss.item()
            #if cnt % print_frequency == print_frequency-1:
            if cnt % 100 == 99:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, cnt + 1, running_loss / print_frequency))
                graph_x.append(counter)
                graph_y.append(running_loss/print_frequency)
                #print(str(running_loss) + " " + str(running_loss/print_frequency))
                counter += 1
                running_loss = 0.0

                #save output file
                speech_prep = np.array(speech)

                #print(speech_prep[0:15])
                wav.write("reconstruction/speech_"+str(epoch), 8000, speech_prep)
                speech = []


print('Finished Training')

plt.plot(graph_x, graph_y)        
plt.show()

exit(0)

