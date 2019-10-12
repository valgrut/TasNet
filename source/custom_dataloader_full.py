from scipy.io import wavfile as wav
import numpy as np
import torch
import torch.utils.data as data_utils
from os import listdir
from os.path import isfile, join

"""
AudioDataset
"""
class AudioDataset(data_utils.Dataset):
    """
    Dataset of speech mixtures for speech separation. 
    """
    def __init__(self, path):
        super(AudioDataset, self).__init__()
        self.path = path
        self.mixtures_path = self.path + "mix/"
        self.source1_path  = self.path + "s1/"
        self.source2_path  = self.path + "s2/"
        self.mixtures = [f for f in listdir(self.mixtures_path) if isfile(join(self.mixtures_path, f))]
        
    def __len__(self):
        return len(self.mixtures)


    def __getitem__(self, index):
        """
        funkce vrati:
        v1: transformovane a nachystane audio v podobe tensoru
        v2: transformovane a nachystane audio, ale pouze jeden segment v podobe tensoru
        """
        #mixture, s1, s2 = getAudio(self.mixtures[index])
        #item = getSegment(self.mixtures[index]) # tato bude, misto return, mit yield
        inputmix  = self.getMixture(self.mixtures[index])
        outputmix = self.getMixture(self.mixtures[index])
        return inputmix[0:30000], outputmix[0:30000] #TODO zde bude output jako inputmix, s1, s2

    def getAudioSamples(self, audio_file_path):
        """
        Vrati vzorky zadaneho audio souboru
        """
        rate, samples = wav.read(audio_file_path)
        print("samples: " + str(len(samples)))
        return samples 

    def getMixture(self, audio_file_path):
        mixture = self.getAudioSamples(self.mixtures_path + audio_file_path)
        return self.transform(mixture)
        
    def transform(self, samples):
        # normalisation - zero mean & jednotkova variance (unit variation)
        # to np array / tensor
        numpy = np.array(samples)
        print("transform")
        print(numpy.shape)
        tensor = torch.from_numpy(numpy)
        print(tensor.shape)
        return tensor


# --------------- testing of our custom dataloader and dataset ----------------------------
train_data_path = "/home/valgrut/Documents/full/min/tr/"
trainset = AudioDataset(train_data_path)
print(len(trainset)) 
print(trainset[6])

dataloader = data_utils.DataLoader(trainset, batch_size = 3, shuffle=False)
iterator = iter(dataloader)
inmix, outmix = iterator.next()
print(inmix)

