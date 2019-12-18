from scipy.io import wavfile as wav
import numpy as np
import torch
import torch.utils.data as data_utils
from torch._six import int_classes as _int_classes
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
        self.generator = self.generate_segment()

        self.current_mixture = ""
        self.audioindex = 0
        self.loadNextAudio()

    def __len__(self):
        return len(self.current_mixture)


    def __getitem__(self, index):
        """
        funkce vrati:
        v1: transformovane a nachystane audio v podobe tensoru
        v2: transformovane a nachystane audio, ale pouze jeden segment v podobe tensoru
        """
        #print("f: __getitem__")
        segment = self.getSegment(self.current_mixture) # tato bude, misto return, mit yield
        return segment

    def loadNextAudio(self):
        #print("f: load_next")
        self.current_mixture = self.getAudioSamples(self.mixtures_path + self.mixtures[self.audioindex])
        self.audioindex += 1
        if self.audioindex > len(self.mixtures):
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

    def transform(self, samples):
        # normalisation - zero mean & jednotkova variance (unit variation)
        numpy = np.array(samples)
        tensor = torch.from_numpy(numpy)
        return tensor

    def getSegment(self, path):
        #print("f: get_segment")
        next_segment = next(self.generator)
        return self.transform(next_segment)
    
    def generate_segment(self):
        #print("f: generate_segments")
        samples = self.current_mixture
        segment = []
        
        for i in range(0, len(samples), 20):
            #print("index i: " + str(i))
            segment = [samples[i] for i in range(i, i+20)]
            yield segment
        
        self.load_next() # uz neni co nacitat
        


# --------------- testing of our custom dataloader and dataset ----------------------------
train_data_path = "/home/valgrut/Documents/full/min/tr/"
trainset = AudioDataset(train_data_path)
print(len(trainset)) 

dataloader = data_utils.DataLoader(trainset, batch_size = 3, shuffle=False)
iterator = iter(dataloader)
print("Cyklus:")
#for i in range (1, 10):
minibatch = iterator.next()
print(minibatch)



