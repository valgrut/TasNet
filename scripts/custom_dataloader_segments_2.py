from scipy.io import wavfile as wav
import numpy as np
import torch
import torch.nn.functional as F
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
        self.dataset_len = len(self.mixtures)
        self.generator = self.segment_generator()

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
        # print("f: __getitem__")
        segment = self.getSegment() # tato bude, misto return, mit yield
        return segment

    def loadNextAudio(self):
        print("")
        # print("f: loadNextAudio")
        self.current_mixture = self.getAudioSamples(self.mixtures_path + self.mixtures[self.audioindex])
        self.current_mixture_len = len(self.current_mixture)
        print("New audio len: ", self.current_mixture_len)
        self.audioindex += 1
        self.generator = self.segment_generator()
        # print("1: ", self.transform(self.current_mixture[0]), ", 32k:", self.transform(self.current_mixture[31999]), ", 24k: ", self.transform(self.current_mixture[24000]))
        if self.audioindex >= len(self.mixtures):
            return
        # print("f: konec loadNextAudio")


    def getAudioSamples(self, audio_file_path):
        """
        Vrati vzorky zadaneho audio souboru
        """
        rate, samples = wav.read(audio_file_path)
        # print("f: get_audio_samples")
        return samples

    def transform(self, samples):
        # normalisation - zero mean & jednotkova variance (unit variation)
        numpy = np.array(samples)
        numpy = numpy / 2**15
        tensor = torch.as_tensor(numpy)
        tensor_float32 = torch.tensor(tensor, dtype=torch.float32)
        # tensor = torch.from_numpy(numpy)
        return tensor_float32

    def getSegment(self):
        # print("f: get_segment")
        next_segment = next(self.generator)
        return self.transform(next_segment)

    def segment_generator(self):
        # print("f: segment_generators")
        # print("current audio index: ", self.audioindex)
        print("Beru si dalsi segment")
        segment = []
        seglen = 32000 #4seconds, 32k samples

        segptr = 0
        while(segptr < self.current_mixture_len):
            segment = self.current_mixture[segptr:(segptr+seglen)]
            print("Delka segmentu: ", len(segment))
            segptr += 24000 #32000 - 8000 stride
            print("segment1: ", self.transform(segment[0:2]))
            if(len(segment) == seglen): #TODO overit
                print("yield")
                yield segment
            else:
                self.loadNextAudio() # uz neni co nacitat
                yield segment

        # if(self.current_mixture_len < segptr+seglen): #TODO overit
            # print("Je to delsi, takze dalsi nahravka")
            # self.loadNextAudio() # uz neni co nacitat

        # print("yield-outer")
        # yield segment
        ### Takhle je to skoro dobre, ale posledni segment se Yielduje dvakrat!!!

 # -----------------------------------------------------------------------------------------

# Budu muset predelat pro 3 nahravky - mix, s1, s2
def audio_collate(batch):
    if(len(batch) > 1):
        list_mix = []
        for audio in batch:
            list_mix.append(audio)

        minibatch_mix = torch.nn.utils.rnn.pad_sequence(list_mix, batch_first=True)
    else:
        print("Doplneno nul: ", 32000 - len(batch[0]))
        zero = torch.zeros(32000 - len(batch[0]))
        minibatch_mix = torch.cat((batch[0], zero), 0)

    return minibatch_mix


# --------------- testing of our custom dataloader and dataset ----------------------------
train_data_path = "/root/Documents/full/min/tr/"
trainset = AudioDataset(train_data_path)
print(len(trainset))

dataloader = data_utils.DataLoader(trainset, batch_size = 3, shuffle=False, collate_fn=audio_collate)
iterator = iter(dataloader)


# print("Cyklus:")
# for i in range (1, 10):
    # minibatch = iterator.next()
    # print(minibatch)

maxx = 10
for cnt, data in enumerate(dataloader, 0):
    if(cnt < maxx):
        print(data)
    else:
        break

# minibatch = iterator.next()
# minibatch = iterator.next()
# print(minibatch)
print("")
print("")

# minibatch = iterator.next()
# print(minibatch)
