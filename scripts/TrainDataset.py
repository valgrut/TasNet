from scipy.io import wavfile as wav
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch._six import int_classes as _int_classes
from os import listdir
from os.path import isfile, join

"""
TrainDataset
"""
class TrainDataset(data_utils.Dataset):
    """
    Dataset of speech mixtures for speech separation.
    """
    def __init__(self, path):
        super(TrainDataset, self).__init__()
        self.segment_len = 32000 #4seconds, 32k samples

        self.path = path

        self.mixtures_path = self.path + "mix/"
        self.sources1_path  = self.path + "s1/"
        self.sources2_path  = self.path + "s2/"

        self.mixtures = []
        self.sources1 = []
        self.sources2 = []

        self.mixtures = [f for f in listdir(self.mixtures_path) if isfile(join(self.mixtures_path, f))]
        self.sources1 = [f for f in listdir(self.sources1_path) if isfile(join(self.sources1_path, f))]
        self.sources2 = [f for f in listdir(self.sources2_path) if isfile(join(self.sources2_path, f))]

        # REMOVE DUPLICATES
        # make list unique
        smixtures = set(self.mixtures)
        ssources1 = set(self.sources1)
        ssources2 = set(self.sources2)

        ms1_duplicates = smixtures - ssources1
        ms2_duplicates = smixtures - ssources2
        self.mixtures = list((smixtures - ms1_duplicates) - ms2_duplicates)

        s1m_duplicates = ssources1 - smixtures
        s2m_duplicates = ssources2 - smixtures
        self.sources1 = list(((ssources1 - s1m_duplicates) - s2m_duplicates) - ms2_duplicates)
        self.sources2 = list(((ssources2 - s2m_duplicates) - s1m_duplicates) - ms1_duplicates)

        self.mixtures.sort()
        self.sources1.sort()
        self.sources2.sort()

        self.dataset_len = len(self.mixtures)
        # instantiate generator of segments
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
        """
        # print("f: __getitem__")
        mix_seg, s1_seg, s2_seg = self.getSegment() # tato bude, misto return, mit yield
        mix_seg.unsqueeze_(0)
        s1_seg.unsqueeze_(0)
        s2_seg.unsqueeze_(0)
        return mix_seg, s1_seg, s2_seg


    def loadNextAudio(self):
        # print("")
        # print("f: loadNextAudio")
        self.current_mixture = self.transform(self.getAudioSamples(self.mixtures_path + self.mixtures[self.audioindex]))
        self.current_source1 = self.transform(self.getAudioSamples(self.sources1_path + self.sources1[self.audioindex]))
        self.current_source2 = self.transform(self.getAudioSamples(self.sources2_path + self.sources2[self.audioindex]))
        self.current_mixture_len = len(self.current_mixture)
        # print("New audio len: ", self.current_mixture_len)
        self.audioindex += 1
        self.generator = self.segment_generator()
        if self.audioindex >= len(self.mixtures):
            return


    def getAudioSamples(self, audio_file_path):
        """
        Vrati vzorky zadaneho audio souboru
        """
        rate, samples = wav.read(audio_file_path)
        # print("f: get_audio_samples")
        return samples


    def transform(self, samples):
        """
        normalisation - zero mean & jednotkova variance (unit variation)
        """
        numpy = np.array(samples)
        numpy = numpy / 2**15
        tensor = torch.as_tensor(numpy)
        tensor_float32 = torch.tensor(tensor, dtype=torch.float32)
        return tensor_float32


    def getSegment(self):
        """
        get next segment using segment generator
        """
        mix_seg, s1_seg, s2_seg = next(self.generator)
        return mix_seg, s1_seg, s2_seg


    def segment_generator(self):
        mix_segment = []
        s1_segment = []
        s2_segment = []

        segptr = 0
        while(segptr < self.current_mixture_len):
            mix_segment = self.current_mixture[segptr:(segptr+self.segment_len)]
            s1_segment = self.current_source1[segptr:(segptr+self.segment_len)]
            s2_segment = self.current_source2[segptr:(segptr+self.segment_len)]
            segptr += 24000 #32000 - 8000 stride
            if(len(mix_segment) == self.segment_len):
                yield mix_segment, s1_segment, s2_segment
            else:
                self.loadNextAudio() # uz neni co nacitat
                yield mix_segment, s1_segment, s2_segment

