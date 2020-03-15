from scipy.io import wavfile as wav
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch._six import int_classes as _int_classes
import random
from os import listdir
from os.path import isfile, join

"""
SegmentDataset
"""
class SegmentDataset(data_utils.Dataset):
    """
    Dataset of speech mixtures for speech separation.
    """
    def __init__(self, path):
        # print(">> __init__ ", path)
        super(SegmentDataset, self).__init__()
        self.SEGMENT_LEN = 32000 #4seconds, 32k samples

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
        self.audioindex = 0

        self.prepareNewEpoch()


    def __len__(self):
        """
        """
        # print(">> __len__")
        if(self.audioindex < self.dataset_len):
            if(self.isAudioPrepared == False):
                self.loadNextAudio()
                self.isAudioPrepared = True
                self.generator = self.segment_generator()
            # print("__len__:", self.audioindex, "<", self.dataset_len, " Returns: ", len(self.current_mixture))
            return len(self.current_mixture)
        else:
            # print("POZOR, tohle tu normalne neni: __len__:", self.audioindex, ">=", self.dataset_len, " Returns: ", self.dataset_len)
            return self.dataset_len


    def __getitem__(self, index):
        """
        """
        # print(">> __getitem__")
        try:
            if(self.isAudioPrepared == False):
                # print("    __getitem__(): Audio NOT Prepared.")
                self.isAudioPrepared = True
                self.loadNextAudio()
                self.generator = self.segment_generator()
        except StopIteration:
            # print("    __getitem__(): NELZE NACHYSTAT DALSI AUDIO: StopIteration raised")
            raise StopIteration

        try:
            mix_seg, s1_seg, s2_seg = next(self.generator)
            # print("    __getitem__(): return segments of mix, s1, s2")
            mix_seg.unsqueeze_(0)
            s1_seg.unsqueeze_(0)
            s2_seg.unsqueeze_(0)
            return mix_seg, s1_seg, s2_seg
        except StopIteration:
            # print("    __getitem__(): cant return segments: StopIteration raised")
            raise StopIteration


    def prepareNewEpoch(self):
        """
        """
        # print("Prepare new Epoch")
        self.current_mixture = None
        self.current_source1 = None
        self.current_source2 = None
        self.current_mixture_len = None

        self.isAudioPrepared = False
        self.audioindex = 0

        # #print("Shuffle of mix, s1, s2 array")
        random.shuffle(self.mixtures)
        self.sources1 = self.mixtures
        self.sources2 = self.mixtures


    def segment_generator(self):
        """
        """
        # print(">> segment_generator - NEW")
        mix_segment = []
        s1_segment = []
        s2_segment = []

        segptr = 0
        new_required = False
        # while(not new_required):
        while(self.isAudioPrepared):
            # print("    seg_gen: curr_mix_len: ", self.current_mixture_len)
            # #print("    seg_gen: segptr: ", segptr)

            # nahravka je kratsi nez 4 sekundy (<32k) - nelze vzit 4s od konce.
            if(self.current_mixture_len < self.SEGMENT_LEN):
                # print("    SG: 1.) Nahravka je kratsi nez ", self.SEGMENT_LEN, " (",self.current_mixture_len,"), takze doplnime nulama a vemem dalsi.")
                mix_segment = self.current_mixture[:]
                s1_segment = self.current_source1[:]
                s2_segment = self.current_source2[:]
                # self.loadNextAudio() # uz neni co nacitat

                new_required = True
                self.isAudioPrepared = False
                yield mix_segment, s1_segment, s2_segment

            # jsou li dalsi 4 sekundy k dispozici, nebo je potreba je vzit od konce.
            else:
                # bereme dalsi 4 sekundy
                if(segptr + self.SEGMENT_LEN < self.current_mixture_len):
                    # print("    SG: 2a) (Dalsi) 4s k dispozici: ", (segptr+self.SEGMENT_LEN), "<", self.current_mixture_len)
                    mix_segment = self.current_mixture[segptr:(segptr+self.SEGMENT_LEN)]
                    s1_segment = self.current_source1[segptr:(segptr+self.SEGMENT_LEN)]
                    s2_segment = self.current_source2[segptr:(segptr+self.SEGMENT_LEN)]
                    segptr += self.SEGMENT_LEN
                    self.isAudioPrepared = True
                    yield mix_segment, s1_segment, s2_segment

                # segptr + self.SEGMENT_LEN >= self.current_mixture_len
                else:
                    # print("    SG: 2b) Presahli bychom konec, takze vezmeme 4s od konce a nahrajem dalsi.")
                    mix_segment = self.current_mixture[(self.current_mixture_len - self.SEGMENT_LEN):self.current_mixture_len]
                    s1_segment = self.current_source1[(self.current_mixture_len - self.SEGMENT_LEN):self.current_mixture_len]
                    s2_segment = self.current_source2[(self.current_mixture_len - self.SEGMENT_LEN):self.current_mixture_len]
                    # self.loadNextAudio() # uz neni co nacitat
                    self.isAudioPrepared = False
                    new_required = True
                    yield mix_segment, s1_segment, s2_segment

        # print("#### after while")
        segptr = 0
        new_required = False



    def loadNextAudio(self):
        # print(">> LoadNextAudio: Loaded m,s1,s2 on audioindex: ", self.audioindex, "/", len(self.mixtures))

        # check whether some audio mixtures are available to load
        # All mixtures used.
        if self.audioindex >= self.dataset_len:
            print("    loadNextAudio: POZOR: audioindex >= len(self.mixtures) ", self.audioindex, "/", len(self.mixtures) ,", iterace by mela skoncit a nachystat se nove epocha. (Return None that will rise StopIteration exception.)")
            self.prepareNewEpoch()
            return None #raises StopIteration exception

        # Mixtures are still available so we will load next one.
        else:
            # print("    loadNextAudio: Inicializujeme novy generator pro segmentaci dalsi nahravky.")
            # self.isAudioPrepared = True
            # self.generator = self.segment_generator()

            # print("    loadNextAudio: loading new audio samples for mix, s1, s2 on index: ", self.audioindex)
            self.current_mixture = self.transform(self.getAudioSamples(self.mixtures_path + self.mixtures[self.audioindex]))
            self.current_source1 = self.transform(self.getAudioSamples(self.sources1_path + self.sources1[self.audioindex]))
            self.current_source2 = self.transform(self.getAudioSamples(self.sources2_path + self.sources2[self.audioindex]))
            self.current_mixture_len = len(self.current_mixture)

            ##print("New audio len: ", self.current_mixture_len)
            # print("    This audio will produce ", (int(self.current_mixture_len/self.SEGMENT_LEN)+1), " segments from length: ", self.current_mixture_len)

            # Check that mix,s1,s2 are loaded in corresponding order.
            if((self.mixtures[self.audioindex]) !=
                (self.sources1[self.audioindex]) !=
                (self.sources2[self.audioindex])):
                #print("new mixture: ", self.mixtures_path + self.mixtures[self.audioindex])
                #print("new source1: ", self.sources1_path + self.sources1[self.audioindex])
                #print("new source2: ", self.sources2_path + self.sources2[self.audioindex])
                raise NameError

            self.audioindex += 1



    def getAudioSamples(self, audio_file_path):
        """
        Vrati vzorky zadaneho audio souboru
        """
        # ##print(">> getAudioSamples (Read .wav file)")
        rate, samples = wav.read(audio_file_path)
        return samples


    def transform(self, samples):
        """
        normalisation - zero mean & jednotkova variance (unit variation)
        """
        numpy = np.array(samples)
        numpy = numpy / 2**15
        tensor = torch.as_tensor(numpy)
        # tensor_float32 = torch.tensor(tensor, dtype=torch.float32)
        tensor_float32 = tensor.clone().detach().requires_grad_(True)
        tensor_float32 = tensor_float32.type(torch.float32)
        return tensor_float32

