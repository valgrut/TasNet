from scipy.io import wavfile as wav
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch._six import int_classes as _int_classes
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
        #print(">> __init__ ", path)
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
        self.dataset_len = len(self.mixtures)
        #print("mixtures dataset len: ", self.dataset_len)
        #print("sources1 dataset len: ", len(self.sources1))
        #print("sources2 dataset len: ", len(self.sources2))

        # make list unique
        smixtures = set(self.mixtures)
        ssources1 = set(self.sources1)
        ssources2 = set(self.sources2)

        # #print("mixtures dataset len: ", len(smixtures))
        # #print("sources1 dataset len: ", len(ssources1))
        # #print("sources2 dataset len: ", len(ssources2))

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
        #print("mixtures dataset len: ", self.dataset_len)
        #print("sources1 dataset len: ", len(self.sources1))
        #print("sources2 dataset len: ", len(self.sources2))

        # instantiate generator of segments
        self.generator = self.segment_generator()

        self.current_mixture = ""
        self.audioindex = 0
        self.loadNextAudio()

    def __len__(self):
        #print(">> __len__")
        if(self.audioindex < self.dataset_len):
            #print("__len__:", self.audioindex, "<", self.dataset_len)
            return len(self.current_mixture)
        else:
            #print("__len__:", self.audioindex, ">=", self.dataset_len)
            return self.dataset_len


    def __getitem__(self, index):
        """
        funkce vrati:
        v1: transformovane a nachystane audio v podobe tensoru
        """
        #print(">> __getitem__")
        mix_seg, s1_seg, s2_seg = self.getSegment() # tato bude, misto return, mit yield
        mix_seg.unsqueeze_(0)
        s1_seg.unsqueeze_(0)
        s2_seg.unsqueeze_(0)
        return mix_seg, s1_seg, s2_seg


    def loadNextAudio(self):
        # #print("")
        #print(">> loadNextAudio")
        #print("LoadNextAudio: audioindex of current mixture: ", self.audioindex)
        #print("LoadNextAudio: number of mixtures: ", len(self.mixtures))

        if self.audioindex >= len(self.mixtures):
            #print("POZOR: audioindex >= len(self.mixtures), iterace by mela skoncit")
            return None #raises StopIteration exception
        else: #jeste je co prochazet
            self.generator = self.segment_generator()


        #print("loading new audio samples for mix, s1, s2 on index: ", self.audioindex)
        self.current_mixture = self.transform(self.getAudioSamples(self.mixtures_path + self.mixtures[self.audioindex]))
        self.current_source1 = self.transform(self.getAudioSamples(self.sources1_path + self.sources1[self.audioindex]))
        self.current_source2 = self.transform(self.getAudioSamples(self.sources2_path + self.sources2[self.audioindex]))
        self.current_mixture_len = len(self.current_mixture)
        #print("New audio len: ", self.current_mixture_len)

        # #print("new mixture: ", self.mixtures_path + self.mixtures[self.audioindex])
        # #print("new source1: ", self.sources1_path + self.sources1[self.audioindex])
        # #print("new source2: ", self.sources2_path + self.sources2[self.audioindex])

        # takze ja to nactu, abych to mohl zpracovat po segmentech, ale
        # zde se to ukonci, protoze inkrementuju audioindex a tim zacne platit podminka
        # zabranujici, abych nacetl z pole dalsi nahravku, kdyz by byl idnex mimo.
        # Dokud mimo neni, tak nastavi novej generator a

        self.audioindex += 1
        # if self.audioindex >= len(self.mixtures):
        #     #print("POZOR: audioindex >= len(self.mixtures), iterace by mela skoncit")
        #     return None #raises StopIteration exception
        # else: #jeste je co prochazet
        #     self.generator = self.segment_generator()


    def getAudioSamples(self, audio_file_path):
        """
        Vrati vzorky zadaneho audio souboru
        """
        # #print(">> getAudioSamples (Read .wav file)")
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


    def getSegment(self):
        """
        get next segment using segment generator
        """
        #print(">> getSegment")
        try:
            mix_seg, s1_seg, s2_seg = next(self.generator)
            return mix_seg, s1_seg, s2_seg
        except StopIteration:
            #init for next epoch
            #print("segmentDataset - segmentGenerator Prepare Next Epoch!")
            self.audioindex = 0
            self.loadNextAudio()
            self.generator = self.segment_generator()
            raise StopIteration


    def segment_generator(self):
        """
        """
        #print(">> segment_generator")
        mix_segment = []
        s1_segment = []
        s2_segment = []

        segptr = 0
        new_required = False
        while(not new_required):
            #print("seg_gen: curr_mix_len: ", self.current_mixture_len)
            #print("seg_gen: segptr: ", segptr)

            # nahravka je kratsi nez 4 sekundy (<32k) - nelze vzit 4s od konce.
            if(self.current_mixture_len < self.SEGMENT_LEN):
                #print("Nahravka je kratsi nez ", self.SEGMENT_LEN, ", takze doplnime nulama a vemem dalsi.")
                mix_segment = self.current_mixture[(self.current_mixture_len - self.SEGMENT_LEN):self.current_mixture_len]
                s1_segment = self.current_source1[(self.current_mixture_len - self.SEGMENT_LEN):self.current_mixture_len]
                s2_segment = self.current_source2[(self.current_mixture_len - self.SEGMENT_LEN):self.current_mixture_len]
                self.loadNextAudio() # uz neni co nacitat
                new_required = True
                yield mix_segment, s1_segment, s2_segment

            # jsou li dalsi 4 sekundy k dispozici, nebo je potreba je vzit od konce.
            else:
                # bereme dalsi 4 sekundy
                if(segptr + self.SEGMENT_LEN < self.current_mixture_len):
                    #print("if Dalsi 4s k dispozici: ", (segptr+self.SEGMENT_LEN), "<", self.current_mixture_len)
                    mix_segment = self.current_mixture[segptr:(segptr+self.SEGMENT_LEN)]
                    s1_segment = self.current_source1[segptr:(segptr+self.SEGMENT_LEN)]
                    s2_segment = self.current_source2[segptr:(segptr+self.SEGMENT_LEN)]
                    segptr += self.SEGMENT_LEN
                    yield mix_segment, s1_segment, s2_segment

                # segptr + self.SEGMENT_LEN >= self.current_mixture_len
                else:
                    #print("else presahli bychom konec, takze vezmeme 4s od konce.")
                    mix_segment = self.current_mixture[(self.current_mixture_len - self.SEGMENT_LEN):self.current_mixture_len]
                    s1_segment = self.current_source1[(self.current_mixture_len - self.SEGMENT_LEN):self.current_mixture_len]
                    s2_segment = self.current_source2[(self.current_mixture_len - self.SEGMENT_LEN):self.current_mixture_len]
                    self.loadNextAudio() # uz neni co nacitat
                    new_required = True
                    yield mix_segment, s1_segment, s2_segment

