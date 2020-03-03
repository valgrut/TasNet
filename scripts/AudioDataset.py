import torch
import torch.utils.data as data_utils
import sys
from os import listdir
from os.path import isfile, join

from tools import *


class AudioDataset(data_utils.Dataset):
    """
    Dataset of speech mixtures for speech separation.
    """
    def __init__(self, path, transform=None, DEBUG=False):
        super(AudioDataset, self).__init__()
        self.path = path
        self.DEBUG=DEBUG
        self.mixtures_path = self.path + "mix/"
        self.sources1_path  = self.path + "s1/"
        self.sources2_path  = self.path + "s2/"

        self.mixtures = []
        self.sources1 = []
        self.sources2 = []

        # self.mixtures je vektor, kde jsou ulozeny nazvy vsech audio nahravek urcenych k uceni site.
        self.mixtures = [mix for mix in listdir(self.mixtures_path) if isfile(join(self.mixtures_path, mix))]
        self.sources1 = [s1 for s1 in listdir(self.sources1_path) if isfile(join(self.sources1_path, s1))]
        self.sources2 = [s2 for s2 in listdir(self.sources2_path) if isfile(join(self.sources2_path, s2))]

        # ono by vlastne stacilo rozkopirovat unique mixture do zbylych dvou sources1 a sources2 misto tech zbylych rozdilu.
        # V obou totiz ma byt to same a ve stejnem poctu.

        # REMOVE DUPLICATES
        if self.DEBUG:
            print("audiodataset mixture size: ", len(self.mixtures))
            print("audiodataset sources1 size: ", len(self.sources1))
            print("audiodataset sources1 size: ", len(self.sources2))
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

        if self.DEBUG:
            print("audiodataset mixture size: ", len(self.mixtures))
            print("audiodataset sources1 size: ", len(self.sources1))
            print("audiodataset sources1 size: ", len(self.sources2))

    def __len__(self):
        """
        Vraci celkovy pocet dat, ktere jsou zpracovavane
        """
        return len(self.mixtures)

    def __getitem__(self, index):
        """
        v2: transformovane a nachystane audio, ale pouze jeden segment v podobe tensoru
        """
        if self.DEBUG:
            print("getItem: index:",index, " path: ", self.mixtures_path + self.mixtures[index])
            print("getItem: index:",index, " path: ", self.sources1_path + self.sources1[index])
            print("getItem: index:",index, " path: ", self.sources2_path + self.sources2[index])
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
        # tensor_float32 = torch.tensor(tensor, dtype=torch.float32)
        tensor_float32 = tensor.clone().detach().requires_grad_(True)
        tensor_float32 = tensor_float32.type(torch.float32)
        return tensor_float32

