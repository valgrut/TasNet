from scipy.io import wavfile as wav
import numpy as np
import torch
import torch.utils.data as data_utils
import torchvision
import torchvision.transforms as transforms
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
        # ZDE JSOU nazvy souboruu

    def __len__(self):
        return len(self.mixtures)


    def __getitem__(self, index):
        #segment = self.getSegment(self.mixtures[index])
        print("__getitem__")
        print(self.mixtures[index])
        samples = self.getAudioSamples(self.mixtures[index])
        print(len(samples))
        return self.transform(samples)

    def getAudioSamples(self, audio_file_path):
        """
        Vrati vzorky zadaneho audio souboru
        """
        rate, samples = wav.read(self.mixtures_path + audio_file_path)
        print("get audio samples")
        print(samples[0:7])
        return samples

    def transform(self, samples):
        # normalisation - zero mean & jednotkova variance (unit variation)
        numpy = np.array(samples)
        tensor = torch.from_numpy(numpy)
        return tensor

# ----------------------------------------------------------------------------------
class BatchSampler(data_utils.Sampler):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler, batch_size, drop_last):
        if not isinstance(sampler, data_utils.Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size should be a positive integeral value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


# --------------- testing of our custom dataloader and dataset ----------------------------
train_data_path = "/home/valgrut/Documents/full/min/tr/"
trainset = AudioDataset(train_data_path)
print(len(trainset))
print(trainset[0])
print("...")

minibatchsampler = BatchSampler(data_utils.SequentialSampler(range(10)), batch_size=20, drop_last=True)
dataloader = data_utils.DataLoader(trainset, batch_size = 3, shuffle=False, sampler=minibatchsampler)
iterator = iter(dataloader)
print("Cyklus:")
#for i in range (1, 10):
minibatch = iterator.next()
#print(minibatch)




