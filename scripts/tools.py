from scipy.io import wavfile as wav
import numpy as np
import torch
import scipy.fftpack
from scipy.linalg import toeplitz
from scipy.signal import fftconvolve
import collections
import itertools
import warnings

def siSNRloss(output, target):
    # check dimensions of output and target, should be 1xT
    s_target = (torch.dot(output, target)*target) / ((torch.dot(target, target)))
    e_noise = output - s_target
    loss = 10*torch.log10((torch.dot(s_target, s_target))/(torch.dot(e_noise, e_noise)))
    return loss


def audio_collate(batch):
    print("collate")
    print(batch) #list listuu s tenzory mix,s1,s2
    print("")
    list_mix = []
    list_s1 = []
    list_s2 = []
    for audio in batch:
        print("Audio mix: ", audio[0].shape) #tensor mix
        print("Audio s1 : ", audio[1].shape) #tensor s1
        print("Audio s2 : ", audio[2].shape) #tensor s2

        list_mix.append(audio[0][0])
        list_s1.append(audio[1][0])
        list_s2.append(audio[2][0])

    minibatch_mix = torch.nn.utils.rnn.pad_sequence(list_mix, batch_first=True)
    minibatch_s1 = torch.nn.utils.rnn.pad_sequence(list_s1, batch_first=True)
    minibatch_s2 = torch.nn.utils.rnn.pad_sequence(list_s2, batch_first=True)

    print("minibatch_mix: ", minibatch_mix)
    print("minibatch_mix unsqueezed: ", minibatch_mix.unsqueeze(1))

    print("konec_collate")
    return minibatch_mix.unsqueeze(1), minibatch_s1.unsqueeze(1), minibatch_s2.unsqueeze(1)


def train_collate(batch):
    list_mix = []
    list_s1 = []
    list_s2 = []

    if(len(batch) > 1):
        for audio in batch:
            list_mix.append(audio[0][0])  # pripadne bez te posledni [0], pokud bych oddelal squ      eeze v __get_item__()
            list_s1.append(audio[1][0])
            list_s2.append(audio[2][0])

        padded_mix = torch.nn.utils.rnn.pad_sequence(list_mix, batch_first=True)
        padded_s1 = torch.nn.utils.rnn.pad_sequence(list_s1, batch_first=True)
        padded_s2 = torch.nn.utils.rnn.pad_sequence(list_s2, batch_first=True)
    else:
        print("Doplneno nul: ", 32000 - len(batch[0]))
        zero = torch.zeros(32000 - len(batch[0]))
        minibatch_mix = torch.cat((batch[0], zero), 0)

    return padded_mix, padded_s1, padded_s2
    # return padded_mix.unsqueeze_(0), padded_s1.unsqueeze_(0), padded_s2.unsqueeze_(0)


def saveAudio():
    print("saveAudio")

def normalizeAudio(samples):
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

def getAudioSamples(audio_file_path):
    rate, samples = wav.read(audio_file_path)
    return normalizeAudio(samples)
