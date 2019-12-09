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
    s_target = (torch.dot(output, target)*target) / ((torch.dot(target, target))**2)
    e_noise = output - s_target
    loss = 10*torch.log10((torch.dot(s_target, s_target)**2)/(torch.dot(e_noise, e_noise)**2))
    return loss

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
