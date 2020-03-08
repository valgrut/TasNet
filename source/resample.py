import librosa
from scipy.io import wavfile as wav
import argparse
import numpy as np

def transform(samples):
    """
    normalisation - zero mean & jednotkova variance (unit variation)
    """
    numpy = np.array(samples)
    numpy = numpy / 2**15
    # tensor = torch.as_tensor(numpy)
    # tensor_float32 = torch.tensor(tensor, dtype=torch.float32)
    # tensor_float32 = tensor.clone().detach().requires_grad_(True)
    # tensor_float32 = tensor_float32.type(torch.float32)
    # return tensor_float32
    return numpy


parser = argparse.ArgumentParser(description='Setup and init neural network')

parser.add_argument('--mixture',
    dest='input_mixture',
    type=str,
    help='number of epochs for training')

args = parser.parse_args()

mixture = args.input_mixture
print(mixture)

# Downsample to 8kHZ
N, SR = librosa.load(mixture, sr=8000)
# resampled_signal = scipy.signal.resample(mixture, 8000)

print(N)
print(SR)

# transformed_mixture = transform(N)
# print(transformed_mixture)

# Save as wav
wav.write(args.input_mixture+"-resampled.wav", 8000, N)



