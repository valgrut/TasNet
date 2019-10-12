import os
import struct
from scipy.io import wavfile as wav
import matplotlib.pyplot as plt
import IPython.display as ipd

def wav_plotter(full_path, class_label):   
    rate, wav_sample = wav.read(full_path)
    wave_file = open(full_path,"rb")
    riff_fmt = wave_file.read(36)
    bit_depth_string = riff_fmt[-2:]
    bit_depth = struct.unpack("H",bit_depth_string)[0]
    print('sampling rate: ',rate,'Hz')
    print('bit depth: ',bit_depth)
    print('number of channels: ',wav_sample.shape[1])
    print('duration: ',wav_sample.shape[0]/rate,' second')
    print('number of samples: ',len(wav_sample))
    print('class: ',class_label)
    plt.figure(figsize=(12, 4))

    #plt.plot(wav_sample[0:100])  # z tohoto by slo lehce ziskat tech 20 vzorku jako input do NN
    plt.plot(wav_sample)
    
    plt.show()
    return [rate, wav_sample]

rate, samples = wav_plotter("data/strings.wav", "kill")

print(samples[10100])    # vypis signalu na konkretnim vzorku (vzpise vsechny kanaly)
print(samples[10100][0]) # vypise pouze prvni kanal 
print(samples[10100][1]) # vypise pouze druhy kanal

# vypis vzorku po 20ti
for i in range(0, len(samples), 20):
    segment = samples[i:i+20]
    pass

segment = samples[1000:10000]
print (segment)
wav.write("segment.wav", rate, segment)

