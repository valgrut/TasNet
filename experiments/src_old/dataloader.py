import os
import struct
from scipy.io import wavfile as wav
import matplotlib.pyplot as plt
import IPython.display as ipd
import numpy as np

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
    #plt.plot(wav_sample)
    #plt.show()
    
    return [rate, wav_sample]

rate, samples = wav_plotter("data/strings.wav", "kill")
#rate, samples = wav_plotter("data/spatialize_wsj0-mix/405o0319_2.3824_01xo030w_-2.3824.wav", "kill")

print(samples[10100])    # vypis signalu na konkretnim vzorku (vzpise vsechny kanaly)
print(samples[10100][0]) # vypise pouze prvni kanal 
#print(samples[10100][1]) # vypise pouze druhy kanal



########################################################################################

# vypis vzorku po 20ti
def gen():
    for i in range(0, len(samples), 20):
        segment = samples[i:i+20]
        print(i)
#        print(segment)
#        print("")
        if i > 200:
            break
        
        yield segment

#for seg in gen():
#    print(seg)

a = gen()
print(next(a))
print(next(a))

########################################################################################
########################### TOHLE JE FUNGL #############################################

# TODO dodelat jeste dalsi vnejsi funkci, ktera teto bude predavat postupne vsechny vstupni pisnickz
# TODO dodelat funkci ktera bude balit segmenty do mini_bash, treba po 4 atd

segment = []
# vypis vzorku po 20ti - pouze prvni kanal
def gen_single_channel():
    segment = []
    for i in range(0, len(samples), 20):
        segment = [[samples[i][0]] for i in range(i, i+20)]
        yield segment

aa = gen_single_channel()
print(next(aa))
print(next(aa))
print(next(aa))
print(next(aa))
print(next(aa))
print(next(aa))

out = []
for i in range(1, 1000):
    seg = next(aa)
    for s in seg:
        out.append(s)
#print(out)

#jeste to je potreba previst na numpu array a na svislej vector
nump_out =  np.array(out)
print(nump_out)
wav.write("segs.wav", rate, nump_out)

######################################################################################
# oba kanaly ale pouze kousek vstupniho zvuku
segment = samples[1000:10000]
print(type(segment))
print(segment) # numpy.ndarray
wav.write("segment.wav", rate, segment)



