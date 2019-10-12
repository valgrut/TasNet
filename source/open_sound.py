from __future__ import print_function
import scipy
import matplotlib.pyplot as plt
import librosa
import numpy as np
from scipy.io import wavfile
from librosa import display


rate, data = wavfile.read("strings.wav")

print(rate)
print(data)

# 1. Get the file path to the included audio example
filename = librosa.util.example_audio_file()

# 2. Load the audio as a waveform `y`
#    Store the sampling rate as `sr`
y, sr = librosa.load("implode.wav")
print(sr) # sample rate
#print(y) # signal as waveform
print(y[0])
print(y[1000])


#segment = np.ndarray(shape=(1, 1000), dtype=float, order='F')
data = []
for i in range(1000, 2000):
    data.append(y[i])

segment = np.ndarray((1,), buffer=np.array(data),offset=np.float_().itemsize,dtype=float)

librosa.output.write_wav('file_trim_5s.wav', segment, sr) 


y, sr = librosa.load("file_trim_5s.wav")
print(y[0])

#input_data = []
#for sample in y:
#    if len(input_data) < 20:
#        input_data.append(sample)
#    else:
#        print(input_data)
#        input_data = []


## 3. Run the default beat tracker
#tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
#
#print('Estimated tempo: {:.2f} beats per minute'.format(tempo))
#
## 4. Convert the frame indices of beat events into timestamps
#beat_times = librosa.frames_to_time(beat_frames, sr=sr)
#
#print('Saving output to beat_times.csv')
#librosa.output.times_csv('beat_times.csv', beat_times)
#
##y, sr = librosa.load(librosa.util.example_audio_file(), duration=10)
#plt.figure()
#plt.subplot(3, 1, 1)
#display.waveplot(y, sr=sr)
#plt.title('Monophonic')
