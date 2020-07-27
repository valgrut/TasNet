# Load the required libraries:
#   * scipy
#   * numpy
#   * matplotlib
from scipy.io import wavfile
from matplotlib import pyplot as plt
import numpy as np

# Load the data and calculate the time of each sample
samplerate, data = wavfile.read('mixA.wav')
times = np.arange(len(data))/float(samplerate)

# Make the plot
# You can tweak the figsize (width, height) in inches
plt.figure(figsize=(30, 4))
# plt.fill_between(times, data[:,0], data[:,1], color='k') 
plt.fill_between(times, data)
plt.xlim(times[0], times[-1])
plt.xlabel('Čas (s)')
plt.ylabel('Amplituda')
# You can set the format by changing the extension
# like .pdf, .svg, .eps
# plt.savefig('s2.png', dpi=100)
plt.savefig('mix.png', spi=100, bbox_inches='tight')
plt.show()
