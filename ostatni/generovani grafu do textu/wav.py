# Load the required libraries:
#   * scipy
#   * numpy
#   * matplotlib
import matplotlib
from scipy.io import wavfile
from matplotlib import pyplot as plt
import numpy as np


name="As2"

# Global setting of fonts
font = {'family' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)
plt.rc('legend', fontsize=20) 

# Load the data and calculate the time of each sample
samplerate, data = wavfile.read(name+'.wav')
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
plt.savefig(name+'.png', spi=100, bbox_inches='tight')
plt.show()
