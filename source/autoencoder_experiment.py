import torch
import torch.nn as nn
import torch.nn.functional as fc

# constants
batch_size = 10    #
input_channels = 1 #
length_of_sig_seq = 10 #
padding = 5 #aby se zachoval puvodni rozmer vstupniho signalu (length of sig = 10)

num_of_filters = 2 # 256 
filter_size = 5    # 20 samples

# operations
conv = nn.Conv1d(input_channels, num_of_filters, filter_size, padding=padding)
relu = nn.ReLU(num_of_filters)

# inicializace dat
# pocet vstupnich dat, pocet kanalu, delka sekvence signalu
data = torch.rand(batch_size, input_channels, length_of_sig_seq) 
#(r1 - r2) * torch.rand(a, b) + r2
print(data)
data = (-1 - 1) * data + 1
print(data)

# convolving
output = conv(data)

print("conv")
print(conv.weight)
print(output.size())
print(output)

output = relu(output)
print(output)  # tohle je vlastne nonnegativni reprezentace vstupniho signalu


