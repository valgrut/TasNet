import torch
import torch.nn as nn
import torch.nn.functional as fc

# ####################################################################
#
#
class ResBlock(nn.Module):
    def __init__(self, in_channels, dilation):
        super(ResBlock, self).__init__()
        self.dilation = dilation
        self.conv1 = nn.Conv1d(1, 256, kernel_size=3, padding=0, dilation=self.dilation, stride=1)
        self.batch1 = nn.BatchNorm1d(256)
        self.batch2 = nn.BatchNorm1d(256)
        self.prelu1 = nn.PReLU(256)
        self.prelu2 = nn.PReLU(256)
        self.conv = nn.Conv1d(256, 1) #d-conv
        self.conv2 = nn.Conv1d(in_channels, 256, kernel_size=3)
    
    def forward(self, input_data):
        #...
        x = self.conv1(input_data)
        x = self.prelu1(x)
        x = self.batch1(x)
        #x = self.d_conv()
        torch.cat(x, 1)
        x = self.prelu2(x)
        x = self.batch2(x)
        x = self.conv2(x)
        #outputs = [x ,input_data]
        #return torch.cat(outputs, 1)
        return (x + input_data)


# #####################################################################
# Separation - base
#
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 256, 20)

    def forward(self, input_data):
        # jakekoli operace s tensorem
        input_data = self.conv1(input_data);
        input_data = fc.relu(input_data); 
        return input_data

#######################################################################

# instanciace site
net = Net()
print(net)

# learnable parameters
params = list(net.parameters())
print("Pocet parametru: " + str(len(params)))
print("parametry:")
print(params)
print(params[0].size())  # self.conv1's .weight
#print(params.size())

# testing
input_data = torch.randn(1, 50)
input_data = input_data.unsqueeze(0)
output = net(input_data) 
print(output)

output = net(input)
target = torch.randn(1, 50)  # a dummy target, for example
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)
