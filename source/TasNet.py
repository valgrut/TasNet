import torch
import torch.nn as nn
import torch.nn.functional as fc

from ResBlock import ResBlock



class Net(nn.Module):
    """
    Trida reprezentujici model TasNet. Jejim zavolanim na smes mluvcich vygeneruje data reprezentujici separovane mluvci.
    Trida pouziva konvolucni bloky implementovane ve tride ResBlock.
    """
    def __init__(self, X, R, nn_stride, padd, batch_size, DEBUG):
        super(Net, self).__init__()
        self.DEBUG=DEBUG
        self.batch_size = batch_size
        # treti hodnota (20) predstavuje hyperparametr L ze studie. 
        self.conv1 = nn.Conv1d(1, 256, 20, bias=False, stride=nn_stride, padding=padd)
        self.deconv = nn.ConvTranspose1d(512, 2, 20, padding=padd, bias=False, stride=nn_stride, groups=2)

        self.layer_norm = nn.LayerNorm([256, 1601])
        # self.layer_norm = nn.LayerNorm(256)
        self.bottleneck1 = nn.Conv1d(256, 256, 1) #TODO padding, stride???
        self.bottleneck2 = nn.Conv1d(256, 512, 1) #TODO 512 = NxC
        self.softmax = nn.Softmax(2)

        self.TCN = nn.Sequential()

        dilation = 0
        for i in range(0, R):
            for i in range(0, X):
                self.TCN.add_module("ResBlock"+str(dilation),ResBlock(256, 2**dilation))
                dilation += 1
            dilation = 0

        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.deconv.weight)

    def forward(self, input_data):
        # encoder
        if self.DEBUG:
            print("Net start - create representation from input_data: ", input_data.shape)
        #input_data = torch.unsqueeze(input_data, 1) #TODO ??
        if self.DEBUG:
            print("Net start - create representation from input_data: ", input_data.shape)
        representation = self.conv1(input_data)
        representation = fc.relu(representation)
        if self.DEBUG:
            print("Net: advanced representation created", representation.shape)

        # separation - estimate masks
        # representation = self.layer_norm(representation) # TODO -layer norm
        data = self.bottleneck1(representation)

        data = self.TCN(data)

        data = self.bottleneck2(data)
        current_batch_size = data.shape[0] # for case if batch is not complete
        data = torch.reshape(data, (current_batch_size, 256, 2, -1,))
        # data = torch.reshape(data, (self.batch_size, 256, 2, -1,))
        masks = self.softmax(data)
        if self.DEBUG:
            print("NN: Masks: ", masks.shape)

        # multiply masks and representation
        # print("TasNet: representation shape: ", representation.shape)
        # print("TasNet: masks shape:", masks.shape)
        masked_representation = torch.mul(representation[:,:,None,:], masks)
        masked_representation = torch.reshape(masked_representation, (current_batch_size, 512, -1))
        # masked_representation = torch.reshape(masked_representation, (self.batch_size, 512, -1))

        # decoder
        separate_data = self.deconv(masked_representation)
        # print("TasNet: separated data OUTPUT shape", separate_data.shape)
        return separate_data

