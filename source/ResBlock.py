import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """
    Trida reprezentuje jeden konvolucni blok.
    Tyto bloky jsou v hlavni tride TasNet skladany za sebe.
    Trida je konstruovana s urcitou casovou dilataci, ktera ovlivnuje kontextualni okno, ktere sit vidi.
    """
    def __init__(self, in_channels, dilation, DEBUG=False):
        super(ResBlock, self).__init__()
        self.DEBUG=DEBUG
        self.dilation = dilation

        self.conv1 = nn.Conv1d(256, 512, kernel_size=1)
        self.D_conv = nn.Conv1d(512, 512, kernel_size=3, padding=self.dilation, groups=512, dilation=self.dilation)
        self.conv2 = nn.Conv1d(512, 256, kernel_size=1)

        self.batch1 = nn.BatchNorm1d(512)
        self.batch2 = nn.BatchNorm1d(512)

        self.prelu1 = nn.PReLU(512)
        self.prelu2 = nn.PReLU(512)

    def forward(self, input_data):
        if self.DEBUG: 
            print("ResBlock Start: shape of input_data:", input_data.shape)
        x = self.conv1(input_data)
        x = self.prelu1(x)
        x = self.batch1(x)
        x = self.D_conv(x)
        if self.DEBUG:
            print("ResBlock middle shape:", x.shape)
        #x = torch.reshape(x, (1, -1,))
        #x = torch.reshape(x, (-1,))
        if self.DEBUG:
            print("ResBlock po concatenaci:", x.shape)
        x = self.prelu2(x)
        x = self.batch2(x)
        x = self.conv2(x)
        if self.DEBUG:
            print("ResBlock after conv2: ", x.shape)
            print("ResBlock end: input_data: ", input_data.shape)
        return torch.add(x, input_data)

