import torch
import torch.utils.data as data_utils

class PowerDataset(data_utils.Dataset):
    """
    Dataset of speech mixtures for speech separation. 
    Dataset contains a list of numbers of the range [a,b] inclusive. // DELETE
    """
    def __init__(self, a=0, b=1):
        super(PowerDataset, self).__init__()
        assert a <= b
        self.a = a
        self.b = b

    def __len__(self):
        return self.b - self.a + 1

    def __getitem__(self, index):
        """
        1. Make appropriate assertions on the "index" argument. Python allows slices as well, so it is important to be clear of what arguments to support. Just supporting integer indices works well most of the times.
        2. This is the place to load large data on-demand. DONOT ever load all data in the constructor as that unnecessarily bloats memory.
        3. This method should be as fast as possible and should only be using certain pre-computed values. e.g. When loading images, the path directory should be handled during the constructor and this method should only load the file into memory and apply relevant transforms.
        4. Whenever lazy loading is possible, this is the place to be. e.g. Loading images only when called should be here. Keeps the memory footprint low.
        5. Subsequently, this also becomes the place for any input transforms (like resizing, cropping, conversion to tensor and so on)
        """
        assert self.a <= index <= self.b

        return index, index**2

# testing of our custom dataloader and dataset
dataset = PowerDataset(a=1, b=128)
data_loader = data_utils.DataLoader(dataset, batch_size = 64, shuffle=True)
print(len(dataset))  # 128
print(dataset[6]) # 6^2

class PowerFourDataset(PowerDataset):
    def __init__(self, a=0, b=1):
        super(PowerFourDataset, self).__init__(a,b)

    def __getitem__(self,index):
        index, value = super(PowerFourDataset, self).__getitem__(index)
        return index, value**4


# testing of our custom dataloader and dataset
dataset4 = PowerFourDataset(a=1, b=128)
data_loader = data_utils.DataLoader(dataset4, batch_size = 64, shuffle=True)
print(len(dataset4))  # 128
print(dataset4[6]) # 6^4

