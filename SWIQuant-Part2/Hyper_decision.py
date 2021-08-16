import os
from numpy.lib.stride_tricks import DummyArray
import torch

from SignalSegNet import SignalSegNet, Basicblock
from torch.utils.data import DataLoader, dataset
from Dataset_test import Dataset_test
from utils import adjust_window, label2Spair, pair2label
from PrelabelEvalu import evalu
from Model_test import inital_net, plot_demo, test


if __name__ == "__main__":

    root = 'Seg5data/testData'
    filelist = os.listdir(root)
    DataPath = os.path.join(root, filelist[0])

    layers = [2,2,2,2,2]
    net = SignalSegNet(Basicblock, layers)
    dataset = Dataset_test(DataPath=DataPath)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataloader = DataLoader(dataset=dataset, batch_size=8, shuffle=False)

    for i, data in enumerate(dataloader):
        x, label = data

        net = net.to(device)
        x = x.to(device)

        output = net(x=x)
    pass