import os
import torch
import numpy as np
from SignalSegNet import SignalSegNet, Basicblock
from torch.utils.data import DataLoader
from torch import nn
from Dataset_train import Dataset_train

def test(net, root, input_size, stride):

    with torch.no_grad():
        net.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        loss = nn.CrossEntropyLoss()
        loss_all = 0

        dataset = Dataset_train(DataPath=root, type='Test', input_size=input_size, stride=stride)

        
        dataloader = DataLoader(dataset=dataset, batch_size=48, shuffle=False)

        for data in dataloader:
            
            x, label = data['Data'].float(), data['label'].long()

            x = x.to(device)
            label = label.to(device)

            output = net(x=x)
            Loss =  loss(output, label)
            loss_all += Loss.cpu().data.numpy()
            
        loss_all = loss_all/len(dataloader)
    return loss_all
