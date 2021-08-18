import os
import torch
import numpy as np
from SignalSegNet import SignalSegNet, Basicblock
from torch.utils.data import DataLoader
from torch import nn
from Dataset_test import Dataset_test

def test(net, root):

    with torch.no_grad():
        net.eval()
        loss = nn.CrossEntropyLoss()
        loss_all = 0

        filelist = os.listdir(root)
        
        for file in filelist:
            
            DataPath = os.path.join(root, file)
            dataset = Dataset_test(DataPath=DataPath)

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            dataloader = DataLoader(dataset=dataset, batch_size=10, shuffle=False)

            for i, data in enumerate(dataloader):
                if i >= 1:
                    print("!!!!!!!!!!!Wrong dataset!")
                    break

                x, label = data

                net = net.to(device)
                x = x.to(device)
                label = label.to(device)

                output = net(x=x)
                loss_all += loss(output, label[:,0])
            
    loss_all = loss_all/len(filelist)
    loss_all = loss_all.cpu().data.numpy()
    return loss_all
