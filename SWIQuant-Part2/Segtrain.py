import os
import numpy as np
import torch

from SignalSegNet import SignalSegNet, Basicblock
from Train_dataset import Dataset_train
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch import optim
from Segtest import test


# 初始化设定

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("<<<<<<<< Device: ", device," >>>>>>>>>>>")

lr = 1e-3
batch_size = 48
n_epoch = 1000
s_epoch = -1
layers = [2,2]
input_size = 5000
RESUME = False
model_root = "models5000/model-"+str(len(layers))+"-layers"+str(0)

# dataset_train = Dataset_train(DataPath="Seg5data\\trainData", type='Train', input_size=input_size, stride=stride)
dataset_train = Dataset_train(DataPath="Seg5data", type='Train', input_size=input_size)
dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
dataloader_train = list(dataloader_train)

dataset_test = Dataset_train(DataPath="Seg5data", type='Test', input_size=input_size)
dataloader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False)
dataloader_test = list(dataloader_test)

# 定义网络
writer = SummaryWriter('./log/'+model_root, flush_secs=1)

net = SignalSegNet(Basicblock, layers)

optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.9)
loss = nn.CrossEntropyLoss()

net = net.to(device)
loss = loss.to(device)

if RESUME:
    print("<<<<<<<<<<< Resume model from {} epoch. >>>>>>>>>>>".format(s_epoch))
    path_checkpoint = '{0}/model_epoch_{1}.pth.tar'.format(model_root, s_epoch)
    checkpoint = torch.load(path_checkpoint)
    net.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    s_epoch = checkpoint['epoch']

for epoch in range(s_epoch+1, n_epoch):

    loss_all = 0

    net.train()
    for i, data_train in enumerate(dataloader_train):

        x_data = data_train['Data'].float()
        label = data_train['label'].long()
        
        net.zero_grad()

        x_data = x_data.to(device)
        label = label.to(device)

        output = net(x=x_data)
        Loss = loss(output, label)
        loss_all += Loss.cpu().data.numpy()
        Loss.backward()
        optimizer.step()

        i += 1
        print('epoch: %d, [iter: %d / all %d], loss : %f' \
              % (epoch, i, len(dataloader_train), Loss.cpu().data.numpy()))
    
    scheduler.step()
    
    train_loss = loss_all/len(dataloader_train)
    test_loss = test(net, dataloader_test)
    print('epoch: %d,  test_loss : %f' % (epoch, test_loss))
    # save_checkpoint_state(model_root, epoch, model=net, optimizer)
    checkpoint = {
            "net": net.state_dict(),
            'optimizer': optimizer.state_dict(),
            "epoch": epoch,
        }
    if not os.path.isdir(model_root):
        os.mkdir(model_root)
    if epoch % 2 == 0:
        torch.save(checkpoint,'{0}/model_epoch_{1}.pth.tar'.format(model_root, epoch))
        print("Save {0} epoch model in Path [{1}/model_epoch_{2}.pth.tar]".format(epoch, model_root, epoch))
        
        writer.add_scalars('Loss', {'train_loss': train_loss, 'test_loss':test_loss}, epoch)
        writer.close()

pass