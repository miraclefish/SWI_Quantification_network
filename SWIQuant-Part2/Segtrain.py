import os
import numpy as np
import torch

from SignalSegNet import SignalSegNet, Basicblock
from Dataset_train import Dataset_train
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch import optim
from Segtest import test


# 初始化设定

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("<<<<<<<< Device: ", device," >>>>>>>>>>>")

lr = 1e-3
batch_size = 8
n_epoch = 1000
s_epoch = -1
RESUME = False
model_root = "./model-check"

dataset_train = Dataset_train(DataPath="Seg5data\\trainData", input_size=30014, stride=3000)
dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)

# 定义网络
writer = SummaryWriter('./log/runs0', flush_secs=1)

net = SignalSegNet(Basicblock, [2,2,2,2,2])

optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)
loss = nn.CrossEntropyLoss()

net = net.to(device)
loss = loss.to(device)
net.train()

if RESUME:
    print("<<<<<<<<<<< Resume model from {} epoch. >>>>>>>>>>>".format(s_epoch))
    path_checkpoint = '{0}/model_epoch_{1}.pth.tar'.format(model_root, s_epoch)
    checkpoint = torch.load(path_checkpoint)
    net.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    s_epoch = checkpoint['epoch']

for epoch in range(s_epoch+1, n_epoch):
    data_train_iter = iter(dataloader_train)

    i = 0
    
    loss_all = 0

    while i < len(dataloader_train):

        data_train = data_train_iter.next()
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
    test_loss = test(net)
    print('epoch: %d,  test_loss : %f' % (epoch, test_loss))
    # save_checkpoint_state(model_root, epoch, model=net, optimizer)
    checkpoint = {
            "net": net.state_dict(),
            'optimizer': optimizer.state_dict(),
            "epoch": epoch,
        }
    if not os.path.isdir(model_root):
        os.mkdir(model_root)
    if epoch % 5 == 0:
        torch.save(checkpoint,'{0}/model_epoch_{1}.pth.tar'.format(model_root, epoch))
        print("Save {0} epoch model in Path [{1}/model_epoch_{2}.pth.tar]".format(epoch, model_root, epoch))
        
        writer.add_scalars('Loss', {'train_loss': train_loss, 'test_loss':test_loss}, epoch)
        writer.close()

pass