import os
import torch

from SignalSegNet import SignalSegNet, Basicblock
from Train_dataset import Dataset_train
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch import optim


class ModelTrainer(object):

    def __init__(self, Net, model_root,  lr=1e-3, batch_size=256, n_epoch=300, input_size=5000):
        self.Net = Net
        self.mode_flag = {'U-net':'U', 'Score': 'S', 'Att': 'A', 'Score+Att': 'S+A'}
        self.model_root = model_root
        self.model_save_path = os.path.join(model_root, "M-"+str(self.Net.layers_num)+"-layers-"+self.mode_flag[self.Net.mode]+"-"+str(input_size))
        self.lr = lr
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.input_size = input_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.Net_mode = Net.mode

    def load_data(self):
        
        dataset_train = Dataset_train(DataPath="Seg5data", type='Train', input_size=self.input_size)
        dataset_test = Dataset_train(DataPath="Seg5data", type='Test', input_size=self.input_size)
        dataloader_train = DataLoader(dataset=dataset_train, batch_size=self.batch_size, shuffle=True)
        dataloader_test = DataLoader(dataset=dataset_test, batch_size=self.batch_size, shuffle=False)

        return dataloader_train, dataloader_test
    
    def train(self, RESUME=False, s_epoch=-1):

        writer = SummaryWriter('./log/'+self.model_save_path)
        if not os.path.isdir(self.model_save_path):
            os.makedirs(self.model_save_path)

        print('---------------------------------------------------------')
        print("[             Model-{}-layers-mode-{}-{}           ]".format(self.Net.layers_num, self.Net.mode, self.input_size))
        print()
        optimizer = optim.Adam(self.Net.parameters(), lr=self.lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 15, 0.9)
        loss = nn.CrossEntropyLoss()

        self.Net = self.Net.to(self.device)
        loss = loss.to(self.device)

        if RESUME:
            print("<<<<<<<<<<< Resume model from {} epoch. >>>>>>>>>>>".format(s_epoch))
            path_checkpoint = '{0}/model_epoch_{1}.pth.tar'.format(self.model_save_path, s_epoch)
            checkpoint = torch.load(path_checkpoint)
            self.Net.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            s_epoch = checkpoint['epoch']

        dataloader_train, dataloader_test = self.load_data()
        dataloader_train = list(dataloader_train)
        dataloader_test = list(dataloader_test)

        for epoch in range(s_epoch+1, self.n_epoch):

            loss_all = 0

            self.Net.train()
            for i, data_train in enumerate(dataloader_train):

                x_data = data_train['Data'].float()
                label = data_train['label'].long()
                score = data_train['score'].float()
                
                self.Net.zero_grad()

                x_data = x_data.to(self.device)
                label = label.to(self.device)
                score = score.to(self.device)

                output, Att = self.Net(x=x_data, score=score)
                Loss = loss(output, label)
                loss_all += Loss.cpu().data.numpy()
                Loss.backward()
                optimizer.step()

                i += 1
                print('epoch: %d, [iter: %d / all %d], loss : %f' \
                    % (epoch, i, len(dataloader_train), Loss.cpu().data.numpy()))
            
            scheduler.step()
            
            train_loss = loss_all/len(dataloader_train)

            test_loss = self.test(dataloader_test)
            print('epoch: %d,  test_loss : %f' % (epoch, test_loss))
            checkpoint = {
                    "net": self.Net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    "epoch": epoch,
                }

            if epoch % 2 == 0:
                torch.save(checkpoint,'{0}/model_epoch_{1}.pth.tar'.format(self.model_save_path, epoch))
                print("Save {0} epoch model in Path [{1}/model_epoch_{2}.pth.tar]".format(epoch, self.model_save_path, epoch))
                
                writer.add_scalars('Loss', {'train_loss': train_loss, 'test_loss':test_loss}, epoch)
                writer.close()
        return None

    def test(self, dataloader):
        with torch.no_grad():
            self.Net.eval()

            loss = nn.CrossEntropyLoss()
            loss_all = 0

            for data in dataloader:
                
                x, label = data['Data'].float(), data['label'].long()
                score = data['score'].float()

                x = x.to(self.device)
                label = label.to(self.device)
                score = score.to(self.device)

                output, Att = self.Net(x=x, score=score)
                Loss =  loss(output, label)
                loss_all += Loss.cpu().data.numpy()
                
            loss_all = loss_all/len(dataloader)
        return  loss_all

if __name__ == '__main__':

    for input_size in [500, 1000, 2500, 5000, 10000, 15000, 20000, 25000, 30000]:
        layers = 4
        Net = SignalSegNet(Basicblock, layers=layers, mode='U-net')
        modeltrainer = ModelTrainer(Net, 'model', n_epoch=500, batch_size=256, input_size=input_size)
        modeltrainer.train()
        pass
    pass