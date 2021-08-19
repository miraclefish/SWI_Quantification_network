import numpy as np
import pandas as pd
import torch

from SignalSegNet import SignalSegNet, Basicblock
from Dataset_Kfold import Dataset_Kfold
from torch.utils.data import DataLoader
from torch import nn
from torch import optim

if __name__ == "__main__":

    Train_Loss = []
    Valid_Loss = []
    lr = 1e-3
    batch_size = 48
    n_epoch = 100
    s_epoch = 0
    layer_nums = [2, 3, 4, 5]
    K_fold = 5

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root = "Seg5data\\trainData"

    for layer_num in layer_nums:

        layers = [2 for i in range(layer_num)]
        Train_Loss_of_sigle_model = []
        Valid_Loss_of_sigle_model = []

        for ki in range(K_fold):

            net = SignalSegNet(Basicblock, layers)
            optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.9)
            loss = nn.CrossEntropyLoss()

            net = net.to(device)
            loss = loss.to(device)

            print("\n|---------- ", layer_num, " layers Model of the", ki+1, "th fold -----------|")
            print("|  Strat training on device: ", device)
            print("|  #Parameters of the net: ", sum([p.numel() for p in net.parameters()]))

            dataset_train = Dataset_Kfold(DataPath=root, ki=ki+1, K_fold=K_fold, state='Train', input_size=30014, stride=3000)
            dataset_valid = Dataset_Kfold(DataPath=root, ki=ki+1, K_fold=K_fold, state='Valid', input_size=30014, stride=3000)
            dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
            dataloader_valid = DataLoader(dataset=dataset_valid, batch_size=batch_size, shuffle=True)

            for epoch in range(s_epoch, n_epoch):

                '''
                train:
                '''
                net.train()

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
                train_loss = loss_all/len(dataloader_train)
                
                scheduler.step()

                '''
                test:
                '''
                net.eval()

                data_valid_iter = iter(dataloader_valid)

                loss_valid_all = 0

                for data_valid in dataloader_valid:

                    x_data = data_valid['Data'].float()
                    label = data_valid['label'].long()
                    
                    x_data = x_data.to(device)
                    label = label.to(device)

                    output = net(x=x_data)
                    Loss_valid = loss(output, label)
                    loss_valid_all += Loss_valid.cpu().data.numpy()               

                valid_loss = loss_valid_all/len(dataloader_valid)
                print('epoch: %d,  test_loss : %f' % (epoch, valid_loss))

            Train_Loss_of_sigle_model.append(train_loss)
            Valid_Loss_of_sigle_model.append(valid_loss)
            Train_Loss_of_sigle_model = np.array(Train_Loss_of_sigle_model).reshape(-1, 1)
            Valid_Loss_of_sigle_model = np.array(Valid_Loss_of_sigle_model).reshape(-1, 1)

            print("|  Finish training ...")
        
        Train_Loss.append(Train_Loss_of_sigle_model)
        Valid_Loss.append(Valid_Loss_of_sigle_model)

    Train_Loss = np.concatenate(Train_Loss, axis=1)
    Valid_Loss = np.concatenate(Valid_Loss, axis=1)

    tabel = np.concatenate([Train_Loss, Valid_Loss], axis=1)
    columns = ["Train_"+str(i+1) for i in range(len(layer_nums))] \
        +["Valid_"+str(i+1) for i in range(len(layer_nums))]
    tabel = pd.DataFrame(tabel, columns=columns)
    tabel.to_csv(str(K_fold)+"fold_result.csv")