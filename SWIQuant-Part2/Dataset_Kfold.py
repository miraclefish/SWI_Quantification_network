import os
from posixpath import pardir
import torch
import random
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class Dataset_Kfold(Dataset):

    def __init__(self, DataPath, ki, K_fold, state, input_size, stride):

        if os.path.isabs(DataPath):
            self.DataPath = DataPath
        else:
            self.DataPath = os.getcwd() + "\\" + DataPath
        self.input_size = input_size
        self.stride = stride
        self.ki = ki
        self.K_fold = K_fold
        self.state = state

        self.filelist, self.signal_length = self.pre_load_data()
        self.sample_path_list, self.index_list = self.split_data()

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        file = self.sample_path_list[idx]
        start, end = self.index_list[idx]
            
        raw_data = pd.read_csv(file, sep='\t', index_col=0)
        Data = raw_data[raw_data.columns[0]].values.reshape(1, -1)
        Label = raw_data['PreAtn'].values
        length = Data.shape[1]

        data = np.zeros((1, self.input_size))
        label = np.zeros(self.input_size)
        if end < length:
            data = Data[:, start:end]
            label[:] = Label[start:end]
        else:
            data[0, :(length-start)] = Data[:, start:]
            label[:(length-start)] = Label[start:]
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        sample = {"Data":data, "label":label}
        return sample

    def pre_load_data(self):
        filelist = os.listdir(self.DataPath)
        length = []
        for file in filelist:
            path = os.path.join(self.DataPath, file)
            data = pd.read_csv(path, sep='\t', index_col=0)
            length.append(data.shape[0])
        return filelist, length

    def split_data(self):
        path_list = []
        index_list = []
        for file, length in zip(self.filelist, self.signal_length):
            split_num = int(np.ceil((length-(self.input_size-self.stride))/self.stride))
            path = os.path.join(self.DataPath, file)
            for i in range(split_num):
                path_list.append(path)
                index_list.append((i*self.stride, i*self.stride+self.input_size))

        random.seed(1024)
        random.shuffle(path_list)
        random.shuffle(index_list)

        
        path_list, index_list = self.K_fold_ind(path_list, index_list)

        return path_list, index_list

    def K_fold_ind(self, path_list, index_list):
        N = len(path_list)
        n = int(np.ceil(N/self.K_fold))
        ki = self.ki-1
        valid_start = ki*n
        valid_end = (ki+1)*n
        if self.ki == self.K_fold:
            valid_end = N
        if self.state == 'Valid':            
            path_list = path_list[valid_start:valid_end]
            index_list = index_list[valid_start:valid_end]
        if self.state == 'Train':
            path_list = path_list[0:valid_start] + path_list[valid_end:N]
            index_list = index_list[0:valid_start] + index_list[valid_end:N]
        return path_list, index_list

if __name__ == "__main__":

    data_train = Dataset_Kfold(DataPath="Seg5data\\trainData", ki=2, K_fold=5, state='Valid', input_size=30014, stride=3000)
    pass