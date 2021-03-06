import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class Dataset_test(Dataset):

    def __init__(self, DataPath, input_size):

        if os.path.isabs(DataPath):
            self.DataPath = DataPath
        else:
            self.DataPath = os.getcwd() + "\\" + DataPath

        self.input_size = input_size
        self.data, self.s_channel, self.label, self.score = self.load_data()
        self.dataFill, self.labelFill, self.scoreFill, self.index_list, self.length = self.split_as_batch()

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        data = self.dataFill[:, self.index_list[idx][0]:self.index_list[idx][1]]
        label = self.labelFill[:, self.index_list[idx][0]:self.index_list[idx][1]]
        score = self.scoreFill[:, self.index_list[idx][0]:self.index_list[idx][1]]
        data = torch.from_numpy(data).float()
        label = torch.from_numpy(label).long()
        score = torch.from_numpy(score).float()
        return data, label, score

    def load_data(self):
        raw_data = pd.read_csv(self.DataPath, sep='\t', index_col=0)
        s_channel = raw_data.columns[0]
        data = raw_data[raw_data.columns[0]].values.reshape(1, -1)
        label = raw_data[['Atn-0', 'Atn-1', 'Atn-2']].values
        score = raw_data['Score'].values.reshape(1, -1)

        return data, s_channel, label, score

    def split_as_batch(self):

        L = self.data.shape[1]
        batch_size = int(np.ceil(L/self.input_size))
        dataFill = np.zeros((1, batch_size*self.input_size))
        dataFill[0, :L] = self.data
        labelFill = np.zeros((1, batch_size*self.input_size))
        labelFill[0, :L] = self.label[:, 0]
        scoreFill = np.zeros((1, batch_size*self.input_size))
        scoreFill[0, :L] = self.score[:, 0]
        index_list = [(i*self.input_size, (i+1)*self.input_size) for i in range(batch_size)]
        return dataFill, labelFill, scoreFill, index_list, L

if __name__ == "__main__":

    data_test = Dataset_test(DataPath="Seg5data\\testData2\\03-?????????-1.txt", input_size=5000)
    pass