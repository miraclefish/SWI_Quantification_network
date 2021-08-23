import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

class Dataset_train(Dataset):

    def __init__(self, DataPath, type, input_size):

        if os.path.isabs(DataPath):
            self.DataPath = DataPath
        else:
            self.DataPath = os.getcwd() + "\\" + DataPath

        self.type = type
        self.input_size = input_size
        self.filelist = self.get_file_list()

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        file = self.filelist[idx]
        raw_data = pd.read_csv(file, sep='\t', index_col=0)
        Data = raw_data['Data'].values
        Label = raw_data['label'].values
        data = torch.from_numpy(Data)
        label = torch.from_numpy(Label)
        sample = {"Data":data, "label":label}
        Score = raw_data['score'].values.reshape(-1,1)
        Score = np.log1p(np.abs(Score))/np.log(np.max(Score))
        Score = self.dilate(Score, kernel=31)
        score = torch.from_numpy(Score)
        sample['score'] = score
        return sample

    def get_file_list(self):
        if self.type == 'Train':
            path = os.path.join(self.DataPath, 'trainData_'+str(self.input_size))
        if self.type == 'Test':
            path = os.path.join(self.DataPath, 'testData1_'+str(self.input_size))
        file_names = os.listdir(path)
        filelist = [os.path.join(path, file_name) for file_name in file_names]
        return filelist

    def dilate(self, score, kernel=11):
        
        pad_width = ((int((kernel-1)/2),int(np.ceil((kernel-1)/2))), (0,0))
        score_pad = np.pad(score, pad_width=pad_width, mode='constant', constant_values=0)
        stride = 1
        n = int((len(score_pad)-(kernel-stride))/stride)
        out = np.zeros((n, kernel))
        for i in range(kernel-1):
            out[:,i] = score_pad[i:-(kernel-i-1)].squeeze()
        out[:,-1] = score_pad[kernel-1:].squeeze()
        out = np.max(out, axis=1)
        return out




if __name__ == "__main__":

    data_train = Dataset_train(DataPath="Seg5data", type='Train', input_size=5000)
    data_test = Dataset_train(DataPath="Seg5data", type='Test', input_size=5000)
    sample = data_train[1]
    pass