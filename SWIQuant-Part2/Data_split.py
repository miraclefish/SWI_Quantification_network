import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

class Dataset_train(Dataset):

    def __init__(self, DataPath, type, input_size, stride):

        if os.path.isabs(DataPath):
            self.DataPath = DataPath
        else:
            self.DataPath = os.getcwd() + "\\" + DataPath
        self.input_size = input_size
        self.stride = stride
        self.type = type

        self.filelist, self.signal_length = self.pre_load_data()
        self.sample_path_list, self.index_list = self.split_data()

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        file = self.sample_path_list[idx]
        start, end = self.index_list[idx]
            
        raw_data = pd.read_csv(file, sep='\t', index_col=0)
        Data = raw_data[raw_data.columns[0]].values

        if self.type == "Train":
            Label = raw_data['PreAtn'].values
        if self.type == "Test":
            Label = raw_data['Atn-0'].values
        Score = raw_data['Score'].values

        length = Data.shape[0]

        data = np.zeros(self.input_size)
        label = np.zeros(self.input_size)
        score = np.zeros(self.input_size)
        if end < length:
            data = Data[start:end]
            label[:] = Label[start:end]
            score[:] = Score[start:end]
        else:
            data[:(length-start)] = Data[start:]
            label[:(length-start)] = Label[start:]
            score[:(length-start)] = Score[start:]

        sample = {"Data":data, "label":label, "score":score}
        save_sample = pd.DataFrame(sample)
        path, filename = os.path.split(file)
        path = path + '_'+ str(self.input_size)
        if not os.path.isdir(path):
            os.mkdir(path)
        save_path = os.path.join(path, "{}-{}-{}.txt".format(filename[:-4], start, end))
        save_sample.to_csv(save_path, sep='\t')
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

                '''
                0 3 6 9 12 15 18 21 24 27 30 31
                0-9 3-12 6-15 9-18 12-21 15-24 18-27 21-30 24-31
                '''
        return path_list, index_list

if __name__ == "__main__":

    data_train = Dataset_train(DataPath="Seg5data\\trainData", type="Train", input_size=5000, stride=2500)
    data_test = Dataset_train(DataPath="Seg5data\\testData1", type="Test", input_size=5000, stride=2500)
    with tqdm(total=len(data_train)) as pbar:
        pbar.set_description("Split training set:")
        for i in range(len(data_train)):
            sample = data_train[i]
            pbar.update(1)
    with tqdm(total=len(data_test)) as pbar:
        pbar.set_description("Spile test set:")
        for i in range(len(data_test)):
            sample = data_test[i]
            pbar.update(1)
    pass