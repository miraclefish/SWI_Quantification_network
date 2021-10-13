import os
import pandas as pd

class Dataset(object):

    def __init__(self, Path):

        if os.path.isabs(Path):
            self.Path = Path
        else:
            self.Path = os.getcwd() + "\\" + Path
        self.filepath, self.filename = self.pre_load()
    
    def __getitem__(self, index):
        Data = DataLoad(file=self.filepath[index], name=self.filename[index])
        return Data

    def __len__(self):
        return len(self.filename)

    def pre_load(self):

        filelist = os.listdir(self.Path)
        filepath = []
        filename = []
        for file in filelist:
            filepath.append(os.path.join(self.Path, file))
            filename.append(file[:-4])
        return filepath, filename

class DataLoad(object):

    def __init__(self, file, name):
        self.path = file
        self.name = name
        self.data, self.label, self.length = self.load_data()

    def load_data(self):
        raw_data = pd.read_csv(self.path, sep='\t', index_col=0)
        data = raw_data[raw_data.columns[0]].values
        label = raw_data['Atn-0'].values
        length = len(data)
        return data, label, length
