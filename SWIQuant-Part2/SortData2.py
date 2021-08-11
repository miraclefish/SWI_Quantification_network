from EDFreader import EDFreader
import os
import pandas as pd
import numpy as np


class SortData2(EDFreader):

    def __init__(self, filePath, id, expert_num, print_log):
        super(SortData2, self).__init__(filePath=filePath, print_log=print_log)

        self.id = id
        self.num = expert_num
        self.root_path = filePath.split(os.sep)[0]
        self.atn_path_list = self._get_atn_path_list()


    def _get_atn_path_list(self):

        atn_path_list = []
        for i in range(self.num):
            root = self.root_path.replace('0', str(i))
            file_list = self.get_all_file_in_path(root)
            atn_path_i = file_list[self.id][:-3] + 'atn'
            atn_path_list.append(atn_path_i)

        return atn_path_list
    
    def save(self, save_path):
        
        SpikeChannelData = self.data[self.channel_ind, :]
        SpikeChannelData = pd.DataFrame({self.spike_channel: SpikeChannelData})

        mylog = open('log.log', mode = 'a', encoding='utf-8')

        print(' ', file=mylog)
        print('File <{0}> will been saved >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>;'.format(self.seg_path), file=mylog)
        print('Signal length: ', SpikeChannelData.shape[0], 'ms', file=mylog)

        for i, path in enumerate(self.atn_path_list):

            atn_path = '{}\\Atn-{}'.format(save_path, i)
            data_path = '{}\\Data'.format(save_path)
            if not os.path.exists(atn_path):
                os.makedirs(atn_path)

            atn_DF = self.get_atn(path)
            atn_DF, s_pairs, spikes, start_name, end_name = self.confirm_atn(atn_DF)

            atn_DF.to_csv('{}\\{}-{}-{}.txt'.format(atn_path, self.data_path.split('-')[1], self.seg_path.split('-')[1], self.seg_path.split('-')[2]), sep='\t', index=True)
            label = self.get_label(atn_DF, start_name, end_name)
            SpikeChannelData['Atn-{}'.format(i)] = label
            swi = np.mean(label)*100

            print('   ---Atn-{} 【 S pairs : [{:d}]; Spikes : [{:d}]; SWI : [{:.2f}%]】'.format(i, s_pairs, spikes, swi), file=mylog)

        if not os.path.exists(data_path):
            os.makedirs(data_path)

        SpikeChannelData.to_csv('{}\\{}-{}-{}.txt'.format(data_path, self.data_path.split('-')[1], self.seg_path.split('-')[1], self.seg_path.split('-')[2]), sep='\t', index=True)
        return None

    def get_all_file_in_path(self, rt):

        file_list = []
        for root, dirs, files in os.walk(rt):
            for file in files:
                if file[-3:] == 'edf':
                    file_list.append(os.path.join(root, file))
        return file_list


if __name__ == "__main__":

    file_list = []

    path = 'Dataset2-0-病例1-15'

    for root, dirs, files in os.walk(path):
        for file in files:
            if file[-3:] == 'edf':
                file_list.append(os.path.join(root, file))
                # print(os.path.join(root,file))

    for i, file in enumerate(file_list):
        sort3 = SortData2(filePath=file, id=i, expert_num=3, print_log=False)
        sort3.save(save_path='Seg5data')
    
    # sort3 = SortData2(filePath=file_list[4], id=4, expert_num=3, print_log=False)
    # sort3.save(save_path='Seg5data')

    pass


