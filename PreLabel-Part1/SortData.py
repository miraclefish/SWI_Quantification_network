from EDFreader import EDFreader
import os
import pandas as pd
import numpy as np


class SortData(EDFreader):

    def __init__(self, filePath, expert_num, print_log):
        super(SortData, self).__init__(filePath=filePath, print_log=print_log)

        self.num = expert_num
        self.atn_path_list = self._get_atn_path_list()


    def _get_atn_path_list(self):

        atn_path_i = self.absPath[:-3] + 'atn'
        atn_path_list = []
        for i in range(self.num):
            path = atn_path_i.replace('0\\', str(i)+'\\')
            atn_path_list.append(path)
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
            atn_DF, s_pairs, spikes = self.confirm_atn(atn_DF)

            atn_DF.to_csv('{}\\{}-{}.txt'.format(atn_path, self.data_path[3:6], self.seg_path[-5:]), sep='\t', index=True)
            label = self.get_label(atn_DF)
            SpikeChannelData['Atn-{}'.format(i)] = label
            swi = np.mean(label)

            print('   ---Atn-{} 【 S pairs : [{:d}]; Spikes : [{:d}]; SWI : [{:.2f}%]】'.format(i, s_pairs, spikes, swi), file=mylog)

        if not os.path.exists(data_path):
                os.makedirs(data_path)

        SpikeChannelData.to_csv('{}\\{}-{}.txt'.format(data_path, self.data_path[3:6], self.seg_path[-5:]), sep='\t', index=True)
        return None



if __name__ == "__main__":

    file_list = []

    path = 'Data-0'

    for root, dirs, files in os.walk(path):
        for file in files:
            if file[-3:] == 'edf':
                file_list.append(os.path.join(root, file))
                # print(os.path.join(root,file))

    for file in file_list:
        sort3 = SortData(filePath=file, expert_num=3, print_log=False)
        sort3.save(save_path='Pre5data')
    
    # sort3 = SortData(filePath=file_list[10], expert_num=3, print_log=False)
    # sort3.save(save_path='Pre5data')

    pass


