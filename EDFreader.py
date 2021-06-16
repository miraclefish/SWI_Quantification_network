import os
import numpy as np
import re
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import mne # version of mne=0.17.1

"""
You can use "conda install mne==0.17.1" or "pip install mne==0.17.1" to install the library "mne".
If you don't want to print the information in the process of reading .edf files, you need to manually 
modify the source code of the library "mne".

There are some tips:
1. File <mne/io/edf/edf.py>  line 188 needs to be commented out;
2. File <mne/io/edf/edf.py>  line 392 needs to be commented out;
3. File <mne/io/edf/edf.py>  line 436 needs to be commented out;
4. File <mne/io/edf/edf.py>  line 462,463 need to be commented out;
5. File <mne/io/edf/edf.py>  line 467,468 need to be commented out;
6. File <mne/io/edf/edf.py>  line 479,480,496 need to be commented out;
7. File <mne/io/edf/edf.py>  line 193 needs to be commented out;
8. File <mne/io/base.py>     line 663,664 need to be commented out;
9. File <mne/annotations.py> line 787,788 need to be commented out;
"""


class EDFreader(object):
    """A class for you to simplize the reading process of EDF files.
    If you have saved the EEG data as .csv files, maybe you don't need
    to use the EDFreader any more, unless you need to restore the .csv files.
    
    Parameters:
    -----------
    filepath: char
        The global path of the current file.
    filename: char
        The filename includes the patient's name abbreviation and the file 
        number, or sometimes the name of the particular channel.
    NAME: char
        The type of the label need to be load.
    raw: object
        Base object for Raw data, defined by the library "mne".
    event_times: list[float]
        It storages the time indices when the labeled events happend, 
        corresponding with the event_names (have the same length).
    event_names: list[char]
        It storages the labeled events name, corresponding with the event_times.
    Avdata: DataFrame
        It storages the sorted EEG data with Average Lead Method.
        The voltage unit of each channel is microvolts.
    """

    def __init__(self, filePath, print_log = True):

        '''
        The raw data files are orgnized as the form followed:
        -------------------------------------------------------------------------------------
        |>root<|  |>>>>>>> data_path <<<<<<<<|  |> seg_path <|   |>>>>>>>> metadata <<<<<<<<|
        > data  > Channel-Name-Any-Expert_num   >Name_clip_num  > Code+Name.atn (Label information)
                                                                > Code+Name.din
                                                                > Code+Name.edf (Raw data)
                                                                > Code+Name.pit (Some information of patient)
        -------------------------------------------------------------------------------------
        '''
        # Whether print the log information?
        self.print_log = print_log

        # Judge whether the 'filePath' is an absolute path
        if os.path.isabs(filePath):
            self.absPath = filePath
        else:
            self.absPath = os.getcwd() + "\\" + filePath

        # Extract file name information
        self.dir, self.metadata_path = os.path.split(self.absPath)
        self.dir, self.seg_path = os.path.split(self.dir)
        self.dir, self.data_path = os.path.split(self.dir)
        
        # Extract spike channel (in data_path)
        self.spike_channel = self.data_path[:2]

        # Load raw data from the metadata_path
        # Get Av traced data (EEG, ECG and EMG)
        self.raw = mne.io.read_raw_edf(self.absPath, preload=True)
        self.ElectrodeData = self._get_raw_data()
        self.data = self._data_sort()

        # Get channel name and trace name (pre defined)
        self.ch_names, self.trace_names = self._get_channels()
        self.channel_ind = self.trace_names.index(self.spike_channel+'-Av')
        # Get annotations from the .atn file (named  same as metadata_file)
        # self.annotations = self._get_atn()
        # self.s_pairs, self.spikes = self._confirm_atn()

        # self.label = self._get_label()
        
        if self.print_log:
            print("load " + self.metadata_path + " finished.")

    def _get_raw_data(self):
        data = self.raw.get_data()*1000000
        if self.print_log:
            print("Raw data shape: ", data.shape)
        return data

    def get_atn(self, atn_path):

        f = open(atn_path, mode='r', encoding='utf-8')
        strs = f.readline()
        
        pattern = re.compile('\{.+?\}')
        str_list = pattern.findall(strs)

        time2text = {}
        for s in str_list:
            flag = re.search(r'"Event":"([A-Za-z|-]+?)"', s)
            if flag == None:
                continue
            else:
                text = flag.groups()[0]
                time = int(float(re.search(r'"Onset":(-?\d+\.\d+)?', s).groups()[0])*1000)
            time2text[time] = text
        
        atn_DF = pd.DataFrame([time2text.values()], columns=time2text.keys()).T
        atn_DF.columns = ['atn']

        return atn_DF
    
    def confirm_atn(self, atn):
        
        start_flag = True
        l = len(atn)
        for i, ind, atnn in zip(range(l), atn.index, atn.atn):
            if start_flag:
                if atnn == "S-end":
                    atn.drop(index=ind, inplace=True)
                elif atnn == "S-start":
                    start_flag = False
            else:
                if atnn == "S-end":
                    start_flag = True
                elif atnn == "S-start":
                    atn.drop(index=ind, inplace=True)

        start_list = atn[atn.atn=="S-start"].index.tolist()
        end_list = atn[atn.atn=="S-end"].index.tolist()
        spike_list = atn[atn.atn=="Spike"].index.tolist()

        if not start_flag:
            atn.drop(index=start_list[-1], inplace=True)
            start_list.pop()

        if all(np.array(end_list)-np.array(start_list)>0):
            return atn, len(start_list), len(spike_list)
        else:
            assert(len(start_list) == len(end_list))
    
    def get_label(self, atn):
        
        data = self.data[self.channel_ind, :]
        on_points = atn[atn.atn=="S-start"].index.tolist()
        off_points = atn[atn.atn=="S-end"].index.tolist()
        s_points = atn[atn.atn=="Spike"].index.tolist()

        label = np.zeros(data.shape)
        for on_point, off_point in zip(on_points, off_points):
            label[on_point:off_point] = 1
        
        d_data = data[1:] - data[:-1]
        
        peak_inds = np.where(np.int8(d_data[:-1]<0) * np.int8(d_data[1:]>0))[0]+1
        for s_point in s_points:
            if label[s_point]==0:
                on_ind = np.sum(peak_inds < s_point)-1
                on_point = peak_inds[on_ind]
                off_point = peak_inds[on_ind+1]
                label[on_point:off_point] = 1

        return label

    def _get_channels(self):
        ch_names = self.raw.ch_names
        trace = ["Fp1-Av", "Fp2-Av", "F3-Av", "F4-Av", "C3-Av", "C4-Av",
                "P3-Av", "P4-Av", "O1-Av", "O2-Av", "F7-Av", "F8-Av", 
                "T3-Av", "T4-Av", "T5-Av", "T6-Av", "Fz-Av", "Cz-Av", 
                "Pz-Av", "EMG-L", "EMG-R", "EMG-L", "EMG-R", "ECG",]
        return ch_names, trace

    def _get_EEG_data(self, data):
        eeg_data = data[0:19,:]
        eeg_data = self._band_pass_filter(0.53, 80, eeg_data, 2)
        # eeg_data = self._band_stop_filter(48, 52, eeg_data, 2)
        eeg_data = eeg_data - np.mean(eeg_data, axis=0)
        return -eeg_data

    def _get_EMG_data(self, data):
        emg_data = data[26:30, :]
        emg_data[0:3,:] = - emg_data[0:3,:]
        emg_data = self._band_pass_filter(5.3, 120, emg_data, 2)
        return emg_data

    def _get_ECG_data(self, data):
        ecg_data = data[30:32, :]
        ecg_data = ecg_data[1,:] - ecg_data[0,:]
        ecg_data = ecg_data.reshape(1, -1)
        ecg_data = self._band_pass_filter(0.53, 50, ecg_data, 4)
        return ecg_data

    def _data_sort(self):
        
        data = self.ElectrodeData

        eeg_data = self._get_EEG_data(data)
        emg_data = self._get_EMG_data(data)*0.8
        ecg_data = self._get_ECG_data(data)*0.05
        sorted_data = np.vstack([eeg_data, emg_data, ecg_data])

        return sorted_data

    def _band_pass_filter(self, LowHz, HigHz, data, order):
        N, L = data.shape
        hf = HigHz * 2.0 / 1000
        lf = LowHz * 2.0 / 1000
        b, a = butter(order, [lf, hf], "bandpass")
        filted_data = filtfilt(b, a, data)
        return filted_data

    def _band_stop_filter(self, LowHz, HigHz, data, order):
        N, L = data.shape
        hf = HigHz * 2.0 / 1000
        lf = LowHz * 2.0 / 1000
        b, a = butter(order, [lf, hf], "bandstop")
        filted_data = filtfilt(b, a, data)
        return filted_data

    def save_as_txt(self):
        print('********************************')
        SpikeChannelData = self.data[self.channel_ind, :]

        save_path, _ = os.path.split(self.absPath)

        SpikeChannelData = pd.DataFrame({self.spike_channel: SpikeChannelData})
        SpikeChannelData.to_csv(save_path + '\\' + 'data.txt', sep='\t', index=False)
        self.annotations.to_csv(save_path + '\\' + 'atn.txt', sep='\t', index=True)

        print('File <{0}> has been saved in path <{1}>;'.format(self.metadata_path, save_path))
        print('Signal length: ', SpikeChannelData.shape[0])
        print('S pairs : [{:d}]; Spikes : [{:d}]'.format(self.s_pairs, self.spikes))
        return


    def plot_data(self, ind=None, Sens=2):

        if ind==None:
            ind = [0, 5000]
        
        data = self.data[:, ind[0]:ind[1]]
        label = self.label[ind[0]:ind[1]]
        data = data[self.channel_ind, :].reshape(1, -1)
        N, _ = data.shape

        atn_in_ind = [idx for idx in self.annotations.index if idx<ind[1] and idx>=ind[0]]

        fig = plt.figure(figsize=[15,5])
        for i in range(N):
            ax = fig.add_subplot(N, 1, i+1)
            ax.plot(np.arange(ind[0], ind[1]), data[i])
            ax.scatter(np.arange(ind[0], ind[1]), label*data[i])
            ax.vlines(atn_in_ind, -100*Sens, 100*Sens, color='r')
            ax.set_facecolor('none')
            
            ax.set_yticks([0])
            ax.set_yticklabels([self.trace_names[self.channel_ind]], rotation=0)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if i < N-1:
                ax.spines['bottom'].set_visible(False)
                ax.set_xticks([])
            else:
                ax.set_xticks(list(np.arange(ind[0], ind[1], 1000)))
                for idx in atn_in_ind:
                    ax.text(idx+30, -100*Sens+50, self.annotations.loc[idx, 'atn'], fontsize=10, color='red')
        plt.subplots_adjust(hspace=-0.75)
        plt.show()
        return None

if __name__ == "__main__":

    file_list = []
    # path = 'Data-1'
    # i = 22

    # path = 'Data-2'
    # i = 22

    path = 'Data-0'
    i = 22

    for root, dirs, files in os.walk(path):
        for file in files:
            if file[-3:] == 'edf':
                file_list.append(os.path.join(root, file))
                # print(os.path.join(root,file))

    # for file in file_list:
    #     edf = EDFreader(filePath=file, print_log=False)
    #     edf.save_as_txt()
    # pass

    edf = EDFreader(filePath=file_list[i], print_log=False)
    edf.plot_data()
    pass