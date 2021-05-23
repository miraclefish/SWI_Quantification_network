import os
import numpy as np
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

    def __init__(self, filePath):
        if os.path.isabs(filePath):
            self.absPath = filePath
        else:
            self.absPath = os.getcwd() + "\\" + filePath
        self.dirname, self.filename = os.path.split(self.absPath)
        self.raw = mne.io.read_raw_edf(self.absPath, preload=True)
        self.ch_names, self.trace_names = self._get_channels()
        self.annotations = self._get_atn()
        self.ElectrodeData = self._get_raw_data()
        
        print("load " + self.filename + " finished.")

    def _get_raw_data(self):
        data = self.raw.get_data()*1000000
        print("Data shape: ", data.shape)
#         data = pd.DataFrame(data.T, columns=self.ch_names)
        return data

    def _get_atn(self):
        atn_array = self.raw.find_edf_events()

        atn = list(self.raw.annotations.description)
        atn_pair = [(int(float(atn[i])*1000), atn[i+1]) for i, val in enumerate(atn) if atn[i][0]=='+' and atn[i+1][0]!='+' and i<len(atn)-1]
        atn_DF = pd.DataFrame(atn_pair, columns=['ind', 'atn'])
        annotations = atn_DF.set_index(['ind'])
        return annotations

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

    # def _data_sort(self, data):
        
    #     columns = ["Fp1","Fp2","F3","F4","C3","C4",
    #                 "P3","P4","O1","O2","F7","F8",
    #                 "T3","T4","T5","T6","Fz","Cz",
    #                 "Pz"]

    #     eeg_signal = data[0:19, :]
    #     ref_signal = data[22:24, :]

    #     # EEG data filter
    #     eeg_filted = self._lowpass(HigHz=200, data=eeg_signal)
    #     ref_filted = self._lowpass(HigHz=200, data=ref_signal)

    #     # eeg_filted_Av = eeg_filted
    #     eeg_filted_Av = eeg_filted - np.mean(ref_filted, axis=0, keepdims=True)

    #     filted_data = -eeg_filted_Av
    #     sorted_data = pd.DataFrame(filted_data.T, columns=columns)
    #     return sorted_data[self.channel]

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

    # def save_txt(self, path):
    #     if len(self.OriginalData)!=30014:
    #         print('length: ', len(self.OriginalData))
    #     self.OriginalData.to_csv('./'+path+'/'+self.filename+'.txt', sep='\t', index=False)


    def plot_data(self, ind=None, Sens=2):
        
        if ind==None:
            ind = [0, 5000]
        data = self.ElectrodeData[:, ind[0]:ind[1]]

        eeg_data = self._get_EEG_data(data)
        emg_data = self._get_EMG_data(data)*0.8
        ecg_data = self._get_ECG_data(data)*0.05
        data = np.vstack([eeg_data, emg_data, ecg_data])
        N, L = data.shape

        atn_in_ind = [idx for idx in self.annotations.index if idx<ind[1] and idx>=ind[0]]

        fig = plt.figure(figsize=[15,15])
        for i in range(N):
            ax = fig.add_subplot(N, 1, i+1)
            ax.plot(np.arange(ind[0], ind[1]), data[i])
            ax.vlines(atn_in_ind, -100*Sens, 100*Sens, color='r')
            ax.set_ylim([-100*Sens,100*Sens])
            ax.set_facecolor('none')
            
            ax.set_yticks([0])
            ax.set_yticklabels([self.trace_names[i]], rotation=0)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if i < N-1:
                ax.spines['bottom'].set_visible(False)
                ax.set_xticks([])
            else:
                ax.set_xticks(list(np.arange(ind[0], ind[1], 1000)))
                for idx in atn_in_ind:
                    ax.text(idx+30, -100*Sens+50, self.annotations.loc[idx, 'atn'], fontsize=16, color='red')
        plt.subplots_adjust(hspace=-0.75)
        plt.show()
        return None

if __name__ == "__main__":

    # filelist = os.listdir('Pre5data')
    # for file in filelist:
    #     print("***********************************")
    #     filePath = 'Pre5data\\' + file
    #     edf = EDFreader(filePath)
    #     print(edf.raw.annotations.description)
    

    edf = EDFreader('Pre5data\\T4-syn-2-1.edf')
    print(edf.raw.annotations.description)
    edf.plot_data(ind=[5000,10000])
    pass