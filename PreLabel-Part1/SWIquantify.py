import os

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

class SWIquantify(object):

    def __init__(self, filepath, Spike_width, print_log=False):

        self.filepath = filepath
        self.dir, self.filename = os.path.split(self.filepath)
        self.print_log = print_log

        self.data, self.s_channel, self.label = self.load_data()
        self.bandPassData = self._band_pass_filter(LowHz=0.5, HigHz=40, data=self.data)
        self.maskPassData = self._band_pass_filter(LowHz=0.5, HigHz=8, data=self.data)

        self.Spike_width = Spike_width
        self.wavelet = self._get_wave()
        self.swi_label = self._get_swi_label()

        self.score = None
        self.spike_ind = None
        self.spike_point = None
        self.spike_score = None
        self.band_pair = None
        self.mask = None

    def fix_threshold(self):
        self.score = self.Adaptive_Decomposition()

        swi = []
        thresholds = np.concatenate([np.linspace(0.30, 1, 71), np.linspace(1.1, 3, 20)])

        for threshold in thresholds:
            
            self.spike_ind, self.spike_score = self.bect_discrimination(score=self.score, threshold=threshold)

            band_ind = self.band_ind_expand(spike_ind=self.spike_ind)

            self.band_pair = self.find_slow_wave(band_ind=band_ind)

            SWI, self.mask = self.get_SWI(band_pair=self.band_pair)

            swi.append(SWI)

        swi = np.array(swi)
        err = np.abs(swi-self.swi_label)
        ind = np.argmin(err)
        min_err = err[ind]
        print("The Optimal threshold index: ", str(ind), " [", str(thresholds[ind]),"]\n")
        optimal_th = thresholds[ind]

        return optimal_th, min_err

    def get_optimal_result(self, threshold):
        self.score = self.Adaptive_Decomposition()

        self.spike_ind, self.spike_score = self.bect_discrimination(score=self.score, threshold=threshold)

        band_ind = self.band_ind_expand(spike_ind=self.spike_ind)

        self.band_pair = self.find_slow_wave(band_ind=band_ind)

        SWI, self.mask = self.get_SWI(band_pair=self.band_pair)

        return SWI
            

    def get_SWI(self, band_pair):
        mask = np.zeros(len(self.data))
        band_pair = band_pair.reshape(-1,2)
        for ind_pair in band_pair:
            mask[ind_pair[0]:ind_pair[1]] = 1
        spike_time = np.sum(mask)
        SWI = spike_time/len(self.data)
        return SWI, mask
        

    def bect_discrimination(self, score, threshold):
        peak_ind, peak_score = self.get_peak_score(score)

        spike_ind = np.where(peak_score - np.mean(peak_score) > np.std(peak_score)*threshold)[0]
        spike_ind = peak_ind[spike_ind]
        return spike_ind, peak_score

    def get_peak_score(self, score):
        dscore = score[1:]-score[:-1]
        peak_ind = np.where(dscore[:-1]*dscore[1:]<0)[0]+1

        peak_score = score[peak_ind]
        peak_score = np.sign(peak_score)*np.log(np.abs(peak_score)+1)
        return peak_ind, peak_score

    def band_ind_expand(self, spike_ind):
        d_data = self.bandPassData[1:] - self.bandPassData[:-1]
        peak_ind = np.where(d_data[:-1]*d_data[1:]<0)[0]+1
        l = len(peak_ind)
        band_ind = []
        spike_point = []
        for ind in spike_ind:
            loc = int(np.sum(peak_ind<ind))-1
            if loc-1 >=0 and loc+1<=l-1:
                band_ind.append(peak_ind[loc-1])
                spike_point.append(peak_ind[loc])
                band_ind.append(peak_ind[loc+1])
        self.spike_point = spike_point
        return band_ind

    def find_slow_wave(self, band_ind):

        d_mask_data = self.maskPassData[1:] - self.maskPassData[:-1]
        peak_ind = np.where(d_mask_data[:-1]*d_mask_data[1:]<0)[0]+1

        band_pair = np.array(band_ind).reshape(-1,2)
        for i, ind_pair in enumerate(band_pair):
            loc = int(np.sum(peak_ind<ind_pair[1]))-1
            if loc+3 < len(peak_ind):
                
                # 慢波宽度校准
                candidate_wave_length = peak_ind[loc+3] - peak_ind[loc+1]
                low_bound = (ind_pair[1]-ind_pair[0])*1
                high_bound = (ind_pair[1]-ind_pair[0])*5
                length_flag = candidate_wave_length > low_bound and candidate_wave_length < high_bound

                # 慢波高度校准
                candidate_slow_wave = self.maskPassData[peak_ind[loc+1]:peak_ind[loc+3]]
                spike_wave = self.maskPassData[ind_pair[0]:ind_pair[1]]
                candidate_wave_high = np.max(candidate_slow_wave) - np.min(candidate_slow_wave)
                low_bound = (np.max(spike_wave)-np.min(spike_wave))*0.45
                high_flag = candidate_wave_high > low_bound

                if high_flag and length_flag:
                    band_pair[i,1] = peak_ind[loc+3]

        band_pair = band_pair.reshape(-1,1).squeeze()
        return band_pair

    def Adaptive_Decomposition(self):

        # 保证滑窗滤波后信号长度与原信号长度相同，进行Padding操作
        pad_width = ((int((self.Spike_width-1)/2),int(np.ceil((self.Spike_width-1)/2))), (0,0))
        x_pad = np.pad(self.bandPassData, pad_width=pad_width, mode='constant', constant_values=0)

        # 对 data 滑窗的过程矩阵并行化，详情请参考函数 self._window_slide()
        # data_windowed 的每一行是一个 data 的滑窗提取，步长为 1
        data_windowed = self._window_slide(x_pad, self.Spike_width)
        wave = self._Adaptive_wave(Original_wave=self.wavelet, data=data_windowed)

        score = np.sum(data_windowed*wave, axis=1)/data_windowed.shape[1]**2
        return score

    def _get_wave(self, p=2):
        x = np.linspace(0.5,2,self.Spike_width)
        wave = np.exp(-1/(x**p)-x**p)
        wave = (wave-min(wave))/(max(wave)-min(wave))
        return wave

    def _window_slide(self, x, Spike_width):
        stride = 1
        n = int((len(x)-(Spike_width-stride))/stride)
        out = np.zeros((n, Spike_width))
        for i in range(Spike_width-1):
            out[:,i] = x[i:-(Spike_width-i-1)].squeeze()
        out[:,-1] = x[Spike_width-1:].squeeze()
        out = (out.T - np.mean(out, axis=1)).T
        return out

    def _Adaptive_wave(self, Original_wave, data):
        # 小波的尺度根据原始信号的形状做自适应调整
        min_data = np.min(data, axis=1)
        max_data = np.max(data, axis=1)
        Original_wave = Original_wave.reshape(1, -1)
        wave = np.tile(Original_wave, [data.shape[0], 1])
        out = (wave.T*(max_data-min_data)+min_data).T
        return out

    def _get_swi_label(self):
        return np.mean(self.label[:,0])

    def load_data(self):

        if self.print_log:
            print("File loading: \""+self.filepath+"\".")

        raw_data = pd.read_csv(self.filepath, sep='\t', index_col=0)
        s_channel = raw_data.columns[0]
        data = raw_data[raw_data.columns[0]].values
        label = raw_data[['Atn-0', 'Atn-1', 'Atn-2']].values

        if self.print_log:
            print("The length of the data is {:.3f}s.".format(data.shape[0]/1000))

        return data, s_channel, label


    def _band_pass_filter(self, LowHz, HigHz, data):
        data = np.squeeze(data)
        hf = HigHz * 2.0 / 1000
        lf = LowHz * 2.0 / 1000
        N = 2
        b, a = butter(N, [lf, hf], "bandpass")
        filted_data = filtfilt(b, a, data)
        filted_data = filted_data.reshape(-1, 1)
        return filted_data

    def plot_demo(self, time, length, pred=None, save_fig=False):
        
        data = self.data
        label = self.label[:,0]

        L = data.shape[0]
        start, end = self._adjust_window(time, length, L)
        spike_ind = [ind for ind in self.spike_point if ind >= start and ind <= end]
        y_lim_min = min(data[start:end])
        y_lim_max = max(data[start:end])
        ranges = y_lim_max-y_lim_min

        N = 1
        if pred is not None:
            N = 2

        fig = plt.figure(figsize=[15, 6])

        for i in range(N):
            ax = fig.add_subplot(N, 1, i+1)
            ax.plot(np.arange(start, end)/1000, data[start:end])
            ax.set_ylim([y_lim_min-ranges*0.1, y_lim_max+ranges*0.1])
            
            if i > 0:
                s_pair_label = self._label2Spair(label[start:end], start)

                line_label = True
                for s_pair in s_pair_label:
                    if line_label:
                        ax.plot(np.arange(s_pair[0], s_pair[1])/1000, data[s_pair[0]:s_pair[1]], c='r', label='Ground_truth')
                        line_label = False
                    else:
                        ax.plot(np.arange(s_pair[0], s_pair[1])/1000, data[s_pair[0]:s_pair[1]], c='r')

                s_pair_pred = self._label2Spair(label=pred[start:end, i-1], start=start)
                onsets = s_pair_pred[:,0]
                offsets = s_pair_pred[:,1]

                if len(onsets) != 0:
                    ax.scatter(onsets/1000, data[onsets], c='limegreen', marker='>', s = 6**2, label='S_start')
                    ax.scatter(offsets/1000, data[offsets], c='black', marker='s', s=6**2, label='S_end')

                if len(spike_ind) != 0:
                    ax.scatter(np.array(spike_ind)/1000, data[spike_ind]+ranges*0.05, c='maroon', marker='v', s = 4**2, label='S_point')
            
            ax.set_facecolor('none')
            ax.set_ylabel('Amplitude(μV)')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xticks(list(np.arange(start, end, 1000)/1000))
            ax.set_xlabel('Times(s)')
            ax.set_xlim([start/1000, end/1000+1])
        plt.legend(loc=1)

        if save_fig:
            swi_label = np.mean(label[start:end])*100
            swi_pred = np.mean(pred[start:end])*100
            path = os.path.join('Pre5Fig', "{}-{:.2f}-{:.2f}-{}".format(self.filename[:-4], swi_label, swi_pred, '.png'))
            plt.savefig(path, dpi=500, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
        return None

    def _label2Spair(self, label, start):

        d_label = label[1:] - label[:-1]
        inds = list(np.where(d_label != 0)[0]+1)

        if label[0] == 1:
            inds.insert(0, 0)
        if label[-1] == 1:
            inds.append(len(label))

        s_pair = np.array(inds)+start
        s_pair = s_pair.reshape(-1,2)
        return s_pair

    def _adjust_window(self, time, length, L):
        
        time = int(time*1000)
        length = int(length*1000)
        start = time
        end = time + length - 1

        if start < 0:
            start = 0
        elif end > L-1:
            end = L-1
        elif start > L-1:
            start = L-1-length
            end = L-1

        return start, end


if __name__ == "__main__":

    dataPath = "Pre5data\Data\ZZX-clip1.txt"
    SwiQ = SWIquantify(filepath=dataPath, Spike_width=76, print_log=True)
    th, min_err = SwiQ.fix_threshold()
    swi = SwiQ.get_optimal_result(th)
    label_pred = SwiQ.mask.reshape(-1,1)
    SwiQ.plot_demo(time=0, length=10, pred=label_pred, save_fig=True)
    pass