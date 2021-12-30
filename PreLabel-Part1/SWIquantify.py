import os

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, stft
import pywt
import matplotlib.pyplot as plt

class SWIquantify(object):

    def __init__(self, filepath, Spike_width, p=2, print_log=False):

        self.filepath = filepath
        self.dir, self.filename = os.path.split(self.filepath)
        self.print_log = print_log

        self.data, self.s_channel, self.label = self.load_data()
        self.bandPassData = self._band_pass_filter(LowHz=0.5, HigHz=40, data=self.data)
        self.maskPassData = self._band_pass_filter(LowHz=0.5, HigHz=8, data=self.data)

        self.Spike_width = Spike_width
        self.p = p
        self.wavelet = self._get_wave()
        self.swi_label = self._get_swi_label()

        self.score = None
        self.spike_ind = None
        self.spike_point = None
        self.spike_score = None
        self.band_pair = None
        self.mask = None

    def fix_threshold(self, D=0):
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
        if D==0:
            swi_label = self.swi_label
        else:
            swi_label = self.swi_label + D/100
        swi_label = np.max(swi_label, 0)
        
        err = np.abs(swi-swi_label)
        ind = np.argmin(err)
        min_err = err[ind]
        print("The Optimal threshold index: ", str(ind), " [", str(thresholds[ind]),"] with Deviation ", D, "\%\n")
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

    def _AWED_for_plot(self):

        # 保证滑窗滤波后信号长度与原信号长度相同，进行Padding操作
        pad_width = ((int((self.Spike_width-1)/2),int(np.ceil((self.Spike_width-1)/2))), (0,0))
        x_pad = np.pad(self.bandPassData, pad_width=pad_width, mode='constant', constant_values=0)

        # 对 data 滑窗的过程矩阵并行化，详情请参考函数 self._window_slide()
        # data_windowed 的每一行是一个 data 的滑窗提取，步长为 1
        data_windowed = self._window_slide(x_pad, self.Spike_width)
        wave = self._Adaptive_wave(Original_wave=self.wavelet, data=data_windowed)

        score = np.sum(data_windowed*wave, axis=1)/data_windowed.shape[1]**2
        return score, data_windowed, wave

    def _get_wave(self):
        x = np.linspace(0.5,2,self.Spike_width)
        wave = np.exp(-1/(x**self.p)-x**self.p)
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
            ax.plot(np.arange(start, end)/1000, data[start:end], c='black', linewidth=1.5)
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
            path = os.path.join('PlotFig', "{}-{:.2f}-{:.2f}-{}".format(self.filename[:-4], swi_label, swi_pred, '.png'))
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
        end = time + length

        if start < 0:
            start = 0
        elif end > L-1:
            end = L-1
        elif start > L-1:
            start = L-1-length
            end = L-1

        return start, end

    def plot_stfft(self, time, length, vmin=0, vmax=10):
        data = self.data
        L = data.shape[0]
        start, end = self._adjust_window(time, length, L)
        data = data[start:end]
        f, t, nd = stft(data, fs=1000, window='hann', nperseg=200, noverlap=50)
        nd = np.abs(nd)

        fig = plt.figure(figsize=[8,8])
        plt.subplot(2,1,1)
        plt.plot(data)
        plt.xlim(0,5000)
        plt.subplot(2,1,2)
        plt.pcolormesh(t,f[:21],nd[:21,:],vmin=vmin,vmax=vmax)
        # plt.colorbar()
        plt.show()

    def plot_dwt(self, time, length, level=4, save_fig=False):
        data = self.data
        L = data.shape[0]
        start, end = self._adjust_window(time, length, L)
        data = data[start:end]
        data = data[::6]

        wavelet='sym5' #选取的小波基函数
        X = range(len(data))
        wave =pywt.wavedec(data, wavelet, level=level)
        #小波重构
        ya = None
        yds = []
        for i in range(level+1):
            one_hot_list = [0]*(level+1)
            one_hot_list[i] = 1
            if i == 0:
                ya = pywt.waverec(np.multiply(wave, one_hot_list).tolist(),wavelet)#第level层近似分量
            else:
                yd = pywt.waverec(np.multiply(wave, one_hot_list).tolist(), wavelet)
                yds.append(yd)

        fig = plt.figure(figsize=(8, 8))
        num_fig = level+2
        ax = plt.subplot(num_fig, 1, 1)
        plt.plot(X, data, c='black', label='Original EEG Signal')
        plt.legend(bbox_to_anchor=(0.55, 0.8))
        kk = [ax.spines[key].set_visible(False) for key in ax.spines.keys() if key is not 'left']
        ax.set_xticks([])
        ax.set_xlim([0, len(X)])
        ax.set_ylim([-np.max(np.abs(data))-15, np.max(np.abs(data))+15])
        ax.set_facecolor('none')
        # plt.title('Original EEG Signal', fontsize=12)

        pads = [4,4,4,3,2]
        color = ['#1BADD9', '#1584B7', '#0F5A94', '#08316F', '#01074C']
        for i in range(num_fig-1):
            ax = plt.subplot(num_fig, 1, i+2)
            if i == 0:
                plt.plot(X, ya, c=color[i])
                # X_c = np.arange(2**(4-i)/2, len(data), 2**(4-i))
                # w_c = wave[i][pads[i]:-pads[i]]
                # plt.plot(X_c, w_c)
                ax.set_yticks([0])
                ax.set_yticklabels(['A{}'.format(4-i)], rotation=0, fontsize=12)
            else:
                plt.plot(X, yds[i-1], c=color[i])
                # X_c = np.arange(2**(5-i)/2, len(data), 2**(5-i))
                # w_c = wave[i][pads[i]:-pads[i]]
                # plt.plot(X_c, w_c)
                ax.set_yticks([0])
                ax.set_yticklabels(['D{}'.format(5-i)], rotation=0, fontsize=12)
            kk = [ax.spines[key].set_visible(False) for key in ax.spines.keys() if key is not 'left']
            ax.set_xticks([])
            ax.set_xlim([0, len(X)])
            ax.set_facecolor('none')
        plt.tight_layout()
        if save_fig:
            path = os.path.join('PlotFig', "{}-({}-{})-dwt.png".format(self.filename[:-4], time, time+length))
            plt.savefig(path, dpi=500, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
        return None

    def plot_process(self, time, length, num_prc, n=3, save_fig=False):
        
        data = self.data
        L = data.shape[0]
        start, end = self._adjust_window(time, length, L)
        data = data[start:end]
        X = range(len(data))

        score, data_windowed, wave = self._AWED_for_plot()
        score = score[start:end]
        data_windowed = data_windowed[start:end, :]
        wave = wave[start:end, :]

        interval = int((len(data)-n*num_prc*76)/(n*num_prc+1))

        fig = plt.figure(figsize=(8, 8))
        num_fig = num_prc+2
        ax = plt.subplot(num_fig, 1, 1)
        plt.plot(X, data, c='black', label='Original EEG Signal')
        plt.legend(bbox_to_anchor=(0.55, 0.8))
        kk = [ax.spines[key].set_visible(False) for key in ax.spines.keys() if key is not 'left']
        ax.set_xticks([])
        ax.set_xlim([0, len(X)])
        ax.set_ylim([-np.max(np.abs(data))-15, np.max(np.abs(data))+15])
        ax.set_facecolor('none')
        # plt.title('Original EEG Signal', fontsize=12)

        ax_score = plt.subplot(num_fig, 1, num_fig)
        ax_score.plot(X, score, c='b', label='Result of AWED', zorder=1)
        X_marker, score_marker = X[interval+int(self.Spike_width/2):], score[interval+int(self.Spike_width/2):]
        ax_score.scatter(X_marker[::interval+self.Spike_width], score_marker[::interval+self.Spike_width], c='r', s=12, label='Frame Sample', zorder=2)
        kk = [ax_score.spines[key].set_visible(False) for key in ax_score.spines.keys() if key not in ['left', 'bottom']]
        plt.legend(bbox_to_anchor=(0.85, 0.25))
        ax_score.spines['bottom'].set_position(('data', 0))
        ax_score.set_xticks([])
        ax_score.set_xlim([0, len(X)])
        ax_score.set_facecolor('none')

        for i in range(num_prc):
            ax = plt.subplot(num_fig, 1, i+2)
            for j in range(n):
                left_ind = (interval + self.Spike_width)*(i*n+j) + interval
                right_ind = left_ind + self.Spike_width
                ind = int((left_ind + right_ind)/2)
                plt.plot(X[left_ind:right_ind], data_windowed[ind,:], c='black', linewidth=2, label='Local Averaged EEG Signal')
                plt.plot(X[left_ind:right_ind], wave[ind,:], c='r', linewidth=1.5, label='Adaptive Wavelet')
                if j == 0 and i==0:
                    plt.legend()
            kk = [ax.spines[key].set_visible(False) for key in ax.spines.keys() if key not in ['left', 'bottom']]
            ax.spines['bottom'].set_position(('data', 0))
            ax.set_xticks([])
            ax.set_xlim([0, len(X)])
            ax.set_ylim([-np.max(np.abs(data))-15, np.max(np.abs(data))+15])
            ax.set_facecolor('none')

        plt.tight_layout()
        if save_fig:
            path = os.path.join('PlotFig', "{}-({}-{})-AWED.png".format(self.filename[:-4], time, time+length))
            plt.savefig(path, dpi=500, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
        return None

    def plot_structure(self, time, length, k1, k2, save_fig=False):
        
        data_all= self.data
        L = data_all.shape[0]
        start, end = self._adjust_window(time, length, L)
        data = data_all[start:end]
        X = range(start, end)

        score_all, data_windowed, wave = self._AWED_for_plot()
        peak_ind, peak_score = self.get_peak_score(score_all)
        spike_ind, spike_score = self.bect_discrimination(score_all, k1)
        band_ind = self.band_ind_expand(spike_ind)
        band_pair = self.find_slow_wave(band_ind=band_ind)
        SWI, label = self.get_SWI(band_pair)
        score = score_all[start:end]
        th1 = np.std(peak_score)*k1+np.mean(peak_score)
        th2 = np.std(peak_score)*k2+np.mean(peak_score)
        spike_ind = [ind for ind in self.spike_point if ind >= start and ind <= end]
        
        log_spike_ind = peak_ind[np.where(peak_score>th1)[0]]
        log_spike_score = peak_score[np.where(peak_score>th1)[0]]

        peak_ind, peak_score = peak_ind[np.where(peak_ind>=start)[0]], peak_score[np.where(peak_ind>=start)[0]]
        peak_ind, peak_score = peak_ind[np.where(peak_ind<end)[0]], peak_score[np.where(peak_ind<end)[0]]

        num_fig = 6
        place = 0
        fig = plt.figure(figsize=(8, 8))

        '''Plot Original Data'''
        ax_data = plt.subplot(num_fig, 1, 1)
        ax_data.plot(X, data, c='black', label=r'$f(t)$')
        plt.legend(loc=1)
        kk = [ax_data.spines[key].set_visible(False) for key in ax_data.spines.keys() if key not in ['left']]
        ax_data.set_xticks([])
        # ax_data.set_yticks([])
        ax_data.set_xlim([start, end+place])
        # ax_data.set_ylim([-np.max(np.abs(data))-15, np.max(np.abs(data))+15])

        '''Plot AWED result'''
        ax_score = plt.subplot(num_fig, 1, 2)
        ax_score.plot(X, score, c='b', label=r'$S(t)$', zorder=1)
        kk = [ax_score.spines[key].set_visible(False) for key in ax_score.spines.keys() if key not in ['bottom', 'left']]
        # plt.legend(bbox_to_anchor=(0.85, 0.25))
        ax_score.spines['bottom'].set_position(('data', 0))
        ax_score.set_xticks([])
        # ax_score.set_yticks([])
        ax_score.set_xlim([start, end+place])
        plt.legend(loc=1)

        '''Plot peak_score of decomposed Data'''
        ax_pp = plt.subplot(num_fig, 1, 3)
        ax_pp.plot(X, score, linestyle='dashed', c='b', zorder=1)
        score_pp = score_all[list(peak_ind)]
        self.plot_hist(score_pp, height=10, save_fig=True)
        ax_pp.scatter(peak_ind, score_pp, c='r', s=10, zorder=2, label=r'$\mathbf{P}=\{p_j\}$')
        kk = [ax_pp.spines[key].set_visible(False) for key in ax_pp.spines.keys() if key not in ['bottom', 'left']]
        plt.legend(loc=1)
        ax_pp.spines['bottom'].set_position(('data', 0))
        ax_pp.set_xticks([])
        ax_pp.set_xlim([start, end+place])

        '''Plot log peak_score of decomposed Data'''
        ax_logpp = plt.subplot(num_fig, 1, 4)
        ax_logpp.plot(X, np.sign(score)*np.log(np.abs(score)+1), linestyle='dashed', c='gray', zorder=1)
        self.plot_hist(peak_score, height=0.3, th=th1, save_fig=True)
        ax_logpp.scatter(peak_ind, peak_score, c='r', s=10, zorder=2)
        ax_logpp.hlines(th1, xmin=start, xmax=end, color='g', label=r'$k\sigma_{\mathbf{P}}+\overline{\mathbf{P}}$')
        ax_logpp.scatter(log_spike_ind, log_spike_score+0.8, c='maroon', marker='v', s = 6**2, label=r'$\mathbf{S}=\{s_i\}$')
        kk = [ax_logpp.spines[key].set_visible(False) for key in ax_logpp.spines.keys() if key not in ['bottom', 'left']]
        plt.legend(bbox_to_anchor=(0.85,0.6), labelspacing=0)
        ax_logpp.spines['bottom'].set_position(('data', 0))
        ax_logpp.set_xticks([])
        ax_logpp.set_xlim([start, end+place])

        '''Plot Spike points of Original Data'''
        ax_spike = plt.subplot(num_fig, 1, 5)
        ax_spike.plot(X, data, c='black', zorder=1)
        ax_spike.scatter(spike_ind, data_all[spike_ind]+25, c='maroon', marker='v', s = 6**2)
        kk = [ax_spike.spines[key].set_visible(False) for key in ax_spike.spines.keys() if key not in ['left']]
        # plt.legend(bbox_to_anchor=(0.85, 0.25))
        ax_spike.set_xticks([])
        ax_spike.set_xlim([start, end+place])
        ax_spike.set_ylim([np.min(data)-120, np.max(data)+40])

        '''Plot SSW of Original Data'''
        ax_spike = plt.subplot(num_fig, 1, 6)
        ax_spike.plot(X, data, c='black', zorder=1)
        ax_spike.scatter(spike_ind, data_all[spike_ind]+25, c='maroon', marker='v', s=6**2)
        kk = [ax_spike.spines[key].set_visible(False) for key in ax_spike.spines.keys() if key not in ['left']]

        s_pair_pred = self._label2Spair(label=label, start=0)
        onsets = s_pair_pred[:,0]
        offsets = s_pair_pred[:,1]

        line_label = True
        for s_pair in s_pair_pred:
            if line_label:
                ax_spike.plot(np.arange(s_pair[0], s_pair[1]), data_all[s_pair[0]:s_pair[1]], c='r', label=r'$f(d_{i\cdot s}:d_{i\cdot e})$')
                line_label = False
            else:
                ax_spike.plot(np.arange(s_pair[0], s_pair[1]), data_all[s_pair[0]:s_pair[1]], c='r')

        if len(onsets) != 0:
            ax_spike.scatter(onsets, data_all[onsets], c='limegreen', marker='>', s=6**2, label=r'$d_{i\cdot s}$')
            ax_spike.scatter(offsets, data_all[offsets], c='black', marker='s', s=6**2, label=r'$d_{i\cdot e}$')
        ax_spike.set_xticks([])
        ax_spike.set_xlim([start, end+place])
        ax_spike.set_ylim([np.min(data)-120, np.max(data)+40])
        plt.legend(loc=3, ncol=3)



        plt.tight_layout()
        if save_fig:
            path = os.path.join('PlotFig', "{}-({}-{})-Prelabel.png".format(self.filename[:-4], time, time+length))
            plt.savefig(path, dpi=500, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

        return None

    def plot_hist(self, data, height, th=None, save_fig=False):

        num_bins = 25
        hist, bins = np.histogram(data, bins=num_bins)
        y = [(bins[i]+bins[i+1])/2 for i, _ in enumerate(bins[:-1])]
        if th is None:
            mode = 'pp'
            colors = ['r']*num_bins
        else:
            mode = 'logpp'
            colors_r = ['r' for bin in y if bin <= th]
            colors_m = ['maroon' for bin in y if bin > th]
            colors = colors_r + colors_m

        fig1 = plt.figure(figsize=[4,8])
        ax = plt.subplot()
        ax.barh(y, hist, height=height, color=colors)
        if th is not None:
            ax.hlines(th, color='g', xmin=0, xmax=np.max(hist), linewidth=8)
        kk = [ax.spines[key].set_visible(False) for key in ax.spines.keys() if key not in ['left']]
        ax.spines['left'].set_linewidth(5)
        ax.set_xticks([])
        ax.set_yticks([])

        # ax.invert_xaxis()

        if save_fig:
            path = os.path.join('PlotFig', "{}-({}-{})-{}.png".format(self.filename[:-4], time, time+length, mode))
            plt.savefig(path, dpi=500, bbox_inches='tight')
            plt.close(fig1)
        else:
            plt.show()

        return None

if __name__ == "__main__":

    dataPath = "C:\\Users\\yyy96\\Documents\\VScodework\\SWI_Quantification_network\\SWIQuant-Part2\\Seg5data\\tData\\01-刘晓逸-1.txt"
    SwiQ = SWIquantify(filepath=dataPath, Spike_width=76, print_log=True)
    # th, min_err = SwiQ.fix_threshold()
    # swi = SwiQ.get_optimal_result(th)
    # label_pred = SwiQ.mask.reshape(-1,1)
    # SwiQ.plot_demo(time=0, length=10, pred=label_pred, save_fig=False)
    # SwiQ.plot_stfft(time=0, length=5)
    time = 30
    length = 5
    # SwiQ.plot_dwt(time=time, length=length, save_fig=True)
    # SwiQ.plot_process(time=time, length=length, num_prc=4, n=9, save_fig=True)
    SwiQ.plot_structure(time=time, length=length, k1=0.6, k2=3, save_fig=True)

    pass