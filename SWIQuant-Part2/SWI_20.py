import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import argrelextrema, butter, filtfilt
from utils_re import Dataset

class Analyse_TH(object):

    def __init__(self, path):

        self.path = path
        self.dataset = Dataset(Path=self.path)

    def get_TH(self, data):
        hist, bins = np.histogram(data, bins=100)
        hist = hist/data.shape[0]
        cdf_10 = np.array([np.sum(hist[0:i]) for i in range(len(hist))])
        cdf_90 = np.array([np.sum(hist[-i:]) for i in range(len(hist))])
        id_10 = np.argmin(np.abs(cdf_10-0.1))
        id_90 = np.argmin(np.abs(cdf_90-0.1))
        th_l = bins[id_10+1]
        th_u = bins[-(id_90+1)]

        return th_l, th_u

    def get_all_signal(self):
        Bg_data = []
        # len_all = 0
        for i in range(len(self.dataset)):
            Data = self.dataset[i]
            background = self.get_one_signal(Data)
            # len_all = len_all + len(background)
            # print('Data {}: {} ; [All len: {}]'.format(i+1, len(background), len_all))
            Bg_data.append(background)
        Bg_data = np.concatenate(Bg_data)
        return Bg_data
        
    def get_one_signal(self, Data):
        data = Data.data[::2]
        data = self._band_pass_filter(0.5,25,data)
        label = Data.label[::2]
        background = data.take(np.where(label==1)[0])
        return background
    
    def _band_pass_filter(self, LowHz, HigHz, data):
        data = np.squeeze(data)
        hf = HigHz * 2.0 / 500
        lf = LowHz * 2.0 / 500
        N = 2
        b, a = butter(N, [lf, hf], "bandpass")
        filted_data = filtfilt(b, a, data)
        # filted_data = filted_data.reshape(-1, 1)
        return filted_data



class Swi_20(object):

    def __init__(self, Data, fs=500):

        self.data = Data.data
        self.length = Data.length
        self.label = Data.label
        self.name = Data.name
        self.fs = fs
        self.data_d = self.preprocess(self.data, 20, 0)
        self.peak_data = None
        self.pit_data = None
        self.peak_ck = None
        self.pit_ck = None
        self.region_list = None
        self.flag_list = None
        self.event_dict = None

    def analyse(self, W_g1, W_g2, W_g3, W_g4, th_l, th_u):

        self.peak_data = self.peak_detector(self.data_d, W_g1, W_g2)
        self.pit_data = self.pit_detector(self.data_d, W_g3, W_g4)
        self.peak_ck = self.do_FC(self.peak_data)
        self.pit_ck = self.do_FC(self.pit_data)
        peak_region = self.get_extreme_region(self.peak_ck, 'peak')
        pit_region = self.get_extreme_region(self.pit_ck, 'pit')
        self.region_list, self.flag_list = self.peak_pit_list(peak_region, pit_region)
        self.event_dict = self.discriminate_SSW(self.region_list, self.flag_list, th_l, th_u)
        pred, swi = self.compute_swi()
        # print("Pred SWI:{:.2f}; Label SWI:{:.2f}".format(np.mean(swi), np.mean(self.label)))

        return pred, swi

    def compute_swi(self):

        empty = np.zeros(self.data_d.shape)
        for i, event_slice in self.event_dict.items():
            for event in event_slice:
                empty[event[0]:event[1]] = 1
        
        swi = np.mean(empty)
        return empty, swi


    def identify(self, region, th_l, th_u):
        data = self.data_d[region[0]:region[1]]

        id_ts_can = np.where(data > th_u)[0]
        if len(id_ts_can) == 0:
            return False, None
        else:
            id_ts = id_ts_can[0]
            revers_data = data[::-1]
            id_te_can_up = np.where(revers_data > th_l)[0]
            id_te_can_down = np.where(revers_data < th_u)[0]
            if len(id_te_can_down) == 0 or len(id_te_can_up) == 0:
                return False, None
            else: 
                id_te = min(id_te_can_up[0], id_te_can_down[0])
                id_te = len(data) - id_te

        out_region = np.array([[int(id_ts + region[0]), int(id_te + region[0])]])

        return True, out_region
        

    def discriminate_SSW(self, region_list, flag_list, th_l, th_u):

        standard_template = np.array([-1, -1, 1, -1])

        i = 0
        event_dict = {}
        while True:

            standard_template = np.insert(standard_template, 1, 1)
            print('{} : {}'.format(i, standard_template))
            score = np.correlate(flag_list, standard_template, 'valid')
            qualified_ind = np.where(score == 5+i)[0]
            if len(qualified_ind) == 0:
                break

            event_slice = []
            for ind in qualified_ind:
                region = np.array([region_list[ind, 0], region_list[ind+5+i-1, 1]])
                flag, identified_region = self.identify(region, th_l, th_u)
                if flag:
                    event_slice.append(identified_region)
                    # print('>>From {} to {};'.format(region, identified_region))
            
            if len(event_slice) != 0:
                event_slice = np.concatenate(event_slice, axis=0)
            event_dict[i] = event_slice
            i = i+1
            
        return  event_dict

    def peak_pit_list(self, peak_region, pit_region):

        empty = np.zeros(self.data_d.shape[0])
        for i in peak_region[:,0]:
            empty[int(i)] = 1
        for i in pit_region[:,0]:
            
            empty[int(i)] = 1
        
        inds = np.where(empty==1)[0]

        region_list = []
        flag_list = []
        for ind in inds:

            if pit_region.shape[0] == 0:
                # print('Region {}: Ind {}: {} [{}]'.format(len(flag_list)+1, ind, peak_region[0,:], 'Peak'))
                region_list.append(peak_region[0,:].reshape(1, -1))
                flag_list.append(1)
                peak_region = peak_region[1:, :]
            elif peak_region.shape[0] == 0:
                # print('Region {}: Ind {}: {} [{}]'.format(len(flag_list)+1, ind, pit_region[0,:], 'Pit'))
                region_list.append(pit_region[0,:].reshape(1, -1))
                flag_list.append(-1)
                pit_region = pit_region[1:, :]

            elif pit_region[0,0] == peak_region[0,0]:
                l_peak = peak_region[0,1] - peak_region[0,0]
                l_pit = pit_region[0,1] - pit_region[0,0]
                if l_peak > l_pit:
                    # print('Region {}: Ind {}: {} [{}]'.format(len(flag_list)+1, ind, peak_region[0,:], 'Peak'))
                    region_list.append(peak_region[0,:].reshape(1, -1))
                    flag_list.append(1)
                    peak_region = peak_region[1:, :]
                    pit_region = pit_region[1:, :]
                else:
                    # print('Region {}: Ind {}: {} [{}]'.format(len(flag_list)+1, ind, pit_region[0,:], 'Pit'))
                    region_list.append(pit_region[0,:].reshape(1, -1))
                    flag_list.append(-1)
                    pit_region = pit_region[1:, :]
                    peak_region = peak_region[1:, :]
                
            else:

                if ind == peak_region[0,0]:
                    # print('Region {}: Ind {}: {} [{}]'.format(len(flag_list)+1, ind, peak_region[0,:], 'Peak'))
                    region_list.append(peak_region[0,:].reshape(1, -1))
                    flag_list.append(1)
                    peak_region = peak_region[1:, :]
                elif ind == pit_region[0,0]:
                    # print('Region {}: Ind {}: {} [{}]'.format(len(flag_list)+1, ind, pit_region[0,:], 'Pit'))
                    region_list.append(pit_region[0,:].reshape(1, -1))
                    flag_list.append(-1)
                    pit_region = pit_region[1:, :]
        
        region_list = np.concatenate(region_list, axis=0).astype(np.uint64)
        
        return region_list, flag_list

    def get_extreme_region(self, x, type):

        x[np.abs(x)<0.5] = 0

        if type == 'peak':
            win_x = self.window_slide(x, 12)
            x = np.mean(win_x, axis=1)
            max_points, min_points = self.get_extreme_ponits(x)
        elif type == 'pit':
            min_points, max_points = self.get_extreme_ponits(x)

        flags = np.zeros(x.shape)
        for ma in max_points:
            flags[ma] = 1
        for mi in min_points:
            flags[mi] = -1

        out_max_points = []
        out_min_points = []

        current_max = 0
        current_min = 0
        for ma in max_points:

            cut_flag = flags[ma:]
            i = 1
            while i < len(cut_flag) and cut_flag[i] != -1:
                if cut_flag[i] == 1:
                    break
                i = i + 1
            if i >= len(cut_flag):
                break

            if cut_flag[i] == -1:
                current_max = ma
                current_min = ma + i
                out_max_points.append(current_max)
                out_min_points.append(current_min)
        
        out_max_points = np.array(out_max_points)
        out_min_points = np.array(out_min_points)

        if len(out_max_points) == len(out_min_points):
            extreme_region = np.zeros((len(out_max_points), 2))
            extreme_region[:, 0] = out_max_points
            extreme_region[:, 1] = out_min_points
            # take_out_ind = np.where(extreme_region[:, 1] - extreme_region[:, 0] > 2)[0]
            # extreme_region = extreme_region[take_out_ind]
        else:
            raise ValueError('The max_points num is {}, and the min_points num is {}'.format(len(out_max_points), len(out_min_points)))
        
        return extreme_region
     
    def get_extreme_ponits(self, x):
        return argrelextrema(x, np.greater)[0], argrelextrema(x, np.less)[0]

    def do_FC(self, x, r=6):
        pad = np.zeros(r)
        pad[0] = 1
        x = np.correlate(x, pad, 'full')
        x_window = self.window_slide(x, r, mode='valid')
        out = np.sum(x_window[:,1:], axis=1) - 5 * x_window[:,0]
        return out

    def pit_detector(self, x, W_g1, W_g2):
        g1 = self.g_n(L=W_g1)
        g2 = self.g_n(L=W_g2)
        out = self.PTD(x, g1, g2)
        out[out>0] = 0
        return out

    def peak_detector(self, x, W_g3, W_g4):
        g3 = self.g_n(L=W_g3)
        g4 = self.g_n(L=W_g4)
        out = self.PKD(x, g3, g4)
        out[out<0] = 0
        return out

    def erosion(self, x, g):
        L_operator = len(g)
        g = g.reshape(1, -1)
        x_window = self.window_slide(x, L_operator)
        out = np.min((x_window - g), axis=1)
        return out

    def dilation(self, x, g):
        L_operator = len(g)
        g = g.reshape(1, -1)
        x_window = self.window_slide(x, L_operator)
        out = np.max((x_window + g), axis=1)
        return out
    
    def do_open(self, x, g):
        return self.dilation(self.erosion(x, g), g)
    
    def do_close(self, x, g):
        return self.erosion(self.dilation(x, g), g)
    
    def PTD(self, x, g1, g2):
        return self.do_open(x, g1) - self.do_close(self.do_open(x, g1), g2)

    def PKD(self, x, g3, g4):
        return self.do_close(x, g3) - self.do_open(self.do_close(x, g3), g4)

    def window_slide(self, x, L_operator, stride=1, mode='same'):

        if mode == 'same':
            pad = np.zeros(L_operator)
            pad[int(L_operator/2)] = 1
            data = np.correlate(x, pad, 'full')
        else:
            data = x

        n = int((len(data)-(L_operator-1))/1)
        out = np.zeros((n, L_operator))
        for i in range(L_operator-1):
            out[:,i] = data[i:-(L_operator-i-1)].squeeze()
        out[:,-1] = data[L_operator-1:].squeeze()
        out = out[::stride, :]

        return out

    def g_n(self, L):
        W = int(L/2)
        h = 3
        alpha = 1
        out = np.zeros(2*W+1)
        out[:W+1] = h * (1 - np.exp(-alpha * np.arange(0, W+1)))
        out[W+1:] = h * (1 - np.exp(-alpha * np.arange(W-1, -1, -1)))
        return out
    
    def _band_pass_filter(self, LowHz, HigHz, data):
        data = np.squeeze(data)
        hf = HigHz * 2.0 / self.fs
        lf = LowHz * 2.0 / self.fs
        N = 2
        b, a = butter(N, [lf, hf], "bandpass")
        filted_data = filtfilt(b, a, data)
        # filted_data = filted_data.reshape(-1, 1)
        return filted_data

    def preprocess(self, data, HigHz, W):

        print('Data[{}] Analysis ...'.format(self.name))

        data = data[::int(1000/self.fs)]
        out = self._band_pass_filter(0.5, HigHz, data)
        if W != 0:
            out = self.window_slide(out, W)
            out = np.mean(out, axis=1)
        return out

    def plot_process(self, time, length):

        L = len(self.data_d)
        start, end = self.adjust_window(time, length, L, self.fs)

        fig, axes = plt.subplots(5, 1, figsize=[10,6])
        title = ['Original data', 'peak data', 'pit data', 'peak pit detector', 'event_match']

        for i, data in enumerate([self.data[::int(1000/self.fs)], self.peak_data, self.pit_data, self.data[::int(1000/self.fs)], self.data[::int(1000/self.fs)]]):
            ax = axes[i]
            ax.plot(np.arange(start, end)/self.fs, data[start:end])
            ax.set_title(title[i])
            if i == 0:
                pairs = self._label2Spair(self.label)
                for pair in pairs:
                    if pair[0] >= start and pair[1]<end:
                        ax.plot(np.arange(pair[0], pair[1])/self.fs, data[pair[0]:pair[1]], c='g', linewidth=2)
            if i == 3:
                color = {1:'r', -1:'black'}
                for region, flag in zip(self.region_list, self.flag_list):
                    if region[0] >= start and region[1]<end:
                        ax.plot(np.arange(region[0], region[1])/self.fs, data[region[0]:region[1]], c=color[flag], linewidth=2)
            if i == 4:
                for _, event_slice in self.event_dict.items():
                    for event in event_slice:
                        if event[0] >=start and event[1] < end:
                            ax.plot(np.arange(event[0], event[1])/self.fs, data[event[0]:event[1]], c='r', linewidth=2)

        plt.tight_layout()
        plt.show()
        return None

    def _label2Spair(self, label, start=0):

        d_label = label[1:] - label[:-1]
        inds = list(np.where(d_label != 0)[0]+1)

        if label[0] == 1:
            inds.insert(0, 0)
        if label[-1] == 1:
            inds.append(len(label))

        s_pair = np.array(inds)+start
        s_pair = s_pair.reshape(-1,2)
        return s_pair

    def adjust_window(self, time, length, L, fs):

        time = int(time*fs)
        length = int(length*fs)
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



if __name__ == '__main__':

    # th = Analyse_TH(path='Seg5data\\trainData')
    # bg_data = th.get_all_signal()
    # th_l, th_u = th.get_TH(bg_data)

    th_l = -30
    th_u = 40
    dataset = Dataset(Path='Seg5data\\testData2')
    Data = dataset[8]
    swi = Swi_20(Data)
    pred, pred_swi = swi.analyse(W_g1=20, W_g2=40, W_g3=40, W_g4=14, th_l=th_l, th_u=th_u)
    swi.plot_process(time=0, length=5)


    pass

