import numpy as np
import matplotlib.pyplot as plt
from utils_re import Dataset

class Swi_09(object):

    def __init__(self, Data, fs=200):

        self.data = Data.data
        self.length = Data.length
        self.label = Data.label
        self.name = Data.name
        self.fs = fs
        self.data_d = self.data[::int(1000/self.fs)]
        self.derivated_data, self.squared_data, self.smoothed_data = self.preprocess(self.data_d)
        self.template = None
    
    def preprocess(self, x):

        print('Data[{}] Analysis ...'.format(self.name))

        x_pad = np.correlate(x, [0,1,0,0], 'full')
        x_window = self._window_slide(x_pad, 4, 1)
        derivated_data = (2*x_window[:,-1]+x_window[:,-2]-x_window[:,1]-2*x_window[:,0])/8
        squared_data = np.sign(derivated_data)*derivated_data**2
        squared_data_pad = np.correlate(squared_data, [0,0,0,0,1,0,0,0,0,0], 'full')
        smoothed_data = np.mean(self._window_slide(squared_data_pad, 10, 1), axis=1)

        return derivated_data, squared_data, smoothed_data

    def read_step(self, x, window, corr_th, f_th, length=60, mode='One'):
        
        template = self.get_template(length, mode)
        _, _, template = self.preprocess(template)

        L = int(window/(1000/self.fs))
        stride = int(L/10)
        id = np.arange(0, len(x)-L+1, stride)
        x_window = self._window_slide(x, L, stride)

        corr = np.array([np.max(np.correlate(xx, template)/np.sqrt(np.sum(xx**2)*np.sum(template**2))) for xx in x_window])
        ind = np.where(corr>corr_th)[0]

        f_x, ind = self.get_data_feature(x_smooth=x_window, ind=ind)
        f_t = self.get_template_feature(template=template, mode=mode)
        print('Read {} cross-correlation > {} events: {}'.format(mode, corr_th, len(ind)))

        f_x, ind = self.out_feature(f_x=f_x, f_t=f_t, th=f_th, ind=ind)
        print('Read {} features > {} events: {}'.format(mode, f_th, len(ind)))

        id = id[ind]
        x_window = x_window[ind]

        return corr, id, x_window

    def pred_output(self, id, window):
        L = int(window/(1000/self.fs))
        output = np.zeros(self.smoothed_data.shape)
        for i in id:
            output[i:i+L] = 1
        swi = np.mean(output)
        # print("Pred SWI:{:.2f}; Label SWI:{:.2f}".format(np.mean(output), np.mean(self.label)))
        return output, swi

    def out_feature(self, f_x, f_t, th, ind):

        f_max_out = []
        f_min_out = []
        f_cur_out = []
        ind_out = []
        for f_max, f_min, f_cur, id in zip(f_x[0], f_x[1], f_x[2], ind):
            if np.abs(f_max) > np.abs(f_t[0]) * th and np.abs(f_min) > np.abs(f_t[1]) * th and f_cur > f_t[2] * th:
                f_max_out.append(f_max)
                f_min_out.append(f_min)
                f_cur_out.append(f_cur)
                ind_out.append(id)
        
        return (f_max_out, f_min_out, f_cur_out), ind_out

    def get_template(self, length, mode='One'):

        if mode == 'One':
            template = self._triangle_template(length=length)
        elif mode == 'Two':
            template = self.template
        return template

    def get_template_feature(self, template, mode='One'):

        if mode == 'One':
            max_t, max_t_id =self._abs_sqrt(np.max(template)), np.argmax(template)
            min_t, min_t_id = self._abs_sqrt(np.min(template)), np.argmin(template)
            cur_t = (max_t - min_t)/(min_t_id - max_t_id)

        elif mode == 'Two':
            max_id = np.argmax(template)
            template_pad = np.pad(template, (1,1))
            template_pad[-1] = 10e6

            d_template = template_pad[1:] - template_pad[:-1]
            dd_template = d_template[1:] * d_template[:-1]

            pp_id = np.where(dd_template<0)[0]
            max_ = np.where(pp_id == max_id)[0]
            min_ = max_ + 1
            
            max_id = pp_id[max_]
            min_id = pp_id[min_]
            max_t = self._abs_sqrt(template[max_id])
            min_t = self._abs_sqrt(template[min_id])
            cur_t = (max_t - min_t)/float(min_id - max_id)

        return (max_t, min_t, cur_t)

    def get_data_feature(self, x_smooth, ind):

        x_smooth_now = x_smooth[ind]

        max_x, max_x_ids = np.max(x_smooth_now, axis=1), np.argmax(x_smooth_now, axis=1)

        x_smooth_pad = np.pad(x_smooth_now, ((0,0),(1,1)))
        x_smooth_pad[:,-1] = 10e6
        d_x_smooth = x_smooth_pad[:,1:] - x_smooth_pad[:,:-1]
        dd_x_smooth = d_x_smooth[:,1:] * d_x_smooth[:,:-1]

        id_pairs = []
        ind_out = []
        i = 0
        for id, dd_x, iind in zip(max_x_ids, dd_x_smooth, ind):
            pp_id = np.where(dd_x < 0)[0]
            max_id = np.where(pp_id == id)[0]
            i = i+1
            if len(max_id) == 0:
                continue
            min_id = max_id + 1
            # print('{}:f_max- f_min = {}'.format(i, self._abs_sqrt(x_smooth_now[i-1,pp_id[max_id]])-self._abs_sqrt(x_smooth_now[i-1,pp_id[min_id]])))
            id_pairs.append(np.array([pp_id[max_id], pp_id[min_id]]).squeeze())
            ind_out.append(iind)
        
        f_max = []
        f_min = []
        f_cur = []
        for id_pair, x_s in zip(id_pairs, x_smooth[ind_out]):
            mmax = self._abs_sqrt(x_s[id_pair[0]])
            mmin = self._abs_sqrt(x_s[id_pair[1]])
            f_max.append(mmax)
            f_min.append(mmin)
            f_cur.append((self._abs_sqrt(x_s[id_pair[0]]) - self._abs_sqrt(x_s[id_pair[1]]))/float(id_pair[1] - id_pair[0]))
        
        return (f_max, f_min, f_cur), ind_out

    def _abs_sqrt(self, x):
        return np.sign(x)*np.sqrt(np.abs(x))

    def align_spike(self, spike_windows, id):

        n, L = spike_windows.shape
        k = 0
        aligned_data = np.zeros(spike_windows.shape)
        for i, (data, iid) in enumerate(zip(spike_windows, id)):
            
            d_ = data[1:]-data[:-1]
            peak_ind = np.where(d_[1:]*d_[:-1]<0)[0]+1
            max_point_ind = np.argmax(data)

            max_peak_ind = np.where(peak_ind == max_point_ind)[0]
            if len(max_peak_ind) == 0:
                k = k+1
                continue
            elif max_peak_ind == 0:
                ind_s = 0
                ind_e = peak_ind[max_peak_ind][0]
            elif max_peak_ind != 0:
                ind_s = peak_ind[max_peak_ind-1][0]
                ind_e = peak_ind[max_peak_ind][0]
            
            idd = np.argmin(np.abs(data[ind_s:ind_e]))
            ind = np.arange(ind_s, ind_e)[idd]
            # l = L - ind
            if iid+L+ind > self.data_d.shape[0]:
                aligned_data[i,:self.data_d.shape[0]-(iid+L+ind)] = self.data_d[iid+ind:iid+L+ind]
            else:
                aligned_data[i,:] = self.data_d[iid+ind:iid+L+ind]

        template = np.sum(aligned_data, 0)/(n-k)
        
        # for data in aligned_data:
        #     if np.sum(data) != 0:
        #         plt.plot(data, c='b', alpha=0.3)
        # plt.plot(np.mean(aligned_data, axis=0), c='r')
        # plt.show()
        # plt.close()

        self.template = template
        
        return aligned_data


    def _window_slide(self, x, length, stride=1):
        stride = int(stride)

        n = int((len(x)-(length-1))/1)
        out = np.zeros((n, length))
        for i in range(length-1):
            out[:,i] = x[i:-(length-i-1)].squeeze()
        out[:,-1] = x[length-1:].squeeze()
        out = out[::stride, :]
        return out

    def _triangle_template(self, length):
        l = int(length/(1000/self.fs))+1
        template = np.zeros(l)
        if l%2 == 0:
            template[:int(l/2)+1] = np.linspace(0,300,int(l/2)+1)
            template[int(l/2)+1:] = np.linspace(0,300,int(l/2)+1)
        else:
            template[:int(l/2)+1] = np.linspace(0,300,int(l/2)+1)
            template[int(l/2):] = np.linspace(300,0,int(l/2)+1)
        return template

    def plot_process(self, time, length, ind=None):

        L = self.length
        start, end = self.adjust_window(time, length, L)

        fig, axes = plt.subplots(4, 1, figsize=[10,6])

        for i, data in enumerate([self.data_d, self.derivated_data, self.squared_data, self.smoothed_data]):
            ax = axes[i]
            ax.plot(np.arange(start, end)/self.fs, data[start:end])
            if ind is not None:
                for i in ind:
                    if i>=start and i+61<end:
                        ax.plot(np.arange(i,i+61)/self.fs, data[i:i+61], 'r')
            
        plt.show()


    def adjust_window(self, time, length, L):

        time = int(time*self.fs)
        length = int(length*self.fs)
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
    dataset = Dataset(Path='Seg5data\\testData2')
    Data = dataset[8]
    swi = Swi_09(Data)
    window = 300

    corr_th = 0.5
    f_th = 0.4

    corr, id, data_windows = swi.read_step(swi.smoothed_data, window, corr_th, f_th, mode='One')
    align_windows = swi.align_spike(data_windows, id)

    corr_th = 0.75
    f_th = 0.3
    corr, id, data_windows = swi.read_step(swi.smoothed_data, window, corr_th, f_th, mode='Two')

    pred, pred_swi = swi.pred_output(id, window)
    
    swi.plot_process(time=0, length=5, ind=id)
    pass
