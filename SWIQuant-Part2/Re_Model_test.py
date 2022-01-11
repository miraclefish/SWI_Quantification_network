import os
import torch
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

from Re_CNN_Net import SignalSegNet, Basicblock
from torch.utils.data import DataLoader
from Dataset_test import Dataset_test
from utils import *

import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class ModelEvalu(object):

    def __init__(self, layers_num, input_size, epoch, model_root, mode='CNN'):
        super(ModelEvalu, self).__init__()

        self.layers_num = layers_num
        self.input_size = input_size
        self.layers = [2 for i in range(layers_num)]
        self.epoch = epoch
        self.mode = mode
        self.model_root = model_root
        self.Net = SignalSegNet(Basicblock, self.layers, self.mode)
        # self.Netname = '{}-{}-{}'.format(self.layers_num, self.mode_name(), self.epoch)
        self.Netname = '{}-{}-{}'.format(self.layers_num, self.mode_name(), self.input_size)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.initial_net()

    def mode_name(self):
        mode_name = {'U-net': 'U', 'Score': 'S', 'Att': 'A', 'Score+Att': 'S+A', 'CNN': 'C', 'Res': 'R'}
        return mode_name[self.mode]

    def initial_net(self):
        print('initial_net with epoch: ', self.epoch, ' in path (', self.model_root,')')
        checkpoint_path = os.path.join(self.model_root, 'model_epoch_'+str(self.epoch)+'.pth.tar')
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.Net.load_state_dict(checkpoint['net'])
        self.Net = self.Net.to(self.device)
        self.Net.eval()
        return None

    def test_all(self, path):

        filelist = os.listdir(path)

        Sens = []
        Prec = []
        Fp_min = []
        Err = []
        DErr = []
        T = []
        with tqdm(total=len(filelist)) as pbar:

            pbar.set_description('Testing {}:'.format(path))
            for file in filelist:

                file_path = os.path.join(path, file)
                s = time.time()
                data, label, pred = self.test_one(file_path)
                e = time.time()
                t = e-s
                l = len(data)/1000/3600
                t = np.round(t/l/60, 1)
                T.append(t)
                s_pair = label2Spair(pred)
                pred_new = pair2label(s_pair, len(data), 200)
                sens, pre, fp_min= evalu(label=label, pred=pred_new, iou_th=0.3)

                Sens.append(sens*100)
                Prec.append(pre*100)
                Fp_min.append(fp_min)

                swi00 = np.mean(label)*100
                swipp = np.mean(pred_new)*100
                Err.append(abs(swipp-swi00))

                # 每 100s 的 Duration Error
                DErr.append(abs(sum(pred_new) - sum(label))/len(label)*100)

                pbar.update(1)
        
        resultFile = 'TestResult'+str(self.input_size)+'\\'+os.path.split(path)[1]+'-'+str(self.input_size)+'-'
        # resultFile = 'TestResult'+'\\'+os.path.split(path)[1]+'-'
        for metrics, value in zip(['Sens', 'Prec', 'Fp', 'Err', 'DErr'], [Sens, Prec, Fp_min, Err, DErr]):
            resultFileName = resultFile+metrics+'.csv'
            result = pd.read_csv(resultFileName, sep=',', encoding='ANSI')
            result[self.Netname] = value
            result = result.round(2)
            result.to_csv(resultFileName, sep=',', encoding='ANSI', index=0)
        return None

    def test_hyper(self, path):

        # 500-->82
        # 1000-->112
        # 2500-->132
        # 5000-->280
        # 10000-->138
        # 15000-->272
        # 20000-->116
        # 25000-->117
        # 30000-->226
        
        filelist = os.listdir(path)

        input_sizes = [500, 1000, 2500, 5000, 10000, 15000, 20000, 25000, 30000]
        epochs = [82,112,132,280,138,272,116,117,226]
        iou=0.3
        lmins = np.arange(0, 501, 50)

        Sens = np.zeros((len(input_sizes), len(lmins)))
        Prec = np.zeros((len(input_sizes), len(lmins)))
        Fp_min = np.zeros((len(input_sizes), len(lmins)))
        Err = np.zeros((len(input_sizes), len(lmins)))

        Datas = {}
        Labels = {}
        Preds = {}

        for i, input_size in enumerate(input_sizes):
            self.input_size = input_size
            self.epoch = epochs[i]
            self.initial_net()
            with tqdm(total=len(filelist)) as pbar:
                pbar.set_description('Testing {}:'.format(path))
                for file in filelist:

                    file_path = os.path.join(path, file)
                    
                    data, label, pred = self.test_one(file_path)
                    # Labels.append(label)
                    # Preds.append(pred)
                    # Datas.append(data)
                    Labels[file] = label
                    Preds[file] = pred
                    Datas[file] = data

                    pbar.update(1)
            
            for j, lmin in enumerate(lmins):
                sss = []
                ppp = []
                fff = []
                eee = []
                for file in filelist:
                    label, pred, data = Labels[file], Preds[file], Datas[file]
                    s_pair = label2Spair(pred)
                    pred_new = pair2label(s_pairs=s_pair, L=len(data), th=lmin)
                    sens, pre, fp_min= evalu(label=label, pred=pred_new, iou_th=iou)
                    swi00 = np.mean(label)*100
                    swipp = np.mean(pred_new)*100
                    eee.append(np.abs(swi00-swipp))
                    sss.append(sens)
                    ppp.append(pre)
                    fff.append(fp_min)
                Err[i,j] = np.mean(eee)
                Sens[i,j] = np.mean(sss)
                Prec[i,j] = np.mean(ppp)
                Fp_min[i,j] = np.mean(fff)
                print("No.{}:  input_size={}; lmin={}; Finished\n".format(i*j, input_size, lmin))
            pass

        items = ['Sensitivity', 'Precision', 'False positive rate', 'SWI error']
        Results = {}
        for result, item in zip([Sens, Prec, Fp_min, Err], items):
            Results[item] = result

        np.save('Hyper_Results_new_1.npy', Results)
        
        return Results

    def results_initial(self, path):
        # resultFile = 'TestResult'+str(self.input_size)+'\\'+os.path.split(path)[1]+'-'+str(self.input_size)+'-'
        resultFile = 'TestResult'+'\\'+os.path.split(path)[1]+'-'
        for metrics in ['Sens', 'Prec', 'Fp', 'Err', 'DErr']:
            resultFileName = resultFile+metrics+'.csv'
            result = pd.read_csv(resultFileName, sep=',', encoding='ANSI')
            for col in result.columns:
                if col not in ['Name', 'S1', 'S2', 'P1', 'P2', 'Fp1', 'Fp2', 'Err1', 'Err2', 'swi_label', 'Derr1', 'Derr2']:
                    result = result.drop([col], axis=1)
            result.to_csv(resultFileName, sep=',', encoding='ANSI', index=0)
        return None

    def test_one(self, file_path):

        dataset = Dataset_test(file_path, self.input_size)
        dataloader = DataLoader(dataset=dataset, batch_size=300, shuffle=False)

        with torch.no_grad():
            for data in dataloader:
                x, label, score = data

                x = x.to(self.device)

                output = self.Net(x=x)
                output = torch.argmax(output, dim=1)
                output = output.squeeze(dim=0)

                pred = output.cpu().data.numpy()

            signal = dataset.data.squeeze()
            label = dataset.label[:,0]
            pred = pred.reshape(-1, 1)
            pred = pred[:len(signal)]

        return signal, label, pred

    def plot(self, file_path, save_fig=False):

        data, label, pred = self.test_one(file_path)
        s_pair = label2Spair(pred)
        pred_new = pair2label(s_pair, len(data), 200)
        _, file = os.path.split(file_path)
        for i in range(round(len(data)/10000)):
            self.plot_demo(data=data, label=label, time=i*10, length=10, pred=pred_new, filename=file, save_fig=save_fig)
        return None

    def plot_demo(self, data, label, time, length, pred=None, save_fig=False, filename=None):

        L = data.shape[0]
        start, end = adjust_window(time, length, L)
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
                s_pair_label = label2Spair(label[start:end], start)

                line_label = True
                for s_pair in s_pair_label:
                    if line_label:
                        ax.plot(np.arange(s_pair[0], s_pair[1])/1000, data[s_pair[0]:s_pair[1]], c='r', label='Ground_truth')
                        line_label = False
                    else:
                        ax.plot(np.arange(s_pair[0], s_pair[1])/1000, data[s_pair[0]:s_pair[1]], c='r')

                s_pair_pred = label2Spair(label=pred[start:end], start=start)
                onsets = s_pair_pred[:,0]
                offsets = s_pair_pred[:,1]

                if len(onsets) != 0:
                    ax.scatter(onsets/1000, data[onsets], c='limegreen', marker='>', s = 6**2, label='S_start')
                    ax.scatter(offsets/1000, data[offsets], c='black', marker='s', s=6**2, label='S_end')
            
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
            path = os.path.join('Seg5Fig', "{}-{:.2f}-{:.2f}{}".format(filename[:-4], swi_label, swi_pred, '.png'))
            plt.savefig(path, dpi=500, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
        return None

    def plot_hyper(self, Path):

        input_sizes = [500, 1000, 2500, 5000, 10000, 15000, 20000, 25000, 30000]
        Results = np.load(Path, allow_pickle=True).item()

        F_score = 2 * Results['Sensitivity'] * Results['Precision'] / (Results['Sensitivity'] + Results['Precision'])
        Results['F1_score'] = F_score
        fig, axes = plt.subplots(1,3,figsize=[16,5])
        cmap = "RdYlBu"

        for item, id in zip(['F1_score', 'False positive rate', 'SWI error'], np.arange(3)):

            result = Results[item]
            # result = F_score
            ax = axes[id]
            # ax.set_yticks([])
            if item == 'F1_score':
                sns.heatmap(result, annot=True, fmt='.2f', linewidths=0.5, ax=ax, cmap=cmap+'_r', vmax=1, vmin=0.5)
                ax.set_title(item)
            elif item == 'False positive rate':
                sns.heatmap(result, annot=True, fmt='.1f', linewidths=0.5, ax=ax, cmap=cmap, vmin=0, vmax=30)
                ax.set_title(r'False positive rate ($min^{-1}$)')
            elif item == 'SWI error':
                sns.heatmap(result, annot=True, fmt='.1f', linewidths=0.5, ax=ax, cmap=cmap, vmin=0, vmax=10)
                ax.set_title(r'{} ($\%$)'.format(item))
            # ax.set_xticks(np.arange(11))
            # ax.set_yticks(np.arange(9))
            ax.set_xticklabels(np.arange(0,501,50))
            ax.set_yticklabels(input_sizes, rotation=0)
            ax.set_ylabel(r'$L$')
            ax.set_xlabel(r'$l_{min}$')
            

        
        plt.tight_layout()
        plt.savefig(os.path.join('fig', Path[:-4]+'.png'), dpi=500, bbox_inches='tight')
        
        # plt.show()
        plt.close()
        return None

        

if __name__ == "__main__":

    # epoch = 280
    # model_eval = ModelEvalu(layers_num=4, input_size=5000, epoch=epoch, model_root='modelnew', mode='U-net')
    # # Results = model_eval.test_hyper(path='Seg5data\\testData1')
    # model_eval.plot_hyper('Hyper_Results_new_2.npy')

    # input_sizes = [500, 1000, 2500, 5000, 10000, 15000, 20000, 25000, 30000]
    # epochs = [82,112,132,280,138,272,116,117,226]
    input_sizes = [5000]
    epochs = [232]
    for input_size, epoch in zip(input_sizes, epochs):
        model_eval = ModelEvalu(layers_num=4, input_size=input_size, epoch=epoch, model_root='model_5000', mode='Res')
        # model_eval.results_initial(path='Seg5data\\testData1')
        # model_eval.results_initial(path='Seg5data\\testData2')
        
        # model_eval.test_all(path='Seg5data\\testData1')
        model_eval.test_all(path='Seg5data\\testData2')
        # model_eval.plot(file_path='           Seg5data\\testData1\\01-刘晓逸-3.txt')
        # model_eval.plot(file_path='Seg5data\\testD2802\\04-李梓萱-1.txt')

    # pass