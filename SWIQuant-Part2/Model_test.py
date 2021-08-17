import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from SignalSegNetOld import SignalSegNet, Basicblock
from torch.utils.data import DataLoader, dataset
from Dataset_test import Dataset_test
from utils import adjust_window, label2Spair, pair2label
from PrelabelEvalu import evalu


def inital_net(model_root, epoch=0):

    net = SignalSegNet(Basicblock, [2,2,2,2,2])
    checkpoint = torch.load(os.path.join(model_root, 'model_epoch_'+str(epoch)+'.pth.tar'), map_location=torch.device('cpu'))
    net.load_state_dict(checkpoint['net'])
    net = net.eval()

    return net

def plot_demo(data, label, time, length, pred=None, save_fig=False, filename=None):

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

def test(net, dataset):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataloader = DataLoader(dataset=dataset, batch_size=8, shuffle=False)

    for i, data in enumerate(dataloader):
        if i >= 1:
            print("!!!!!!!!!!!Wrong dataset!")
            break

        x, label = data

        net = net.to(device)
        x = x.to(device)

        output = net(x=x)
        output = torch.argmax(output, dim=1)
        output = output.squeeze(dim=0)

        pred = output.cpu().data.numpy()
    
    signal = dataset.data.squeeze()
    label = dataset.label[:,0]
    pred = pred.reshape(-1, 1)
    pred = pred[:len(signal)]

    return signal, label, pred



if __name__ == "__main__":

    root = 'Seg5data/testData2'
    filelist = os.listdir(root)

    
    # filelist = [file for i, file in enumerate(filelist) if i not in [4,5,14]]
    # net = inital_net(model_root='model-check', epoch=155)
    # for file in filelist[11:]:
        
    #     DataPath = os.path.join('Seg5data/testData', file)
    #     dataset = Dataset_test(DataPath=DataPath)
    #     data, label, pred = test(net=net, dataset=dataset)
    #     s_pair = label2Spair(pred)
    #     pred_new = pair2label(s_pair, len(data), 250)
    #     for i in range(round(len(data)/10000)):
    #         plot_demo(data=data, label=label, time=i*10, length=10, pred=pred_new, filename=file)
    #     pass
    # pass
    Sens = []
    Prec = []
    Fp_min = []

    Sens1 = []
    Prec1 = []
    Fp_min1 = []

    Sens2 = []
    Prec2 = []
    Fp_min2 = []

    
    swi_label = []
    swi_1 = []
    swi_2 = []
    swi_pred = []

    Err1 = []
    Err2 = []
    Err = []

    Dur_err = []
    Dur_err1 = []
    Dur_err2 = []

    # filelist = [file for i, file in enumerate(filelist) if i not in [4,5,14]]
    net = inital_net(model_root='model-check', epoch=690)
    with tqdm(total=len(filelist)) as pbar:

        pbar.set_description('Testing:')
        for i, file in enumerate(filelist):
            
            DataPath = os.path.join(root, file)
            dataset = Dataset_test(DataPath=DataPath)
            data, label, pred = test(net=net, dataset=dataset)
            s_pair = label2Spair(pred)
            pred_new = pair2label(s_pair, len(data), 200)
            sens, pre, fp_min= evalu(label=label, pred=pred_new, iou_th=0.3)
            Sens.append(sens*100)
            Prec.append(pre*100)
            Fp_min.append(fp_min)

            label1 = dataset.label[:,1]
            label2 = dataset.label[:,2]
            sens1, pre1, fp_min1 = evalu(label=label, pred=label1, iou_th=0.3)
            Sens1.append(sens1*100)
            Prec1.append(pre1*100)
            Fp_min1.append(fp_min1)

            sens2, pre2, fp_min2 = evalu(label=label, pred=label2, iou_th=0.3)
            Sens2.append(sens2*100)
            Prec2.append(pre2*100)
            Fp_min2.append(fp_min2)

            swi00 = np.mean(label)*100
            swi11 = np.mean(label1)*100
            swi22 = np.mean(label2)*100
            swipp = np.mean(pred_new)*100

            swi_label.append(swi00)
            swi_1.append(swi11)
            swi_2.append(swi22)
            swi_pred.append(swipp)

            Err1.append(abs(swi11-swi00))
            Err2.append(abs(swi22-swi00))
            Err.append(abs(swipp-swi00))

            Dur_err.append(abs(sum(pred_new) - sum(label))/1000)
            Dur_err1.append(abs(sum(label1) - sum(label))/1000)
            Dur_err2.append(abs(sum(label2) - sum(label))/1000)

            pbar.update(1)

    tabel = {'Name':filelist, 
            'S1': Sens1, 'P1':Prec1, 'Fp1':Fp_min1, 
            'S2': Sens2, 'P2':Prec2, 'Fp2':Fp_min2,
            'Sens':Sens, 'Prec':Prec, 'Fp_min':Fp_min, 
            'swi_label': swi_label, 'swi_1':swi_1, 'swi_2':swi_2, 'swi_pred':swi_pred,
            'Err1': Err1, 'Err2':Err2, 'Err':Err,
            'Derr1': Dur_err1, 'Derr2':Dur_err2, 'Derr':Dur_err}
    tabel = pd.DataFrame.from_dict(tabel)
    tabel = tabel.round(2)
    tabel.to_csv('TestResult\\testData2-200-690.csv', index=0)

    pass

