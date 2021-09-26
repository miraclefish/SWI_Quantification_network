import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False


def plot_data(dataPath, time, length, pred=None, Sens=1.5, save_fig=False):

    _, filename = os.path.split(dataPath)

    data_with_label = pd.read_csv(dataPath, sep='\t', index_col=0)

    L, N = data_with_label.shape

    start, end = adjust_window(time, length, L)

    columns = data_with_label.columns

    data = data_with_label[columns[0]].values

    Atn_path_list = parse_dir(dataPath)

    k = 0
    if pred is not None:
        _, k = pred.shape

    fig = plt.figure(figsize=[15,10])


    for i in range(N):
        ax = fig.add_subplot(N, 1, i+1)
        ax.plot(np.arange(start, end), data[start:end])
        ax.set_ylim([-100*Sens, 100*Sens])
        ylabel = columns[0]

        if i > 0:
            ylabel = columns[0]+'-'+str(i-1)
            spike_inds, s_pairs = get_spike_point(Atn_path=Atn_path_list[i-1], start=start, end=end)
            s_pairs = label2Spair(label=data_with_label[columns[i]].values[start:end], start=start)
            ax.vlines(spike_inds, -100*Sens, 100*Sens, color='r')
            for s_pair in s_pairs:
                ax.plot(np.arange(s_pair[0], s_pair[1]), data[s_pair[0]:s_pair[1]], c='r')
            if k>=i:
                s_pair_preds = label2Spair(label=pred[start:end,i-1], start=start)
                for s_pair_pred in s_pair_preds:
                    ax.plot(np.arange(s_pair_pred[0], s_pair_pred[1]), data[s_pair_pred[0]:s_pair_pred[1]]-15, c='g')


        ax.set_facecolor('none')
        ax.set_yticks([0])
        ax.set_yticklabels([ylabel], rotation=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if i < N-1:
            ax.spines['bottom'].set_visible(False)
            ax.set_xticks([])
        else:
            ax.set_xticks(list(np.arange(start, end, 1000)))
            for idx in spike_inds:
                ax.text(idx+30, -100*Sens+50, 'Spike', fontsize=10, color='red')
        
    plt.suptitle('{} : {}s-{}s'.format(filename, time, time+length))
    plt.subplots_adjust(hspace=0.05)
    if save_fig:
        path = os.path.join('PlotDataset2', "{}-{}.png".format(filename[:-4], time))
        plt.savefig(path, dpi=500, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    return None

def pair2label(s_pairs, L, th):

    new_pairs = []
    for s_pair in s_pairs:
        if s_pair[1] - s_pair[0] >= th:
            new_pairs.append(s_pair)

    label = np.zeros(L)
    for s_pair in new_pairs:
        label[s_pair[0]:s_pair[1]] = 1
    return label

def label2Spair(label, start=0):

    d_label = label[1:] - label[:-1]
    inds = list(np.where(d_label != 0)[0]+1)

    if label[0] == 1:
        inds.insert(0, 0)
    if label[-1] == 1:
        inds.append(len(label))

    s_pair = np.array(inds)+start
    s_pair = s_pair.reshape(-1,2)
    return s_pair

def get_spike_point(Atn_path, start, end):
    Atn = pd.read_csv(Atn_path, sep='\t', index_col=0)
    spike_inds = Atn[Atn.atn=='Spike'].index.to_list()
    spike_inds = [idx for idx in spike_inds if idx<end and idx>start]

    start_inds = Atn[Atn.atn=='S-start'].index.to_list()
    end_inds = Atn[Atn.atn=='S-end'].index.to_list()

    s_pairs = []
    for s, e in zip(start_inds, end_inds):
        if s>=start and e<=end:
            s_pairs.append([s, e])
        if s<start and e>start:
            s_pairs.append([start, e])
        if s<end and e>end:
            s_pairs.append([s, end])
    
    return spike_inds, s_pairs

def adjust_window(time, length, L):

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

def parse_dir(dataPath):

    dir1, file = os.path.split(dataPath)
    root, _ = os.path.split(dir1)

    sep_list = os.listdir(root)
    Atn_path_list = [os.path.join(root, Atn, file) for Atn in sep_list if 'Atn' in Atn]
    
    return Atn_path_list

def iou(anchor, pred):
    if pred[1] <= anchor[0] or pred[0] >= anchor[1]:
        iou = 0
    else:
        left = min(anchor[0], pred[0])
        right = max(anchor[1], pred[1])
        in_left = max(anchor[0], pred[0])
        in_right = min(anchor[1], pred[1])
        iou = (in_right-in_left)/(right-left)
    return iou

def match_pairs(label, pred, iou_th):
    label_pairs = label2Spair(label)
    pred_pairs = label2Spair(pred)

    tp_T = 0
    for label_pair in label_pairs:
        for pred_pair in pred_pairs:
            iou_pair = iou(label_pair, pred_pair)
            if iou_pair >= iou_th:
                tp_T = tp_T + 1
    
    tp_P = 0
    for pred_pair in pred_pairs:
        for label_pair in label_pairs:
            iou_pair = iou(label_pair, pred_pair)
            if iou_pair >= iou_th:
                tp_P = tp_P + 1
                break
    
    return label_pairs.shape[0], pred_pairs.shape[0], tp_T, tp_P

def evalu(label, pred, iou_th):
    num_T, num_P, num_TP, num_FP = match_pairs(label, pred, iou_th)
    Sens = num_TP/(num_T+10e-6)
    Prec = num_FP/(num_P+10e-6)
    Fp_min = (num_P - num_FP)/(len(label)/1000/60)

    return Sens, Prec, Fp_min



if __name__ == "__main__":

    filelist = os.listdir('Pre5data\Data')
    length = 10

    for file in filelist:
        path = os.path.join('Pre5data\\Data', file)
        for i in range(5):
            time = i*10
            plot_data(dataPath=path, time=time, length=length, Sens=1, save_fig=False)
        pass


    # dataPath = "Seg5data\\Data\\2-苏涵-1.txt"
    # time = 10
    # length = 10
    # plot_data(dataPath=dataPath, time=time, length=length, Sens=5)

    pass