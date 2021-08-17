from SWIquantify import SWIquantify
from utils import plot_data
import numpy as np
import os
import pandas as pd


def label2Spair(label):

    d_label = label[1:] - label[:-1]
    inds = list(np.where(d_label != 0)[0]+1)

    if label[0] == 1:
        inds.insert(0, 0)
    elif label[-1] == 1:
        inds.append(len(label))

    s_pair = np.array(inds)
    s_pair = s_pair.reshape(-1,2)
    return s_pair

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
                break
    
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
    Prec = num_FP/num_P
    Fp_min = (num_P - num_FP)/(len(label)/1000/60)

    return Sens, Prec, Fp_min


    


if __name__ == "__main__":

    root = 'Seg5data/tData'
    filelist = os.listdir(root)
    # S_num = []
    Swi = []
    # for file in filelist[7:]:
    #     dataPath = os.path.join(root, file)
    #     SwiQ = SWIquantify(filepath=dataPath, Spike_width=76, print_log=True)
    #     th, min_err = SwiQ.fix_threshold()
    #     swi = SwiQ.get_optimal_result(th)
    #     label_pred = SwiQ.mask.reshape(-1,1)
    #     L = len(SwiQ.data)
    #     for i in range(round(L/10000)):
    #         SwiQ.plot_demo(time=i*10, length=10, pred=label_pred, save_fig=False)
    #     pass

    filelist = os.listdir(root)
    Sens = []
    Prec = []
    Fp_min = []
    Swi_label_0 = []

    for file in filelist:

        dataPath = os.path.join(root, file)
        SwiQ = SWIquantify(filepath=dataPath, Spike_width=76, print_log=True)
        th, min_err = SwiQ.fix_threshold()
        swi = SwiQ.get_optimal_result(th)

        label = SwiQ.label[:,0]
        swi_label_0 = np.mean(label)
        label_pred = SwiQ.mask.reshape(-1,1)
        sens, prec, fp_min = evalu(label, label_pred, 0.3)
        
        Sens.append(sens)
        Prec.append(prec)
        Fp_min.append(fp_min)
        Swi.append(swi)
        Swi_label_0.append(swi_label_0)
        Data = pd.read_csv(dataPath, sep='\t')
        Data['PreAtn'] = label_pred
        Data.to_csv(os.path.join('Seg5data/trainData', file), sep='\t', index=0)

        pass
    
    tabel = {'Name':filelist, 'Sens':Sens, 'Prec':Prec, 'Fp_min':Fp_min, 'Swi':Swi, 'Swi_label':Swi_label_0}
    tabel = pd.DataFrame.from_dict(tabel)
    tabel.to_csv('Seg5data/PreLabelResult0-4.csv', index=0)
    
    pass

