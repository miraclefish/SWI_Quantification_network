from SWIquantify import SWIquantify
from utils import *
import numpy as np
import os
import pandas as pd


if __name__ == "__main__":

    # filelist = os.listdir('Pre5data/Data')
    # S_num = []
    # swi = []
    # for file in filelist:
    #     dataPath = os.path.join('Pre5data/Data', file)
    #     SwiQ = SWIquantify(filepath=dataPath, Spike_width=76, print_log=True)
    #     th, min_err = SwiQ.fix_threshold()
    #     swi = SwiQ.get_optimal_result(th)
    #     label_pred = SwiQ.mask.reshape(-1,1)
    #     L = len(SwiQ.data)
    #     for i in range(round(L/10000)):
    #         SwiQ.plot_demo(time=i*10, length=5, pred=label_pred, save_fig=True)
    #     pass

    # filelist = os.listdir('Pre5data/Data')
    # S_num = []
    # swi = []
    # for file in filelist:

    #     dataPath = os.path.join('Pre5data/Data', file)
    #     SwiQ = SWIquantify(filepath=dataPath, Spike_width=81, print_log=True)

    #     label = SwiQ.label[:,0]
    #     s_pair = label2Spair(label)

    #     S_num.append(s_pair.shape[0])
    #     swi.append(np.mean(label)*100)
    #     pass
    
    # tabel = {'Name':filelist, 'S_num':S_num, 'swi':swi}
    # tabel = pd.DataFrame.from_dict(tabel)
    # tabel.to_csv('Dataset1info.csv', index=0)

    # dataPath = "Pre5data\Data\CCW-clip3.txt"
    # SwiQ = SWIquantify(filepath=dataPath, Spike_width=76, print_log=True)
    # th, min_err = SwiQ.fix_threshold()
    # swi = SwiQ.get_optimal_result(th)

    # label = SwiQ.label[:,0]
    # label_pred = SwiQ.mask.reshape(-1,1)
    # Sens, Prec, Fp_min = evalu(label, label_pred, 0.3)
    # plot_data(dataPath=dataPath, time=0, length=30, pred=label_pred, Sens=5.5)

    filelist = os.listdir('Pre5data/Data')
    Sens = []
    Prec = []
    Fp_min = []

    Sens1 = []
    Prec1 = []
    Fp_min1 = []

    Sens2 = []
    Prec2 = []
    Fp_min2 = []

    ths = []
    Min_err = []
    for file in filelist:

        dataPath = os.path.join('Pre5data/Data', file)
        SwiQ = SWIquantify(filepath=dataPath, Spike_width=76, print_log=True)
        th, min_err = SwiQ.fix_threshold()
        ths.append(th)
        swi = SwiQ.get_optimal_result(th)

        label = SwiQ.label[:,0]
        label_pred = SwiQ.mask.reshape(-1,1)
        sens, prec, fp_min = evalu(label, label_pred, 0.3)
        Sens.append(sens)
        Prec.append(prec)
        Fp_min.append(fp_min)

        sens1, prec1, fp_min1 = evalu(label, SwiQ.label[:,1], 0.3)
        sens2, prec2, fp_min2 = evalu(label, SwiQ.label[:,2], 0.3)

        Sens1.append(sens1)
        Prec1.append(prec1)
        Fp_min1.append(fp_min1)

        Sens2.append(sens2)
        Prec2.append(prec2)
        Fp_min2.append(fp_min2)

        Min_err.append(min_err)

        pass
    
    tabel = {'Name':filelist, 'Sens':Sens, 'Prec':Prec, 'Fp_min':Fp_min, 
            'S1': Sens1, 'P1':Prec1, 'Fp1':Fp_min1, 
            'S2': Sens2, 'P2':Prec2, 'Fp2':Fp_min2,
            'th': ths, 'Min_err':Min_err}
    tabel = pd.DataFrame.from_dict(tabel)
    tabel.to_csv('PreLabel_0917.csv', index=0)
    
    pass

