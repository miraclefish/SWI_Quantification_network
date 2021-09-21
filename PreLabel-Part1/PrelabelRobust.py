from SWIquantify import SWIquantify
from utils import *
import numpy as np
import os
import pandas as pd
from tqdm import tqdm


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

    root = 'Pre5data/Data'
    filelist = os.listdir(root)
    l = len(filelist)

    Deviations = [5,10,15,20,25]

    modes = ["-Fix", "Interval", "+Fix"]

    Sens_all = np.zeros((len(filelist),len(Deviations)*len(modes)))
    Prec_all = np.zeros((len(filelist),len(Deviations)*len(modes)))
    Fp_all = np.zeros((len(filelist),len(Deviations)*len(modes)))

    Col = []
    k = 0

    for Deviation in Deviations:

        for mode in modes:

            if mode == 'Interval':
                D = [np.random.randint(-Deviation, Deviation) for i in range(1024, 1024+l*2, 2) if not np.random.seed(i)]
            elif mode == '+Fix':
                D = [Deviation for i in range(l)]
            elif mode == '-Fix':
                D = [-Deviation for i in range(l)]
            elif mode == 'Zero':
                D = [0 for i in range(l)]

            Sens = []
            Prec = []
            Fp = []

            ths = []

            with tqdm(total=len(filelist)) as pbar:

                pbar.set_description('[{}___{}]'.format(Deviation, mode))
                for file, d in zip(filelist, D):

                    dataPath = os.path.join(root, file)
                    SwiQ = SWIquantify(filepath=dataPath, Spike_width=76, print_log=True)
                    th, min_err = SwiQ.fix_threshold(D=d)
                    ths.append(th)
                    swi = SwiQ.get_optimal_result(th)

                    label = SwiQ.label[:,0]
                    label_pred = SwiQ.mask.reshape(-1,1)
                    sens, prec, fp_min = evalu(label, label_pred, 0.3)
                    Sens.append(sens)
                    Prec.append(prec)
                    Fp.append(fp_min)

                    pbar.update(1)
                    pass

            Sens_all[:, k] = Sens
            Prec_all[:, k] = Prec
            Fp_all[:, k] = Fp
            Col.append("{}_{}".format(Deviation, mode))
            k = k+1

    tabel_Sens = pd.DataFrame(Sens_all, columns=Col)
    tabel_Sens.to_csv('Pre_Sens1.csv', index=0)
    tabel_Prec = pd.DataFrame(Prec_all, columns=Col)
    tabel_Prec.to_csv('Pre_Prec1.csv', index=0)
    tabel_Fp = pd.DataFrame(Fp_all, columns=Col)
    tabel_Fp.to_csv('Pre_Fp1.csv', index=0)
    pass

