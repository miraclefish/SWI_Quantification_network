import argparse
from contextlib import ExitStack
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import *
from SWIquantify import SWIquantify

def Awed_with_p(filePath, P):
    
    Score = []
    SwiQ = SWIquantify(filepath=filePath, Spike_width=76, print_log=True)
    data = SwiQ.bandPassData

    Score.append(data)
    for i, p in enumerate(P):
        SwiQ = SWIquantify(filepath=filePath, Spike_width=76, p=p, print_log=True)
        score = SwiQ.Adaptive_Decomposition()
        Score.append(score)
    return Score

def plot_AWED(scores, P, time=0, length=5):

    L = len(scores[0])
    start, end = adjust_window(time, length, L)
    x = np.linspace(start/1000, (end-1)/1000, end-start)

    sns.set_style("darkgrid")
    fig = plt.figure(figsize=[16,7])

    plt.subplot(2, 1, 1)
    plt.plot(x, scores[0][start:end], color='black', label='EEG Signal')
    plt.legend()
    plt.ylabel('$f(t)$', fontsize=14)
    plt.xlabel('Time/s', fontsize=14)

    # i = 1
    plt.subplot(2, 1, 2)
    for p, score in zip(P, scores[1:]):

        # i = i + 1
        # plt.subplot(len(scores),1,i)
        plt.plot(x, score[start:end], label='$p={:.1f}$'.format(p))
        plt.legend()
        plt.ylabel('$S_{p}(t)$', fontsize=14)
        plt.xlabel('Time/s', fontsize=14)

    plt.tight_layout()
    # plt.show()
    plt.savefig('C:\\Users\\yyy96\\Desktop\\AWED_p.png', dpi=500, bbox_inches='tight')
    plt.close(fig)
    return None

def PreLA(filePath):

    SwiQ = SWIquantify(filepath=filePath, Spike_width=76, print_log=True)
    th, min_err = SwiQ.fix_threshold()
    swi = SwiQ.get_optimal_result(th)

    label = SwiQ.label[:,0]
    label_pred = SwiQ.mask.reshape(-1,1)

    ExpertB = SwiQ.label[:,1]
    ExpertC = SwiQ.label[:,2]

    return {'label': label, 'pred': label_pred, 'EB': ExpertB, 'EC': ExpertC}

def plotPR(result):

    iou_ths = np.linspace(0, 1, 50)
    # iou_ths = iou_ths[::-1]
    Sens = np.zeros(len(iou_ths))
    Prec = np.zeros(len(iou_ths))
    SensB = np.zeros(len(iou_ths))
    SensC = np.zeros(len(iou_ths))
    PrecB = np.zeros(len(iou_ths))
    PrecC = np.zeros(len(iou_ths))
    

    for i, iou_th in enumerate(iou_ths):

        sens, prec, fp_min = evalu(result['label'], result['pred'], iou_th)
        sensB, precB, fp_minB = evalu(result['label'], result['EB'], iou_th)
        sensC, precC, fp_minC = evalu(result['label'], result['EC'], iou_th)
        Sens[i] = sens
        Prec[i] = prec
        SensB[i] = sensB
        PrecB[i] = precB
        SensC[i] = sensC
        PrecC[i] = precC

    plt.plot(Sens, Prec, color='black')
    plt.plot(SensB, PrecB, color='blue')
    plt.plot(SensC, PrecC, color='red')
    # plt.axis([0,1,0,1])
    plt.xlabel('Sensitivity')
    plt.ylabel('Precision')
    plt.show()
    return None


if __name__ == '__main__':

    '''
    Revision: PreLA的ROC曲线
    '''
    Path = "C:\\Users\\yyy96\\Documents\\VScodework\\SWI_Quantification_network\\PreLabel-Part1\\Pre5data\\Data"
    filelist = os.listdir(Path)
    i = 2
    filePath = os.path.join(Path, filelist[i])
    result = PreLA(filePath)
    plotPR(result)



    '''
     Revision: 关于AWED为何只取一个p值
    '''
    # Path = "C:\\Users\\yyy96\\Documents\\VScodework\\SWI_Quantification_network\\PreLabel-Part1\\Pre5data\\Data"
    # filelist = os.listdir(Path)
    # i = 20
    # filePath = os.path.join(Path, filelist[i])
    # P = np.linspace(1, 9, 9)
    # output = Awed_with_p(filePath, P)
    # plot_AWED(scores=output, P=P, time=10, length=5.6)

    '''
    终端调用接口
    '''

    # parser = argparse.ArgumentParser(description='Load the data with or without label and plot it.')

    # parser.add_argument('-f', '--file', help='> The path of a data file.')
    # parser.add_argument('-t', '--time', type=int, help='> The start time of the plot figure.')
    # parser.add_argument('-l', '--length', type=int, help='> The length of the signal you would like to plot.')
    # parser.add_argument('-Sens', type=float, default=1, help='> The sensitivity of y-axis, larger Sens means larger range of y-axis')
    # parser.add_argument('-s', '--save', default=False, help='> Save the figure as PNG file or show it in window. \
    #                      Its default value is <False>. If you want to save figure, you need to put a path into --save.')

    # args = parser.parse_args()

    
    # dataPath = args.file
    # time = args.time
    # length = args.length
    # save = args.save
    # Sens = args.Sens
    # plot_data(dataPath=dataPath, time=time, length=length, Sens=Sens, save_fig=save)