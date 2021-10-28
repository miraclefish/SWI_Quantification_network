#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
 
def fft_for_wavelet():
     Fs = 50
     p = 2

     b = 3
     x = np.zeros(2*Fs*b)
     w = np.linspace(0, b, Fs*b)
     x[Fs*b:] = w
     x[:Fs*b] = np.linspace(0, b, Fs*b)[::-1]

     y = np.exp(-1/(x**p)-x**p)
     y[:Fs*b] = -y[:Fs*b]

     n = len(y)
     k = np.arange(n)
     T = n/Fs
     frq = k/T
     frq1 = frq[range(int(n/2))]

     YY = np.fft.fft(y)
     Y = YY/n
     Y1 = Y[range(int(n/2))]

     plt.subplot(2,1,1)
     plt.plot(np.linspace(-b,b,2*Fs*b), y)
     plt.subplot(2,1,2)
     plt.plot(frq1, np.abs(Y1))
     
     plt.show()

     return None
 
if __name__ == '__main__':
     
     metrics = ['Sens', 'Prec', 'Fp']
     Deviations = [5,10,15,20,25]

     modes = ["-Fix", "Interval", "+Fix"]

     result_0 = pd.read_csv('PreLabel_0917.csv')

     Results_0 = {}
     Results_0['Sens'] = result_0['Sens'].mean()*100
     Results_0['Prec'] = result_0['Prec'].mean()*100
     Results_0['Fp'] = result_0['Fp_min'].mean()

     Results = {}
     for metric in metrics:
          result_file = 'Pre_' + metric + '.csv'
          result = pd.read_csv(result_file)
          X = np.zeros((len(modes), len(Deviations)))
          for i, D in enumerate(Deviations):
               for j, mode in enumerate(modes):
                    col = str(D)+'_'+mode
                    if metric is 'Fp':
                         X[j,i] = result[col].mean()
                    else:
                         X[j,i] = result[col].mean()*100
          Results[metric] = X

     with sns.axes_style('darkgrid'):
          metric = 'Prec'
          x_tick_label = ['0'] + ['±'+str(D)+'%' for D in Deviations]
          fig, axes = plt.subplots(1,3,figsize=[12,4])
          # axes.size = [15,10]


          X = Results[metric]
          Y = np.zeros((X.shape[0], X.shape[1]+1))
          Y[:,1:] = X
          Y[:,0] = Results_0[metric]
          ax = axes[0]
          ax.plot(np.arange(Y.shape[1]), Y[1,:], marker='^', label='Precision')
          ax.fill_between(np.arange(Y.shape[1]), Y[2,:], Y[0,:], color='r', alpha=0.2)
          ax.hlines(86.67, xmin=0, xmax=Y.shape[1]-1, color='g', label='Expert B', alpha=0.5)
          ax.hlines(90.09, xmin=0, xmax=Y.shape[1]-1, color='y', label='Expert C', alpha=0.5)
          ax.set_ylim([0,100])
          ax.set_xlabel('Weak label deviation')
          ax.set_ylabel(r'Precision ($\%$)')
          ax.set_xticks(np.arange(Y.shape[1]))
          ax.set_xticklabels(x_tick_label)
          ax.legend(loc=4)

          metric='Sens'
          X = Results[metric]
          Y = np.zeros((X.shape[0], X.shape[1]+1))
          Y[:,1:] = X
          Y[:,0] = Results_0[metric]
          ax = axes[1]
          ax.plot(np.arange(Y.shape[1]), Y[1,:], marker='^', label='Sensitivity')
          ax.fill_between(np.arange(Y.shape[1]), Y[2,:], Y[0,:], color='r', alpha=0.2)
          ax.hlines(79.39, xmin=0, xmax=Y.shape[1]-1, color='g', label='Expert B', alpha=0.5)
          ax.hlines(84.34, xmin=0, xmax=Y.shape[1]-1, color='y', label='Expert C', alpha=0.5)
          ax.set_ylim([0,100])
          ax.set_xlabel('Weak label deviation')
          ax.set_ylabel(r'Sensitivity ($\%$)')
          ax.set_xticks(np.arange(Y.shape[1]))
          ax.set_xticklabels(x_tick_label)
          ax.legend(loc=4)

          metric = 'Fp'
          X = Results[metric]
          Y = np.zeros((X.shape[0], X.shape[1]+1))
          Y[:,1:] = X
          Y[:,0] = Results_0[metric]
          ax = axes[2]
          ax.plot(np.arange(Y.shape[1]), Y[1,:], marker='^', label='False positive rate')
          ax.fill_between(np.arange(Y.shape[1]), Y[2,:], Y[0,:], color='r', alpha=0.2)
          ax.hlines(9.001, xmin=0, xmax=Y.shape[1]-1, color='g', label='Expert B', alpha=0.5)
          ax.hlines(7.087, xmin=0, xmax=Y.shape[1]-1, color='y', label='Expert C', alpha=0.5)
          ax.set_ylim([0,50])
          ax.set_xlabel('Weak label deviation')
          ax.set_ylabel(r'False positive rate ($min^{-1}$)')
          # ax.set_yticks(np.linspace(0,100,6))
          ax.set_xticks(np.arange(Y.shape[1]))
          ax.set_xticklabels(x_tick_label)
          # ax.set_yticklabels([str(y)+'%' for y in np.linspace(0,100,6)])
          ax.legend()
     
     plt.tight_layout()
     plt.savefig(os.path.join('PlotFig', 'Robustness.png'), dpi=500, bbox_inches='tight')
     
     # plt.show()
     plt.close()
     pass


