import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import table


class ModelCheck(object):

    def __init__(self, ResultPath, Dataset, layers, lth=200):
        self.path = ResultPath
        self.dataset = Dataset
        self.layers = layers
        self.lth = lth
        self.path_list = self._get_result_path_list()

    def _get_result_path_list(self):
        path_list = []
        file_list = os.listdir(self.path)
        for file in file_list:
            elements = file.split('-')
            for layer in self.layers:
                if self.dataset == elements[0] and layer == int(elements[2][0]):
                    path_list.append(os.path.join(self.path, file))
                    break
        return path_list

class OneModelAnalysis(object):

    def __init__(self, file):
        self.file = file
        self.Results = self.load_result()
        self.results = None
        self.metric_name = None
        self.col_name = None
        self.plot_ylabel = {'Sens': "Sensitivity ($\%$)",
                            'Pre': "Precision ($\%$)",
                            'Fp_min': "False Positive Rate (min$^{-1}$)",
                            'SwiErr': "SWI Quantification Error ($\%$)",
                            'Derr': "SW and SSW Duration Error (s/min)"}

    def load_result(self):
        Results = pd.read_csv(self.file, sep=',', encoding='ANSI')
        return Results

    def swi_correlation(self):
        self.Swi = self.Results[['swi_label', 'swi_1', 'swi_2', 'swi_pred']]
        corr = self.Swi.corr()
        swi_corr = corr['swi_label'][1:]
        swi_corr.name = 'SWI_corr'
        return swi_corr

    def Analyse(self, metric_name):
        self.metric_name = metric_name
        col_choice = {'Sens': ['S1', 'S2', 'Sens'],
                      'Pre': ['P1', 'P2', 'Prec'],
                      'Fp_min': ['Fp1', 'Fp2', 'Fp_min'],
                      'SwiErr': ['Err1', 'Err2', 'Err'],
                      'Derr': ['Derr1', 'Derr2', 'Derr']}
        self.col_name = col_choice[self.metric_name]
        self.results = self.Results[self.col_name]
        self.results.columns = ['Expert B', 'Expert C', 'SQNN']
        return self.results.mean().round(2), self.results.std().round(2)

    def plot_box(self, save=False):
        fig, ax = plt.subplots(1, 1)
        if self.metric_name == 'SwiErr':
            describe = self.swi_err_describe()
        else:
            describe = np.round(self.results.describe(), 2)
        table(ax, describe, loc='upper right', colWidths=[0.1,0.1,0.1])
        self.results.plot.box(ax=ax)
        plt.ylabel(self.plot_ylabel[self.metric_name])
        plt.grid(linestyle='--', alpha=0.5)
        plt.ylim([0,35])
        if save:
            plt.savefig('fig\\'+os.path.split(self.file)[1][:-3]+'.png', dpi=500, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        return None

    def swi_err_describe(self):
        N = self.results.shape[0]
        mean = self.results.mean().round(2)
        std = self.results.std().round(2)
        min = self.results.min().round(2)
        max = self.results.max().round(2)
        count0_1 = ((self.results <= 1).sum()/N).apply(lambda x: '%.1f%%' % (x))
        count1_5 = ((self.results[self.results > 1] <= 5).sum()/N*100).apply(lambda x: '%.1f%%' % (x))
        count5_10 = ((self.results[self.results > 5] <= 10).sum()/N*100).apply(lambda x: '%.1f%%' % (x))
        count10_ = ((self.results > 10).sum()/N*100).apply(lambda x: '%.1f%%' % (x))
        columns = ['Mean', 'Std', 'Min', 'Max', '0-1%', '1-5%', '5-10%', '>10%']
        describe = pd.DataFrame([mean, std, min, max, count0_1, count1_5, count5_10, count10_], index=columns)
        return describe




if __name__ == '__main__':

    model_check = ModelCheck(ResultPath='TestResult', Dataset='testData1', layers=[2,5])
    # example = OneModelAnalysis(file='TestResult\\testData1-200-2layers-32.csv')
    example = OneModelAnalysis(file='TestResult5000\\testData2-200-2layers-116.csv')
    example.Analyse(metric_name='SwiErr')
    example.plot_box(save=True)
    # example.swi_correlation()
    pass