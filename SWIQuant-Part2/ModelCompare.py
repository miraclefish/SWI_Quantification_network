import os
import time
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from SWI_09 import Swi_09
from SWI_12 import Swi_12
from SWI_20 import Swi_20
from utils_re import *
from utils import *
from Model_test import ModelEvalu

class ModelCompare(object):

    def __init__(self, DataPath):

        self.path = DataPath
        self.dataset = Dataset(Path=self.path)

    def Swi_Test(self, paper_name, Config):

        Err = []
        Sens = []
        Prec = []
        Fp_min = []
        T = []

        if paper_name == 'ours':
            filelist = os.listdir(self.path)
            filepath = [os.path.join(self.path, file) for file in filelist]
            model = self.get_Model_ours(Config)

        with tqdm(total=len(self.dataset)) as pbar:

            pbar.set_description('Testing on model <{}>:'.format(paper_name))
            for i in range(len(self.dataset)):
                Data = self.dataset[i]

                if paper_name == '09':
                    s = time.time()
                    pred, _, label, length = self.swi_09(Data, Config)
                    e = time.time()
                
                elif paper_name == '12':
                    s = time.time()
                    pred, _, label, length = self.swi_12(Data, Config)
                    e = time.time()

                elif paper_name == '20':
                    s = time.time()
                    pred, _, label, length = self.swi_20(Data, Config)
                    e = time.time()

                elif paper_name == 'ours':
                    s = time.time()
                    data, label, pred = model.test_one(filepath[i])
                    e = time.time()
                    length = len(data)

                t = e - s
                t = t/(length/1000/3600)/60
                T.append(t)
                print('')

                err, sens, prec, fp_min = self.compute_matric(pred, label)

                Err.append(err)
                Sens.append(sens)
                Prec.append(prec)
                Fp_min.append(fp_min)

                pbar.update(1)


        Err.append(np.mean(Err))
        Sens.append(np.mean(Sens))
        Prec.append(np.mean(Prec))
        Fp_min.append(np.mean(Fp_min))
        T.append(np.mean(T))

        resultFile = 'Compare'+'\\'+'ModelCompare.csv'
        columns = [metrics+'_'+paper_name for metrics in ['Err', 'T', 'Sens', 'Prec', 'Fp']]
        result = pd.read_csv(resultFile, sep=',', encoding='ANSI')
        for metric, value in zip(columns, [Err, T, Sens, Prec, Fp_min]):
            result[metric] = value
        result = result.round(2)
        result.to_csv(resultFile, sep=',', encoding='ANSI', index=0)
        return None

    def compute_matric(self, pred, label):
        sens, prec, fp_min= evalu(label=label, pred=pred, iou_th=0.3)
        swi_label = np.mean(label)*100
        swi_pred = np.mean(pred)*100
        err = np.abs(swi_label-swi_pred)
        return err, sens*100, prec*100, fp_min

    def swi_09(self, Data, Config):

        model = Swi_09(Data)
        window = Config['window']

        corr_th = Config['corr_th_1']
        f_th = Config['f_th_1']

        corr, id, data_windows = model.read_step(model.smoothed_data, window, corr_th, f_th, mode='One')
        align_windows = model.align_spike(data_windows, id)

        corr_th = Config['corr_th_2']
        f_th = Config['f_th_2']
        corr, id, data_windows = model.read_step(model.smoothed_data, window, corr_th, f_th, mode='Two')

        pred, swi = model.pred_output(id, window)

        label = model.label[::int(1000/model.fs)]

        return pred, swi, label, model.length

    def swi_12(self, Data, Config):
        
        model = Swi_12(Data)
        window = Config['window']

        corr_th = Config['corr_th_1']
        f_th = Config['f_th_1']

        corr, id, data_windows = model.read_step(model.smoothed_data, window, corr_th, f_th, mode='One')

        align_windows = model.align_spike(data_windows, id)
        n = model.cluster_template(align_windows)

        corr_th = Config['corr_th_2']
        f_th = Config['f_th_2']

        Ids = model.multi_read(window, corr_th, f_th, n)
        id = model.merge_output(Ids)
        pred, swi = model.pred_output(id, window)

        label = model.label[::int(1000/model.fs)]

        return pred, swi, label, model.length

    def swi_20(self, Data, Config):

        g1 = Config['W_g1']
        g2 = Config['W_g2']
        g3 = Config['W_g3']
        g4 = Config['W_g4']
        th_l = Config['th_l']
        th_u = Config['th_u']

        model = Swi_20(Data)
        pred, swi = model.analyse(W_g1=g1, W_g2=g2, W_g3=g3, W_g4=g4, th_l=th_l, th_u=th_u)


        label = model.label[::int(1000/model.fs)]

        return pred, swi, label, model.length

    def get_Model_ours(self, Config):

        layers_num = Config['layers_num']
        input_size = Config['input_size']
        epoch = Config['epoch']
        model_root = Config['model_root']
        mode = Config['mode']

        model_eval = ModelEvalu(layers_num, input_size, epoch, model_root, mode)
        return model_eval

if __name__ == '__main__':

    model_compare = ModelCompare(DataPath='Seg5data\\testData2')

    Config_09 = {'window':300,
                 'corr_th_1':0.5,
                 'corr_th_2':0.75,
                 'f_th_1':0.4,
                 'f_th_2':0.3}
    model_compare.Swi_Test(paper_name='09', Config=Config_09)

    Config_12 = {'window':300,
                 'corr_th_1':0.6,
                 'corr_th_2':0.7,
                 'f_th_1':0.3,
                 'f_th_2':0.5}
    model_compare.Swi_Test(paper_name='12', Config=Config_12)

    Config_20 = {'W_g1':20, 'W_g2':40, 'W_g3':40, 'W_g4':14,
                 'th_l':-30,
                 'th_u':40 }
    model_compare.Swi_Test(paper_name='20', Config=Config_20)

    Config_ours = {'layers_num':4,
                   'input_size':5000,
                   'epoch':280,
                   'model_root':'modelnew',
                   'mode':'U-net'}
    model_compare.Swi_Test(paper_name='ours', Config=Config_ours)

    pass