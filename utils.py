import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_data(dataPath, time, length, Sens=0.8):

    data_with_label = pd.read_csv(dataPath, sep='\t', index_col=0)

    L, N = data_with_label.shape

    start, end = adjust_window(time, length, L)

    columns = data_with_label.columns

    data = data_with_label[columns[0]].values

    Atn_path_list = parse_dir(dataPath)

    fig = plt.figure(figsize=[15,10])


    for i in range(N):
        ax = fig.add_subplot(N, 1, i+1)
        ax.plot(np.arange(start, end), data[start:end])
        ax.set_ylim([-100*Sens, 100*Sens])
        ylabel = columns[0]

        if i > 0:
            ylabel = columns[0]+'-'+str(i-1)
            spike_inds, s_pairs = get_spike_point(Atn_path=Atn_path_list[i-1], start=start, end=end)
            ax.vlines(spike_inds, -100*Sens, 100*Sens, color='r')
            for s_pair in s_pairs:
                ax.plot(np.arange(s_pair[0], s_pair[1]), data[s_pair[0]:s_pair[1]], c='r')

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

    plt.subplots_adjust(hspace=0.05)
    plt.show()
    return None

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




if __name__ == "__main__":

    dataPath = "Pre5data\Data\CCW-clip1.txt"
    time = 58
    length = 5
    plot_data(dataPath, time, length)

    pass