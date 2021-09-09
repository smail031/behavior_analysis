import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import core
from scipy.ndimage import gaussian_filter1d

plt.rcParams['pdf.fonttype'] = 42
fig, axs = plt.subplots(
    nrows=1, ncols=2, figsize = (8, 6), constrained_layout = True)

datasets, dataset_names = core.dataset_search()

for dset in range(len(datasets)):
    ant_lick = datasets[dset].get_anticipatory_licking()
    
    for mouse in range(len(datasets[dset].mouse_list)):
        
        axs[0].plot(ant_lick[0][mouse], color='green', label='Correct port')
        axs[0].plot(ant_lick[1][mouse], color='red', label='Incorrect port')
        axs[1].plot(ant_lick[2][mouse], color='green', label='Correct port')
        axs[1].plot(ant_lick[3][mouse], color='red', label='Incorrect port')

axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)
axs[0].set_ylim(0,1)
axs[0].set_xlabel('Training Day')
axs[0].set_ylabel('Fraction Trials with Anticipatory Licking')
axs[0].legend()

axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)
axs[1].set_ylim(0,None)
axs[1].set_xlabel('Training Day')
axs[1].set_ylabel('Anticipatory Lick Rate')
axs[1].legend()

plt.show()
