import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import core

plt.rcParams['pdf.fonttype'] = 42
fig, ax = plt.subplots(
    nrows=1, ncols=1, figsize = (8, 6), constrained_layout = True)

datasets, dataset_names = core.dataset_search()

for dset in range(len(datasets)):

    for mouse in range(len(datasets[dset].get_weights())):
    
        ax.plot(datasets[dset].weights[mouse])
    
ax.set_xlabel('Training Day')
ax.set_ylabel('Weight')
plt.show()
