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

    performance_nested = datasets[dset].get_performance_experiment()
    performance_array = core.as_array(performance_nested)
    mean_performance = np.nanmean(performance_array, axis=0)
    
    ax.plot(mean_performance, label=dataset_names[dset])

ax.set_ylim(0,1)
ax.set_xlabel('Training Day')
ax.set_ylabel('Fraction Correct Trials')
ax.legend()
plt.show()
