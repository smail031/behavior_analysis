import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import core
from scipy.ndimage import gaussian_filter1d

plt.rcParams['pdf.fonttype'] = 42
fig, ax = plt.subplots(
    nrows=1, ncols=1, figsize = (8, 6), constrained_layout = True)

datasets, dataset_names = core.dataset_search()

for dset in range(len(datasets)):
    postrev_performance_nested = (
        datasets[dset].get_post_reversal_performance(1000))

    mouse_mean = np.empty(len(datasets[dset].mouse_list), dtype=np.ndarray)
    
    for mouse in range(len(postrev_performance_nested)):
        if len(postrev_performance_nested[mouse]) > 0:
            performance_array = core.as_array(postrev_performance_nested[mouse])
            mean_performance = np.mean(performance_array, axis=0)
            conv_mean_performance = gaussian_filter1d(mean_performance, sigma=15)
            mouse_mean[mouse] = conv_mean_performance
            ax.plot(conv_mean_performance, color='grey',
                    label=datasets[dset].mouse_list[mouse], lw=0.75)

        else:
            mouse_mean[mouse] = np.array([])

    population_array = core.as_array(mouse_mean)
    population_mean = np.nanmean(population_array, axis=0)

    ax.plot(population_mean, color='red', lw=2, label='Population mean')
    

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(0,1)
ax.set_xlabel('Training Day')
ax.set_ylabel('Fraction Correct Trials')
ax.legend()
plt.show()
