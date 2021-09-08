import h5py
import numpy as np
import matplotlib.pyplot as plt
import core

plt.rcParams['pdf.fonttype'] = 42
fig, ax = plt.subplots(
    nrows=1, ncols=1, figsize = (8, 6), constrained_layout = True)

datasets, dataset_names = core.dataset_search()

for dset in range(len(datasets)):

    performance_experiment = datasets[dset].get_performance_experiment(
        pre_switch=True, switch_type='r')
    for mouse in range(len(performance_experiment)):
        ax.plot(performance_experiment[mouse])

ax.set_ylim(0,1)
ax.set_xlabel('Training Day')
ax.set_ylabel('Fraction Correct Trials')
plt.show()
