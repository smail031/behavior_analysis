import numpy as np
import matplotlib.pyplot as plt
import core
from scipy import stats

# Initialize the plot
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['errorbar.capsize'] = 10
plt.rcParams['font.size'] = 20
fig, axs = plt.subplots(
    nrows=1, ncols=3, figsize = (12, 6), constrained_layout = True)

datasets, dataset_names = core.dataset_search()
colors = ['#782CC8', '#009498']
x_pos = np.arange(0,len(datasets))

first_rev_trials = []
first_rev_means = []
first_rev_sems = []

inter_rev_trials = []
inter_rev_means = []
inter_rev_sems = []

frac_rev = [] # Fraction of mice with at least 1 reversal

for dset in range(len(datasets)): # Iterate through all the datasets
    # Get the trials to criterion for each mouse in the dataset
    switch_trials = datasets[dset].get_switch_trials()
    first_rev_trial = []
    inter_rev_trial = []

    for mouse in range(len(datasets[dset].mouse_list)):

        if len(switch_trials[mouse]) > 0:
            first_rev_trial.append(switch_trials[mouse][0])
            print(datasets[dset].mouse_list[mouse])

        if len(switch_trials[mouse]) > 1:
            inter_rev_trial.append(np.mean(np.diff(switch_trials[mouse])))

    first_rev_trials.append(first_rev_trial)
    first_rev_means.append(np.mean(first_rev_trial))
    
    if len(first_rev_trial) > 1:
        first_rev_sems.append(stats.sem(first_rev_trial, nan_policy='omit'))
            
    else:
        first_rev_sems.append(np.nan)

    inter_rev_trials.append(inter_rev_trial)
    inter_rev_means.append(np.mean(inter_rev_trial))

    if len(inter_rev_trial) > 1:
        inter_rev_sems.append(stats.sem(inter_rev_trial, nan_policy = 'omit'))
    
    else:
        inter_rev_sems.append(np.nan)

    axs[0].scatter(np.full(len(first_rev_trial), x_pos[dset]),first_rev_trial,
                   s=50, edgecolors='k', color=colors[dset], zorder=10)
    axs[1].scatter(np.full(len(inter_rev_trial), x_pos[dset]), inter_rev_trial,
                   s=50, edgecolors='k', color=colors[dset], zorder=10)

    frac_rev.append(len(first_rev_trial)/len(datasets[dset].mouse_list))
    print(len(first_rev_trial))
    print(datasets[dset].mouse_list)

if len(datasets) == 2:
    first_test = stats.ttest_ind(first_rev_trials[0], first_rev_trials[1],
                                 equal_var=False)
    inter_test = stats.ttest_ind(inter_rev_trials[0], inter_rev_trials[1],
                                 equal_var=False)
    print(first_test)
    print(inter_test)
    

axs[0].bar(x_pos, first_rev_means, yerr=first_rev_sems, zorder=0,
           color=colors, tick_label=dataset_names, edgecolor='black')
axs[1].bar(x_pos, inter_rev_means, yerr=inter_rev_sems, zorder=0,
           color=colors, tick_label=dataset_names, edgecolor='black')
axs[2].bar(x_pos, frac_rev, color=colors, tick_label=dataset_names,
           edgecolor='black')


# Formatting axes
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)
axs[0].spines['bottom'].set_visible(False)
axs[0].set_ylabel('Trials to Criterion')
axs[0].set_title('First reversal')

axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)
axs[1].spines['bottom'].set_visible(False)
axs[1].set_ylabel('Trials to Criterion')
axs[1].set_title('Subsequent reversals')

axs[2].spines['top'].set_visible(False)
axs[2].spines['right'].set_visible(False)
axs[2].spines['bottom'].set_visible(False)
axs[2].set_ylabel('Fraction reaching criterion')
axs[2].set_ylim(0,1)

plt.show()
