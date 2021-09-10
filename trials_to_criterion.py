import numpy as np
import matplotlib.pyplot as plt
import core
from scipy import stats

# Initialize the plot
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['errorbar.capsize'] = 10
fig, ax = plt.subplots(
    nrows=1, ncols=1, figsize = (8, 6), constrained_layout = True)

datasets, dataset_names = core.dataset_search() # Get datasets and labels
colors = ['grey', 'red'] # List of colors cor

means = []
sem = []

for dset in range(len(datasets)): # Iterate through all the datasets
    # Get the trials to criterion for each mouse in the dataset
    trials_to_criterion = datasets[dset].trials_to_criterion()
    means.append(np.mean(trials_to_criterion))

    if len(trials_to_criterion) > 0:
        sem.append(stats.sem(trials_to_criterion, nan_policy='omit'))

    else:
        sem.append(np.nan)
    
x_pos = np.arange(0,len(means))
ax.bar(x_pos, means, yerr=sem,
       color=colors, tick_label=dataset_names, edgecolor='black')

# Formatting the plot
ax.spines['top'].set_visible(False) # Remove top and right parts of box
ax.spines['right'].set_visible(False)
#ax.set_ylim(0,2000) # Set Y axis limit
#ax.set_xlim(0,10)
#ax.set_xlabel('Training Day') # Set Y axis label
ax.set_ylabel('Trials to Criterion') # Set X axis label
#ax.legend() # Generate the legend (labels are specified in ax.plot)
plt.show()
