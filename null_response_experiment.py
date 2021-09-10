import numpy as np
import matplotlib.pyplot as plt
import core
from scipy import stats

# Initialize the plot
plt.rcParams['pdf.fonttype'] = 42
fig, ax = plt.subplots(
    nrows=1, ncols=1, figsize = (8, 6), constrained_layout = True)

datasets, dataset_names = core.dataset_search() # Get datasets and labels
colors = ['grey', 'red'] # List of colors cor

for dset in range(len(datasets)): # Iterate through all the datasets
    # Get the response latencies for each mouse in the dataset
    null_responses = datasets[dset].get_null_responses()
    
    #for mouse in range(len(null_responses)): # Iterate through the mice
        #ax.plot(null_responses[mouse],
                #color=colors[dset], lw=1, alpha=0.5) # Plot mean response times

    # Put all the data into an array, and average it to get population mean.
    population_array = core.as_array(null_responses)
    population_mean = np.nanmean(population_array, axis=0)
    population_sem = stats.sem(population_array, axis=0, nan_policy='omit')
    x_axis = np.linspace(0, len(population_mean), len(population_mean))
    
    # Plot population mean
    ax.plot(population_mean,
            color=colors[dset], lw=2,
            label=(f'{dataset_names[dset]} n={len(datasets[dset].mouse_list)}'))

    ax.fill_between(x_axis, population_mean+population_sem,
                    population_mean-population_sem,
                    color=colors[dset], alpha=0.4)

# Formatting the plot
ax.spines['top'].set_visible(False) # Remove top and right parts of box
ax.spines['right'].set_visible(False)
#ax.set_ylim(0,2000) # Set Y axis limits
#ax.set_xlim(0,8) # Set X axis limits
ax.set_xlabel('Training Day') # Set Y axis label
ax.set_ylabel('Fraction of Null Responses (ms)') # Set X axis label
ax.legend() # Generate the legend (labels are specified in ax.plot)
plt.show()
