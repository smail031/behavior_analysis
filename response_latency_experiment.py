import numpy as np
import matplotlib.pyplot as plt
import core

# Initialize the plot
plt.rcParams['pdf.fonttype'] = 42
fig, ax = plt.subplots(
    nrows=1, ncols=1, figsize = (8, 6), constrained_layout = True)

datasets, dataset_names = core.dataset_search() # Get datasets and labels
colors = ['grey', 'red'] # List of colors cor

for dset in range(len(datasets)): # Iterate through all the datasets
    # Get the response latencies for each mouse in the dataset
    response_times = datasets[dset].get_response_latency()
    
    for mouse in range(len(response_times)): # Iterate through the mice
        ax.plot(response_times[mouse],
                color=colors[dset], lw=1, alpha=0.5) # Plot mean response times

    # Put all the data into an array, and average it to get population mean.
    population_array = core.as_array(response_times)
    population_mean = np.nanmean(population_array, axis=0)
    # Plot population mean
    ax.plot(population_mean,
            color=colors[dset], lw=2,
            label=(f'{dataset_names[dset]} n={len(datasets[dset].mouse_list)}'))

# Formatting the plot
ax.spines['top'].set_visible(False) # Remove top and right parts of box
ax.spines['right'].set_visible(False)
#ax.set_ylim(0,2000) # Set Y axis limit
ax.set_xlabel('Training Day') # Set Y axis label
ax.set_ylabel('Response Latency (ms)') # Set X axis label
ax.legend() # Generate the legend (labels are specified in ax.plot)
plt.show()
