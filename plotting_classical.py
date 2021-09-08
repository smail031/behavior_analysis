#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 15:14:33 2021

@author: sebastienmaille
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import time

mouse_number = input('Mouse: ')
date_experiment = input('Date of experiment (yyyy-mm-dd or today): ')
block_number = input('Block number: ')
x_limit = 5000

if date_experiment == 'today':
    date_experiment = time.strftime("%Y-%m-%d", time.localtime(time.time()))

file = ('/Volumes/GoogleDrive/Shared drives/Beique Lab/Data/'
        'Raspberry PI Data/Sebastien/Dual_Lickport/Mice/'
        f'{mouse_number}/{date_experiment}/'
        f'ms{mouse_number}_{date_experiment}_block{block_number}.hdf5')

f = h5py.File(file, 'r') #open HDF5 file

plt.style.use('seaborn-pastel')
fig, ax = plt.subplots(figsize = (8, 6), constrained_layout = True)
#gs = gridspec.GridSpec(nrows = 1, ncols = 2, width_ratios = [1, 1], figure = fig)

#################

num_trials = len(f['lick_l']['volt'])

# Create a raster plot of reward delivery times
plt.eventplot(np.expand_dims(f['rew_l']['t'], 1),
              lw=2.5, linelengths=0.8, color='black')
plt.eventplot(np.expand_dims(f['rew_r']['t'], 1),
              lw=2.5,  linelengths=0.8, color='black')

for trial in range(num_trials):

    # Get lick indices by finding where derivative of lickport voltage is > 0.
    L_lick_d = np.diff(f['lick_l']['volt'][trial])
    R_lick_d = np.diff(f['lick_r']['volt'][trial])
    L_lick_index = np.argwhere(L_lick_d > 0).flatten()
    R_lick_index = np.argwhere(R_lick_d > 0).flatten()

    # Get lick limes by looking at timestamps for associated lick indices
    L_lick_timestamps = np.empty(len(L_lick_index), dtype=float)
    R_lick_timestamps = np.empty(len(R_lick_index), dtype=float)

    for lick in range(len(L_lick_index)):    
        L_lick_timestamps[lick] = (
            f['lick_l']['t'][trial][L_lick_index[lick]])
    
    for lick in range(len(R_lick_index)):
        R_lick_timestamps[lick] = (
            f['lick_r']['t'][trial][R_lick_index[lick]])

    ax.eventplot(L_lick_timestamps,
                 lineoffsets=trial, linelengths=0.8, lw=0.9, color='blue')
    ax.eventplot(R_lick_timestamps,
                 lineoffsets=trial, linelengths=0.8, lw=0.9, color='red')

    # Create a shaded area indicating where sample tones are played.
    sample_tone_on = f['sample_tone']['t'][trial]
    sample_tone_end = f['sample_tone']['end'][trial]

    ax.eventplot([sample_tone_end+250],
                 lineoffsets=trial, linelengths=1, lw=0.9, color='grey')

    if 'L' in str(f['sample_tone']['type'][trial]): #if L trial
        ax.fill_between([sample_tone_on, sample_tone_end],
                        [trial-0.5, trial-0.5], [trial+0.5, trial+0.5],
                        edgecolor='black',lw=0.5, facecolor='blue', alpha=0.3)

    elif 'R' in str(f['sample_tone']['type'][trial]): #if R trial                     
        ax.fill_between([sample_tone_on, sample_tone_end],
                        [trial-0.5, trial-0.5], [trial+0.5, trial+0.5],
                        edgecolor='black',lw=0.5, facecolor='red', alpha=0.3)

ax.set_ylim(-0.5,None)
ax.set_xlim(0,None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylabel('Trials')
ax.set_xlabel('Time (ms)')

ax.set_yticks(np.arange(0, ax.get_ylim()[1], 5))

plt.show()
