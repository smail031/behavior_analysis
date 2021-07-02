#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 01 15:14:33 2021

@author: sebastienmaille
"""
#The goal of this analysis script is to compare the rate of errors on conflicting and non-conflicting trials before and after a rule switch.

import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter1d

data_path = '/Volumes/GoogleDrive/Shared drives/Beique Lab/Data/Raspberry PI Data/Sebastien/Dual_Lickport/Mice/' #Directory with all data files organized as mouse/date/mouse_date_block.hdf5

dataset_repo = './datasets/' #path to hdf5 file with experiments to be analyzed

file_search = True
while file_search == True:
    
    fname = input('Enter dataset name (ls:list): ')
    
    if fname == 'ls':
        
        print(sorted(os.listdir(dataset_repo)))
        
    elif f'{fname}.hdf5' in os.listdir(dataset_repo):
        
        print(f'Opening {fname}.hdf5')
        fd = h5py.File(f'{dataset_repo}{fname}.hdf5', 'a') #create hdf5 file
        file_search = False #exit file_search loop
        
    else:
        print('Dataset file not found.')

mouse_list = [i for i in list(fd.keys()) if i != 'Activity log'] #generate list of mice in dataset

freq_rule = np.empty(len(mouse_list), dtype = np.ndarray) #Relevant dimension
left_port = np.empty(len(mouse_list), dtype = np.ndarray) #Port assignment

freq = np.empty(len(mouse_list), dtype=np.ndarray) #will store tone frequency
multipulse = np.empty(len(mouse_list), dtype=np.ndarray) #will store whether pulsing (1) or solid (0) tone

performance = np.empty(len(mouse_list), dtype=np.ndarray) #will store correct/incorrect trials

set_shift = np.empty(len(mouse_list), dtype=np.ndarray) #will whether each trial had a set shift (1) or not (0)
shift_trials =  np.empty(len(mouse_list), dtype=np.ndarray)

all_freq_learning = [] #Nested lists of post-shift learning when shifting to freq rule
all_pulse_learning = [] #Nested lists of post-shift learning when shifting to pulse rule

plt.rcParams['pdf.fonttype'] = 42
fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (8, 6), constrained_layout = True)

for mouse in range(len(mouse_list)):

    mouse_number = mouse_list[mouse]
    mouse_group = fd[mouse_number]
    date_list = list(mouse_group.keys())

    freq_rule[mouse] = np.array([])
    left_port[mouse] = np.array([])

    freq[mouse] = np.array([])
    multipulse[mouse] = np.array([])

    performance[mouse] = np.array([])

    set_shift[mouse] = np.array([])
    
    for date in range(len(date_list)):
        
        date_experiment = date_list[date]

        block_list = mouse_group[date_list[date]]['blocks']

        for block in range(len(block_list)):

            block_number = str(block_list[block])[-2]
        
            data_file = f'{mouse_number}/{date_experiment}/ms{mouse_number}_{date_experiment}_block{block_number}.hdf5'

            print(f'Opening mouse {mouse_number}, {date_experiment}, block {block_number}')
            f = h5py.File(data_path+data_file, 'r') #open HDF5 file
            
            total_trials = len(f['lick_l']['volt']) #get number of trials

            performance_block = np.zeros(total_trials, dtype=int) #will store performance for each trial (1/0) for this block

            for trial in range(total_trials):

                if f['response'][trial] == f['sample_tone']['type'][trial]: #if these are equal, it's a correct trial

                    performance_block[trial] = 1


            freq_rule[mouse] = np.append(freq_rule[mouse], f['rule']['freq_rule'])
            left_port[mouse] = np.append(left_port[mouse], f['rule']['left_port'])

            freq[mouse] = np.append(freq[mouse], f['sample_tone']['freq'])
            multipulse[mouse] = np.append(multipulse[mouse], f['sample_tone']['multipulse'])

            set_shift[mouse] = np.append(set_shift[mouse], np.diff(f['rule']['freq_rule'])) #will be zero everywhere except set shifts (1 or -1)
            set_shift[mouse] = np.append(set_shift[mouse], 0)  #add a zero at the end to match number of trials


            performance[mouse] = np.append(performance[mouse], performance_block) #add performance data for this block


    freq[mouse][freq[mouse] == 4000] = 1 #turn all 4000 khz into 1 
    freq[mouse][freq[mouse] == 1000] = 0 #turn all 1000 khz into 1


    shift_trials[mouse] = np.where(set_shift[mouse] != 0) #set shifts will be where the diff of the rule will not be 0

    for shift in shift_trials[mouse][0]:

        post_trials = 350 #how many trials to analyze following the shift

        prev_rule = np.empty(2, dtype=int) #empty array that will represent the rule prior to switch
        new_rule = np.empty(2, dtype=int) #empty array that will represent rule after switch
        
        prev_rule[0] = freq_rule[mouse][shift-1] #Get value of freq_rule prior to shift
        prev_rule[1] = left_port[mouse][shift-1] #Get value of left_port prior to shift
 
        new_rule[0] = freq_rule[mouse][shift+1] #Get value of freq_rule after shift
        new_rule[1] = left_port[mouse][shift+1] #Get value of left_port after shift

        shift_performance = performance[mouse][int(shift):int(shift) + post_trials]

        if new_rule[0] == 1:

            all_freq_learning.append(shift_performance)

            conv_perform = gaussian_filter1d(shift_performance, sigma=5, mode='mirror')

            if len(mouse_list) > 1:

                ax.plot(conv_perform, color = 'red', alpha=0.5, lw=0.75)

            else:

                ax.plot(conv_perform, color='red', label='Shift to Frequency Rule')
            

        elif new_rule[0] == 0:

            all_pulse_learning.append(shift_performance)
            
            conv_perform = gaussian_filter1d(shift_performance, sigma=5, mode='mirror')

            if len(mouse_list) > 1:

                ax.plot(conv_perform, color = 'blue', alpha=0.5, lw=0.75)

            else:

                ax.plot(conv_perform, color='blue', label='Shift to Pulse Rule')



if len(mouse_list) > 1:
    
    freq_learning_array = np.empty((len(all_freq_learning), post_trials))
    freq_learning_array.fill(np.nan)
    pulse_learning_array = np.empty((len(all_pulse_learning), post_trials))
    pulse_learning_array.fill(np.nan)

    for shift in range(len(all_freq_learning)):

        freq_learning_array[shift,:len(all_freq_learning[shift])] = all_freq_learning[shift]

    for shift in range(len(all_pulse_learning)):

        pulse_learning_array[shift,:len(all_pulse_learning[shift])] = all_pulse_learning[shift]

    freq_avg = np.nanmean(freq_learning_array, axis=0)
    pulse_avg = np.nanmean(pulse_learning_array, axis=0)

    freq_avg_conv = gaussian_filter1d(freq_avg, sigma=5, mode='mirror')
    pulse_avg_conv = gaussian_filter1d(pulse_avg, sigma=5, mode='mirror')
    
    ax.plot(freq_avg_conv, color='red', label=f'Shift to Frequency Rule (n = {len(all_freq_learning)})', lw=2)
    ax.plot(pulse_avg_conv, color='blue', label=f'Shift to Pulse Rule (n = {len(all_pulse_learning)})', lw=2)


ax.set_xlabel('Trials After Rule Switch')
ax.set_ylabel('Performance')

ax.set_ylim(0, 1)
plt.legend()


plt.show()

