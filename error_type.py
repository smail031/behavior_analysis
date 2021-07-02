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

all_conflicting = []
all_nonconflicting = []

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

        post_trials = 500 #how many trials to analyze following the shift

        prev_rule = np.empty(2, dtype=int) #empty array that will represent the rule prior to switch
        curr_rule = np.empty(2, dtype=int) #empty array that will represent rule after switch
        
        prev_rule[0] = freq_rule[mouse][shift-1]
        prev_rule[1] = left_port[mouse][shift-1]

        curr_rule[0] = freq_rule[mouse][shift+1]
        curr_rule[1] = left_port[mouse][shift+1]

        conflicting = np.zeros(post_trials) #will store whether 500 post-shift trials were conflicting (1) or not (0)

        shift_freq = freq[mouse][int(shift):int(shift) + post_trials]
        shift_multipulse = multipulse[mouse][int(shift):int(shift) + post_trials]

        shift_performance = performance[mouse][int(shift):int(shift) + post_trials]

        if prev_rule[1] == curr_rule[1]:

            conflicting[shift_freq != shift_multipulse] = 1

        elif prev_rule[1] != curr_rule[1]:

            conflicting[shift_freq == shift_multipulse] = 1

        #Regardless of the previous and subsequent rules, there are only two possible combinations of conflicting/non-conflicting trials:
        # 1) 4khz pulsing (1,1) and 1khz solid (0,0) are conflicting, and 4khz solid (1,0) and 1khz pulsing (0,1) are non conflicting.
        # 2) 4khz solid (1,0) and 1khz pulsing (0,1) are conflicting, and 4khz pulsing (1,1) and 1khz solid (0,0) are non-conflicting.
        #Which of these is true depends on whether the port assignment(left_port) changes across the rule switch (which is determined randomly).
        #If it changes, then we get 1); if it stays constant, we get 2).

        correct_conflicting = np.array([])
        correct_nonconflicting = np.array([])

        for trial in range(post_trials):
            
            if conflicting[trial] == 1:
                correct_conflicting = np.append(correct_conflicting, shift_performance[trial])

            else:
                correct_nonconflicting = np.append(correct_nonconflicting, shift_performance[trial])

        all_conflicting.append(correct_conflicting)
        all_nonconflicting.append(correct_nonconflicting)

        conv_conflicting = gaussian_filter1d(correct_conflicting, sigma=9, mode='mirror')
        conv_nonconflicting = gaussian_filter1d(correct_nonconflicting, sigma=9, mode='mirror')

        if len(mouse_list) > 1:
            
            ax.plot(conv_conflicting, color='red', alpha=0.5, lw=0.75)
            ax.plot(conv_nonconflicting, color='blue', alpha=0.5, lw=0.75)

        else:

            ax.plot(conv_conflicting, color='red', label='Conflicting Trials')
            ax.plot(conv_nonconflicting, color='blue', label='Non-Conflicting Trials')

if len(mouse_list) > 1:
    
    conflicting_array = np.empty((len(all_conflicting), post_trials))
    conflicting_array.fill(np.nan)
    nonconflicting_array = np.empty((len(all_nonconflicting), post_trials))
    nonconflicting_array.fill(np.nan)
    
    for shift in range(len(all_conflicting)):
        
        conflicting_array[shift,:len(all_conflicting[shift])] = all_conflicting[shift]
        nonconflicting_array[shift,:len(all_nonconflicting[shift])] = all_nonconflicting[shift]
        
    conflicting_avg = np.nanmean(conflicting_array, axis=0)
    nonconflicting_avg = np.nanmean(nonconflicting_array, axis=0)
    
    conflicting_avg_conv = gaussian_filter1d(conflicting_avg, sigma=5, mode='mirror')
    nonconflicting_avg_conv = gaussian_filter1d(nonconflicting_avg, sigma=5, mode='mirror')
    
    ax.plot(conflicting_avg_conv, color='red', label='Conflicting Trials', lw=2)
    ax.plot(nonconflicting_avg_conv, color='blue', label='Non-Conflicting Trials', lw=2)
    

ax.set_xlabel('Trials After Rule Switch')
ax.set_ylabel('Performance')

ax.set_ylim(0, 1)
plt.legend()


plt.show()

