#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 01 15:14:33 2021

@author: sebastienmaille
"""

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

mouse_search = True

while mouse_search  == True: #ask user to choose a mouse from the dataset (or use all)
    
    all_mice = input('Enter mouse number (a:all mice, ls:list mice): ')

    if all_mice == 'a': #if user wants all mice, mouse_list stays the same

        mouse_search = False

    elif all_mice == 'ls': #will list all available mice for the user

        print(mouse_list)

    elif all_mice in mouse_list:

        #if user inputs a ms, mouse_list will only contain that ms
        mouse_list = []
        mouse_list.append(all_mice)
        mouse_search = False

    else:

        print('Not recognized')

performance = np.empty(len(mouse_list), dtype=np.ndarray) #will store correct (1) and incorrect (0) trials

conv_performance = np.empty(len(mouse_list), dtype=np.ndarray) #will store correct (1) and incorrect (0) trials

reversal = np.empty(len(mouse_list), dtype=np.ndarray) #will store trial indices of set shifts



plt.rcParams['pdf.fonttype'] = 42
fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (8, 6), constrained_layout = True)

for mouse in range(len(mouse_list)):

    mouse_number = mouse_list[mouse]
    mouse_group = fd[mouse_number]
    date_list = list(mouse_group.keys())

    performance_mouse = np.array([])
    reversal_mouse = np.array([])
    
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

            reversal_block = np.diff(f['rule']['left_port']) #will be zero everywhere except reversals (1 or -1)
            reversal_block = np.append(reversal_block, 0) #add a zero at the end to match number of trials


            performance_mouse = np.append(performance_mouse, performance_block) #add performance data for this block
            reversal_mouse = np.append(reversal_mouse, reversal_block) #add reversal data for this block
            

    performance[mouse] = performance_mouse
    conv_performance[mouse] = gaussian_filter1d(performance_mouse, sigma=15, mode='mirror')

    correct_trials = np.where(performance[mouse] == 1)
    incorrect_trials = np.where(performance[mouse] == 0)

    reversal[mouse] = np.nonzero(reversal_mouse)
    
    ax.plot(conv_performance[mouse], color='grey') #plot performance
    
    ax.eventplot(reversal[mouse], color='blue', lineoffsets=0.5, linelengths=1) #plot reversals
    ax.eventplot(correct_trials, color = 'green', lineoffsets=1.05, linelengths=0.05, lw=0.15)
    ax.eventplot(incorrect_trials, color = 'red', lineoffsets=1.05, linelengths=0.05, lw=0.15)
    
    ax.set_xlabel('Trials')
    ax.set_ylabel('Performance')

    ax.set_ylim(0, 1.075)
    ax.set_xlim(0, len(conv_performance[mouse]))


plt.show()
