#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 15:14:33 2021

@author: sebastienmaille
"""
#This script will take a dataset file in .hdf5 format, with references to behavior data
#from dual-port classical conditioning training. From this data, it will measure anticipatory
#licking (i.e. licking between stim presentation and reward delivery) and plot the fraction
#of trials with any anticipatory licking, in addition to the mean rate (Hz) of anticipatory
#licking. This will be computed separately for anticipatory licking on the 'correct' port
#(i.e. the port that will eventually deliver reward), licking on the 'incorrect' port, and
#the difference between the two.

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


mouse_search = True
mouse_list = [i for i in list(fd.keys()) if i != 'Activity log'] #generate list of mice in dataset

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


lick_rates = np.empty(len(mouse_list), dtype=np.ndarray) #will store a_lick rates for each mouse and each block
incorr_lick_rates = np.empty(len(mouse_list), dtype=np.ndarray) #will store lick rates on the incorrect port
diff_lick_rates = np.empty(len(mouse_list), dtype=np.ndarray)# will store difference in rate between correct vs incorrect ports

lick_trials = np.empty(len(mouse_list), dtype=np.ndarray) #will store proportion of a_lick trials for each mouse and each experiment
incorr_lick_trials = np.empty(len(mouse_list), dtype=np.ndarray) #will store trials with incorrect a_licks.
diff_lick_trials = np.empty(len(mouse_list), dtype=np.ndarray)

plt.rcParams['pdf.fonttype'] = 42
fig, axs = plt.subplots(nrows=1, ncols=2, figsize = (8, 6), constrained_layout = True)

for mouse in range(len(mouse_list)):

    mouse_number = mouse_list[mouse]
    mouse_group = fd[mouse_number]
    date_list = list(mouse_group.keys())

    lick_rates[mouse] = np.empty(len(date_list))
    incorr_lick_rates[mouse] = np.empty(len(date_list))
    diff_lick_rates[mouse] = np.empty(len(date_list))
    
    lick_trials[mouse] = np.empty(len(date_list))
    incorr_lick_trials[mouse] = np.empty(len(date_list))
    diff_lick_trials[mouse] = np.empty(len(date_list))
    
    for date in range(len(date_list)):
        
        date_experiment = date_list[date]

        block_list = mouse_group[date_list[date]]['blocks']

        for block in range(len(block_list)):

            block_number = str(block_list[block])[-2]
        
            data_file = f'{mouse_number}/{date_experiment}/ms{mouse_number}_{date_experiment}_block{block_number}.hdf5'

            print(f'Opening mouse {mouse_number}, {date_experiment}, block {block_number}')
            f = h5py.File(data_path+data_file, 'r') #open HDF5 file
            
            total_trials = len(f['lick_l']['volt']) #get number of trials

            a_licking = np.zeros(total_trials) #will store whether there was anticipatory licking
            a_licking.fill(np.nan)
            incorr_a_licking = np.zeros(total_trials)
            incorr_a_licking.fill(np.nan)   
            
            a_licking_rate = np.zeros(total_trials) #will store a lick rate for each trial
            a_licking_rate.fill(np.nan)
            incorr_a_licking_rate = np.zeros(total_trials) #ant lick rates for incorrect port
            incorr_a_licking_rate.fill(np.nan)

            for trial in range(total_trials):

                if 'L' in str(f['sample_tone']['type'][trial]): #if L trial
                    
                    lick_v = f['lick_l']['volt'][trial] #raw voltage traces from left (correct) port
                    incorr_lick_v = f['lick_r']['volt'][trial] #voltage traces from right (incorrect) port
                    
                    rew_time = f['rew_l']['t'][trial] #get L reward delivery time
                    
                elif 'R' in str(f['sample_tone']['type'][trial]): #if R trial
                    
                    lick_v = f['lick_r']['volt'][trial] #raw voltage traces from right (correct) port
                    incorr_lick_v = f['lick_l']['volt'][trial] #raw voltage traces from left (incorrect) port
                    
                    rew_time = f['rew_r']['t'][trial] #get R reward delivery time
                    
                lick_d = np.diff(lick_v) #1st derivative of correct lickport voltage
                incorr_lick_d = np.diff(incorr_lick_v) #derivative of incorrect lickport
                
                licks = np.argwhere(lick_d > 0).flatten() #Get correct lick timestamps
                incorr_licks = np.argwhere(incorr_lick_d > 0).flatten() #Get incorrect lick timestamps
                
                sample_tone_end = f['sample_tone']['end'][trial] #Find sample tone end time
                
                delay = rew_time - sample_tone_end #delay between tone end and rew delivery

                if delay > 250: #only interested in delays greater than 250ms
                    
                    a_licks = licks[(licks > sample_tone_end) & (licks < rew_time)]
                    incorr_a_licks = incorr_licks[(incorr_licks>sample_tone_end)&(incorr_licks<rew_time)]
                    
                    a_licking_rate[trial] = len(a_licks) / (delay/1000) #lick rate in Hz
                    incorr_a_licking_rate[trial] = len(incorr_a_licks) / (delay/1000)
                    
                    
                    if len(a_licks) > 0: #Store whether there were any licks at all
                        a_licking[trial] = True
                        
                    elif len(a_licks) == 0: #False if there were no licks
                        a_licking[trial] = False
                        
                    if len(incorr_a_licks) > 0: #Store whether there were any incorrect licks
                        incorr_a_licking[trial] = True
                        
                    elif len(incorr_a_licks) == 0: #False if there were no licks
                        incorr_a_licking[trial] = False


        lick_rates[mouse][date] = np.nanmean(a_licking_rate)
        incorr_lick_rates[mouse][date] = np.nanmean(incorr_a_licking_rate)
        diff_lick_rates[mouse][date] = lick_rates[mouse][date] - incorr_lick_rates[mouse][date]
        
        lick_trials[mouse][date] = (int(np.nansum(a_licking))/total_trials) * 100
        incorr_lick_trials[mouse][date] = (int(np.nansum(incorr_a_licking))/total_trials) * 100
        diff_lick_trials[mouse][date] = lick_trials[mouse][date] - incorr_lick_trials[mouse][date]
    
    axs[0].plot(lick_trials[mouse], color='green', alpha=0.3, lw=1.5)
    axs[0].plot(incorr_lick_trials[mouse], color='red', alpha=0.3, lw=1.5)
    axs[0].plot(diff_lick_trials[mouse], color='steelblue', alpha=0.3, lw=1.5)
    
    axs[1].plot(lick_rates[mouse], color='green', alpha=0.3, lw=1.5)
    axs[1].plot(incorr_lick_rates[mouse], color='red', alpha=0.3, lw=1.5)
    axs[1].plot(diff_lick_rates[mouse], color='steelblue', alpha=0.3, lw=1.5)

#----------------- Calculate averages across mice ------------------------

max_exp = np.amax([len(mouse) for mouse in lick_trials])

lick_trials_array = np.empty((len(mouse_list),max_exp), dtype=float)
lick_trials_array.fill(np.nan)
incorr_lick_trials_array = np.empty((len(mouse_list),max_exp), dtype=float)
incorr_lick_trials_array.fill(np.nan)
diff_lick_trials_array = np.empty((len(mouse_list),max_exp), dtype=float)
diff_lick_trials_array.fill(np.nan)


lick_rates_array = np.empty((len(mouse_list),max_exp), dtype=float)
lick_rates_array.fill(np.nan)
incorr_lick_rates_array = np.empty((len(mouse_list),max_exp), dtype=float)
incorr_lick_rates_array.fill(np.nan)
diff_lick_rates_array = np.empty((len(mouse_list),max_exp), dtype=float)
diff_lick_rates_array.fill(np.nan)


for mouse in range(len(mouse_list)):

    mouse_number = mouse_list[mouse]
    mouse_group = fd[mouse_number]
    date_list = list(mouse_group.keys())
    
    for experiment in range(len(date_list)):
        lick_trials_array[mouse,experiment] = lick_trials[mouse][experiment]
        incorr_lick_trials_array[mouse,experiment] = incorr_lick_trials[mouse][experiment]
        diff_lick_trials_array[mouse,experiment] = diff_lick_trials[mouse][experiment]

        lick_rates_array[mouse,experiment] = lick_rates[mouse][experiment]
        incorr_lick_rates_array[mouse,experiment] = incorr_lick_rates[mouse][experiment]
        diff_lick_rates_array[mouse,experiment] = diff_lick_rates[mouse][experiment]

avg_lick_trials = np.nanmean(lick_trials_array, axis=0)
avg_incorr_lick_trials = np.nanmean(incorr_lick_trials_array, axis=0)
avg_diff_lick_trials = np.nanmean(diff_lick_trials_array, axis=0)


avg_lick_rates = np.nanmean(lick_rates_array, axis=0)
avg_incorr_lick_rates = np.nanmean(incorr_lick_rates_array, axis=0)
avg_diff_lick_rates = np.nanmean(diff_lick_rates_array, axis=0)

axs[0].plot(avg_lick_trials, color='green', lw=2.5, label='Correct port')
axs[0].plot(avg_incorr_lick_trials, color='red', lw=2.5,label='Incorrect port')
axs[0].plot(avg_diff_lick_trials, color='steelblue', lw=2.5, label='Correct - Incorrect')

axs[1].plot(avg_lick_rates, color='green', lw=2.5, label='Correct port')
axs[1].plot(avg_incorr_lick_rates, color='red', lw=2.5, label='Incorrect port')
axs[1].plot(avg_diff_lick_rates, color='steelblue', lw=2.5, label='Correct - Incorrect')

axs[0].set_ylabel('% Trials with anticipatory licking')
axs[0].set_xlabel('Training Day')
axs[0].set_ylim(0,100)
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)
axs[0].legend()

axs[1].set_ylabel('Mean anticipatory lick rate (Hz)')
axs[1].set_xlabel('Training Day')
axs[1].set_ylim(0,None)
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)
axs[1].legend()

plt.show()
