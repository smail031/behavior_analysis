#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 15:14:33 2021

@author: sebastienmaille
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import time

data_path = '/Volumes/GoogleDrive/Shared drives/Beique Lab/Data/Raspberry PI Data/Sebastien/Dual_Lickport/Mice/'

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

correct_trials = np.empty(len(mouse_list), dtype=np.ndarray) #will store correct (1) and incorrect (0) trials
null_responses = np.empty(len(mouse_list), dtype=np.ndarray) #will store % null responses

bias = np.empty(len(mouse_list), dtype=np.ndarray) #will store L(0) and R(1) responses
response_time = np.empty(len(mouse_list), dtype=np.ndarray) #will store response times

plt.rcParams['pdf.fonttype'] = 42
fig, axs = plt.subplots(nrows=2, ncols=2, figsize = (8, 6), constrained_layout = True)

for mouse in range(len(mouse_list)):

    mouse_number = mouse_list[mouse]
    mouse_group = fd[mouse_number]
    date_list = list(mouse_group.keys())

    correct_trials[mouse] = np.empty(len(date_list))
    null_responses[mouse] = np.empty(len(date_list))
    bias[mouse] = np.empty(len(date_list))
    response_time[mouse] = np.empty(len(date_list))

    
    for date in range(len(date_list)):
        
        date_experiment = date_list[date]

        block_list = mouse_group[date_list[date]]['blocks']

        for block in range(len(block_list)):

            block_number = str(block_list[block])[-2]
        
            data_file = f'{mouse_number}/{date_experiment}/ms{mouse_number}_{date_experiment}_block{block_number}.hdf5'

            print(f'Opening mouse {mouse_number}, {date_experiment}, block {block_number}')
            f = h5py.File(data_path+data_file, 'r') #open HDF5 file
            
            total_trials = len(f['lick_l']['volt']) #get number of correct responses
            
            correct_trials[mouse][date] = 0 #start count of correct responses
            
            null_responses[mouse][date] = 0 #start count of null responses
            
            bias[mouse][date] = 0 #start count of R responses
            
            reward_times = np.empty(total_trials)
            reward_times.fill(np.nan)
            
            for trial in range(total_trials):
                
                if 'N' in str(f['response'][trial]): #check for null responses
                    null_responses[mouse][date] += 1
                    
                elif 'R' in str(f['response'][trial]): #check for null responses
                    bias[mouse][date] += 1
                        
                if np.isnan(f['rew_l']['t'][trial]) == False: #choose trials where reward time isn't NaN
                            
                    #correct_trials[mouse][experiment] += 1 #count this as a correct trial
                    
                    rew_time = f['rew_l']['t'][trial] #find time where reward was delivered
                    sample_tone_end = f['sample_tone']['end'][trial] #Find sample tone end time
                    reward_times[trial] = rew_time - sample_tone_end #Calculate difference
                    
                elif np.isnan(f['rew_r']['t'][trial]) == False:
                            
                    #correct_trials[mouse][experiment] += 1 #count this as a correct trial
                            
                    rew_time = f['rew_r']['t'][trial] #find time where reward was delivered
                    sample_tone_end = f['sample_tone']['end'][trial] #Find sample tone end time
                    reward_times[trial] = rew_time - sample_tone_end
                
                if f['response'][trial] == f['sample_tone']['type'][trial]:
                    #If response matches trial type, it's a correct response
                    correct_trials[mouse][date] += 1 #count this as a correct response

            correct_trials[mouse][date] /= (total_trials)
            
            bias[mouse][date] /= (total_trials - null_responses[mouse][date])
            
            null_responses[mouse][date] /= total_trials                     
            
            response_time[mouse][date] = np.nanmean(reward_times)

    #axs[0,0].plot(correct_trials[mouse]*100, label=mouse_list[mouse])
    axs[0,1].plot(null_responses[mouse]*100, label=mouse_list[mouse])
    axs[1,0].plot(bias[mouse]*100, label=mouse_list[mouse])
    axs[1,1].plot(response_time[mouse], label=mouse_list[mouse])

if len(mouse_list) > 1:
    
    correct_array = np.empty((len(mouse_list), 100))
    correct_array.fill(np.nan)

    for mouse in range(len(mouse_list)):

        correct_array[mouse,:len(correct_trials[mouse])] = correct_trials[mouse]

    correct_avg = np.nanmean(correct_array, axis=0)

    axs[0,0].plot(correct_avg)
    

axs[0,0].set_ylabel('% Correct trials')
axs[0,0].set_xlabel('Training Day')
axs[0,0].set_ylim(0,1)

axs[0,0].set_xlim(0,10)
#axs[0].set_xlim(None, 15)
axs[0,0].spines['top'].set_visible(False)
axs[0,0].spines['right'].set_visible(False)
axs[0,0].legend()

axs[0,1].set_ylabel('% Null responses')
axs[0,1].set_xlabel('Training Day')
axs[0,1].set_ylim(0,100)
#axs[1].set_xlim(None, 15)
axs[0,1].spines['top'].set_visible(False)
axs[0,1].spines['right'].set_visible(False)
#axs[1].legend()

axs[1,0].set_ylabel('Response bias (%R responses)')
axs[1,0].set_xlabel('Training Day')
axs[1,0].set_ylim(0,100)
#axs[2].set_xlim(None, 15)
axs[1,0].spines['top'].set_visible(False)
axs[1,0].spines['right'].set_visible(False)
#axs[2].legend()

axs[1,1].set_ylabel('Response time (ms)')
axs[1,1].set_xlabel('Training Day')
axs[1,1].set_ylim(0,None)
#axs[3].set_xlim(None, 15)
axs[1,1].spines['top'].set_visible(False)
axs[1,1].spines['right'].set_visible(False)
#axs[3].legend()

#fig.suptitle('Pulse rule (5Hz)')

plt.show()
