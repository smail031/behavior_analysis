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

plt.set_cmap('Set1')
plt.rcParams['pdf.fonttype'] = 42
fig, axs = plt.subplots(nrows=1, ncols=3, figsize = (9, 3.5), constrained_layout = True)

#mouse_list = np.array(['5300','5305','5307','5308'])
mouse_list = np.array(['5300']) #ms 5300 with reverse rule

training_days = np.empty(len(mouse_list), dtype=np.ndarray)

training_days[0] = np.array(['2021-03-16', '2021-03-17', '2021-03-18', '2021-03-19', '2021-03-20', '2021-03-21','2021-03-23','2021-03-24','2021-03-25']) #for ms5300 reverse (pulse) rule
#training_days[0] = np.array(['2021-03-02', '2021-03-03', '2021-03-04', '2021-03-05', '2021-03-08', '2021-03-09']) #for ms5300
#training_days[1] = np.array(['2021-03-02', '2021-03-03', '2021-03-04', '2021-03-05', '2021-03-08', '2021-03-09', '2021-03-10', '2021-03-11', '2021-03-15']) #for ms5305
#training_days[2] = np.array(['2021-03-02', '2021-03-03', '2021-03-04', '2021-03-05', '2021-03-08', '2021-03-09', '2021-03-10', '2021-03-11', '2021-03-15']) #for ms5307
#training_days[3] = np.array(['2021-03-02', '2021-03-03', '2021-03-04', '2021-03-05', '2021-03-08', '2021-03-09', '2021-03-10', '2021-03-11']) #for ms 5308

lick_rates = np.empty(len(mouse_list), dtype=np.ndarray) #will store a_lick rates for each mouse and each experiment
incorr_lick_rates = np.empty(len(mouse_list), dtype=np.ndarray) #will store lick rates on the incorrect port
diff_lick_rates = np.empty(len(mouse_list), dtype=np.ndarray)# will store difference in rate between correct vs incorrect ports

lick_trials = np.empty(len(mouse_list), dtype=np.ndarray) #will store proportion of a_lick trials for each mouse and each experiment
incorr_lick_trials = np.empty(len(mouse_list), dtype=np.ndarray) #will store trials with incorrect a_licks.
diff_lick_trials = np.empty(len(mouse_list), dtype=np.ndarray)

only_corr_trials = np.empty(len(mouse_list), dtype=np.ndarray) #trials with only licks on correct port
only_incorr_trials = np.empty(len(mouse_list), dtype=np.ndarray) #trials with only licks on incorrect port
                              
for mouse in range(len(mouse_list)):

    lick_rates[mouse] = np.empty(len(training_days[mouse]))
    incorr_lick_rates[mouse] = np.empty(len(training_days[mouse]))
    diff_lick_rates[mouse] = np.empty(len(training_days[mouse]))
    
    lick_trials[mouse] = np.empty(len(training_days[mouse]))
    incorr_lick_trials[mouse] = np.empty(len(training_days[mouse]))
    diff_lick_trials[mouse] = np.empty(len(training_days[mouse]))

    only_corr_trials[mouse] = np.empty(len(training_days[mouse]), dtype=int)
    only_incorr_trials[mouse] = np.empty(len(training_days[mouse]), dtype=int)

    for experiment in range(len(training_days[mouse])):

        mouse_number = mouse_list[mouse]
        date_experiment = training_days[mouse][experiment]
        block_number = '1'
        
        file = f'/Volumes/GoogleDrive/Shared drives/Beique Lab/Data/Raspberry PI Data/Sébastien/Dual_Lickport/Mice/{mouse_number}/{date_experiment}/ms{mouse_number}_{date_experiment}_block{block_number}.hdf5'

        f = h5py.File(file, 'r') #open HDF5 file

        #################

        num_trials = len(f['lick_l']['volt']) #get number of trials

        a_licking = np.zeros(num_trials) #will store whether there was anticipatory licking
        a_licking.fill(np.nan)
        incorr_a_licking = np.zeros(num_trials)
        incorr_a_licking.fill(np.nan)   
        
        a_licking_rate = np.zeros(num_trials) #will store a lick rate for each trial
        a_licking_rate.fill(np.nan)
        incorr_a_licking_rate = np.zeros(num_trials) #ant lick rates for incorrect port

        for trial in range(num_trials):

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

            lick_rates[mouse][experiment] = np.nanmean(a_licking_rate)
            incorr_lick_rates[mouse][experiment] = np.nanmean(incorr_a_licking_rate)
            diff_lick_rates[mouse][experiment] = lick_rates[mouse][experiment] - incorr_lick_rates[mouse][experiment]
            
            lick_trials[mouse][experiment] = (int(np.nansum(a_licking))/num_trials) * 100
            incorr_lick_trials[mouse][experiment] = (int(np.nansum(incorr_a_licking))/num_trials) * 100
            diff_lick_trials[mouse][experiment] = lick_trials[mouse][experiment] - incorr_lick_trials[mouse][experiment]

            only_corr_trials[mouse][experiment] = np.sum((a_licking==True) & (incorr_a_licking==False))
            only_incorr_trials[mouse][experiment] = np.sum((a_licking==False) & (incorr_a_licking==True))

    axs[0].plot(lick_trials[mouse], color='steelblue', alpha=0.3, lw=1.5)
    axs[0].plot(incorr_lick_trials[mouse], color='red', alpha=0.3, lw=1.5)
    #axs[0].plot(diff_lick_trials[mouse], color='green', alpha=0.3, lw=1.5)

    axs[1].plot(only_corr_trials[mouse], color='steelblue', alpha=0.3, lw=1.5)
    axs[1].plot(only_incorr_trials[mouse], color='red', alpha=0.3, lw=1.5)
    
    axs[2].plot(lick_rates[mouse], color='steelblue', alpha=0.3, lw=1.5)
    axs[2].plot(incorr_lick_rates[mouse], color='red', alpha=0.3, lw=1.5)
    #axs[2].plot(diff_lick_rates[mouse], color='green', alpha=0.3, lw=1.5)


#----------------- Calculate averages across mice ------------------------

max_exp = np.amax([len(mouse) for mouse in lick_trials])

lick_trials_array = np.empty((len(mouse_list),max_exp), dtype=float)
lick_trials_array.fill(np.nan)
incorr_lick_trials_array = np.empty((len(mouse_list),max_exp), dtype=float)
incorr_lick_trials_array.fill(np.nan)
diff_lick_trials_array = np.empty((len(mouse_list),max_exp), dtype=float)
diff_lick_trials_array.fill(np.nan)

only_corr_array = np.empty((len(mouse_list),max_exp), dtype=float)
only_corr_array.fill(np.nan)
only_incorr_array = np.empty((len(mouse_list),max_exp), dtype=float)
only_incorr_array.fill(np.nan)

lick_rates_array = np.empty((len(mouse_list),max_exp), dtype=float)
lick_rates_array.fill(np.nan)
incorr_lick_rates_array = np.empty((len(mouse_list),max_exp), dtype=float)
incorr_lick_rates_array.fill(np.nan)
diff_lick_rates_array = np.empty((len(mouse_list),max_exp), dtype=float)
diff_lick_rates_array.fill(np.nan)


for mouse in range(len(mouse_list)):
    for experiment in range(len(training_days[mouse])):
        lick_trials_array[mouse,experiment] = lick_trials[mouse][experiment]
        incorr_lick_trials_array[mouse,experiment] = incorr_lick_trials[mouse][experiment]
        diff_lick_trials_array[mouse,experiment] = diff_lick_trials[mouse][experiment]

        only_corr_array[mouse,experiment] = only_corr_trials[mouse][experiment]
        only_incorr_array[mouse,experiment] = only_incorr_trials[mouse][experiment]

        lick_rates_array[mouse,experiment] = lick_rates[mouse][experiment]
        incorr_lick_rates_array[mouse,experiment] = incorr_lick_rates[mouse][experiment]
        diff_lick_rates_array[mouse,experiment] = diff_lick_rates[mouse][experiment]

avg_lick_trials = np.nanmean(lick_trials_array, axis=0)
avg_incorr_lick_trials = np.nanmean(incorr_lick_trials_array, axis=0)
avg_diff_lick_trials = np.nanmean(diff_lick_trials_array, axis=0)

avg_only_corr = np.nanmean(only_corr_array, axis=0)
avg_only_incorr = np.nanmean(only_incorr_array, axis=0)

avg_lick_rates = np.nanmean(lick_rates_array, axis=0)
avg_incorr_lick_rates = np.nanmean(incorr_lick_rates_array, axis=0)
avg_diff_lick_rates = np.nanmean(diff_lick_rates_array, axis=0)

axs[0].plot(avg_lick_trials, color='steelblue', lw=2.5, label='Correct port')
axs[0].plot(avg_incorr_lick_trials, color='red', lw=2.5,label='Incorrect port')
#axs[0].plot(avg_diff_lick_trials, color='green', lw=2.5, label='Correct - Incorrect')

axs[1].plot(avg_only_corr, color='steelblue', lw=2.5, label='Only correct port')
axs[1].plot(avg_only_incorr, color='red', lw=2.5, label='Only incorrect port')

axs[2].plot(avg_lick_rates, color='steelblue', lw=2.5, label='Correct port')
axs[2].plot(avg_incorr_lick_rates, color='red', lw=2.5, label='Incorrect port')
#axs[2].plot(avg_diff_lick_rates, color='green', lw=2.5, label='Correct - Incorrect')

axs[0].set_ylabel('% Trials with anticipatory licking')
axs[0].set_xlabel('Training Day')
axs[0].set_ylim(0,100)
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)
axs[0].legend(frameon=False)

axs[1].set_ylabel('% Trials with anticipatory licking')
axs[1].set_xlabel('Training Day')
axs[1].set_ylim(0,100)
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)
axs[1].legend(frameon=False)

axs[2].set_ylabel('Mean anticipatory lick rate (Hz)')
axs[2].set_xlabel('Training Day')
axs[2].set_ylim(0,None)
axs[2].spines['top'].set_visible(False)
axs[2].spines['right'].set_visible(False)
axs[2].legend(frameon=False)

plt.show()
