#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 03 15:14:33 2021

@author: sebastienmaille
"""

import h5py
import numpy as np
import os

data_path = '/Volumes/GoogleDrive/Shared drives/Beique Lab/Data/Raspberry PI Data/Sebastien/Dual_Lickport/Mice/' #Directory with all data files organized as mouse/date/mouse_date_block.hdf5



dataset_path = './datasets/' #path to hdf5 file with experiments to be analyzed
print(os.listdir(dataset_path))

dataset = input('Dataset: ') #get name of dataset file


if dataset in os.listdir(dataset_path): #Check that the file exists, so h5py doesn't create a new file
    fd = h5py.File((dataset_path+dataset), 'r') #open hdf5 file with read/write access
    print('yaaaa')
else:
    print('Dataset file not found')

mouse_list = [i for i in list(fd.keys()) if i != 'Activity log'] #generate list of mice in dataset

performance = np.empty(len(mouse_list), dtype=np.ndarray) #will store correct (1) and incorrect (0) trials

conv_performance = np.empty(len(mouse_list), dtype=np.ndarray) #will store correct (1) and incorrect (0) trials

set_shift = np.empty(len(mouse_list), dtype=np.ndarray) #will store trial indices of set shifts

for mouse in range(len(mouse_list)):

    mouse_number = mouse_list[mouse]
    mouse_group = fd[mouse_number]
    date_list = list(mouse_group.keys())

    
    freq = int(input('What should the new freq_rule be?: '))
    port = int(input('What should the new port assignment be?: '))
    
    for date in range(len(date_list)):
        
        date_experiment = date_list[date]

        block_list = mouse_group[date_list[date]]['blocks']

        for block in range(len(block_list)):

            block_number = str(block_list[block])[-2]
        
            data_file = f'{mouse_number}/{date_experiment}/ms{mouse_number}_{date_experiment}_block{block_number}.hdf5'

            print(f'Opening mouse {mouse_number}, {date_experiment}, block {block_number}')

            d = h5py.File(data_path+data_file, 'a') #open HDF5 file

            if 'rule' in d.keys():
                
                print('There is already rule data.')
               
                
            else:

                n_trials = len(d['lick_l']['volt']) #get number of trials

                rule_group = d.create_group('rule')

                freq_array = np.empty(n_trials)
                freq_array.fill(freq)

                port_array = np.empty(n_trials)
                port_array.fill(port)

                freq_ds = rule_group.create_dataset('freq_rule', data=freq_array, dtype=int)
                port_ds = rule_group.create_dataset('left_port', data=port_array, dtype=int)
