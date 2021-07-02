import h5py
import os
import datetime
import time
import numpy as np

#This script allows a user to generate a .hdf5 file with references to experimental data that will be used for
#later analysis. The experimental data must be stored in a common repository in the format: 
#(mouse) / (date) / ms(mouse)_(date)_block(block).hdf5 with date format yyyy-mm-dd. This code will iterate through
#the data repository and allow the user to select data files to be included. The resulting hdf5 file is structured
#as: [mouse][date][block]

data_repo = '/Volumes/GoogleDrive/Shared drives/Beique Lab/Data/Raspberry PI Data/Sebastien/Dual_Lickport/Mice/'
dataset_repo = './datasets/'

create_edit = input('Create(c) new dataset or edit(e) existing?: ')



if create_edit == 'e':

    file_search = True
    while file_search == True:
        
        fname = input('Enter dataset name (ls:list): ')
        
        if fname == 'ls':
            
            print(sorted(os.listdir(dataset_repo)))
            
        elif f'{fname}.hdf5' in os.listdir(dataset_repo):
            
            print(f'Opening {fname}.hdf5')
            description = input('Describe this edit: ')
            
            f = h5py.File(f'{dataset_repo}{fname}.hdf5', 'a') #create hdf5 file
            file_search = False #exit file_search loop
            
        else:
            print('Dataset file not found.')

elif create_edit == 'c':

    fname = input('Enter new dataset name: ')

    print(f'Creating {fname}.hdf5')
    description = input('Describe this file: ')
    
    f = h5py.File(f'{dataset_repo}{fname}.hdf5', 'a') #create hdf5 file

log = f.require_group('Activity log') #create activity log, if it doesn't already exist

date = datetime.datetime.now() #get the time and date (to store as attribute with description
log.attrs[date.strftime('%c')] = 'Comment: ' + description 


mice = sorted(os.listdir(data_repo))
mice.remove('test')

choose_mice = True
while choose_mice == True:

    mouse = input('Enter mouse number (q:quit, h:help): ')

    if mouse == 'h':
        print('q:quit, ls:list all mice, lsf:list mice in file, rm:delete mode')

    elif mouse == 'q':
        choose_mice = False

    elif mouse == 'ls':
        print(mice)

    elif mouse == 'lsf':
        print([i for i in list(f.keys()) if i != 'Activity log']) #show all subgroups that aren't 'Activity log'

    elif mouse == 'rm':

        mice_file = [i for i in list(f.keys()) if i != 'Activity log']
        del_mode = True
        
        while del_mode == True:
            del_mouse = input('*DELETE MODE* Enter mouse to delete (q:quit, lsf:list mice in file):')

            if del_mouse == 'lsf':
                print(mice_file)

            elif del_mouse == 'q':
                del_mode = False

            elif del_mouse in mice_file:
                del f[del_mouse]
                print(f'Deleted mouse {del_mouse}')
                now = datetime.datetime.now() #get time and date (to store in log)
                datestring = now.strftime('%c') #convert to string
                log.attrs[datestring] = f'Deleted ms{del_mouse}'
                mice_file = [i for i in list(f.keys()) if i != 'Activity log'] #remake the list
                

    elif mouse in mice:

        ms = f.require_group(mouse) #create hdf subgroup for this mouse (if not already created)
        
        dates = sorted(os.listdir(f'{data_repo}/{mouse}/'))
        
        choose_dates = True
        while choose_dates == True:

            date = input('Enter date  yyyy-mm-dd (q:quit, h:help): ')

            if date == 'h':
                print('q:quit, ls:list all dates, lsf:list dates in file, rm:delete mode, rg:range mode')

            elif date == 'q':
                choose_dates= False

            elif date == 'ls':
                print(dates)

            elif date == 'lsf':
                print(list(ms.keys()))

            elif date == 'rm':

                dates_file = list(ms.keys()) #generate a list of dates in the file that can be deleted
                del_mode = True
                
                while del_mode == True:
                    del_date = input('*DELETE MODE* Enter date to delete (q:quit, lsf:list dates in file):')
                    
                    if del_date == 'lsf':
                        print(dates_file)
                        
                    elif del_date == 'q':
                        del_mode = False
                        
                    elif del_date in dates_file:
                        del ms[del_date]
                        print(f'Deleted ms{mouse}_{del_date}')
                        now = datetime.datetime.now() #get time and date (to store in log)
                        datestring = now.strftime('%c') #convert to string
                        log.attrs[datestring] = f'Deleted ms{mouse}_{del_date}'
                        dates_file = list(ms.keys()) #remake the list

            elif date == 'rg':

                print('*RANGE MODE*')
                start_date = input('Enter start date: ')
                end_date = input('Enter end date: ')

                if start_date not in dates or end_date not in dates:

                    print('Dates not recognized.')
                    break

                else:

                    start_index = int(np.where(np.array(dates) == start_date)[0])
                    end_index = int(np.where(np.array(dates) == end_date)[0] + 1) #+1 so that it includes the last date.

                    selected_dates = dates[start_index : end_index]

                    for date in selected_dates:

                         print(date)

                         dt = ms.require_group(date) #create hdf subgroup for this date

                         experiments = sorted(os.listdir(f'{data_repo}/{mouse}/{date}/'))
                         #^filenames in format 'ms(mouse)_(date)_block(block number).hdf5'
                         blocks = [name[-6] for name in experiments] #get block number from filename
                         
                         selected_blocks = []
                         
                         if len(blocks) == 1:
                             selected_blocks.append(blocks[0])
                             
                         else:
                             
                             choose_blocks = True
                             while choose_blocks == True:
                                 
                                 block = input('Enter block number (q:quit, ls:list blocks) :')
                                 
                                 if block == 'q':
                                     choose_blocks = False
                                     
                                 elif block == 'ls':
                                     print(blocks)
                                     
                                 elif block in blocks:
                                     selected_blocks.append(block)
                                     
                                 else:
                                     print('Block number not recognized') #block not recognized
                                    
                         attribute_counter = 0 #will make sure that attributes don't have the same name (overwrites them)
                         if 'blocks' in dt.keys():
                             existing_blocks = dt['blocks'] #get existing blocks
                    

                             for block in existing_blocks:
                                 block_number = str(block)[-2] #block is stored as e.g. "b'1'"
                                 print(f'Deleted existing block {block_number}')
                                 now = datetime.datetime.now() #get time and date (to store in log)
                                 datestring = now.strftime('%c') #store as string
                                 log.attrs[f'{datestring} ({attribute_counter})'] = f'Deleted ms{mouse}_{date}_block{block_number}' #store in activity log
                                 attribute_counter += 1
                                 
                                 del dt['blocks'] #Delete the blocks
                                     
                         for block in selected_blocks:
                             print(f'Added block {block}')
                             now = datetime.datetime.now() #get time and date (to store in log)
                             datestring = now.strftime('%c') #convert to string
                             log.attrs[f'{datestring} ({attribute_counter})'] = f'Added ms{mouse}_{date}_block{block}' #store this in activity log
                             attribute_counter += 1
                            
                         dt.create_dataset('blocks', data = np.array(selected_blocks, dtype='S1'), dtype='S1') #create the dataset with the blocks
                        

            elif date in dates:

                dt = ms.require_group(date) #create hdf subgroup for this date

                experiments = sorted(os.listdir(f'{data_repo}/{mouse}/{date}/'))
                #^filenames in format 'ms(mouse)_(date)_block(block number).hdf5'
                blocks = [name[-6] for name in experiments] #get block number from filename

                selected_blocks = []
                

                if len(blocks) == 1:
                    selected_blocks.append(blocks[0])

                else:
                    
                    choose_blocks = True
                    while choose_blocks == True:

                        block = input('Enter block number (q:quit, ls:list blocks) :')

                        if block == 'q':
                            choose_blocks = False

                        elif block == 'ls':
                            print(blocks)

                        elif block in blocks:
                            selected_blocks.append(block)

                        else:
                            print('Block number not recognized') #block not recognized

                attribute_counter = 0 #will make sure that attributes don't have the same name (overwrites them)
                if 'blocks' in dt.keys():
                    existing_blocks = dt['blocks'] #get existing blocks
                    

                    for block in existing_blocks:
                        block_number = str(block)[-2]
                        print(f'Deleted existing block {block_number}')
                        now = datetime.datetime.now() #get time and date (to store in log)
                        datestring = now.strftime('%c')
                        log.attrs[f'{datestring} ({attribute_counter})'] = f'Deleted ms{mouse}_{date}_block{block_number}'
                        attribute_counter += 1
                    del dt['blocks']
                
                for block in selected_blocks:
                    print(f'Added block {block}')
                    now = datetime.datetime.now() #get time and date (to store in log)
                    datestring = now.strftime('%c')
                    log.attrs[f'{datestring} ({attribute_counter})'] = f'Added ms{mouse}_{date}_block{block}'
                    attribute_counter += 1
                    
                dt.create_dataset('blocks', data = np.array(selected_blocks, dtype='S1'), dtype='S1')

            else:
               print('Date not recognized') #date not recognized

    else:
        print('Mouse number not recognized')
