import h5py
import numpy as np
import os

data_repo_path = '/Volumes/GoogleDrive/Shared drives/Beique Lab/Data/Raspberry PI Data/Sebastien/Dual_Lickport/Mice/'
dataset_repo_path = './datasets/'

class Experiment():
    '''
    A base class to work with a .hdf5 file containing data recorded in a single
    behavior experiment. Is instantiated by an object of class Mouse.

    Attributes:
    -----------
    self.mouse: str
        Number corresponding to this mouse's ID.
    self.date: str
        Date on which the experiment took place.
    self.block: str
        Number corresponding to the experimental block for that mouse/date.
    self.data: h5py.File object
        The h5py object containing all experimental data.
    self.num_trials: int
        The number of trials in the corresponding experiment.

    Methods:
    --------
    get_weight():
        Gets the weight of the mouse recorded on that day from the 
        attributes in self.data.
    get_rule_switch(switch_type):
        Indicates trials in which there was a rule switch. Will look for either
        a reversal or a set shift.
    get_lick_timestamps(lickport_voltage, lickport_timestamps):
        Returns the trial-relative timestamps for each lick contact on a
        given lickport.
    get_anticipatory_licking():

    get_performance():
        Generates a vector self.performance that indicates, for each trial,
        whether the mouse chose correctly(1) or incorrectly(0).
    '''

    
    def __init__(self, mouse, date, block):
        '''
        Arguments:
        ----------
        mouse: str
            Number corresponding to this mouse's ID.
        date: str
            Date on which the experiment took place.
        block: str
            Number corresponding to the experimental block for that mouse/date.
        '''
        
        self.mouse = mouse
        self.date = date
        self.block = block

        data_path = (f'{self.mouse}/{self.date}/'
            f'ms{self.mouse}_{self.date}_block{self.block}.hdf5')
        full_path = data_repo_path + data_path

        self.data = h5py.File(full_path, 'r')
        print(f'Opening mouse {self.mouse}, {self.date},'
            f'block {self.block}')

        self.num_trials = len(self.data['lick_l']['volt'])

    def get_weight(self):
        '''
        Returns the weight of the mouse in grams, prior to this experiment.

        Returns:
        --------
        weight: float
            weight of the mouse in grams
        '''
        weight = self.data.attrs['mouse_weight']

        return weight
    

    def get_rule_switch(self, switch_type):
        '''
        Find trials in which there was a reversal or a set shift. A reversal
        (in reversal learning paradigms) leads to a change in port assignment
        (L cue -> R and vice versa), which is stored in rule/left_port. 
        A set shift (in attentional set shifting paradigms) leads to a change
        in the relevant/instrutive dimension (e.g. frequency rule -> location
        rule), which is stored in rule/freq_rule. The method can handle
        experiments with multiple rule switches, but this should never occurr.

        Arguments:
        ----------
        switch_type: str
            Indicates whether the method should look for a reversal ('r') or
            a set shift ('s').

        Returns:
        --------
        rule_switch: 1D numpy array
            Indicates, on each trial, whether there was a rule switch(1)
            or not(0).
        '''

        if switch_type == 'r':       
            rule_switch = np.diff(self.data['rule']['left_port'])

        elif switch_type == 's':
            rule_switch = np.diff(self.data['rule']['freq_rule'])
            
        rule_switch = np.append(rule_switch, 0)

        return rule_switch

    def get_lick_timestamps(self, lickport_voltage, lickport_timestamps):
        '''
        Get timestamps for licking contacts between mouse and lickport.
        
        Arguments:
        ----------
        lickport_voltage: 1D array 
            A vector of 'voltage' recorded from the lickport circuit for one
            of the lickports
        lickport_timestamps: 1D array
            A vector of timestamps (relative to start of trial for each observation
            in lickport_voltage.
        
        Returns:
        --------
        lick_timestamps: np.array 
            An array of timestamps for each detected lick.
        '''
        
        lick_derivative = np.diff(lick_voltage)
        lick_derivative = np.append(lick_derivative, 0)
        lick_index = np.argwhere(lick_derivative > 0).flatten()
        lick_timestamps = np.empty(len(lick_index), dtype=float)
        
        for lick in range(len(lick_index)):
            lick_timestamps[lick] = lickport_timestamps[lick_index[lick]]
            
        return lick_timestamps


    def get_anticipatory_licking(self):
        '''
        '''

        self.corr_ant_lick_trials = np.empty(len(self.num_trials), dtype=bool)
        self.corr_ant_lick_trials.fill(np.nan)
        self.incorr_ant_lick_trials = np.empty(len(self.num_trials), dtype=bool)
        self.incorr_ant_lick_trials.fill(np.nan)
        self.corr_ant_lick_rate = np.empty(len(self.num_trials), dtype=float)
        self.corr_ant_lick_rate.fill(np.nan)
        self.incorr_ant_lick_rate = np.empty(len(self.num_trials), dtype=float)
        self.incorr_ant_lick_rate.fill(np.nan)

        for trial in range(self.num_trials):

            if 'L' in str(self.data['sample_tone']['type'][trial]):
                    
                correct_lick_group = self.data['lick_l']
                incorr_lick_group = self.data['lick_r']
                reward_time = self.data['rew_l']['t'][trial]
                    
            elif 'R' in str(self.data['sample_tone']['type'][trial]):
                    
                correct_lick_group = self.data['lick_r']
                incorrect_lick_group = self.data['lick_l']
                reward_time = self.data['rew_r']['t'][trial]

            correct_lick_timestamps = self.get_lick_timestamps(
                correct_lick_group['volt'][trial],
                correct_lick_group['t'][trial])
            incorrect_lick_timestamps = self.get_lick_timestamps(
                incorrect_lick_group['volt'][trial],
                incorrect_lick_group['t'][trial])

            sample_tone_end = self.data['sample_tone']['end'][trial]
            trace_period = reward_time - sample_tone_end

            if trace_period > 250:
                # If trace period is too short, there is not enough time
                # for the mouse to lick in anticipation.
                corr_ant_licks = correct_lick_timestamps[
                    (correct_lick_timestamps > sample_tone_end)
                    & (correct_lick_timestamps < reward_time)]
                incorr_ant_licks = incorrect_lick_timestamps[
                    (incorrect_lick_timestamps > sample_tone_end)
                    & (incorrect_lick_timestamps < reward_time)]
                
                self.corr_ant_lick_trials[trial] = len(corr_ant_licks) > 0
                self.incorr_ant_lick_trials[trial] = len(incorr_ant_licks) > 0

                self.corr_ant_lick_rate[trial] = (
                    len(corr_ant_licks)
                    / (trace_period/1000)) # Divide by 1000 to get rate in Hz
                self.incorr_ant_lick_rate[trial] = (
                    len(incorr_ant_licks)
                    / (trace_period/1000))

    def get_performance(self):                       
        
        self.performance = np.zeros(self.num_trials, dtype=int)

        for trial in range(self.num_trials):
            if (self.data['response'][trial]
                == self.data['sample_tone']['type'][trial]):
                # If true, answer was correct.
                self.performance[trial] = 1

        return self.performance

                
class Mouse():
    '''
    A class to work with all experimental data for a single mouse in a
    dataset file. Is instantiated by a DataSet object, and creates instances
    of class Experiment.

    Attributes:
    -----------
    self.mouse_number: str
        A number (in str) corresponding to this mouse's ID.
    self.mouse_group: h5py group object
        Object to work with the mouse's group in the dataset file.
    self.experiments: list of obj (Experiment)
        A list containing an instance of class Expriment for each experiment 
        on this mouse in the dataset.

    Methods:
    --------
    self.get_experiments(date_list):
        Returns a list containing an instance of class Experiment for each
        experiment associated with this mouse.
    self.get_weights():
        Returns daily weights(g) for this mouse for each experiment in the 
        dataset.
    self.get_performance():
        Returns nested 1D numpy arrays indicating, for each trial, for each
        experiment, whether the mouse answered correctly (1) or not (0).
    self.get_performance_experiment():
        Returns the fraction of correct trials for each experiment in the
        dataset for this mouse.
    '''

    
    def __init__(self, mouse_number, mouse_group):
        '''
        '''
        self.mouse_number = mouse_number
        self.mouse_group = mouse_group

        date_list = list(self.mouse_group.keys())
        self.experiments = self.get_experiments(date_list)

    def get_experiments(self, date_list):
        '''
        '''
        experiments = []

        for date in date_list:
            block_list = self.mouse_group[date]['blocks']

            for block in block_list:
                block_number = block.decode('utf-8')

                experiments.append(Experiment(
                    self.mouse_number, date, block_number))
        return experiments

    def get_weights(self):
        '''
        '''
        self.weights = np.empty(len(self.experiments), dtype=float)

        for exp in range(len(self.experiments)):
            self.weights[exp] = self.experiments[exp].get_weight()

        return self.weights

    def get_performance(self):
        '''
        '''
        performance = np.empty(len(self.experiments), dtype=np.ndarray)

        for exp in range(len(self.experiments)):
            performance[exp] = self.experiments[exp].get_performance()

        return performance

    def get_performance_experiment(self):
        '''
        '''
        performance = self.get_performance()
        self.performance_experiment = np.empty(
            len(self.experiments), dtype=float)
        
        for exp in range(len(self.experiments)):
            corr_trials = np.sum(performance[exp])
            num_trials = self.experiments[exp].num_trials
            self.performance_experiment[exp] = (corr_trials/num_trials)

        return self.performance_experiment

    def get_performance_trials(self):
        '''
        '''
        performance = self.get_performance()
        performance_trials = as_vector(performance)
        
        return performance_trials
    
    def get_reversal_vector(self):
        '''
        '''
        
        reversal  = np.empty(len(self.experiments), dtype=np.ndarray)
        
        for exp in range(len(self.experiments)):
            reversal[exp] = self.experiments[exp].get_rule_switch('r')
            
        reversal_vector = as_vector(reversal)
        
        return reversal_vector
    
    def get_post_switch_trials(self, trial_vector, rule_switch_trials, n_trials):
        '''
        '''
        
        post_switch_trials = np.empty(len(rule_switch_trials), dtype=np.ndarray)
        
        for switch in range(len(rule_switch_trials)):
            switch_trial = rule_switch_trials[switch]
            post_switch_trials[switch] = (
                trial_vector[switch_trial:(switch_trial+n_trials)])
            
        return post_switch_trials
    
    def get_post_reversal_performance(self, n_trials):
        '''
        '''
        performance_trials = self.get_performance_trials()
        reversal_vector = self.get_reversal_vector()
        reversal_trials = np.nonzero(reversal_vector)[0]
        reversal_performance = self.get_post_switch_trials(
            performance_trials, reversal_trials, n_trials)

        return reversal_performance
   
   
class DataSet():
    '''
    A class to handle datasets containing data from several experiments from
    several different mice. Creates instances of class Mouse.

    Attributes:
    -----------
    self.dataset: h5py File object

    self.mouse_list: list of str

    self.mouse_objects: list of obj (Mouse)
    
    Methods:
    --------
    self.get_mice():
        Returns self.mouse_list, a list of all mice in the dataset.
    self.get_mouse_objects():
        Returns a list containing an instance of class Mouse for each
        mouse in the dataset.
    self.get_weights():
        Returns the daily weights for each mouse in mouse_objects.
    self.get_performance_experiment():
        Returns the fraction of correct trials for each experiment for 
        each mouse in the dataset.
    '''
    
    
    def __init__(self, filename):
        '''
        '''
        print(f'Opening {filename}.hdf5')
        self.dataset = h5py.File(f'{dataset_repo_path}{filename}.hdf5','r')
        self.mouse_list = self.get_mice()
        self.mouse_objects = self.get_mouse_objects(self.mouse_list)

    def get_mice(self):
        '''
        '''

        mouse_list = [i for i in list(self.dataset.keys())
                      if i != 'Activity log']

        mouse_search = True
        while mouse_search ==True:
            chosen_mice = input(
                'Enter mouse number (a:all mice, ls:list mice): ')
            
            if chosen_mice == 'a':
                mouse_search = False
                
            elif chosen_mice == 'ls':
                print(mouse_list)
                
            elif chosen_mice in mouse_list:
                mouse_list = []
                mouse_list.append(chosen_mice)
                mouse_search = False
                
            else:
                print('Not recognized')

        return mouse_list

    def get_mouse_objects(self, mouse_list):
        '''
        '''

        mouse_objects = []
        for mouse in mouse_list:
            mouse_group = self.dataset[mouse]
            mouse_objects.append(Mouse(mouse, mouse_group))

        return mouse_objects

    def get_weights(self):
        '''
        '''
        self.weights = np.empty(len(self.mouse_objects), dtype=np.ndarray)

        for mouse in range(len(self.mouse_objects)):
            self.weights[mouse] = self.mouse_objects[mouse].get_weights()

        return self.weights

    def get_performance_experiment(self):
        '''
        '''
        self.performance_experiment = np.empty(
            len(self.mouse_objects), dtype=np.ndarray)

        for mouse in range(len(self.mouse_objects)):
            self.performance_experiment[mouse] = (
                self.mouse_objects[mouse].get_performance_experiment())

        return self.performance_experiment

    def get_post_reversal_performance(self, n_trials):
        '''
        '''
        reversal_performance = np.empty(len(self.mouse_list), dtype=np.ndarray)

        for mouse in range(len(self.mouse_objects)):
            reversal_performance[mouse] = (
                self.mouse_objects[mouse].get_post_reversal_performance(n_trials))

        return reversal_performance

    
def as_array(nested_vectors):
    '''
    Converts a 1D numpy array with nested 1D numpy arrays of variable length
    into a single 2D numpy array, with width equal to length of the longest
    nested array. Missing values are np.nan.

    Arguments:
    ----------
    nested_vectors: 1D numpy array, dtype=np.ndarray
        1D numpy array with nested 1D numpy arrays (can be different lengths).

    Returns:
    --------
    output_array: 2D numpy array
        2D numpy array containing all nested vectors, with np.nan in missing
        places.
    '''

    max_length = max([len(i) for i in nested_vectors])
    output_array = np.empty(shape=[len(nested_vectors), max_length])
    output_array.fill(np.nan)

    for vector in range(len(nested_vectors)):
        vector_length = len(nested_vectors[vector])
        output_array[vector,0:vector_length] = nested_vectors[vector]

    return output_array

def as_vector(nested_vectors):
    '''
    Converts a 1D numpy array with nested 1D numpy arrays of variable length
    into a single 1D numpy array with all data concatenated.

    Arguments:
    ----------
    nested_vectors: 1D numpy array, dtype=np.ndarray
        1D numpy array with nested 1D numpy arrays (can be different lengths).

    Returns:
    --------
    output_array: 1D numpy array
        1D numpy array containing all vectors concatenated together. 
    '''
    output_vector = np.array([])

    for vector in nested_vectors:
        output_vector = np.append(output_vector, vector)

    return output_vector

def dataset_search():
    '''
    '''

    datasets = []
    dataset_names = []
    file_search = True
    while file_search == True:   
        fname = input('Enter dataset name (ls:list): ')
        
        if fname == 'ls':  
            print(sorted(os.listdir(dataset_repo_path)))
            
        elif f'{fname}.hdf5' in os.listdir(dataset_repo_path):   
            datasets.append(DataSet(fname))
            dataset_names.append(input('Enter label for this dataset: '))
            
            if input('Add another dataset?(y/n): ') == 'n':
                file_search = False
                
        else:
            print('Dataset file not found.')

    return datasets, dataset_names
