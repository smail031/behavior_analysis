import h5py
import numpy as np
import os

data_repo_path = ('/Volumes/GoogleDrive/Shared drives/Beique Lab/'
                  'Data/Raspberry PI Data/Sebastien/Dual_Lickport/Mice/')
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
    get_rule_switch():
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

    
    def __init__(self, mouse, date, block, exp_type):
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
        self.exp_type = exp_type

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
    

    def get_rule_switch(self):
        '''
        Find trials in which there was a reversal or a set shift. A reversal
        (in reversal learning paradigms) leads to a change in port assignment
        (L cue -> R and vice versa), which is stored in rule/left_port. 
        A set shift (in attentional set shifting paradigms) leads to a change
        in the relevant/instrutive dimension (e.g. frequency rule -> location
        rule), which is stored in rule/freq_rule. The method can handle
        experiments with multiple rule switches, but this should never occurr.

        Returns:
        --------
        rule_switch: 1D numpy array
            Indicates, on each trial, whether there was a rule switch(1)
            or not(0).
        '''

        if self.exp_type == 'r':       
            rule_switch = np.diff(self.data['rule']['left_port'])

        elif self.exp_type == 's':
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
        
        lick_derivative = np.diff(lickport_voltage)
        lick_derivative = np.append(lick_derivative, 0)
        lick_index = np.argwhere(lick_derivative > 0).flatten()
        lick_timestamps = np.empty(len(lick_index), dtype=float)
        
        for lick in range(len(lick_index)):
            lick_timestamps[lick] = lickport_timestamps[lick_index[lick]]
            
        return lick_timestamps


    def get_anticipatory_licking(self):
        '''
        For classical conditioning experiments. For trials in which the trace
        period is >250ms, calculates the proportion of trials in which there
        is any anticipatory licking on the correct and incorrect ports. Also
        calculates mean anticipatory lick rates (in Hz) for correct and
        incorrect ports, not including trials in which there is no anticipatory
        licking on that port.

        Anticipatory licking is defined as any licking contact that occurrs
        between the end of the sample tone and the water delivery.
        
        Returns list containing:
        -------------------------
        [0]corr_ant_lick_trials: 1D numpy array (float)
            Numpy vector indicating, for each trial, whether(1) or not(0) there
            was any anticipatory licking on the correct port, or whether the 
            trace period was too short (np.nan).
        [1]incorr_ant_lick_trials: 1D numpy array (float)
            Numpy vector indicating, for each trial, whether(1) or not(0) there
            was any anticipatory licking on the incorrect port, or whether the 
            trace period was too short (np.nan).
        [2]corr_ant_lick_rate: 1D numpy array (float)
            Numpy vector indicating the rate (in Hz) of anticipatory licking on
            the correct port for each trial. If the trace period is too short,
            the value is set to np.nan.
        [3]incorr_ant_lick_rate: 1D numpy array (float)
            Numpy vector indicating the rate (in Hz) of anticipatory licking on
            the incorrect port for each trial. If the trace period is too short,
            the value is set to np.nan.
        '''

        corr_ant_lick_trials = np.empty(self.num_trials)
        corr_ant_lick_trials.fill(np.nan)
        incorr_ant_lick_trials = np.empty(self.num_trials)
        incorr_ant_lick_trials.fill(np.nan)
        corr_ant_lick_rate = np.empty(self.num_trials, dtype=float)
        corr_ant_lick_rate.fill(np.nan)
        incorr_ant_lick_rate = np.empty(self.num_trials, dtype=float)
        incorr_ant_lick_rate.fill(np.nan)

        for trial in range(self.num_trials):

            if 'L' in str(self.data['sample_tone']['type'][trial]):
                    
                correct_lick_group = self.data['lick_l']
                incorrect_lick_group = self.data['lick_r']
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

            # If trace period is too short, there is not enough time
            # for the mouse to lick in anticipation.
            if trace_period > 250:
                corr_ant_licks = correct_lick_timestamps[
                    (correct_lick_timestamps > sample_tone_end)
                    & (correct_lick_timestamps < reward_time)]
                incorr_ant_licks = incorrect_lick_timestamps[
                    (incorrect_lick_timestamps > sample_tone_end)
                    & (incorrect_lick_timestamps < reward_time)]

                corr_ant_lick_trials[trial] = (len(corr_ant_licks) > 0)
                incorr_ant_lick_trials[trial] = (len(incorr_ant_licks) > 0)

                corr_ant_lick_rate[trial] = (
                    len(corr_ant_licks)
                    / (trace_period/1000)) # Divide by 1000 to get rate in Hz
                incorr_ant_lick_rate[trial] = (
                    len(incorr_ant_licks)
                    / (trace_period/1000))

        output_list =  [corr_ant_lick_trials,
                        incorr_ant_lick_trials,
                        corr_ant_lick_rate,
                        incorr_ant_lick_rate]
        
        return output_list
            
    def get_performance(self):
        '''
        Returns a 1D numpy array indicating, for each trial, whether the mouse
        responded correctly(1) or incorrectly(0), or failed to respond(0)

        Returns:
        --------
        performance: np.array, dtype=int
            Contains the performance of the mouse on each trial.
        '''
        
        performance = np.zeros(self.num_trials, dtype=int)

        for trial in range(self.num_trials):
            if (self.data['response'][trial]
                == self.data['sample_tone']['type'][trial]):
                # If true, answer was correct.
                performance[trial] = 1

        return performance

    def get_response_latency(self):
        '''
        '''

        resp_period = 2000
        response_times = np.empty(self.num_trials, dtype=float)
        # If no response, latency is set to length of response period
        response_times.fill(resp_period) 
        
        for trial in range(self.num_trials):
            tone_end = self.data['sample_tone']['end'][trial]
            left_licks = self.get_lick_timestamps(
                self.data['lick_l']['volt'][trial],
                self.data['lick_l']['t'][trial])
            right_licks = self.get_lick_timestamps(
                self.data['lick_r']['volt'][trial],
                self.data['lick_r']['t'][trial])
            
            all_licks = np.append(left_licks,right_licks)
            response_licks = all_licks[
                (all_licks > tone_end)
                &(all_licks < tone_end+resp_period)]

            if len(response_licks) > 0:
                response_times[trial] = np.amin(response_licks) - tone_end
            
            
        return response_times

                
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
    self.get_performance_experiment():
        Returns the fraction of correct trials for each experiment in the
        dataset for this mouse.
    '''

    
    def __init__(self, mouse_number, mouse_group, exp_type):
        '''
        '''
        self.mouse_number = mouse_number
        self.mouse_group = mouse_group
        self.exp_type = exp_type

        self.date_list = list(self.mouse_group.keys())
        self.experiments = self.get_experiments(self.date_list)
        self.initial_experiments = self.get_initial_experiments()

    def get_experiments(self, date_list):
        '''
        Instantiates an object of class Experiment for each experiment in
        the dataset for this mouse. Returns a list of these objects. This
        method is called in __init__.

        Returns:
        --------
        experiments: list
            A list of objects from the Experiments class.
        '''
        experiments = []

        for date in self.date_list:
            block_list = self.mouse_group[date]['blocks']

            for block in block_list:
                block_number = block.decode('utf-8')

                experiments.append(Experiment(
                    self.mouse_number, date, block_number, self.exp_type))
        return experiments

    def get_initial_experiments(self):
        '''
        '''

        rule_switch_experiments = np.empty(len(self.experiments), dtype=int)

        for exp in range(len(self.experiments)):
            rule_switch_experiments[exp] =np.sum(
                self.experiments[exp].get_rule_switch())

        switch_experiments = np.nonzero(rule_switch_experiments)[0]

        if len(switch_experiments) > 0:
            initial_experiments = self.experiments[0:switch_experiments[0]]

        else: # In case there are no switches, all experiments are taken.
            initial_experiments = self.experiments

        return initial_experiments

    def get_weights(self):
        '''
        Returns a 1D array containing the weight (in grams) of the mouse
        recorded prior to each experiment in this dataset.
        
        Returns:
        weights: 1D numpy array (float)
            A vector containing the weight of the mouse for each experiment.
        '''
        
        weights = np.empty(len(self.experiments), dtype=float)

        for exp in range(len(self.experiments)):
            weights[exp] = self.experiments[exp].get_weight()

        return weights

    def get_performance_experiment(self, pre_switch=False):
        '''
        Returns a 1D array containing, for each experiment for this mouse, the
        fraction of correctly answered trials. Can ignore trials after either a
        reversal or a set shift.
        
        Arguments: 
        ----------
        pre_switch: bool, default = False
            Indicates whether trials after a rule switch (reversal or set shift)
            should be ignored (True) or not (False).

        Returns:
        --------
        performance_experiment: np.array, dtype=float
            Indicates the fraction of correct trials for each experiment.
        
        '''

        performance_experiment = np.empty(len(self.experiments), dtype=float)
        
        for exp in range(len(self.experiments)):

            num_trials = self.experiments[exp].num_trials
            
            if pre_switch:
                switch_vector = (
                    self.experiments[exp].get_rule_switch())
                switch_trials = np.nonzero(switch_vector)[0]
                
                if len(switch_trials) > 0:
                    num_trials = switch_trials[0] # Get first switch, if several
                    
            performance = self.experiments[exp].get_performance()[0:num_trials]
            performance_experiment[exp] = (np.sum(performance)/num_trials)

        return performance_experiment

    def get_performance_trials(self):
        '''
        Returns a single 1D numpy array indicating, for each trial of each
        experiment (concatenated together), whether the mouse answered
        correctly(1) or incorrectly(0), or failed to respond(0)

        Returns:
        --------
        performance_trials: np.array, dtype = int
            Indicates whether each trial is correct(1) or incorrect(0).
        '''

        performance = np.empty(len(self.experiments), dtype=np.ndarray)
        
        for exp in range(len(self.experiments)):
            performance[exp] = self.experiments[exp].get_performance()
            
        performance_trials = as_vector(performance)
        
        return performance_trials
    
    def get_reversal_vector(self):
        '''
        Returns a 1D numpy array indicating, for each trial of each experiment,
        whether a reversal was triggered(1) or not(0). Note that the reversal
        trial is the one in which the reversal was triggered, so the rule only
        changes on the subsequent trial.

        Returns:
        --------
        reversal_vector: 1D np.array, dtype=int
            With all trials from all experiments concatenated together,
            is filled with 0 everywhere except trials in which a reversal
            was triggered(1).
        '''
        
        reversal  = np.empty(len(self.experiments), dtype=np.ndarray)
        
        for exp in range(len(self.experiments)):
            reversal[exp] = self.experiments[exp].get_rule_switch()
            
        reversal_vector = as_vector(reversal)
        
        return reversal_vector
    
    def get_post_switch_trials(self, trial_vector, rule_switch_trials, n_trials):
        '''
        Returns data from a given number of trials after a rule switch.

        Arguments:
        ----------
        trial_vector: 1D numpy array
            For each trial of each experiment, contains relevant information
            to be returned (e.g. performance(1/0), choice(L/R/N)).
        rule_switch_trials: 1D numpy array
            Lists the trial numbers corresponding to rule switches.
        n_trials: int
            Indicates how many trials to return after the switch.

        Returns:
        --------
        post_switch_trials: np.array, dtype=np.array
            For each rule switch, contains data from trial_vector for 
            n_trials after the rule switch.
        '''
        
        post_switch_trials = np.empty(len(rule_switch_trials), dtype=np.ndarray)
        
        for switch in range(len(rule_switch_trials)):
            switch_trial = rule_switch_trials[switch]
            post_switch_trials[switch] = (
                trial_vector[switch_trial:(switch_trial+n_trials)])
            
        return post_switch_trials
    
    def get_post_reversal_performance(self, n_trials):
        '''
        For each reversal, calculates the performance of the mouse for a given
        number of subsequent trials. 
        
        Arguments:
        ----------
        n_trials: int
            Indicates how many post-reversal trials to consider. 
        
        Returns:
        --------
        reversal_performance: np.array, dtype=np.array
            For each reversal, contains performance of the mouse (1/0)
            for (n_trials) subsequent trials.
        
        '''
        performance_trials = self.get_performance_trials()
        reversal_vector = self.get_reversal_vector()
        reversal_trials = np.nonzero(reversal_vector)[0]
        reversal_performance = self.get_post_switch_trials(
            performance_trials, reversal_trials, n_trials)

        return reversal_performance

    def get_anticipatory_licking(self):
        '''
        For each experiment, calculates the fraction of trials with anticipatory
        licking on the correct and incorrect port, and the mean anticipatory 
        lick rate for correct and incorrect ports.

        Returns:
        --------
        output_list, containing:
        [0]corr_ant_lick_trials: np.array, dtype=float
            For each experiment, stores the fraction of trials with anticipatory
            licking on the correct port.
        [1]incorr_ant_lick_trials: np.array, dtype=float
            For each experiment, stores the fraction of trials with anticipatory
            licking on the incorrect port.
        [2]corr_ant_lick_rates: np.array, dtype=float
            For each experiment, stores mean anticipatory
            lick rate on the correct port.
        [3]incorr_ant_lick_rates: np.array, dtype=float
            For each experiment, stores mean anticipatory
            lick rate on the incorrect port.
        '''
        corr_ant_lick_trials = np.empty(len(self.experiments), dtype=float)
        incorr_ant_lick_trials = np.empty_like(corr_ant_lick_trials)
        corr_ant_lick_rates = np.empty_like(corr_ant_lick_trials)
        incorr_ant_lick_rates = np.empty_like(corr_ant_lick_trials)

        for exp in range(len(self.experiments)):
            ant_lick_exp = self.experiments[exp].get_anticipatory_licking()
            corr_ant_lick_trials[exp] = (
                np.nansum(ant_lick_exp[0])
                /self.experiments[exp].num_trials)
            incorr_ant_lick_trials[exp] = (
                np.nansum(ant_lick_exp[1])
                /self.experiments[exp].num_trials)
            corr_ant_lick_rates[exp] = np.nanmean(ant_lick_exp[2])
            incorr_ant_lick_rates[exp] = np.nanmean(ant_lick_exp[3])

        output_list =  [corr_ant_lick_trials,
                        incorr_ant_lick_trials,
                        corr_ant_lick_rates,
                        incorr_ant_lick_rates]

        return output_list

    def get_response_latency(self, initial_learning=True):
        '''
        '''

        if initial_learning:
            experiments = self.initial_experiments

        else:
            experiments = self.experiments

        response_times = np.empty(len(experiments), dtype=float)

        for exp in range(len(experiments)):
            print(f'Date {experiments[exp].date}')
            response_times[exp] = np.nanmean(
                experiments[exp].get_response_latency())

        return response_times

    def get_null_responses(self, initial_learning=True):
        '''
        '''

        if initial_learning:
            experiments = self.initial_experiments

        else:
            experiments = self.experiments

        null_responses = np.empty(len(experiments), dtype=float)

        for exp in range(len(experiments)):
            null = ['N' in str(i) for i in experiments[exp].data['response']]
            null_responses[exp] = sum(null) / experiments[exp].num_trials

        return null_responses

    def get_port_bias(self, initial_learning=True):
        '''
        '''

        if initial_learning:
            experiments = self.initial_experiments

        else:
            experiments = self.experiments
        
        port_bias = np.empty(len(experiments), dtype=float)

        for exp in range(len(experiments)):
            left_resp = ['L' in str(i) for i in experiments[exp].data['response']]
            port_bias[exp] = sum(left_resp) / experiments[exp].num_trials

        return port_bias
        
   
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
    self.get_post_reversal_performance:
        For each reversal for each mouse, returns a nested vectors for 
        performance for a given number of trials post-reversal.
    '''
    
    
    def __init__(self, filename, exp_type):
        '''
        '''
        print(f'Opening {filename}.hdf5')
        self.dataset = h5py.File(f'{dataset_repo_path}{filename}.hdf5','r')
        self.exp_type = exp_type
        self.mouse_list = self.get_mice()
        self.mouse_objects = self.get_mouse_objects(self.mouse_list)


    def get_mice(self):
        '''
        Guides the user through a mouse selection process; either choosing
        a specific mouse from the dataset or choosing all mice.
        
        Returns:
        --------
        mouse_list: list(str)
            Contains the ID numbers of either the selected mouse, or all 
            mice in the dataset.
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
        For each mouse in mouse_list, instantiates an object of class
        Mouse. 
        
        Arguments:
        ----------
        mouse_list: list(str)
            A list containing the ID numbers of each relevant mouse.

        Returns:
        mouse_objects: list(Mouse objects)
            A list containing an instance of class Mouse for each mouse
            in mouse_list
        '''

        mouse_objects = []
        for mouse in mouse_list:
            mouse_group = self.dataset[mouse]
            mouse_objects.append(Mouse(mouse, mouse_group, self.exp_type))

        return mouse_objects

    def get_weights(self):
        '''
        For each mouse, gets the recorded weight (in grams) for each experiment
        day in the dataset. Note that these weight curves only include dates on
        which a behavior experiment was run.

        Returns:
        --------
        weights: np.array, dtype=np.array
            Nested arrays containing, for each mouse, the recorded weight
            (in grams, float) recorded on each experiment day corresponding
            to that mouse.
        '''
        weights = np.empty(len(self.mouse_objects), dtype=np.ndarray)

        for mouse in range(len(self.mouse_objects)):
            weights[mouse] = self.mouse_objects[mouse].get_weights()

        return weights

    def get_performance_experiment(self, pre_switch):
        '''
        For each mouse, gets the fraction of correctly answered trials on
        each experiment day corresponding to that mouse.

        Arguments:
        ----------
        pre_switch: bool
            Indicates whether trials after a rule switch should be
            ignored(True) or not(False)
        
        Returns:
        --------
        performance_experiment: np.array, dtype=np.array
            Stores the fraction of correct trials (float) for each mouse on
            each experiment day.
        '''
        performance_experiment = np.empty(
            len(self.mouse_objects), dtype=np.ndarray)

        for mouse in range(len(self.mouse_objects)):
            performance_experiment[mouse] = (
                self.mouse_objects[mouse].get_performance_experiment(
                    pre_switch=pre_switch))

        return performance_experiment

    def get_post_reversal_performance(self, n_trials):
        '''
        For each mouse, for each reversal, returns the performance of the mouse
        on a given number of trials after a reversal.

        Arguments:
        ----------
        n_trials: int
            Specifies how many post-reversal trials to consider.
        
        Returns:
        --------
        reversal_performance: np.array, dtype=np.array
            Nested 1D numpy arrays indicating, for each mouse, for each
            reversal, for each trial, whether the mouse answered correctly(1)
            or incorrectly(0), or failed to respond(0).
        '''
        reversal_performance = np.empty(len(self.mouse_list), dtype=np.ndarray)

        for mouse in range(len(self.mouse_objects)):
            reversal_performance[mouse] = (
                self.mouse_objects[mouse].get_post_reversal_performance(n_trials))

        return reversal_performance

    def get_anticipatory_licking(self):
        '''
        For each mouse, gets the fraction of trials with anticipatory licking
        on the correct and incorrect ports, and the mean anticipatory lick rate
        on the correct and incorrect ports for each associated experiment. 

        Returns:
        --------
        output_list, containing:
        [0]corr_ant_lick_trials: np.array, dtype=np.array
            For each mouse, stores the fraction of trials with anticipatory
            licking on the correct port for each experiment.
        [1]incorr_ant_lick_trials: np.array, dtype=np.array
            For each mouse, stores the fraction of trials with anticipatory
            licking on the incorrect port for each experiment.
        [2]corr_ant_lick_rates: np.array, dtype=np.array
            For each mouse, stores the mean anticipatory lick rate on the 
            correct port for each experiment.
        [3]incorr_ant_lick_rates: np.array, dtype=np.array
            For each mouse, stores the mean anticipatory lick rate on the 
            incorrect port for each experiment.
        '''
        corr_ant_lick_trials = np.empty(len(self.mouse_list), dtype=np.ndarray)
        incorr_ant_lick_trials = np.empty_like(corr_ant_lick_trials)
        corr_ant_lick_rates = np.empty_like(corr_ant_lick_trials)
        incorr_ant_lick_rates = np.empty_like(corr_ant_lick_trials)

        for mouse in range(len(self.mouse_list)):
            ant_lick_mouse = self.mouse_objects[mouse].get_anticipatory_licking()

            corr_ant_lick_trials[mouse] = ant_lick_mouse[0]
            incorr_ant_lick_trials[mouse] = ant_lick_mouse[1]
            corr_ant_lick_rates[mouse] = ant_lick_mouse[2]
            incorr_ant_lick_rates[mouse] = ant_lick_mouse[3]

        output_list = [corr_ant_lick_trials,
                  incorr_ant_lick_trials,
                  corr_ant_lick_rates,
                  incorr_ant_lick_rates]

        return output_list

    def get_response_latency(self, initial_learning=True):
        '''
        '''

        response_times = np.empty(len(self.mouse_list), dtype=np.ndarray)

        for mouse in range(len(self.mouse_list)):
            print(f'Mouse {self.mouse_list[mouse]}')
            response_times[mouse] = (
                self.mouse_objects[mouse].get_response_latency(
                    initial_learning=initial_learning))

        return response_times

    def get_null_responses(self, initial_learning=True):
        '''
        '''

        null_responses = np.empty(len(self.mouse_list), dtype=np.ndarray)

        for mouse in range(len(self.mouse_list)):
            null_responses[mouse] = (
                self.mouse_objects[mouse].get_null_responses(
                    initial_learning=initial_learning))

        return null_responses

    def get_port_bias(self, initial_learning=True):
        '''
        '''

        port_bias = np.empty(len(self.mouse_list), dtype=np.ndarray)

        for mouse in range(len(self.mouse_list)):
            port_bias[mouse] = (self.mouse_objects[mouse].get_port_bias(
                initial_learning=initial_learning))

        return port_bias
        
    
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
    Gets the user to choose one of many dataset files in dataset_repo_path.
    Once these datasets are chosen, a corresponding DataSet object is 
    instantiated and appended to a list.

    Returns:
    --------
    datasets: list (DataSet objects)
        A list containing a DataSet object for each selected dataset file.
    dataset_names: list (str)
        A list containing a user input name for each selected dataset.
        To be used as label in figures.
    '''

    datasets = []
    dataset_names = []
    file_search = True
    while file_search == True:   
        fname = input('Enter dataset name (ls:list): ')
        
        if fname == 'ls':  
            print(sorted(os.listdir(dataset_repo_path)))
            
        elif f'{fname}.hdf5' in os.listdir(dataset_repo_path):

            exp_type = input('What kind of experiment?'
                             '(r:reversal learning, s:set shifting): ')
            datasets.append(DataSet(fname, exp_type))
            dataset_names.append(input('Enter label for this dataset: '))
            
            if input('Add another dataset?(y/n): ') == 'n':
                file_search = False
                
        else:
            print('Dataset file not found.')

    return datasets, dataset_names
