"""
---------------------------------------------------------------------------------
cognitive_science_learning_model_base.py

Cog Sci Learning model base class.

Source:
-------
    : https://ccn.studentorg.berkeley.edu/pdfs/papers/WilsonCollins_modelFitting.pdf
---------------------------------------------------------------------------------
"""

import plotly.graph_objects as go
from scipy.optimize import minimize  # finding optimal params in models
import numpy as np                   # matrix/array functions
from abc import ABC, abstractmethod
import os
import scipy.io
import json
import numpy as np
from collections import deque, OrderedDict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# " +-- Helpers ----------------------------------------------------------------->> "
def load_matlab_data():
    # Define file paths relative to project root
    matlab_file = os.path.join(PROJECT_ROOT, 'data', 'raw_matlab_data', 'attention_behaviorals_v3.mat')
    output_dir = os.path.join(PROJECT_ROOT, 'data', 'processed_data')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, 'data', 'raw_matlab_data'), exist_ok=True)

    # Load MATLAB file
    data = scipy.io.loadmat(matlab_file)
    data = data['dat']


    rat_ids = data['ratID']
    flat_rat_ids = [id[0][0] for id in rat_ids.flatten()]
    list_subjects = np.unique(flat_rat_ids)


    EDS_Easy = {}
    EDS_Hard = {}
    for subject in list_subjects:
        EDS_Easy[subject]=[[],[]]
        EDS_Hard[subject]=[[],[]]


    data = data[0]
    for entry in data:
        session_type = entry['SessionType'][0]
        ratID = entry['ratID'][0][0]

        resp_code = entry['RespCode'].flatten()
        new_stimuli_trial = entry['NewStimuli_Trial']
        ToneCode = entry['ToneCode'].flatten()
        VisCode = entry['VisCode'].flatten()
        stim_code = np.array(list(zip(ToneCode, VisCode)))
        if new_stimuli_trial.size>0:
            new_stimuli_trial = new_stimuli_trial[0][0]
        else:
            new_stimuli_trial = 0

        if session_type == "EDS_BL_Easy":
            last_BL = "Easy"
        if session_type == "EDS_BL_Hard":
            last_BL = "Hard"

        if session_type == "EDS":
            if previous_session == "EDS_BL_Easy" or previous_session == "EDS_BL_Hard":
                chuck = int(entry['NewStimuli_Trial'])
                stim_code = stim_code[chuck:]
                resp_code = resp_code[chuck:]
            if last_BL == "Easy":
                EDS_Easy[ratID][0].extend(stim_code)
                EDS_Easy[ratID][1].extend(resp_code)
            elif last_BL == "Hard":
                EDS_Hard[ratID][0].extend(stim_code)
                EDS_Hard[ratID][1].extend(resp_code)
                
        previous_session = session_type
    return EDS_Easy, EDS_Hard

def load_pre_data():
     # Define file paths relative to project root
    matlab_file = os.path.join(PROJECT_ROOT, 'data', 'raw_matlab_data', 'attention_behaviorals_v3.mat')
    output_dir = os.path.join(PROJECT_ROOT, 'data', 'processed_data')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, 'data', 'raw_matlab_data'), exist_ok=True)

    # Load MATLAB file
    data = scipy.io.loadmat(matlab_file)
    data = data['dat']


    rat_ids = data['ratID']
    flat_rat_ids = [id[0][0] for id in rat_ids.flatten()]
    list_subjects = np.unique(flat_rat_ids)

    QUEUE = deque(maxlen=4)

    pre_EASY = {}
    pre_HARD = {}
    EDS_Easy = {}
    EDS_Hard = {}
    for subject in list_subjects:
        pre_EASY[subject]=[[],[]]
        pre_HARD[subject]=[[],[]]
        EDS_Easy[subject]=[[],[]]
        EDS_Hard[subject]=[[],[]]

    data = data[0]
    for entry in data:
        #Adding the entry to the queue to keep entries for Q-value intialisation
        QUEUE.append(entry)

        #Extracting relevant information from the entry
        session_type = entry['SessionType'][0]
        ratID = entry['ratID'][0][0]
        resp_code = entry['RespCode'].flatten()
        new_stimuli_trial = entry['NewStimuli_Trial']
        ToneCode = entry['ToneCode'].flatten()
        VisCode = entry['VisCode'].flatten()
        stim_code = np.array(list(zip(ToneCode, VisCode)))

        #Extracting from which trial EDS actually starts
        if new_stimuli_trial.size>0:
            new_stimuli_trial = new_stimuli_trial[0][0]
        else:
            new_stimuli_trial = 0

        #Using baseline to infer easy or hard grouping
        if session_type == "EDS_BL_Easy":
            last_BL = "Easy"
        if session_type == "EDS_BL_Hard":
            last_BL = "Hard"

        if session_type == "EDS":
            #If start of EDS session, need to remove the baseline trials from the data as well as capture pre-EDS sessions for Q-value initialisation
            if previous_session == "EDS_BL_Easy" or previous_session == "EDS_BL_Hard":

                #Iterates through queue and extracts data for pre-training trials to ensure correct Q-value initialisation
                for i in range(len(QUEUE)):
                    Q_entry = QUEUE[i]
                    Q_ratID = Q_entry['ratID'][0][0]
                    Q_resp_code = Q_entry['RespCode'].flatten()
                    Q_ToneCode = Q_entry['ToneCode'].flatten()
                    Q_VisCode = Q_entry['VisCode'].flatten()
                    Q_stim_code = np.array(list(zip(Q_ToneCode, Q_VisCode)))

                    if last_BL == "Easy":
                        pre_EASY[Q_ratID][0].extend(Q_stim_code)
                        pre_EASY[Q_ratID][1].extend(Q_resp_code)
                    elif last_BL == "Hard":
                        pre_HARD[Q_ratID][0].extend(Q_stim_code)
                        pre_HARD[Q_ratID][1].extend(Q_resp_code)

                # Adds the pre-EDS trials in the first EDS session to the pre-training data
                if last_BL == "Easy":
                    pre_EASY[ratID][0].extend(stim_code[:new_stimuli_trial])
                    pre_EASY[ratID][1].extend(resp_code[:new_stimuli_trial])
                elif last_BL == "Hard":
                    pre_HARD[ratID][0].extend(stim_code[:new_stimuli_trial])
                    pre_HARD[ratID][1].extend(resp_code[:new_stimuli_trial])

                #Throws away the pre-EDS trials
                stim_code = stim_code[new_stimuli_trial:]
                resp_code = resp_code[new_stimuli_trial:]

            if last_BL == "Easy":
                EDS_Easy[ratID][0].extend(stim_code)
                EDS_Easy[ratID][1].extend(resp_code)
            elif last_BL == "Hard":
                EDS_Hard[ratID][0].extend(stim_code)
                EDS_Hard[ratID][1].extend(resp_code)
                        
        previous_session = session_type

    return pre_EASY, pre_HARD, EDS_Easy, EDS_Hard

def load_data():

    #+______________________________________________________________________________________________________________________________________+
    #DATA PREPROCESSING

    file_path = os.path.join(PROJECT_ROOT, 'data', 'extracting_relevant_data', 'attention_behaviorals.json')

    # Open and load the JSON file
    with open(file_path, 'r') as file:
        jsondata = json.load(file)


    list_subjects = list(OrderedDict.fromkeys([entry['ratID'] for entry in jsondata])) #gets list of rat IDs    

    EDS_Easy = {}
    EDS_Hard = {}
    last_BL = None
    previous_session = None

    for subject in list_subjects:
        EDS_Easy[subject]=[[],[],[]]
        EDS_Hard[subject]=[[],[],[]]

    #Extracts EDS data, splitting into the Easy and Hard to distinguish groups
    for entry in jsondata:
        session_type = entry['SessionType']
        ratID = entry['ratID']
        stim_code = entry['StimCode']
        resp_code = entry['RespCode']
        rele_mode = entry['ReleMode']

        #print(f"ratID: {ratID}, session_type: {session_type}, rele_mode: {rele_mode}")
        if session_type == "EDS_BL_Easy":
            last_BL = "Easy"
        if session_type == "EDS_BL_Hard":
            last_BL = "Hard"

        if session_type == "EDS":
            if previous_session == "EDS_BL_Easy" or previous_session == "EDS_BL_Hard":
                chuck = int(entry['NewStimuli_Trial'])
                stim_code = stim_code[chuck:]
                resp_code = resp_code[chuck:]
            if last_BL == "Easy":
                EDS_Easy[ratID][0].extend(stim_code)
                EDS_Easy[ratID][1].extend(resp_code)
                EDS_Easy[ratID][2] = rele_mode
                #print(f"ratID: {ratID}, EDS Easy rele_mode: {rele_mode}")
            elif last_BL == "Hard":
                EDS_Hard[ratID][0].extend(stim_code)
                EDS_Hard[ratID][1].extend(resp_code)
                EDS_Hard[ratID][2] = rele_mode
                #print(f"ratID: {ratID}, EDS Hardrele_mode: {rele_mode}")
                
        previous_session = session_type

    return jsondata, list_subjects, EDS_Easy, EDS_Hard

    #+______________________________________________________________________________________________________________________________________+

def extract_order():
    
    (jsondata, list_subjects, EDS_Easy, EDS_Hard) = load_data()

    first_Easy = []
    first_Hard = []
    dict = {ratID:None for ratID in list_subjects}

    for entry in jsondata:
        session_type = entry['SessionType']
        ratID = entry['ratID']
        stim_code = entry['StimCode']
        resp_code = entry['RespCode']
        #print(session_type)
        if session_type == "EDS_BL_Easy" and dict[ratID] == None:
            dict[ratID] = "Easy"

        elif session_type == "EDS_BL_Hard" and dict[ratID] ==None:
            dict[ratID] = "Hard"

    for k,v in dict.items():
        #print(k,v)
        if v == "Easy":
            first_Easy.append(k)
        else:
            first_Hard.append(k)

    return first_Easy, first_Hard

def get_actions_rewards(response_codes):
    """Function to get the action and rewards from the response codes"""
    _actions = np.isin(np.array(response_codes), [2,3]).astype(int)
    _rewards = np.isin(np.array(response_codes), [1,2]).astype(int)
    return (_actions, _rewards)

def add_diag_line(fig, xmin=0, xmax=1, marker_color='black', row=1, col=1):
    _x = np.linspace(xmin, xmax, 100)
    _y = _x
    fig.add_trace(go.Scatter(x=_x, y=_y, mode='lines', marker_color=marker_color, name='y=x', showlegend=False),row=row, col=col)

# " +-- Base Class -------------------------------------------------------------->> "
class MultiArmedBanditModels(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def simulate(self, **kwargs):
        self.simulated_params = kwargs

    @abstractmethod
    def neg_log_likelihood(self):
        pass
    
    @abstractmethod
    def perform_sensitivity_analysis(self):
        pass

    @abstractmethod
    def optimize_brute_force(self, loss_function, loss_kwargs, parameter_ranges):

        neg_log_likelihoods = [loss_function(**params) for params in loss_kwargs]
        min_loss_func_idx = np.argmin(neg_log_likelihoods)
        min_loss_value = min(neg_log_likelihoods)
        return min_loss_func_idx, loss_kwargs[min_loss_func_idx], min_loss_value
        # best_epsilon = epsilon_values[np.argmin(neg_log_likelihoods)]
        # return best_epsilon

    def optimize_scikit(self, loss_function, init_guess, args, bounds):
        result = minimize(
                        loss_function,
                        init_guess,
                        args=args,
                        bounds=bounds
            )
        res_nll = result.fun
        param_fits = result.x
        return result, res_nll, param_fits

    @abstractmethod
    def plot_neg_log_likelihood(self):
        pass

    def compare_fitting_procedures(self):
        pass

    def compute_BIC(self, negLL, T, k_params):
        return 2 * negLL + k_params * np.log(T)
    
