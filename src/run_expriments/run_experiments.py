import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np
from collections import deque, OrderedDict
import seaborn as sns
import json
import os
import math
import scipy.io
import copy
import sys

# Add parent directory to path to allow imports from models/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#Loading models
from models.cog_sci_learning_model_base import MultiArmedBanditModels, get_actions_rewards
from models.rr import RandomResponseModel
from models.crr import ContextualRandomResponseModel
from models.wsls import WinStayLoseShiftModel
from models.rw import RoscorlaWagnerModel
from models.crw import ContextualRoscorlaWagnerModel
from models.ck import ChoiceKernelModel
from models.crwck import ContextualRascorlaWagnerChoiceKernelModel
from models.crwck_plus import ContextualRascorlaWagnerChoiceKernelPlusModel
from models.crwcknb import ContextualRascorlaWagnerChoiceKernelNoBetaModel
from models.crwck_all import ContextualRascorlaWagnerChoiceKernelAllModel
from models.crwck_pre import ContextualRascorlaWagnerChoiceKernelPreModel
from models.crwcksa import ContextualRescorlaWagnerChoiceKernelSharedAlphaModel
from models.crwck_init import ContextualRascorlaWagnerChoiceKernelInitModel
from models.crwck_plus_init import ContextualRascorlaWagnerChoiceKernelPlusInitModel
from models.crwcksa_plus import ContextualRascorlaWagnerChoiceKernelPlusSharedAlphaModel
from models.crwck_plus_pre import ContextualRascorlaWagnerChoiceKernelPlusPreModel

#Gets project directory (one level above src)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#Runs the experiments for the different models
def run_experiments(data_frames = []):
    exp_functions = {
        'rr': [RandomResponseModel, 'rr_exp'],
        'crr': [ContextualRandomResponseModel, 'crr_exp'],
        'wsls': [WinStayLoseShiftModel, 'wsls_exp'],
        'rw': [RoscorlaWagnerModel, 'rw_exp'],
        'crw': [ContextualRoscorlaWagnerModel, 'crw_exp'],
        'ck': [ChoiceKernelModel, 'ck_exp'],
        'crwck': [ContextualRascorlaWagnerChoiceKernelModel, 'crwck_exp'],
        'crwck_plus': [ContextualRascorlaWagnerChoiceKernelPlusModel, 'crwck_plus_exp'],
        'crwcknb': [ContextualRascorlaWagnerChoiceKernelNoBetaModel, 'crwcknb_exp'],
        'crwck_all': [ContextualRascorlaWagnerChoiceKernelAllModel, 'crwck_all_exp'],
        'crwck_pre': [ContextualRascorlaWagnerChoiceKernelPreModel, 'crwck_pre_exp'],
        'crwcksa': [ContextualRescorlaWagnerChoiceKernelSharedAlphaModel, 'crwcksa_exp'],
        'crwck_init': [ContextualRascorlaWagnerChoiceKernelInitModel, 'crwck_init_exp'],
        'crwck_plus_init': [ContextualRascorlaWagnerChoiceKernelPlusInitModel, 'crwck_plus_init_exp'],
        'crwcksa_plus': [ContextualRascorlaWagnerChoiceKernelPlusSharedAlphaModel, 'crwcksa_plus_exp'],
        'crwck_plus_pre': [ContextualRascorlaWagnerChoiceKernelPlusPreModel, 'crwck_plus_pre_exp']
    }

    for name in data_frames:
        if name in exp_functions:
            model_class, exp_function = exp_functions[name]
            model_instance = model_class()
            getattr(model_instance, exp_function)()
    
#Extracts data from json file, returning the full dataset, a list of subjects, and the EDS data (containing stim and response codes) split into Easy and Hard groups
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

            elif last_BL == "Hard":
                EDS_Hard[ratID][0].extend(stim_code)
                EDS_Hard[ratID][1].extend(resp_code)
                EDS_Hard[ratID][2] = rele_mode
                
        previous_session = session_type

    return jsondata, list_subjects, EDS_Easy, EDS_Hard

    #+______________________________________________________________________________________________________________________________________+

#Extracts the order of the first task for each subject, returning a list of subjects who did the first task in Easy, and a list of subjects who did the first task in Hard
def extract_order():
    jsondata, list_subjects, EDS_Easy, EDS_Hard = load_data()

    first_Easy = []
    first_Hard = []
    dict = {ratID:None for ratID in list_subjects}

    for entry in jsondata:
        session_type = entry['SessionType']
        ratID = entry['ratID']
        stim_code = entry['StimCode']
        resp_code = entry['RespCode']
        if session_type == "EDS_BL_Easy" and dict[ratID] == None:
            dict[ratID] = "Easy"

        elif session_type == "EDS_BL_Hard" and dict[ratID] ==None:
            dict[ratID] = "Hard"

    for k,v in dict.items():
        if v == "Easy":
            first_Easy.append(k)
        else:
            first_Hard.append(k)

    return first_Easy, first_Hard

#Plots the parameters of the models, comparing the Easy and Hard groups
def plot_params(dataframes=[], remove=['subject', 'group', 'negLL', 'BIC', 'optimal_init_params', 'Q_init','CK_init','n'], format = (), colors = ()):

    (_alpha, error_size, _width) = format
    (color1,color2) = colors
    base_path=os.path.join(PROJECT_ROOT, 'data', 'data_frames')
    save_path=os.path.join(PROJECT_ROOT, 'plots', 'param_comparison')
    
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    })

    dfs = {}

    for name in dataframes:
        file_path = os.path.join(base_path, f"df_{name}.csv")
        dfs[name] = pd.read_csv(file_path)
    
    for name in dataframes:
        df = pd.DataFrame(dfs[name])
        df_melted = df.melt(id_vars=['subject', 'group'], var_name='metric', value_name='value')
        
        metrics = list(df.columns)
        metrics = [item for item in metrics if item not in remove]

        n = len(metrics)
        if n == 1:
            num_rows = 1
            num_cols = 1
        else: 
            num_cols = 2
            num_rows = math.ceil(n/num_cols)

        plt.figure(figsize=(16, 10))
        
        for i, metric in enumerate(metrics, 1):
            plt.subplot(num_rows, num_cols, i)

            # Calculate mean and standard deviation for each group
            summary = df_melted[df_melted['metric'] == metric].groupby('group')['value'].agg(['mean', 'std','size']).reset_index()

            summary['std_error'] = summary['std']/np.sqrt(summary['size'])
            
            # Set width for bars (increased width)
            width = _width  
            
            # Create positions for bars with reduced spacing
            x = np.arange(len(summary['group']))  # Reduced spacing between positions
            
            # Plot bars
            plt.bar(x, summary['mean'], width, 
                   color=[color1, color2], alpha=_alpha)
            
            # Add error bars
            plt.errorbar(x, summary['mean'], yerr=summary['std_error'], 
                        fmt='none', color='black', capsize=error_size, capthick=2)
            
            # Add individual data points
            for idx, group in enumerate(summary['group']):
                group_data = df_melted[(df_melted['metric'] == metric) & 
                                     (df_melted['group'] == group)]['value']
                plt.scatter([idx] * len(group_data), group_data, 
                          color='black', alpha=0.5, s=50)
                
                # Add connecting lines for each subject
                for subject in df['subject'].unique():
                    subject_data = df_melted[(df_melted['subject'] == subject) & 
                                           (df_melted['metric'] == metric)]
                    if len(subject_data) == 2:
                        plt.plot([0, 1], subject_data['value'], 
                               color='gray', linestyle='-', alpha=0.5, linewidth=1)

            plt.title(f'{metric} comparison', pad=20, fontweight='bold')
            plt.xlabel('Group', labelpad=10, fontweight='bold')
            plt.ylabel('Value', labelpad=10, fontweight='bold')
            
            # Set x-axis ticks
            plt.xticks(x, summary['group'])
            
            # Adjust plot limits to reduce empty space
            plt.margins(x=0.15)  # Reduced from default
            
            # Increase tick label padding
            plt.tick_params(axis='both', which='major', pad=8)
            
            # Add grid for better readability
            plt.grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout(pad=3.0)
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, f'{name}_param_comparison.eps'), format = 'eps', dpi=300, bbox_inches='tight')

#Plots the parameters of the models, comparing groups who first did the Easy task vs those who first did the Hard task
def plot_order_effects_comp_first(dataframes=[], remove = [],  first_Easy=None, first_Hard=None, format = (), colors = ()):
    base_path=os.path.join(PROJECT_ROOT, 'data', 'data_frames') 
    save_path=os.path.join(PROJECT_ROOT, 'plots', 'order_effects_comp_first') 
    (_alpha, error_size, _width) =  format
    (color1, color2) = colors

    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    })

    easy_hard = ["First_Easy", "First_Hard"]
    dfs = {}

    for name in dataframes:
        file_path = os.path.join(base_path, f"df_{name}.csv")
        dfs[name] = pd.read_csv(file_path)

    for name in dataframes:
        df = pd.DataFrame(dfs[name])
        df['first'] = np.where(df['subject'].isin(first_Easy), 'First_Easy', 'First_Hard')
        df['group_first'] = df['first']

        df_melted = df.melt(id_vars=['subject', 'group', 'first', 'group_first'], 
                           var_name='metric', value_name='value')

        metrics = list(df.columns)
        metrics = [item for item in metrics if item not in remove]

        n = len(metrics)
        if n == 1:
            num_rows = 1
            num_cols = 1
        else: 
            num_cols = 2
            num_rows = math.ceil(n / num_cols)

        plt.figure(figsize=(16, 10))

        for i, metric in enumerate(metrics, 1):
            plt.subplot(num_rows, num_cols, i)
            
            summary = df_melted[df_melted['metric'] == metric].groupby(['group_first', 'group'])['value'].agg(['mean', 'std', 'size']).reset_index()

            summary['std_error'] = summary['std']/np.sqrt(summary['size'])

            width = _width/2
            x = np.arange(len(easy_hard))
            
            easy_data = summary[summary['group'] == 'Easy']
            hard_data = summary[summary['group'] == 'Hard']
            
            plt.bar(x - width/2, easy_data['mean'], width, label='Easy', 
                   color=color1, alpha=_alpha)
            plt.bar(x + width/2, hard_data['mean'], width, label='Hard',
                   color=color2, alpha=_alpha)
            
            plt.errorbar(x - width/2, easy_data['mean'], yerr=easy_data['std_error'], 
                        fmt='none', color='black', capsize=error_size, capthick=2)
            plt.errorbar(x + width/2, hard_data['mean'], yerr=hard_data['std_error'], 
                        fmt='none', color='black', capsize=error_size, capthick=2)
            
            # Plot points and connecting lines for each subject within First_Easy and First_Hard groups
            for group_first in easy_hard:
                group_first_data = df_melted[(df_melted['metric'] == metric) & 
                                           (df_melted['group_first'] == group_first)]
                
                x_pos = easy_hard.index(group_first)
                
                # Plot points and connecting lines for each subject
                for subject in group_first_data['subject'].unique():
                    subject_data = group_first_data[group_first_data['subject'] == subject]
                    if len(subject_data) == 2:  # Make sure we have both Easy and Hard data
                        # Get the values for Easy and Hard
                        easy_val = subject_data[subject_data['group'] == 'Easy']['value'].values[0]
                        hard_val = subject_data[subject_data['group'] == 'Hard']['value'].values[0]
                        
                        # Plot points
                        plt.scatter(x_pos - width/2, easy_val, color='black', alpha=0.5, s=50)
                        plt.scatter(x_pos + width/2, hard_val, color='black', alpha=0.5, s=50)
                        
                        # Draw connecting line
                        plt.plot([x_pos - width/2, x_pos + width/2], [easy_val, hard_val], 
                               color='gray', alpha=0.5, linewidth=1)

            plt.title(f'{metric} comparison', pad=20, fontweight='bold')
            plt.xlabel('First Task', labelpad=10, fontweight='bold')
            plt.ylabel('Value', labelpad=10, fontweight='bold')
            plt.legend(title='Group', title_fontsize=12)
            
            plt.xticks(x, easy_hard)
            plt.grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout(pad=3.0)

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, f'{name}_param_comparison.eps'), 
                    dpi=300, bbox_inches='tight')

#Processes the dataframes, extracting the optimal parameters and saving them to a new dataframe where each fitted parameter has an individual column
def process_df(data_frames =[]):
    base_path = os.path.join(PROJECT_ROOT, 'data', 'data_frames')
    # Ensure the directory exists before saving files
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        
    for name in data_frames:
        path = os.path.join(base_path, f'df_{name}_original.csv')
        df = pd.read_csv(path)

        # Extract param_opt values into separate columns
        df[['alpha_opt', 'beta_opt']] = df['param_opt'].str.extract(r'\[([0-9.]+)\s+([0-9.]+)\]').astype(float)
        # Drop the original param_opt column
        df = df.drop(columns=['param_opt'])
        df.to_csv(os.path.join(PROJECT_ROOT, 'data', 'data_frames', 'df_ck.csv'), index = False)
        return df

#Prints the stats of the dataframes
def print_stats(data_frames = []):
    base_path = os.path.join(PROJECT_ROOT, 'data', 'data_frames')
    for name in data_frames:
        print(f'MODEL: {name}')
        path = os.path.join(base_path, f'df_{name}.csv')
        df = pd.read_csv(path)
        print(df)

#Renames the variables in the dataframes to be more readable
def rename_variable(data_frames = [], remove = []):
    base_path = os.path.join(PROJECT_ROOT, 'data', 'data_frames')
    # Ensure the directory exists before saving files
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        
    alpha = '\u03B1'  # Greek letter alpha
    beta = '\u03B2'   # Greek letter beta

    epsilon = '\u03B5'
    b_0 = '\u0062\u2080'  # beta with subscript 0
    b_1 = '\u0062\u2081'  # beta with subscript 1

    # Update the subscripts to use proper Unicode characters
    alpha_rw = '\u03B1\u02B3\u02B7'  # alpha with subscript rw
    alpha_ck = '\u03B1\u1D9C\u1D4F'  # alpha with subscript ck
    beta_rw = '\u03B2\u02B3\u02B7'  # beta with subscript rw
    beta_ck = '\u03B2\u1D9C\u1D4F'  # beta with subscript ck

    for name in data_frames:
        path = os.path.join(base_path, f'df_{name}.csv')
        df = pd.read_csv(path)
        columns = list(df.columns)
        columns = [item for item in columns if item not in remove]

        if name == 'rr':
            df = df.rename(columns = {columns[0]: f'Fitted b'})

        elif name == 'crr':
            df = df.rename(columns = {columns[0]: f'Fitted {b_0}', columns[1]: f'Fitted {b_1}'})  

        elif name == 'wsls':
            df = df.rename(columns = {columns[0]: f'Fitted {epsilon}'})

        elif name in ('rw','crw','ck', 'crwcksa'):
            df = df.rename(columns = {columns[0]: f'Fitted {alpha}', columns[1]: f'Fitted {beta}'})

        elif name in ('crwck', 'crwck_all', 'crwck_pre', 'crwck_init'):
            df = df.rename(columns = {columns[0]: f'Fitted {alpha_rw}', columns[1]: f'Fitted {alpha_ck}', columns[2]: f'Fitted {beta}'})

        elif name in ('crwck_plus', 'crwck_plus_pre', 'crwck_plus_init'):
            df = df.rename(columns = {columns[0]: f'Fitted {alpha_rw}', columns[1]: f'Fitted {alpha_ck}', columns[2]: f'Fitted {beta_rw}', columns[3]: f'Fitted {beta_ck}'})

        elif name == 'crwcknb':
            df = df.rename(columns = {columns[0]: f'Fitted {alpha_rw}', columns[1]: f'Fitted {alpha_ck}'})

        elif name == 'crwcksa_plus':
            df = df.rename(columns = {columns[0]: f'Fitted {alpha}', columns[1]: f'Fitted {beta_rw}', columns[2]: f'Fitted {beta_ck}'})

        df.to_csv(os.path.join(PROJECT_ROOT, 'data', 'data_frames', f'df_{name}.csv'), index = False)

#Plots the negative log likelihood and BIC for the different models, ordered by BIC
def plot_LL(dataframes=[]):
    base_path=os.path.join(PROJECT_ROOT, 'data', 'data_frames')
    save_path=os.path.join(PROJECT_ROOT, 'plots', 'LL')

    labels = dataframes
    dfs = {}

    for name in dataframes:
        file_path = os.path.join(base_path, f"df_{name}.csv")
        dfs[name] = pd.read_csv(file_path)

    # Calculate means for all metrics
    negLL_means = {name: np.mean(dfs[name]['negLL']) for name in dataframes}

    BIC_means = {name: np.mean(dfs[name]['BIC']) for name in dataframes}
    
    # Sort models by increasing BIC
    sorted_models = sorted(BIC_means.keys(), key=lambda x: BIC_means[x])
    
    # Calculate statistics for sorted models
    Easy_negLL_means = [np.mean(dfs[name][dfs[name]['group'] == "Easy"]['negLL']) for name in sorted_models]
    Easy_negLL_devs = [np.std(dfs[name][dfs[name]['group'] == "Easy"]['negLL']) for name in sorted_models]
    
    Hard_negLL_means = [np.mean(dfs[name][dfs[name]['group'] == "Hard"]['negLL']) for name in sorted_models]
    Hard_negLL_devs = [np.std(dfs[name][dfs[name]['group'] == "Hard"]['negLL']) for name in sorted_models]
    
    negLL_means_sorted = [negLL_means[name] for name in sorted_models]
    negLL_std_devs = [np.std(dfs[name]['negLL']) for name in sorted_models]
    
    BIC_means = [np.mean(dfs[name]['BIC']) for name in sorted_models]
    BIC_std_devs = [np.std(dfs[name]['BIC']) for name in sorted_models]

    # First plot: Easy vs Hard groups
    plt.figure(figsize=(10, 6))
    
    plt.errorbar(sorted_models, Easy_negLL_means, yerr=Easy_negLL_devs, 
                fmt='-o', capsize=5, capthick=2, ecolor='red', color='blue', 
                label='Easy negLL')

    plt.errorbar(sorted_models, Hard_negLL_means, yerr=Hard_negLL_devs, 
                fmt='-o', capsize=5, capthick=2, ecolor='green', color='orange', 
                label='Hard negLL')

    plt.title('Mean and Standard Deviation of negLL for Easy and Hard groups for Each Model')
    plt.xlabel('Model')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)  # Rotate labels for better readability

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, 'negLL_EZ_HARD_plot.eps'), format = 'eps', bbox_inches='tight', dpi = 300)

    # Second plot: Overall negLL and BIC
    plt.figure(figsize=(10, 6))

    plt.errorbar(sorted_models, negLL_means_sorted, yerr=negLL_std_devs, 
                fmt='-o', capsize=5, capthick=2, ecolor='red', color='blue', 
                label='negLL')

    plt.errorbar(sorted_models, BIC_means, yerr=BIC_std_devs, 
                fmt='-s', capsize=5, capthick=2, ecolor='green', color='orange', 
                label='BIC')

    plt.title('Mean and Standard Deviation of negLL and BIC for Each Model')
    plt.xlabel('Model')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)  # Rotate labels for better readability

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, 'negLL_BIC_plot.eps'), format = 'eps', bbox_inches='tight', dpi = 300)

#Plots the simulation of the model vs the actual decisions made by rats, for the last N trials
def sim_vs_reality(data_frames = ['crwcksa_plus'], remove=[], N=500):
    base_path = os.path.join(PROJECT_ROOT, 'data', 'data_frames')
    save_path = os.path.join(PROJECT_ROOT, 'plots', 'sim_vs_reality')
    # Function which compares simulation under fitted variables of a model to that of the actual decisions made by rats.
    
    # Ensure the main save directory exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    models = {
        'crwck_plus': ContextualRascorlaWagnerChoiceKernelPlusModel,
        'crwcksa_plus': ContextualRascorlaWagnerChoiceKernelPlusSharedAlphaModel
    }

    _,_,EDS_Easy, EDS_Hard = load_data()

    easy_hard = {"Easy":EDS_Easy, 
                 "Hard": EDS_Hard}

    dfs = {}

    for name in data_frames:
        file_path = os.path.join(base_path, f"df_{name}.csv")
        dfs[name] = pd.read_csv(file_path)
        
    for name in data_frames:

        for subject in EDS_Easy.keys():

            model = models[name]()
            df = pd.DataFrame(dfs[name])

            _param_names = list(df.columns)
            _param_names = [item for item in _param_names if item not in remove]


            for (type, data) in easy_hard.items():
                _params = [df[(df['subject'] == subject)&(df['group']==type)][param_name].values[0] for param_name in _param_names]

                (_stim_codes, _resp_codes, _rele_mode) = data[subject]
                _actions, _rewards = get_actions_rewards(_resp_codes)

                GO_probs = model.probs_given_data(resp_codes=_resp_codes, stim_codes= _stim_codes, actions = _actions, rewards = _rewards, params = _params)


                #Extracting last N
                _actions = _actions[-N:]
                _rewards = _rewards[-N:]
                _stim_codes = _stim_codes[-N:]
                GO_probs = GO_probs[-N:]
                            # Plotting
                plt.figure(figsize=(20, 10))

                # Plot Actions (step plot for discrete data)
                plt.plot(_actions, label = 'Actions', color =  '#C0C0C0', linewidth = 2)
                #plt.step(range(len(_actions)), _actions, where='post', label='Actions', color='blue', linewidth=2)

                # Plot Rewards (step plot for discrete data)
                #plt.step(range(len(_rewards)), _rewards, where='post', label='Rewards', color='green', linewidth=2)

                # Plot Stim Codes (step plot for discrete data)
                #plt.step(range(len(_stim_codes)), _stim_codes, where='post', label='Stim Codes', color='red', linewidth=2)

                # Plot Stim Codes as horizontal line segments
                for i in range(len(_stim_codes)):
                    # Create a horizontal line segment centered on the time point
                    plt.hlines(_stim_codes[i], i-0.5, i+0.5, colors='red', linewidth=2, label='Stim Codes' if i==0 else "")
                    
                # Plot GO Probabilities (continuous line)
                plt.plot(GO_probs, label='GO Probabilities', color='#B87333' , linewidth=2)


                # Add labels, title, and legend
                plt.title(f'{type} - Actions, Stimuli and GO Probabilities Over Last {N} Trials')
                plt.xlabel('Trial Number')
                plt.ylabel('Value')
                plt.legend(loc='upper right')

                plt.grid(True, linestyle='--', alpha=0.5)
                plt.tight_layout()
                
                # Create subdirectory for the model if it doesn't exist
                model_save_path = os.path.join(save_path, name)
                if not os.path.exists(model_save_path):
                    os.makedirs(model_save_path)
                    
                plt.savefig(os.path.join(model_save_path, f'{name}_{type}_{subject}_last_sim_vs_reality.eps'), format = 'eps', dpi=300, bbox_inches='tight')
                plt.close()
        
#Plots the correlation matrix and scatterplots of the fitted parameters for the different models
def plot_correlation(data_frames = ['rw','crw','ck', 'crwck','crwcksa', 'crwcknb','crwck_all','crwck_pre','crwck_init', 'crwck_plus', 'crwcksa_plus', 'crwck_plus_pre', 'crwck_plus_init'], remove = []):
    base_path = os.path.join(PROJECT_ROOT, 'data', 'data_frames')
    dfs = {}

    for name in data_frames:
        file_path = os.path.join(base_path, f"df_{name}.csv")
        dfs[name] = pd.read_csv(file_path)

    for name in data_frames:
        df = pd.DataFrame(dfs[name])

        # Get the metrics (parameters) to plot
        metrics = list(df.columns)
        metrics = [item for item in metrics if item not in remove]

        # Create correlation matrix
        corr_matrix = df[metrics].corr()

        # Create a mask for the upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        # Set up the matplotlib figure
        plt.figure(figsize=(12, 10))

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt='.2f')

        plt.title(f'Correlation Matrix of Fitted Parameters - {name}', pad=20, fontweight='bold')
        plt.tight_layout()

        # Save the correlation matrix plot
        corr_save_path = os.path.join(PROJECT_ROOT, 'plots', 'correlation', 'correlation_matrices')
        if not os.path.exists(corr_save_path):
            os.makedirs(corr_save_path)
        plt.savefig(os.path.join(corr_save_path, f'{name}_correlation.eps'), format = 'eps', dpi=300, bbox_inches='tight')
        plt.close()

        # Calculate number of scatterplots needed (n choose 2)
        n_scatterplots = len(metrics) * (len(metrics) - 1) // 2
        
        # Create a figure with the exact number of subplots needed
        fig, axes = plt.subplots(1, n_scatterplots, figsize=(5*n_scatterplots, 5))
        if n_scatterplots == 1:
            axes = [axes]  # Convert single axis to list for consistency
        
        # Create scatterplots for each parameter pair
        plot_idx = 0
        for i, param1 in enumerate(metrics):
            for j, param2 in enumerate(metrics):
                if i < j:  # Only plot upper triangle
                    ax = axes[plot_idx]
                    sns.scatterplot(data=df, x=param1, y=param2, ax=ax, alpha=0.7)
                    ax.set_title(f'{param1} vs {param2}')
                    ax.grid(True, linestyle='--', alpha=0.3)
                    plot_idx += 1
        
        plt.suptitle(f'Parameter Scatterplots - {name}', fontsize=16, y=1.05)
        plt.tight_layout()
        
        # Save the scatterplot figure
        scatter_save_path = os.path.join(PROJECT_ROOT, 'plots', 'correlation', 'parameter_scatterplots')
        if not os.path.exists(scatter_save_path):
            os.makedirs(scatter_save_path)
        plt.savefig(os.path.join(scatter_save_path, f'{name}_scatterplots.eps'), format = 'eps', dpi=300, bbox_inches='tight')
        plt.close()

#Generates the various plots for the dataframes
def generate_plots(dfs = []):
    # Define silver and copper colors
    silver_color = '#C0C0C0'  # Silver
    copper_color = '#B87333'  # Copper
    _format = (0.8, 20, 0.6)  # alpha(opaque), error_bar_size, width_of_bars
    _colors = (silver_color, copper_color)

    first_Easy, first_Hard = extract_order()

    _remove=['subject', 'group', 'negLL', 'BIC', 'optimal_init_params', 'first', 'group_first', 'Q_init', 'CK_init', 'n']

    rename_variable(data_frames=dfs, remove = _remove)
    plot_LL(dataframes = dfs)
    plot_params(dataframes = dfs, format = _format, colors = _colors, remove = _remove)
    plot_order_effects_comp_first(dataframes=dfs, first_Easy=first_Easy, first_Hard= first_Hard, format = _format, colors =_colors, remove = _remove)
    sim_vs_reality(remove = _remove, N = 500)
    plot_correlation(data_frames = ['rw','crw','ck', 'crwck','crwcksa', 'crwcknb','crwck_all','crwck_pre','crwck_init', 'crwck_plus', 'crwcksa_plus', 'crwck_plus_pre', 'crwck_plus_init'], remove = _remove)

def main():
    dfs = ['rr', 'crr','wsls','rw','crw','ck', 'crwck','crwcksa', 'crwcknb','crwck_all','crwck_pre','crwck_init', 'crwck_plus', 'crwcksa_plus', 'crwck_plus_pre', 'crwck_plus_init']

    #run_experiments(data_frames=dfs)
    #generate_plots(dfs)
    _remove=['subject', 'group', 'negLL', 'BIC', 'optimal_init_params', 'first', 'group_first', 'Q_init', 'CK_init', 'n']

    plot_correlation(data_frames = ['rw','crw','ck', 'crwck','crwcksa', 'crwcknb','crwck_all','crwck_pre','crwck_init', 'crwck_plus', 'crwcksa_plus', 'crwck_plus_pre', 'crwck_plus_init'], remove = _remove)


if __name__ == "__main__":
    main()
