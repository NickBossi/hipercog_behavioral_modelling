"""
Contextual Rescorla-Wagner with Choice Kernel Model (No Beta)
-------------------------------------------------
This model combines two learning mechanisms:
1. Rescorla-Wagner learning: Updates action values based on reward prediction errors
2. Choice Kernel learning: Tracks choice history independently of rewards

The model maintains separate Q-values and choice kernels for each context (stimulus),
allowing it to learn context-specific action values and choice preferences.

Unlike the standard model, this version does not use a beta parameter for softmax,
instead using direct probability calculations.

Source:
-------
- Rescorla-Wagner model: https://ccn.studentorg.berkeley.edu/pdfs/papers/WilsonCollins_modelFitting.pdf
- Choice Kernel model: https://www.sciencedirect.com/science/article/pii/S0893608009000520
"""

import plotly.graph_objects as go
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
from plotly.subplots import make_subplots

from .cog_sci_learning_model_base import (MultiArmedBanditModels, add_diag_line, load_data, get_actions_rewards)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ContextualRascorlaWagnerChoiceKernelNoBetaModel(MultiArmedBanditModels):
    """
    A reinforcement learning model that combines Rescorla-Wagner learning with a choice kernel,
    without using a beta parameter for softmax.
    
    The model learns in two ways:
    1. Through reward prediction errors (Rescorla-Wagner)
    2. Through choice history (Choice Kernel)
    
    Parameters:
    -----------
    alpha_rw : float
        Learning rate for Rescorla-Wagner updates (0-1)
    alpha_choice : float
        Learning rate for choice kernel updates (0-1)
    """
    def __init__(self):
        """Initialize the model with default parameters."""
        pass

    def predict(self):
        """Predict actions given current model parameters."""
        pass

    def simulate(self, alpha_rw, alpha_choice, N=100, Q_init=[0, 0], noise=True):
        """
        Simulate the Rescorla-Wagner with Choice Kernel model over N trials.

        Parameters:
        -----------
        alpha_rw : float
            Learning rate for Rescorla-Wagner updates
        alpha_choice : float
            Learning rate for choice kernel updates
        N : int
            Number of trials to simulate
        Q_init : list[float, float]
            Initial Q-values for each action
        noise : bool
            Whether to add noise to choices (True) or choose deterministically (False)

        Returns:
        --------
        self : ContextualRascorlaWagnerChoiceKernelNoBetaModel
            The model instance with simulation results stored in simulated_experiment
        """
        # Store simulation parameters
        self.simulated_params = {'alpha_rw': alpha_rw, 'alpha_choice': alpha_choice, 'N': N, 'Q_init': Q_init, 'noise': noise}

        # Initialize arrays to store choices, rewards, and stimulus codes
        c = np.zeros((N), dtype=int)  # choices
        r = np.zeros((N), dtype=int)  # rewards
        s = np.zeros((N), dtype=int)  # stimulus codes

        # Initialize Q-values and choice kernels for both contexts
        Q0 = Q_init.copy()  # Q-values for context 0
        Q1 = Q_init.copy()  # Q-values for context 1
        CK0 = [0, 0]  # Choice kernel for context 0
        CK1 = [0, 0]  # Choice kernel for context 1

        # Initialize arrays to store Q and CK values over time
        Q_stored = np.zeros((2, 2, N), dtype=float)  # [context, action, trial]
        CK_stored = np.zeros((2, 2, N), dtype=float)  # [context, action, trial]

        # Run simulation for N trials
        for t in range(N):
            # Randomly select context (stimulus)
            if np.random.rand() < 0.5:
                s[t] = 0  # Context 0
                # Store current Q and CK values
                Q_stored[0, :, t] = Q0
                CK_stored[0, :, t] = CK0
                
                # Compute choice probabilities using both Q and CK values
                # Direct probability calculation without beta
                total = np.exp(Q0[0] + CK0[0]) + np.exp(Q0[1] + CK0[1])
                p0 = np.exp(Q0[0] + CK0[0]) / total
                p1 = 1 - p0

                # Make choice with or without noise
                if noise:
                    if np.random.random_sample(1) < p0:
                        c[t] = 0
                    else:
                        c[t] = 1
                else:
                    c[t] = np.argmax([p0, p1])
                
                # Generate reward (1 if choice matches context, 0 otherwise)
                if c[t] == s[t]:
                    r[t] = 1

                # Update Q-values using Rescorla-Wagner rule
                delta = r[t] - Q0[c[t]]  # Prediction error
                Q0[c[t]] = Q0[c[t]] + alpha_rw * delta

                # Update choice kernel based on chosen action
                if c[t] == 0:
                    CK0[0] = CK0[0] + alpha_choice * (1 - CK0[0])  # Increase for chosen action
                    CK0[1] = CK0[1] + alpha_choice * (0 - CK0[1])  # Decrease for unchosen action
                else:
                    CK0[0] = CK0[0] + alpha_choice * (0 - CK0[0])
                    CK0[1] = CK0[1] + alpha_choice * (1 - CK0[1])

            else:
                s[t] = 1  # Context 1
                # Store current Q and CK values
                Q_stored[1, :, t] = Q1
                CK_stored[1, :, t] = CK1
                
                # Compute choice probabilities using both Q and CK values
                total = np.exp(Q1[0] + CK1[0]) + np.exp(Q1[1] + CK1[1])
                p0 = np.exp(Q1[0] + CK1[0]) / total
                p1 = 1 - p0

                # Make choice with or without noise
                if noise:
                    if np.random.random_sample(1) < p0:
                        c[t] = 0
                    else:
                        c[t] = 1
                else:
                    c[t] = np.argmax([p0, p1])
                
                # Generate reward
                if c[t] == s[t]:
                    r[t] = 1

                # Update Q-values using Rescorla-Wagner rule
                delta = r[t] - Q1[c[t]]
                Q1[c[t]] = Q1[c[t]] + alpha_rw * delta

                # Update choice kernel based on chosen action
                if c[t] == 0:
                    CK1[0] = CK1[0] + alpha_choice * (1 - CK1[0])
                    CK1[1] = CK1[1] + alpha_choice * (0 - CK1[1])
                else:
                    CK1[0] = CK1[0] + alpha_choice * (0 - CK1[0])
                    CK1[1] = CK1[1] + alpha_choice * (1 - CK1[1])

        # Store simulation results
        self.simulated_experiment = {
            'action': c,          # Sequence of actions taken
            'reward': r,          # Sequence of rewards received
            'Q_stored': Q_stored, # Q-values over time for both contexts
            'CK_stored': CK_stored, # Choice kernel values over time for both contexts
            'stim_codes': s       # Sequence of contexts
        }
        return self

    def neg_log_likelihood(self, parameters, stim_codes, actions, rewards, Q_init = [0,0]):
        """
        Compute the negative log likelihood of the model given parameters and data.

        Parameters:
        -----------
        parameters : tuple
            (alpha_rw, alpha_choice) model parameters
        stim_codes : array
            Sequence of stimulus codes (contexts)
        actions : array
            Sequence of actions taken
        rewards : array
            Sequence of rewards received
        Q_init : list[float, float]
            Initial Q-values

        Returns:
        --------
        negLL : float
            Negative log likelihood of the data under the model
        """
        # Extract parameters
        alpha_rw, alpha_choice = parameters
        
        # Initialize Q-values and choice kernels
        Q = [Q_init,Q_init]
        N = len(actions)
        CK = [[0,0],[0,0]]
        log_likelihood = 0
        choice_probs = np.zeros((N), dtype = float)

        # Compute likelihood for each trial
        for i in range(N):
            if stim_codes[i] == 0:
                # Compute choice probabilities using direct probability calculation
                total = np.exp(Q[0][0] + CK[0][0]) + np.exp(Q[0][1] + CK[0][1])
                p0 = np.exp(Q[0][0] + CK[0][0]) / total
                p = [p0, 1-p0]

                # Store choice probability for actual choice
                choice_probs[i] = p[actions[i]]

                # Update choice kernel
                if actions[i] == 0:
                    CK[0][0] = CK[0][0] + alpha_choice * (1-CK[0][0])
                    CK[0][1] = CK[0][1] + alpha_choice * (0-CK[0][1])
                else:
                    CK[0][0] = CK[0][0] + alpha_choice * (0-CK[0][0])
                    CK[0][1] = CK[0][1] + alpha_choice * (1-CK[0][1])

                # Update Q-values using Rescorla-Wagner rule
                delta = rewards[i]-Q[0][actions[i]]
                Q[0][actions[i]] = Q[0][actions[i]] + alpha_rw*delta

            else:
                # Same process for context 1
                total = np.exp(Q[1][0] + CK[1][0]) + np.exp(Q[1][1] + CK[1][1])
                p0 = np.exp(Q[1][0] + CK[1][0]) / total
                p = [p0, 1-p0]

                choice_probs[i] = p[actions[i]]

                if actions[i] == 0:
                    CK[1][0] = CK[1][0] + alpha_choice * (1-CK[1][0])
                    CK[1][1] = CK[1][1] + alpha_choice * (0-CK[1][1])
                else:
                    CK[1][0] = CK[1][0] + alpha_choice * (0-CK[1][0])
                    CK[1][1] = CK[1][1] + alpha_choice * (1-CK[1][1])

                delta = rewards[i]-Q[1][actions[i]]
                Q[1][actions[i]] = Q[1][actions[i]] + alpha_rw*delta
        
        # Compute negative log likelihood
        negLL = -np.sum(np.log(choice_probs))
        return negLL  

    def generate_parameter_init_range(self, alpha_range, log_progress=False):
        """
        Generate parameter combinations for grid search.

        Parameters:
        -----------
        alpha_range : array
            Range of alpha values to test
        log_progress : bool
            Whether to show progress bar

        Yields:
        -------
        tuple
            (alpha_rw, alpha_choice) parameter combinations
        """
        alpha_iterable = tqdm(alpha_range) if log_progress else alpha_range

        for _alpha_rw in alpha_iterable:
            for _alpha_ck in alpha_range:
                yield _alpha_rw, _alpha_ck

    def compute_BIC(self, LL, T, k_params=2):
        """
        Compute Bayesian Information Criterion.

        Parameters:
        -----------
        LL : float
            Log likelihood
        T : int
            Number of trials
        k_params : int
            Number of parameters in model

        Returns:
        --------
        float
            BIC value
        """
        return super().compute_BIC(LL, T, k_params=k_params)
    
    def optimize_scikit_model_over_init_parameters(
        
        self,
        stim_codes,
        actions,
        rewards,
        loss_function=None,
        alpha_init_range=np.linspace(0, 1, 5),
        bounds=((0, 1), (0, 1)),
        log_progress=True
        ):
        """
        Optimize model parameters using scikit-learn with multiple initial conditions.

        Parameters:
        -----------
        stim_codes : array
            Sequence of stimulus codes
        actions : array
            Sequence of actions
        rewards : array
            Sequence of rewards
        loss_function : callable
            Loss function to optimize
        alpha_init_range : array
            Range of initial alpha values
        bounds : tuple
            Parameter bounds
        log_progress : bool
            Whether to show progress bar

        Returns:
        --------
        dict
            Optimization results including optimal parameters and BIC
        """
        if loss_function is None:
            loss_function = self.neg_log_likelihood
        
        # Initialize optimization
        negLL = np.inf
        optimal_init_params = (None, None)

        # Generate parameter combinations
        gen_experiments = self.generate_parameter_init_range(alpha_range=alpha_init_range, log_progress=log_progress)

        # Try each initial condition
        for _alpha_rw, _alpha_ck in gen_experiments:              
            result, res_nll, param_fits = self.optimize_scikit(
                loss_function=loss_function,
                init_guess=[_alpha_rw, _alpha_ck],
                args=(stim_codes, actions, rewards),
                bounds=bounds)

            if result.fun < negLL:
                negLL = result.fun
                params_opt = result.x
                optimal_init_params = (_alpha_rw, _alpha_ck)
      
        # Compute BIC
        BIC = self.compute_BIC(negLL, len(actions), 2)
        return {'negLL': negLL, 'param_opt': params_opt, 'BIC': BIC, 'optimal_init_params': optimal_init_params} 
    
    def optimize_brute_force():
        pass

    def perform_sensitivity_analysis():
        pass

    def plot_neg_log_likelihood():
        pass

    def crwcknb_exp(self, m=5):
        # Load data using the data_loader function
        jsondata, list_subjects, EDS_Easy, EDS_Hard = load_data()

        crwcknb_easy_results = {}
        crwcknb_hard_results = {}

        alpha_bounds = (0.01,1)
        alpha_init = np.linspace(0,1,m)
        _bounds = (alpha_bounds,alpha_bounds)
        counter = 0

        for k,v in EDS_Easy.items():
            _stim_codes = v[0]
            _resp_codes = v[1]
            _actions, _rewards = get_actions_rewards(_resp_codes)
            _args = (_stim_codes, _actions, _rewards)
            res = self.optimize_scikit_model_over_init_parameters(
                stim_codes = _stim_codes,
                actions=_actions,
                rewards=_rewards,
                loss_function=None,
                alpha_init_range=alpha_init,
                bounds=_bounds,
                log_progress=True
                )
            
            #Extracting optimal parameters in to individual variables
            new_res = {
                'negLL': res['negLL'],
                'alpha_rw_opt': res['param_opt'][0],
                'alpha_ck_op': res['param_opt'][1],
                'BIC': res['BIC'],
                'optimal_init_params': res['optimal_init_params']
            }

            crwcknb_easy_results[k]=new_res
            counter +=1
            print(f"Progress:{(counter/30)*100}%")

        for k,v in EDS_Hard.items():
            _stim_codes = v[0]
            _resp_codes = v[1]
            _actions, _rewards = get_actions_rewards(_resp_codes)
            _args = (_stim_codes, _actions, _rewards)
            res = self.optimize_scikit_model_over_init_parameters(
                stim_codes = _stim_codes,
                actions=_actions,
                rewards=_rewards,
                loss_function=None,
                alpha_init_range=alpha_init,
                bounds=_bounds,
                log_progress=True
                )
            new_res = {
                'negLL': res['negLL'],
                'alpha_rw_opt': res['param_opt'][0],
                'alpha_ck_op': res['param_opt'][1],
                'BIC': res['BIC'],
                'optimal_init_params': res['optimal_init_params']
            }
            crwcknb_hard_results[k]=new_res
            counter +=1
            print(f"Progress:{(counter/30)*100}%")

        # Convert dictionaries to DataFrames
        df1 = pd.DataFrame.from_dict(crwcknb_easy_results, orient='index').reset_index().rename(columns={'index': 'subject'})
        df2 = pd.DataFrame.from_dict(crwcknb_hard_results, orient='index').reset_index().rename(columns={'index': 'subject'})

        # Add a column to distinguish between the two groups
        df1['group'] = 'Easy'
        df2['group'] = 'Hard'

        # Combine the DataFrames
        df_crwcknb = pd.concat([df1, df2])
        # Save to the correct path using PROJECT_ROOT
        output_path = os.path.join(PROJECT_ROOT, 'data', 'data_frames', 'df_crwcknb.csv')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_crwcknb.to_csv(output_path, index = False)
