"""
---------------------------------------------------------------------------------
cig_sci_win_stay_lose_shift_model.py


Parameter Interpretation:
-------------------------
    :
    :

Source:
-------
    : https://ccn.studentorg.berkeley.edu/pdfs/papers/WilsonCollins_modelFitting.pdf


Modifaction Logs:
: 03 July 24     : zachcolinwolpe@gmail.com      : init
---------------------------------------------------------------------------------
"""

import plotly.graph_objects as go
from scipy.optimize import minimize # finding optimal params in models
from scipy import stats             # statistical tools
import numpy as np                  # matrix/array functions
import pandas as pd                 # loading and manipulating data
import ipywidgets as widgets        # interactive display
import matplotlib.pyplot as plt     # plotting
from tqdm import tqdm
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os
from .cog_sci_learning_model_base import (MultiArmedBanditModels, add_diag_line, load_data, get_actions_rewards)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class WinStayLoseShiftModel(MultiArmedBanditModels):
    """
    -------------------------------------------------
    Win Stay Lose Shift Model
    -------------------------

    Consistent with the name, the win-stay-lose-shift model repeats rewarded actions and switches away from unrewarded actions.
    In the noisy version of the model, the win-stay-lose-shift rule is applied probabilistically,
    such that the model applies the win-stay-lose-shift rule with probability 1-ε, and chooses randomly with probability ε.

    Free Parameters:
    ----------------
        : epsilon (ε) (float). Probability of switching.

            P_t^k = { 
                : 1 - ε   : IF (c_{t_1} == k & r_t=1) OR (c_{t-1} != k & r_{t-1}=0)
                : ε       : IF (c_{t_1} != k & r_t=1) OR (c_{t-1} == k & r_{t-1}=0)
            }
            : 
        
    -------------------------------------------------
    """

    def __init__(self):
        pass

    def simulate(self, epsilon, N=100, mu=[0.2, 0.8], noise=0.1):
        self.simulated_params = {'epsilon': epsilon, 'N': N, 'mu':mu, 'noise': noise}
        action = []
        reward = []

        assert epsilon <= 1, 'epsilon out of range.'
        assert epsilon >= 0, 'epsilon out of range.'

        _action = np.random.choice([0,1]) # init random action
        for t in range(N):

            # reward == A if action == A with probability mu[A]
            _reward = int(np.random.rand() < mu[_action])

            # probability of next action
            if _reward == 1:
                prob_stay = 1 - epsilon/2
            if _reward == 0:
                prob_stay = epsilon/2
            prob_change = 1 - prob_stay

            # sample action space: IN TERMS OF FIXED CHOICES, NOT CURRENT ACTIONS
            _next_action = np.random.choice([_action, 1-_action], p=[prob_stay, prob_change])

            # inject noise
            if np.random.rand() < noise:
                _next_action = np.random.choice([0,1])

            # update
            action.append(_action)
            reward.append(int(_reward))
            _action = _next_action

        self.simulated_experiment = {'action': action, 'reward': reward}
        return self
       
    def predict(self, epsilon, reward_vector, noise=0):
        """Given parameter:b predict a sequence of actions."""

        actions = []
        _action = np.random.choice([0,1]) # init random action
        for _reward in reward_vector:
            # probability of next action
            if _reward == 1:
                prob_stay = 1 - epsilon
            if _reward == 0:
                prob_stay = epsilon
            prob_change = 1 - prob_stay

            # sample action space
            _next_action = np.random.choice([_action, 1-_action], p=[prob_stay, prob_change])

            # inject noise
            if np.random.rand() < noise:
                _next_action = 1 - _next_action

            # update
            _action = _next_action
            actions.append(_action)

        return actions

    def neg_log_likelihood(self, epsilon, actions, rewards, epsilon_clip=1e-10):
        """
        Compute the negative log likelihood of the data given the model (function, parameters). 
        """
        N = len(actions)
        assert N == len(rewards), "actions and rewards must be the same length."
    
        log_likelihood = 0      #assumes first action taken randomly.
        for i in range(1, N):

            #Changed to epsilon
            # probability of each action.
            if rewards[i-1] == 1:
                prob_stay = 1 - epsilon 
            else:
                prob_stay = epsilon 
            prob_change = 1 - prob_stay

            # probability of the action taken.
            if actions[i] == actions[i-1]:
                action_prob = prob_stay
            else:
                action_prob = prob_change
            
            action_prob = np.clip(action_prob, epsilon_clip, 1 - epsilon_clip)  # Ensure action_prob is not 0 or 1
            log_likelihood += np.log(action_prob)
        
        return -log_likelihood
    
    def optimize_calculus(self, args):
        _stim_codes, _actions, _rewards = args

        k = 0
        N = len(_stim_codes)
        n=N-1

        prev_choice = _actions[0]

        for i in range(1,N):
            if (_rewards[i-1] == 1 and _actions[i] == prev_choice) or (_rewards[i-1] == 0 and _actions[i] != prev_choice):
                k+=1
            prev_choice = _actions[i]
        
        epsilon_pred = (n-k)/n

        negll = self.neg_log_likelihood(epsilon=epsilon_pred, actions = _actions, rewards = _rewards)

        BIC = self.compute_BIC(negLL = negll, T = N)

        results = {'negLL': negll, 'epsilon_pred': epsilon_pred, 'BIC': BIC}
        return results


    def optimize_scikit(self, init_guess, args, bounds, loss_function=None):
        """
        Optimize the loss function using scikit-learn minimize.
        """
        if loss_function is None:
            loss_function = self.neg_log_likelihood

        result_object, negLL, param_opt = super().optimize_scikit(loss_function, init_guess, args, bounds)

        T = len(args[0])
        BIC = self.compute_BIC(negLL, T)

        return {'negLL': negLL, 'param_opt': param_opt, 'BIC': BIC}

    def optimize_brute_force(self, bounds, actions, rewards, loss_function=None):
        """
        Optimize the loss function using brute force search.
        """
        if loss_function is None:
            loss_function = self.neg_log_likelihood

        # perform parameter search
        epsilon_values = np.linspace(bounds[0], bounds[1], 100)
        neg_log_likelihoods = [loss_function(epsilon, actions, rewards) for epsilon in epsilon_values]
        
        # select param
        negLL_min_idx = np.argmin(neg_log_likelihoods)
        negLL_min = neg_log_likelihoods[negLL_min_idx]
        epsilon_optimal = epsilon_values[negLL_min_idx]
        
        # compute BIC
        T = len(actions)
        BIC = self.compute_BIC(negLL_min, T)

        results = {'negLL': negLL_min, 'epsilon_pred': epsilon_optimal, 'BIC': BIC}

        return results

    def plot_neg_log_likelihood(self, epsilon_true=None, action=None, reward=None, _plt=None):

        if action is None:
            action = self.simulated_experiment['action']
        if reward is None:
            reward = self.simulated_experiment['reward']

        # evaluate likelihood
        negll = []
        for _epsilon in np.linspace(0, 1, 100):
            negll.append(self.neg_log_likelihood(_epsilon, action, reward))
        epsilon_pred = np.linspace(0, 1, 100)[np.argmin(negll)]

        if _plt is None:
            _plt = plt.figure()
        plt.plot(np.linspace(0, 1, 100), negll)

        if epsilon_true is not None:
            plt.axvline(epsilon_true, color='green', linestyle='--', label='epsilon (true):{:.2f}'.format(epsilon_true))
        plt.axvline(epsilon_pred, color='red', linestyle='--', label='epsilon (pred):{:.2f}'.format(epsilon_pred))
        plt.xlabel('epsilon')
        plt.ylabel('Negative Log Likelihood')
        plt.title('Negative Log Likelihood')
        plt.legend()
        return plt

    def compare_fitting_procedures(self, log_progress=True):
        """
        Compute brute force & scikit optim for a range of values.
        """
        epsilon_range = np.linspace(0, 1, 100)
        epsilon_range = tqdm(epsilon_range) if log_progress else epsilon_range

        results = {'epsilon (true)': [], 'epsilon (pred - brute force)': [], 'epsilon (pred - scikit-optim)': []}
        for epsilon_true in epsilon_range:
            self.simulate(epsilon=epsilon_true)

            actions = self.simulated_experiment['action']
            rewards = self.simulated_experiment['reward']

            epsilon_bounds = (0, 1)

            # Brute force and scikit optim
            res_brute = self.optimize_brute_force(bounds=epsilon_bounds, actions=actions, rewards=rewards)
            epsilon_hat_brute_force = res_brute['epsilon_pred']

            res_opt = self.optimize_scikit(init_guess=[0.5], args=(actions, rewards), bounds=[epsilon_bounds])
            epsilon_hat_scikit = res_opt['param_opt'][0]

            results['epsilon (true)'].append(epsilon_true)
            results['epsilon (pred - brute force)'].append(epsilon_hat_brute_force)
            results['epsilon (pred - scikit-optim)'].append(epsilon_hat_scikit)

        results = pd.DataFrame(results)
        return results, results.plot()
        
    def perform_sensitivity_analysis(self, log_progress=True):
        results = {'epsilon (true)': [], 'N': [], 'epsilon (pred)': []}
        
        epsilon_range = np.linspace(0, 1, 100)
        epsilon_range = tqdm(epsilon_range) if log_progress else epsilon_range
        for epsilon_true in epsilon_range:
            for N in [10, 100, 500, 1000]:
                self.simulate(epsilon=epsilon_true, N=N)
            
                # Estimate parameter: epsilon
                actions = self.simulated_experiment['action']
                rewards = self.simulated_experiment['reward']

                res_opt = self.optimize_scikit(init_guess=[0.5], args=(actions, rewards), bounds=[(0,1)])
                epsilon_hat_scikit = res_opt['param_opt'][0]
                
                results['N'].append(N)
                results['epsilon (true)'].append(epsilon_true)
                results['epsilon (pred)'].append(epsilon_hat_scikit)
        
        results = pd.DataFrame(results)

        # Plot
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Scatter(x=results['epsilon (true)'], y=results['epsilon (pred)'], mode='markers', name='epsilon estimates'), row=1, col=1)
        add_diag_line(fig)  # Add diagonal line

        fig.update_layout(height=600, width=800, title_text="Sensitivity Analysis: Parameter Stability Estimate", template='none')
        fig.update_layout(xaxis_title='epsilon (true)', yaxis_title='epsilon (pred)')
        fig.show()


    def compute_BIC(self, negLL, T):

        """
        Compute the Bayesian Information Criterion (BIC) for the model.
        """
        return super().compute_BIC(negLL, T, k_params=1)
    
    def wsls_exp(self):
        # Load data using the data_loader function
        jsondata, list_subjects, EDS_Easy, EDS_Hard = load_data()

        wsls_easy_results = {}
        wsls_hard_results = {}

        for k,v in EDS_Easy.items():
            stim_code = v[0]
            resp_code = v[1]
            actions, rewards = get_actions_rewards(resp_code)
            _args = (stim_code, actions, rewards)
            res = self.optimize_calculus(args = _args)
            wsls_easy_results[k]=res

        for k,v in EDS_Hard.items():
            stim_code = v[0]
            resp_code = v[1]
            actions, rewards = get_actions_rewards(resp_code)
            _args = (stim_code, actions, rewards)
            res = self.optimize_calculus(args = _args)
            wsls_hard_results[k] = res

        # Convert dictionaries to DataFrames
        df1 = pd.DataFrame.from_dict(wsls_easy_results, orient='index').reset_index().rename(columns={'index': 'subject'})
        df2 = pd.DataFrame.from_dict(wsls_hard_results, orient='index').reset_index().rename(columns={'index': 'subject'})

        # Add a column to distinguish between the two groups
        df1['group'] = 'Easy'
        df2['group'] = 'Hard'

        # Combine the DataFrames
        df_wsls = pd.concat([df1, df2])

        # Save to the correct path using PROJECT_ROOT
        output_path = os.path.join(PROJECT_ROOT, 'data', 'data_frames', 'df_wsls.csv')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_wsls.to_csv(output_path, index = False)


    
