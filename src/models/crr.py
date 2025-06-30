"""
---------------------------------------------------------------------------------
crr.py

Contextual Random Response Model

Source:
-------
    : https://ccn.studentorg.berkeley.edu/pdfs/papers/WilsonCollins_modelFitting.pdf


---------------------------------------------------------------------------------
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


class ContextualRandomResponseModel(MultiArmedBanditModels):
    """
    -------------------------------------------------
    Contextual Random Response Model
    ---------------------

    We assume that participants do not engage with the task at all
    and simply press buttons at random, perhaps with a bias (b) for one 
    option over the other. 

    Free Parameters:
    ----------------
        : p0 (float): NOGO probability given NOGO stimulus
        : p1 (float): GO probability given NOGO stimulus
    
    -------------------------------------------------
    """
    def __init__(self):
        pass

    def predict(self, stim_codes,  params = (0.5, 0.5), noise=0):
        """Given parameters predict a sequence of actions."""
        N = len(stim_codes)
        p0 = params[0]
        p1 = params[1]
        actions = []
        for i in range(N):
            if stim_codes[i] == 0:
                action = np.random.choice([0, 1], p=[p0, 1-p0])
            else:
                action = np.random.choice([0, 1], p=[p1, 1-p1])
            if np.random.rand() < noise:
                _action = 1 - _action
            actions.append(_action)
        return actions
        
    def simulate(self, b, N=100, mu=[0.2, 0.8], noise=0.1):
        self.simulated_params = {'b': b, 'N': N, 'mu':mu, 'noise': noise}

        assert b <= 1, 'b out of range.'
        assert b >= 0, 'b out of range.'

        action = []
        reward = []
        for _ in range(N):
            _action = np.random.choice([0, 1], p=[b, 1-b])

            if np.random.rand() < noise:
                _action = 1 - _action
            action.append(_action)

            # reward == A if action == A with probability mu[A]
            _reward = np.random.rand() < mu[_action]
            reward.append(int(_reward))
       
        self.simulated_experiment = {'action': action, 'reward': reward}
        return self

    def neg_log_likelihood(self, params, stim_codes, actions, rewards, epsilon=1e-10):
        """
        Compute the negative log likelihood of the data given the model (function, parameters). 
        """
        p0 = params[0]
        p1 = params[1]
        #stim_codes, actions, rewards = args
        N = len(actions)
        assert N == len(rewards), "actions and rewards must be the same length."

        log_likelihood = 0
        for i in range(N):

            if stim_codes[i] == 0:
                action_prob = p0 if actions[i] == 0 else 1 - p0
                action_prob = np.clip(action_prob, epsilon, 1 - epsilon)  # Ensure action_prob is not 0 or 1

            else:
                action_prob = p1 if actions[i] == 0 else 1 - p1
                action_prob = np.clip(action_prob, epsilon, 1 - epsilon)  # Ensure action_prob is not 0 or 1

            log_likelihood += np.log(action_prob)
    
        return -log_likelihood
    
    def optimize_calculus(self, args):
        stim_codes, actions, rewards = args

        # Counters to track number of correct responses for each context
        k0 =0
        n0 = 0
        k1 = 0
        n1 = 0

        for i in range(len(stim_codes)):
            if stim_codes[i] == 0:
                n0 +=1
                if actions[i] == 0:
                    k0 += 1
            else:
                n1 += 1
                if actions[i] == 0:
                    k1 += 1
        
        p0_pred = k0/n0
        p1_pred = k1/n1

        _params = (p0_pred,p1_pred)
        negll = self.neg_log_likelihood(params = _params, stim_codes = stim_codes, actions = actions, rewards= rewards)

        T = len(actions)
        BIC = self.compute_BIC(negll,T)

        results = {'negLL': negll, 'p0_pred': p0_pred, 'p1_pred': p1_pred, 'BIC': BIC}
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
    
    def optimize_brute_force(self, args, bounds=(0,1), loss_function=None):
        """
        Optimize the loss function using brute force search.
        """
        stim_codes, _actions, _rewards = args
        if loss_function is None:
            loss_function = self.neg_log_likelihood

        # perform parameter search
        p0_values = np.linspace(bounds[0], bounds[1], 100)
        p1_values = np.linspace(bounds[0], bounds[1], 100)
        neg_log_likelihoods = [loss_function(p0, p1, args) for p0 in p0_values for p1 in p1_values]
        
        # select param
        negLL_min_idx = np.argmin(neg_log_likelihoods)
        negLL_min = neg_log_likelihoods[negLL_min_idx]

        # Convert the flattened index to 2D indices
        p0_idx = negLL_min_idx // 100  # Integer division by the size of p1_values
        p1_idx = negLL_min_idx % 100   # Remainder gives us the p1 index
        p0_optimal = p0_values[p0_idx]
        p1_optimal = p1_values[p1_idx]

        T = len(_actions)
        BIC = self.compute_BIC(negLL_min, T)

        results = {'negLL': negLL_min, 'p_pred': (p0_optimal, p1_optimal), 'BIC': BIC}

        return results

    def compute_BIC(self, negLL, T):

        """
        Compute the Bayesian Information Criterion (BIC) for the model.
        """
        return super().compute_BIC(negLL, T, k_params=2)
    
    def plot_neg_log_likelihood(self, args = None, _plt=None):

        if args is None:
            actions = self.simulated_experiment['action']
            rewards = self.simulated_experiment['reward']

        p0_range = np.linspace(0,1,100)
        p1_range  = np.linspace(0,1,100)

        negll = [self.neg_log_likelihood(p0, p1, args) for p0 in p0_range for p1 in p1_range]

        min_index = np.argmin(negll)
        negll = np.array(negll).reshape((100,100))
        p1_idx, p0_idx = np.unravel_index(min_index, negll.shape)

        p0_pred = p0_range[p0_idx]
        p1_pred = p1_range[p1_idx]

        if _plt is None:
            _plt = plt.figure(figsize = (8,6))

        plt.contourf(p0_range, p1_range, negll, levels=50, cmap='viridis')
        plt.colorbar(label='Negative Log Likelihood')
        plt.scatter(p0_pred, p1_pred, color='red', label=f'Pred (p0_pred:{round(p0_pred,2)}, p1_pred:{round(p1_pred,2)})', edgecolors='black', s=100)
        plt.xlabel('p0')
        plt.ylabel('p1')
        plt.title(f'Negative Log Likelihood (p0 vs p1)')
        plt.legend()

        plt.tight_layout()
        return plt, negll, p0_pred, p1_pred

    def compare_fitting_procedures(self, log_progress=True):
        """
        Compute brute force & scikit optim for a range of values.
        """
        b_range = np.linspace(0, 1, 100)
        b_range = tqdm(b_range) if log_progress else b_range

        results = {'b (true)': [], 'b (pred - brute force)': [], 'b (pred - scikit-optim)': []}
        for _b in b_range:
            self.simulate(b=_b)

            action = self.simulated_experiment['action']
            reward = self.simulated_experiment['reward']

            # brute force and scikit optim
            res_brute = self.optimize_brute_force(actions=action, rewards=reward)
            b_hat_brute_force = res_brute['b_pred']

            b_bounds = (0, 1)
            res_opt = self.optimize_scikit(init_guess=[0.5], args=(action, reward), bounds=[b_bounds])
            b_hat_scikit = res_opt['param_opt'][0]

            results['b (true)'].append(_b)
            results['b (pred - brute force)'].append(b_hat_brute_force)
            results['b (pred - scikit-optim)'].append(b_hat_scikit)

        results = pd.DataFrame(results)
        return results, results.plot()

    def perform_sensitivity_analysis(self, log_progress=True):
        
        results = {'b (true)': [], 'N': [], 'b (pred)': []}
        
        b_range = np.linspace(0, 1, 100)
        b_range = tqdm(b_range) if log_progress else b_range
        for _b in b_range:
            for _n in [10, 100, 500, 1000]:
                self.simulate(_b, N=_n)
            
                # estimate parameter: b
                action = self.simulated_experiment['action']
                reward = self.simulated_experiment['reward']

                # use brute force
                # res_brute = self.optimize_brute_force(actions=action, rewards=reward)
                # b_hat = res_brute['b_pred']

                # use scikit
                res_opt = self.optimize_scikit(init_guess=[0.5], args=(action, reward), bounds=[(0,1)])
                b_hat = res_opt['param_opt'][0]

                results['N'].append(_n)
                results['b (true)'].append(_b)
                results['b (pred)'].append(b_hat)
        
        results = pd.DataFrame(results)

        # plot
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Scatter(x=results['b (true)'], y=results['b (pred)'], mode='markers', name='b estimates'), row=1, col=1)
        add_diag_line(fig)  # add linear line

        fig.update_layout(height=600, width=800, title_text="Sensitivity Analysis: Parameter Stability Estimate", template='none')
        fig.update_layout(xaxis_title='b (true)', yaxis_title='b (pred)')
        fig

        return results, fig
    
    def crr_exp(self):

        # Load data using the data_loader function
        jsondata, list_subjects, EDS_Easy, EDS_Hard = load_data()

        crr_easy_results = {}
        crr_hard_results = {}

        for k,v in EDS_Easy.items():
            stim_code = v[0]
            resp_code = v[1]
            actions, rewards = get_actions_rewards(resp_code)
            _args = (stim_code, actions, rewards)
            res = self.optimize_calculus(args = _args)
            crr_easy_results[k]=res

        for k,v in EDS_Hard.items():
            stim_code = v[0]
            resp_code = v[1]
            actions, rewards = get_actions_rewards(resp_code)
            _args = (stim_code, actions, rewards)
            res = self.optimize_calculus(args = _args)
            crr_hard_results[k] = res

        # Convert dictionaries to DataFrames
        df1 = pd.DataFrame.from_dict(crr_easy_results, orient='index').reset_index().rename(columns={'index': 'subject'})
        df2 = pd.DataFrame.from_dict(crr_hard_results, orient='index').reset_index().rename(columns={'index': 'subject'})

        # Add a column to distinguish between the two groups
        df1['group'] = 'Easy'
        df2['group'] = 'Hard'

        # Combine the DataFrames
        df_crr = pd.concat([df1, df2])
        
        # Save to the correct path using PROJECT_ROOT
        output_path = os.path.join(PROJECT_ROOT, 'data', 'data_frames', 'df_crr.csv')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_crr.to_csv(output_path, index = False)
