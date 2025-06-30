"""
---------------------------------------------------------------------------------
cog_sci_random_response_model.py

CogSci Random Respones Model

Source:
-------
    : https://ccn.studentorg.berkeley.edu/pdfs/papers/WilsonCollins_modelFitting.pdf


Modifaction Logs:
: 03 July 24     : zachcolinwolpe@gmail.com      : init
---------------------------------------------------------------------------------
"""

import plotly.graph_objects as go
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import os
from .cog_sci_learning_model_base import (MultiArmedBanditModels, add_diag_line, load_data, get_actions_rewards)

# Get the project root directory (parent of models/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RandomResponseModel(MultiArmedBanditModels):
    """
    -------------------------------------------------
    Random Response Model
    ---------------------

    We assume that participants do not engage with the task at all
    and simply press buttons at random, perhaps with a bias (b) for one 
    option over the other. 

    Free Parameters:
    ----------------
        : b (float): choice bias. Bias for option A.
    
    -------------------------------------------------
    """
    def __init__(self):
        pass

    def predict(self, b, N, noise=0):
        """Given parameter:b predict a sequence of actions."""
        actions = []
        for _ in range(N):
            _action = np.random.choice([0, 1], p=[b, 1-b])

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

    def neg_log_likelihood(self, b, actions, rewards, epsilon=1e-10):
        """
        Compute the negative log likelihood of the data given the model (function, parameters). 
        """
        N = len(actions)
        assert N == len(rewards), "actions and rewards must be the same length."

        log_likelihood = 0
        for i in range(N):
            action_prob = b if actions[i] == 0 else 1 - b
            action_prob = np.clip(action_prob, epsilon, 1 - epsilon)  # Ensure action_prob is not 0 or 1

            log_likelihood += np.log(action_prob)
    
        return -log_likelihood
    
    def optimize_calculus(self, args):
        stim_codes, actions, rewards = args

        k = sum(actions)
        n = len(stim_codes)

        b_pred = 1-k/n

        negll = self.neg_log_likelihood(b_pred, actions, rewards)

        BIC = self.compute_BIC(negll, n)

        results = {'negLL':negll, 'b_pred': b_pred, 'BIC': BIC}

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
    
    def optimize_brute_force(self, actions, rewards, bounds=(0,1), loss_function=None):
        """
        Optimize the loss function using brute force search.
        """
        if loss_function is None:
            loss_function = self.neg_log_likelihood

        # perform parameter search
        b_values = np.linspace(bounds[0], bounds[1], 100)
        neg_log_likelihoods = [loss_function(b, actions, rewards) for b in b_values]
        
        # select param
        negLL_min_idx = np.argmin(neg_log_likelihoods)
        negLL_min = neg_log_likelihoods[negLL_min_idx]
        b_optimal = b_values[negLL_min_idx]

        T = len(actions)
        BIC = self.compute_BIC(negLL_min, T)

        results = {'negLL': negLL_min, 'b_pred': b_optimal, 'BIC': BIC}

        return results

    def compute_BIC(self, negLL, T):

        """
        Compute the Bayesian Information Criterion (BIC) for the model.
        """
        return super().compute_BIC(negLL, T, k_params=1)
    
    def plot_neg_log_likelihood(self, b_true=None, actions = None, rewards = None, _plt=None):

        if actions is None:
            actions = self.simulated_experiment['action']
        if rewards is None:
            rewards = self.simulated_experiment['reward']

        # evaluate likelihood
        negll = []
        for _b in np.linspace(0, 1, 100):
            negll.append(self.neg_log_likelihood(_b, actions, rewards))
        b_pred = np.linspace(0, 1, 100)[np.argmin(negll)]

        if _plt is None:
            _plt = plt.figure()
        plt.plot(np.linspace(0, 1, 100), negll)

        if b_true is not None:
            plt.axvline(b_true, color='green', linestyle='--', label='b (true):{:.2f}'.format(b_true))
        plt.axvline(b_pred, color='red', linestyle='--', label='b_hat:{:.2f}'.format(b_pred))
        plt.xlabel('b')
        plt.ylabel('Negative Log Likelihood')
        plt.title('Negative Log Likelihood')
        plt.legend()
        return plt

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

    def rr_exp(self):

        # Load data using the data_loader function
        jsondata, list_subjects, EDS_Easy, EDS_Hard = load_data()

        rr_easy_results = {}
        rr_hard_results = {}

        for k,v in EDS_Easy.items():
            stim_code = v[0]
            resp_code = v[1]
            actions, rewards = get_actions_rewards(resp_code)
            _args = (stim_code, actions, rewards)
            res = self.optimize_calculus(args = _args)
            rr_easy_results[k]=res

        for k,v in EDS_Hard.items():
            stim_code = v[0]
            resp_code = v[1]
            actions, rewards = get_actions_rewards(resp_code)
            _args = (stim_code, actions, rewards)
            res = self.optimize_calculus(args = _args)
            rr_hard_results[k] = res

        # Convert dictionaries to DataFrames
        df1 = pd.DataFrame.from_dict(rr_easy_results, orient='index').reset_index().rename(columns={'index': 'subject'})
        df2 = pd.DataFrame.from_dict(rr_hard_results, orient='index').reset_index().rename(columns={'index': 'subject'})

        # Add a column to distinguish between the two groups
        df1['group'] = 'Easy'
        df2['group'] = 'Hard'

        # Combine the DataFrames
        df_rr = pd.concat([df1, df2])
        
        # Save to the correct path using PROJECT_ROOT
        output_path = os.path.join(PROJECT_ROOT, 'data', 'data_frames', 'df_rr.csv')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_rr.to_csv(output_path, index = False)
