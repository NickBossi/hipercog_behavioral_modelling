"""
---------------------------------------------------------------------------------
cog_sci_random_response_model.py

CogSci Roscorla Wagner Model

Source:
-------
    : https://ccn.studentorg.berkeley.edu/pdfs/papers/WilsonCollins_modelFitting.pdf

Modifaction Logs:
: 05 July 24     : zachcolinwolpe@gmail.com      : init
---------------------------------------------------------------------------------
"""

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

from .cog_sci_learning_model_base import (MultiArmedBanditModels, add_diag_line, load_data, get_actions_rewards)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class RoscorlaWagnerModel(MultiArmedBanditModels):
    """
    -------------------------------------------------
    Roscorla Wagner Model
    ---------------------

    Rescorla Wagner Model model implementation & optimization (training).

    #### Config

    - *`N`* trails
    - *`K`* number of states (multi-armed bandits)
    - Goal is to maximise reward


    During each trial $t$ each action $k$ pays out reward $r_t$ with probability $µ^k_t$ and reward $0$ with probability $1-µ^k_t$.

    The goal is to maximise the reward over the $T$ trials.

    Parameter Interpretation:
    -------------------------
        : alpha: [0,1] : learning rate: how much the value of the action is updated after each trial.
            0 : no learning
            1 : deterministic updating (Q-value = reward)

        : beta: [0, inf] : inverse temperature: how much the agent exploits vs explores.
            0 : random choice
            inf : deterministic choice

    Source:
    -------
        : 
        : https://shawnrhoads.github.io/gu-psyc-347/module-03-01_Models-of-Learning.html


    -------------------------------------------------
    """
    def __init__(self, alpha_range=(0,1), theta_range=(0.1, 10)):
        self.alpha_range = alpha_range
        self.theta_range = theta_range

    def simulate(self, alpha, theta, N=100, mu=[0.2, 0.8], Q_init=[0, 0], noise=True):
        """
        Simulate the Rescorla-Wagner model

        Parameters:
        -----------
            : params (list[float]) : list of model parameters (alpha, theta)
                : alpha (float) : learning rate
                : theta (float) : inverse temperature
            : T (int) : number of trials.
            : mu (list[float]) : list of reward probabilities for each state.
            : Q_init list[float, float]: initial parameter bias.
            : noisy_choice (bool) : add noise to choice. If 0 do not inject any noise.
            : update_properties (bool): whether or not to update internal properties.
        """
        self.simulated_params = {'alpha': alpha, 'theta':theta, 'N': N, 'mu':mu, 'Q_init':Q_init, 'noise': noise}

        c = np.zeros((N), dtype=int)
        r = np.zeros((N), dtype=int)

        Q_stored = np.zeros((2, N), dtype=float)
        Q = Q_init  # no starting bias

        for t in range(N):
            # store values for Q_{t+1}
            Q_stored[:, t] = Q
            
            # compute choice probabilities
            p0 = np.exp(theta*Q[0]) / (np.exp(theta*Q[0]) + np.exp(theta*Q[1]))
            p1 = 1 - p0

            # sample choice K with probability p(k)
            if noise:
                if np.random.random_sample(1) < p0:
                    c[t] = 0
                else:
                    c[t] = 1
            else:
                # make choice without noise
                c[t] = np.argmax([p0, p1])
                
            # generate reward based on reward probability
            r[t] = np.random.rand() < mu[c[t]]

            # update values
            delta = r[t] - Q[c[t]]
            Q[c[t]] = Q[c[t]] + alpha * delta

        self.simulated_experiment = {'action': c, 'reward': r, 'Q_stored': Q_stored}
        return self

    def neg_log_likelihood(self, parameters, actions, rewards, Q_init=[0, 0], epsilon_clip=1e-10):
        """
        Compute the Negative Log-Likelihood of the Rescorla-Wagner model

        Parameters:
        -----------
            : model_parameters (list[float]) : list of model parameters (alpha, theta)
                : alpha (float) : learning rate
                : theta (float) : inverse temperature
            : actions (array) : choices
            : rewards (array) : rewards

        Returns:
        --------
            : negLL (float) : negative log-likelihood

        Source:
        -------
            : https://shawnrhoads.github.io/gu-psyc-347/module-03-01_Models-of-Learning.html
        """

        # extract params
        alpha, theta = parameters

        # assuming no starting bias
        Q = Q_init
        T = len(actions)
        choiceProb = np.zeros((T), dtype=float)
        
        for t in range(T):
            
            # compute choice probabilities for k=2
            p0 = np.exp(theta*Q[0]) / (np.exp(theta*Q[0]) + np.exp(theta*Q[1]))
            p = [p0, 1-p0]

            # compute choice probability for actual choice
            choiceProb[t] = p[actions[t]]

            # update values
            delta = rewards[t] - Q[actions[t]]
            Q[actions[t]] = Q[actions[t]] + alpha * delta

        negLL = -np.sum(np.log(choiceProb))
        
        return negLL
    
    def generate_parameter_init_range(self, alpha_range, theta_range, log_progress=False):
        """
        Generate params: (alpha, theta) pairs.
        """
        alpha_iterable = tqdm(alpha_range) if log_progress else alpha_range

        for _alpha in alpha_iterable:
            for _theta in theta_range:
                yield _alpha, _theta

    def compute_BIC(self, LL, T, k_params=2):
        return super().compute_BIC(LL, T, k_params=2)
        # bic = k * np.log(N) + 2 * neg_log_likelihood

    def optimize_brute_force(self, actions, rewards, bounds=((0,1), (0.1, 10)), n = 10, loss_function=None, log_progress=True):
        """
        Optimize the loss function using brute force search.
        """
        if loss_function is None:
            loss_function = self.neg_log_likelihood

        # extact parameter range
        alpha_bounds, theta_bounds = bounds
        alpha_values = np.linspace(alpha_bounds[0], alpha_bounds[1], n)
        theta_values = np.linspace(theta_bounds[0], theta_bounds[1], n)

        # generate experiments
        gen_experiments = self.generate_parameter_init_range(
            alpha_range=alpha_values,
            theta_range=theta_values,
            log_progress=log_progress
            )

        neg_log_likelihoods = []
        for _alpha, _theta in gen_experiments:
            _loss = loss_function((_alpha, _theta), actions, rewards)
            neg_log_likelihoods.append((_alpha, _theta, _loss))
        
        # Find the set with the minimum _loss
        alpha_optima, theta_optima, loss = min(neg_log_likelihoods, key=lambda x: x[2])

        # compute BIC
        BIC = self.compute_BIC(loss, len(actions), 2)

        results = {
            'negLL': loss,
            'alpha_pred': alpha_optima,
            'theta_pred': theta_optima,
            'BIC': BIC
        }

        return results

    def optimize_scikit(self, init_guess, args, bounds=((0,1), (0.1, 10)), loss_function=None, single = False):
        """
        Optimize the loss function using scikit-learn minimize.
        """
        if loss_function is None:
            loss_function = self.neg_log_likelihood


        if single == True:

            result_object, negLL, param_opt = super().optimize_scikit(loss_function, init_guess, args, bounds)

            T = len(args[0])
            BIC = self.compute_BIC(negLL, T)

            return {'negLL': negLL, 'param_opt': param_opt, 'BIC': BIC}
        
        else:
            return super().optimize_scikit(loss_function, init_guess, args, bounds)

    def optimize_scikit_model_over_init_parameters(
        self,
        actions,
        rewards,
        loss_function=None,
        alpha_init_range=np.linspace(0, 1, 5),
        theta_init_range=np.linspace(0.1, 10, 7),
        bounds=((0, 1), (0.1, 15)),
        log_progress=True
        ):

        if loss_function is None:
            loss_function = self.neg_log_likelihood
        
        # init log likelihood
        negLL = np.inf
        optimal_init_params = (None, None)

        # generate experiments
        gen_experiments = self.generate_parameter_init_range(alpha_range=alpha_init_range, theta_range=theta_init_range, log_progress=log_progress)

        # run experiments
        for _alpha, _theta in gen_experiments:              
            result, res_nll, param_fits = self.optimize_scikit(
                loss_function=loss_function,
                init_guess=[_alpha, _theta],
                args=(actions, rewards),
                bounds=bounds)

            if result.fun < negLL:
                negLL = result.fun
                params_opt = result.x
                optimal_init_params = (_alpha, _theta)
      
        # compute BIC

        BIC = self.compute_BIC(negLL, len(actions), 2)
        return {'negLL': negLL, 'param_opt': params_opt, 'BIC': BIC, 'optimal_init_params': optimal_init_params}

    def predict(self, alpha, theta, reward_vector, noise=0):
        """
        Given parameters alpha and theta, predict a sequence of actions.
        """
        raise Exception('Implementation not logically sound.')
        actions = []
        Q = [0.5, 0.5]  # Initialize Q-values
        for reward in reward_vector:
            # Compute choice probabilities
            p0 = np.exp(theta * Q[0]) / (np.exp(theta * Q[0]) + np.exp(theta * Q[1]))
            p1 = 1 - p0

            # Make choice with noise
            # if np.random.rand() < noise:
            # if np.random.rand() < noise:
            if noise is not None and noise > 0:
                if np.random.random_sample(1) < p0:
                    action = 0
                else:
                    action = 1
            else:
                # make choice without noise
                action = np.argmax([p0, p1])


            # Update Q-values
            delta = reward - Q[action]
            Q[action] = Q[action] + alpha * delta

            actions.append(action)

        return actions

    def perform_sensitivity_analysis(
        self,
        alpha_range=np.linspace(0, 1, 10),
        theta_range=np.linspace(0, 10, 10),
        N=100,
        bounds=((0, 1), (1, 10)),
        log_progress=True
        ):
        """
        Perform sensitivity analysis to evaluate parameter stability.
        """
        results = {'alpha (true)': [], 'theta (true)': [], 'N': [], 'alpha (pred)': [], 'theta (pred)': []}
        
        param_grid = self.generate_parameter_init_range(alpha_range=alpha_range, theta_range=theta_range, log_progress=log_progress)
        
        for alpha_true, theta_true in param_grid:
            # for N in [10, 100, 500, 1000]:
            self.simulate(alpha=alpha_true, theta=theta_true, N=N)
        
            # Estimate parameters: alpha and theta
            actions = self.simulated_experiment['action']
            rewards = self.simulated_experiment['reward']


            # use multiple init conditions
            results_dict = self.optimize_scikit_model_over_init_parameters(actions=actions, rewards=rewards, bounds=bounds, log_progress=False)
            res_nll = results_dict['negLL']
            param_fits = results_dict['param_opt']
            BIC = results_dict['BIC']
            optimal_init_params = results_dict['optimal_init_params']
            alpha_hat_scikit, beta_hat_scikit = param_fits

            
            results['N'].append(N)
            results['alpha (true)'].append(alpha_true)
            results['theta (true)'].append(theta_true)
            results['alpha (pred)'].append(alpha_hat_scikit)
            results['theta (pred)'].append(beta_hat_scikit)
        
        results = pd.DataFrame(results)

        # Save results to file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(current_dir, '..', 'data', 'data_frames', 'sensitivity_analysis')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'rw_sensitivity_analysis.csv')
        results.to_csv(save_path, index=False)
        print("RW DONE")

        # Plot
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Alpha Sensitivity", "Beta Sensitivity"))
        fig.add_trace(go.Scatter(x=results['alpha (true)'], y=results['alpha (pred)'], mode='markers', name='alpha estimates'), row=1, col=1)

        fig.add_trace(go.Scatter(x=results['theta (true)'], y=results['theta (pred)'], mode='markers', name='theta estimates'), row=1, col=2)
        
        # Add diagonal lines for reference
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],   mode='lines', line=dict(dash='dash'), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=[0, 12], y=[0, 12], mode='lines', line=dict(dash='dash'), showlegend=False), row=1, col=2)

        fig.update_layout(height=600, width=1200, title_text="Sensitivity Analysis: Parameter Stability Estimate", template='none')
        fig.update_xaxes(title_text='alpha (true)', row=1, col=1)
        fig.update_yaxes(title_text='alpha (pred)', row=1, col=1)
        fig.update_xaxes(title_text='theta (true)', row=1, col=2)
        fig.update_yaxes(title_text='theta (pred)', row=1, col=2)
        
        return results, fig

    def compare_fitting_procedures(
        self,
        alpha_range=np.linspace(0, 1, 10), 
        theta_range=np.linspace(0, 10, 10),
        fit_brute_force=True,
        fit_scikit=True,
        bounds=[(0, 1), (1, 10)],
        N=100,
        log_progress=True
        ):
        """
        Compare brute force and scikit optimization for a range of values.
        """
    
        # generate experiments
        gen_experiments = self.generate_parameter_init_range(alpha_range=alpha_range, theta_range=theta_range, log_progress=log_progress)

        results = {'alpha (true)': [], 'theta (true)': [], 'alpha (pred - brute force)': [], 'theta (pred - brute force)': [], 'alpha (pred - scikit-optim)': [], 'theta (pred - scikit-optim)': []}
        for alpha_true, theta_true in gen_experiments:

            # Simulate Data
            self.simulate(alpha=alpha_true, theta=theta_true, N=N)
            actions = self.simulated_experiment['action']
            rewards = self.simulated_experiment['reward']

            # Brute force and scikit optim
            if fit_brute_force:
                brute_force_results = self.optimize_brute_force(bounds=bounds, actions=actions, rewards=rewards, log_progress=False)
                alpha_hat_brute_force = brute_force_results['alpha_pred']
                theta_hat_brute_force = brute_force_results['theta_pred']
                BIC_brute_force = brute_force_results['BIC']
            else:
                alpha_hat_brute_force, theta_hat_brute_force, BIC_brute_force = None, None, None
            
            # use multiple init conditions (scikit)
            if fit_scikit:
                results_dict = self.optimize_scikit_model_over_init_parameters(actions=actions, rewards=rewards, bounds=bounds, log_progress=False)
                res_nll = results_dict['negLL']
                param_fits = results_dict['param_opt']
                BIC = results_dict['BIC']
                optimal_init_params = results_dict['optimal_init_params']
                alpha_hat_scikit, beta_hat_scikit = param_fits
            else:
                alpha_hat_scikit, beta_hat_scikit = None, None
            
            results['alpha (true)'].append(alpha_true)
            results['theta (true)'].append(theta_true)
            results['alpha (pred - brute force)'].append(alpha_hat_brute_force)
            results['theta (pred - brute force)'].append(theta_hat_brute_force)
            results['alpha (pred - scikit-optim)'].append(alpha_hat_scikit)
            results['theta (pred - scikit-optim)'].append(beta_hat_scikit)

        results = pd.DataFrame(results)

        # Create subplots
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Alpha Estimate", "Theta Estimate"))
        fig.add_trace(go.Scatter(x=results['alpha (true)'], y=results['alpha (pred - brute force)'], mode='markers', name='alpha pred (brute force)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=results['theta (true)'], y=results['theta (pred - brute force)'], mode='markers', name='theta pred (brute force)'), row=1, col=2)
        fig.add_trace(go.Scatter(x=results['alpha (true)'], y=results['alpha (pred - scikit-optim)'], mode='markers', name='alpha pred (scikit)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=results['theta (true)'], y=results['theta (pred - scikit-optim)'], mode='markers', name='theta pred (scikit)'), row=1, col=2)
        
        # Add diagonal lines for reference
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],   mode='lines', line=dict(dash='dash'), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=[0, 11], y=[0, 11], mode='lines', line=dict(dash='dash'), showlegend=False), row=1, col=2)

        fig.update_layout(height=600, width=1200, title_text="Compare Parameter Recovery: Brute Force vs Scikit", template='none')
        fig.update_xaxes(title_text='alpha (true)', row=1, col=1)
        fig.update_yaxes(title_text='alpha (pred)', row=1, col=1)
        fig.update_xaxes(title_text='theta (true)', row=1, col=2)
        fig.update_yaxes(title_text='theta (pred)', row=1, col=2)

        # fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        # # Plot for alpha
        # axes[0].scatter(results['alpha (true)'], results['alpha (pred - brute force)'], label='Brute Force', alpha=0.5)
        # axes[0].scatter(results['alpha (true)'], results['alpha (pred - scikit-optim)'], label='Scikit Optim', alpha=0.5)
        # axes[0].plot([0, 1], [0, 1], 'k--', label='Ideal')
        # axes[0].set_xlabel('Alpha (True)')
        # axes[0].set_ylabel('Alpha (Predicted)')
        # axes[0].set_title('Alpha: True vs Predicted')
        # axes[0].legend()
        # axes[0].grid(True)

        # # Plot for theta
        # axes[1].scatter(results['theta (true)'], results['theta (pred - brute force)'], label='Brute Force', alpha=0.5)
        # axes[1].scatter(results['theta (true)'], results['theta (pred - scikit-optim)'], label='Scikit Optim', alpha=0.5)
        # axes[1].plot([0, 10], [0, 10], 'k--', label='Ideal')
        # axes[1].set_xlabel('Theta (True)')
        # axes[1].set_ylabel('Theta (Predicted)')
        # axes[1].set_title('Theta: True vs Predicted')
        # axes[1].legend()
        # axes[1].grid(True)

        # plt.tight_layout()
        return results, fig

    def plot_neg_log_likelihood(self, _plt=None):
        """
        Plot the negative log likelihood as a function of alpha and theta.
        """

        # extract params from simulation
        alpha_true, theta_true = self.simulated_params['alpha'], self.simulated_params['theta']
        actions, rewards = self.simulated_experiment['action'], self.simulated_experiment['reward']

        negll = []
        alpha_range = np.linspace(0, 1, 100)
        theta_range = np.linspace(0, 10, 100)

        # generate experiments
        gen_experiments = self.generate_parameter_init_range(alpha_range=alpha_range, theta_range=theta_range, log_progress=False)

        # run experiments
        for _alpha, _theta in gen_experiments: 
            negll.append(self.neg_log_likelihood((_alpha, _theta), actions, rewards))
        
        # updated
        min_index = np.argmin(negll)
        negll = np.array(negll).reshape((100, 100))
        alpha_idx, theta_idx = np.unravel_index(min_index, negll.shape)
        theta_idx, alpha_idx = np.unravel_index(min_index, negll.shape)

        # Get the corresponding alpha and theta values
        alpha_pred = alpha_range[alpha_idx]
        theta_pred = theta_range[theta_idx]

        print(f'alpha_idx:           {alpha_idx}')
        print(f'theta_idx:           {theta_idx}')
        print(f'min nll:             {np.min(negll)}')
        print()
        print(f"Minimum negll value: {negll[alpha_idx, theta_idx]}")
        print(f"Corresponding alpha: {alpha_pred}")
        print(f"Corresponding theta: {theta_pred}")

        if _plt is None:
            _plt = plt.figure(figsize=(8, 6))

        plt.contourf(alpha_range, theta_range, negll, levels=50, cmap='viridis')
        plt.colorbar(label='Negative Log Likelihood')
        if alpha_true is not None and theta_true is not None:
            plt.scatter(alpha_true, theta_true, color='orange', label=f'True (alpha:{round(alpha_true,2)}, theta:{round(theta_true,2)})', edgecolors='black', s=100)
        plt.scatter(alpha_pred, theta_pred, color='red', label=f'Pred (alpha:{round(alpha_pred,2)}, theta:{round(theta_pred,2)})', edgecolors='black', s=100)
        plt.xlabel('alpha')
        plt.ylabel('theta')
        plt.title(f'Negative Log Likelihood (alpha vs theta)')
        plt.legend()

        plt.tight_layout()
        return plt, negll, theta_pred, theta_range, alpha_pred, alpha_range
  
    def rw_exp(self, m=10):
        # Load data using the data_loader function
        jsondata, list_subjects, EDS_Easy, EDS_Hard = load_data()

        rw_easy_results = {}
        rw_hard_results = {}

        alpha_bounds = (0.01,1)
        theta_bounds = (0.01,10)
        _bounds = (alpha_bounds, theta_bounds)
        counter = 0

        for k,v in EDS_Easy.items():
            _stim_codes = v[0]
            _resp_codes = v[1]
            _actions, _rewards = get_actions_rewards(_resp_codes)
            _args = (_stim_codes, _actions, _rewards)
            res = self.optimize_scikit_model_over_init_parameters(
                actions=_actions,
                rewards=_rewards,
                loss_function=None,
                alpha_init_range=np.linspace(alpha_bounds[0], alpha_bounds[1], m),
                theta_init_range=np.linspace(theta_bounds[0], theta_bounds[1], m),
                bounds=_bounds,
                log_progress=True
                )
            
            #Extracting optimal parameters in to individual variables
            new_res = {
                'negLL': res['negLL'],
                'alpha_opt': res['param_opt'][0],
                'beta_opt': res['param_opt'][1],
                'BIC': res['BIC'],
                'optimal_init_params': res['optimal_init_params']
            }

            rw_easy_results[k]=new_res
            counter +=1
            print(f"Progress:{(counter/30)*100}%")

        for k,v in EDS_Hard.items():
            _stim_codes = v[0]
            _resp_codes = v[1]
            _actions, _rewards = get_actions_rewards(_resp_codes)
            _args = (_stim_codes, _actions, _rewards)

            res = self.optimize_scikit_model_over_init_parameters(
                actions=_actions,
                rewards=_rewards,
                loss_function=None,
                alpha_init_range=np.linspace(alpha_bounds[0], alpha_bounds[1], m),
                theta_init_range=np.linspace(theta_bounds[0], theta_bounds[1], m),
                bounds=_bounds,
                log_progress=True
                )
                            #Extracting optimal parameters in to individual variables
            new_res = {
                'negLL': res['negLL'],
                'alpha_opt': res['param_opt'][0],
                'beta_opt': res['param_opt'][1],
                'BIC': res['BIC'],
                'optimal_init_params': res['optimal_init_params']
            }
            rw_hard_results[k]=new_res

            counter +=1
            print(f"Progress:{(counter/30)*100}%")

        # Convert dictionaries to DataFrames
        df1 = pd.DataFrame.from_dict(rw_easy_results, orient='index').reset_index().rename(columns={'index': 'subject'})
        df2 = pd.DataFrame.from_dict(rw_hard_results, orient='index').reset_index().rename(columns={'index': 'subject'})

        # Add a column to distinguish between the two groups
        df1['group'] = 'Easy'
        df2['group'] = 'Hard'

        # Combine the DataFrames
        df_rw = pd.concat([df1, df2])

        # Save to the correct path using PROJECT_ROOT
        output_path = os.path.join(PROJECT_ROOT, 'data', 'data_frames', 'df_rw.csv')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_rw.to_csv(output_path, index = False)