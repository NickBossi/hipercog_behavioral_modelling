"""
Contextual Rescorla-Wagner with Choice Kernel Model (Shared Alpha)
----------------------------------------------------------------
This model combines two learning mechanisms with a shared learning rate:
1. Rescorla-Wagner learning: Updates action values based on reward prediction errors
2. Choice Kernel learning: Tracks choice history independently of rewards

The model maintains separate Q-values and choice kernels for each context (stimulus),
but uses a single learning rate (alpha) for both learning mechanisms.

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

class ContextualRescorlaWagnerChoiceKernelSharedAlphaModel(MultiArmedBanditModels):
    """
    A reinforcement learning model that combines Rescorla-Wagner learning with a choice kernel,
    using a single learning rate for both mechanisms.
    
    The model learns in two ways:
    1. Through reward prediction errors (Rescorla-Wagner)
    2. Through choice history (Choice Kernel)
    
    Parameters:
    -----------
    alpha : float
        Learning rate shared between Rescorla-Wagner and Choice Kernel updates (0-1)
    beta : float
        Inverse temperature parameter controlling exploration/exploitation
    """
    def __init__(self):
        """Initialize the model with default parameters."""
        pass

    def predict(self):
        """Predict actions given current model parameters."""
        pass

    def simulate(self, alpha, beta, N=100, Q_init=[0, 0], noise=True):
        """
        Simulate the Rescorla-Wagner with Choice Kernel model over N trials.

        Parameters:
        -----------
        alpha : float
            Learning rate shared between Rescorla-Wagner and Choice Kernel updates
        beta : float
            Inverse temperature parameter
        N : int
            Number of trials to simulate
        Q_init : list[float, float]
            Initial Q-values for each action
        noise : bool
            Whether to add noise to choices (True) or choose deterministically (False)

        Returns:
        --------
        self : ContextualRescorlaWagnerChoiceKernelSharedAlphaModel
            The model instance with simulation results stored in simulated_experiment
        """
        # Store simulation parameters
        self.simulated_params = {'alpha': alpha, 'beta': beta, 'N': N, 'Q_init': Q_init, 'noise': noise}

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
                p0 = np.exp(beta*(Q0[0] + CK0[0])) / (np.exp(beta*(Q0[0] + CK0[0])) + np.exp(beta*(Q0[1] + CK0[1])))
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

                # Update Q-values using Rescorla-Wagner rule with shared alpha
                delta = r[t] - Q0[c[t]]  # Prediction error
                Q0[c[t]] = Q0[c[t]] + alpha * delta

                # Update choice kernel based on chosen action with shared alpha
                if c[t] == 0:
                    CK0[0] = CK0[0] + alpha * (1 - CK0[0])  # Increase for chosen action
                    CK0[1] = CK0[1] + alpha * (0 - CK0[1])  # Decrease for unchosen action
                else:
                    CK0[0] = CK0[0] + alpha * (0 - CK0[0])
                    CK0[1] = CK0[1] + alpha * (1 - CK0[1])

            else:
                s[t] = 1  # Context 1
                # Store current Q and CK values
                Q_stored[1, :, t] = Q1
                CK_stored[1, :, t] = CK1
                
                # Compute choice probabilities using both Q and CK values
                p0 = np.exp(beta*(Q1[0] + CK1[0])) / (np.exp(beta*(Q1[0] + CK1[0])) + np.exp(beta*(Q1[1] + CK1[1])))
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

                # Update Q-values using Rescorla-Wagner rule with shared alpha
                delta = r[t] - Q1[c[t]]
                Q1[c[t]] = Q1[c[t]] + alpha * delta

                # Update choice kernel based on chosen action with shared alpha
                if c[t] == 0:
                    CK1[0] = CK1[0] + alpha * (1 - CK1[0])
                    CK1[1] = CK1[1] + alpha * (0 - CK1[1])
                else:
                    CK1[0] = CK1[0] + alpha * (0 - CK1[0])
                    CK1[1] = CK1[1] + alpha * (1 - CK1[1])

        # Store simulation results
        self.simulated_experiment = {
            'action': c,          # Sequence of actions taken
            'reward': r,          # Sequence of rewards received
            'Q_stored': Q_stored, # Q-values over time for both contexts
            'CK_stored': CK_stored, # Choice kernel values over time for both contexts
            'stim_codes': s       # Sequence of contexts
        }
        return self

    def perform_sensitivity_analysis(
        self,
        alpha_range=np.linspace(0, 1, 10),
        beta_range=np.linspace(0, 10, 10),
        N=100,
        bounds=((0, 1), (1, 10)),
        log_progress=True
        ):
        """
        Perform sensitivity analysis to evaluate parameter stability.
        """
        results = {
            'alpha (true)': [], 
            'beta (true)': [], 
            'N': [], 
            'alpha (pred)': [], 
            'beta (pred)': []
        }
        
        param_grid = self.generate_parameter_init_range(
            alpha_range=alpha_range, 
            theta_range=beta_range, 
            log_progress=log_progress
        )
        
        for alpha_true, beta_true in param_grid:
            # Simulate data with current parameters
            self.simulate(
                alpha=alpha_true, 
                beta=beta_true, 
                N=N
            )
        
            # Get simulated data
            actions = self.simulated_experiment['action']
            rewards = self.simulated_experiment['reward']
            stim_codes = self.simulated_experiment['stim_codes']

            # Estimate parameters using multiple init conditions
            results_dict = self.optimize_scikit_model_over_init_parameters(
                stim_codes=stim_codes,
                actions=actions,
                rewards=rewards,
                bounds=bounds,
                log_progress=False
            )
            
            res_nll = results_dict['negLL']
            param_fits = results_dict['param_opt']
            BIC = results_dict['BIC']
            optimal_init_params = results_dict['optimal_init_params']
            alpha_hat, beta_hat = param_fits
            
            # Store results
            results['N'].append(N)
            results['alpha (true)'].append(alpha_true)
            results['beta (true)'].append(beta_true)
            results['alpha (pred)'].append(alpha_hat)
            results['beta (pred)'].append(beta_hat)
        
        results = pd.DataFrame(results)

        # Create plots
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Alpha Sensitivity", "Beta Sensitivity"))
        
        # Plot alpha estimates
        fig.add_trace(
            go.Scatter(
                x=results['alpha (true)'], 
                y=results['alpha (pred)'], 
                mode='markers', 
                name='alpha estimates'
            ), 
            row=1, 
            col=1
        )

        # Plot beta estimates
        fig.add_trace(
            go.Scatter(
                x=results['beta (true)'], 
                y=results['beta (pred)'], 
                mode='markers', 
                name='beta estimates'
            ), 
            row=1, 
            col=2
        )
        
        # Add diagonal lines for reference
        fig.add_trace(
            go.Scatter(
                x=[0, 1], 
                y=[0, 1],   
                mode='lines', 
                line=dict(dash='dash'), 
                showlegend=False
            ), 
            row=1, 
            col=1
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 12], 
                y=[0, 12], 
                mode='lines', 
                line=dict(dash='dash'), 
                showlegend=False
            ), 
            row=1, 
            col=2
        )

        # Update layout
        fig.update_layout(
            height=600, 
            width=1200, 
            title_text="Sensitivity Analysis: Parameter Stability Estimate", 
            template='none'
        )
        fig.update_xaxes(title_text='alpha (true)', row=1, col=1)
        fig.update_yaxes(title_text='alpha (pred)', row=1, col=1)
        fig.update_xaxes(title_text='beta (true)', row=1, col=2)
        fig.update_yaxes(title_text='beta (pred)', row=1, col=2)
        
        # Save results to file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(current_dir, '..', 'data', 'data_frames', 'sensitivity_analysis')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'crwcksa_sensitivity_analysis.csv')
        results.to_csv(save_path, index=False)
        print("DONE!")
        
        return results, fig

    def plot_neg_log_likelihood(self):
        """Plot the negative log likelihood surface."""
        return super().plot_neg_log_likelihood()

    def neg_log_likelihood(self, parameters, stim_codes, actions, rewards, Q_init = [0,0]):
        """
        Compute the negative log likelihood of the model given parameters and data.

        Parameters:
        -----------
        parameters : tuple
            (alpha, beta) model parameters
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
        alpha, beta = parameters
        
        # Initialize Q-values and choice kernels
        Q = [Q_init,Q_init]
        N = len(actions)
        CK = [[0,0],[0,0]]
        log_likelihood = 0
        choice_probs = np.zeros((N), dtype = float)

        # Compute likelihood for each trial
        for i in range(N):
            if stim_codes[i] == 0:
                # Compute choice probabilities using softmax
                p0 = np.exp(beta*(Q[0][0]+CK[0][0]))/(np.exp(beta*(Q[0][0]+CK[0][0]))+np.exp(beta*(Q[0][1]+CK[0][1])))
                p = [p0, 1-p0]

                # Store choice probability for actual choice
                choice_probs[i] = p[actions[i]]

                # Update choice kernel with shared alpha
                if actions[i] == 0:
                    CK[0][0] = CK[0][0] + alpha * (1-CK[0][0])
                    CK[0][1] = CK[0][1] + alpha * (0-CK[0][1])
                else:
                    CK[0][0] = CK[0][0] + alpha * (0-CK[0][0])
                    CK[0][1] = CK[0][1] + alpha * (1-CK[0][1])

                # Update Q-values using Rescorla-Wagner rule with shared alpha
                delta = rewards[i]-Q[0][actions[i]]
                Q[0][actions[i]] = Q[0][actions[i]] + alpha*delta

            else:
                # Same process for context 1
                p0 = np.exp(beta*(Q[1][0]+CK[1][0]))/(np.exp(beta*(Q[1][0]+CK[1][0]))+np.exp(beta*(Q[1][1]+CK[1][1])))
                p = [p0, 1-p0]

                choice_probs[i] = p[actions[i]]

                if actions[i] == 0:
                    CK[1][0] = CK[1][0] + alpha * (1-CK[1][0])
                    CK[1][1] = CK[1][1] + alpha * (0-CK[1][1])
                else:
                    CK[1][0] = CK[1][0] + alpha * (0-CK[1][0])
                    CK[1][1] = CK[1][1] + alpha * (1-CK[1][1])

                delta = rewards[i]-Q[1][actions[i]]
                Q[1][actions[i]] = Q[1][actions[i]] + alpha*delta
        
        # Compute negative log likelihood
        negLL = -np.sum(np.log(choice_probs))
        return negLL  
    
    def generate_parameter_init_range(self, alpha_range, theta_range, log_progress=False):
        """
        Generate parameter combinations for grid search.

        Parameters:
        -----------
        alpha_range : array
            Range of alpha values to test
        theta_range : array
            Range of theta (beta) values to test
        log_progress : bool
            Whether to show progress bar

        Yields:
        -------
        tuple
            (alpha, theta) parameter combinations
        """
        alpha_iterable = tqdm(alpha_range) if log_progress else alpha_range

        for _alpha in alpha_iterable:
            for _theta in theta_range:
                yield _alpha, _theta

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
    
    def optimize_brute_force(self, stim_codes, actions, rewards, bounds=((0,1), (0.1, 10)), n = 10, loss_function=None, log_progress=True):
        """
        Optimize model parameters using brute force grid search.

        Parameters:
        -----------
        stim_codes : array
            Sequence of stimulus codes
        actions : array
            Sequence of actions
        rewards : array
            Sequence of rewards
        bounds : tuple
            Parameter bounds for optimization
        n : int
            Number of grid points per parameter
        loss_function : callable
            Loss function to optimize
        log_progress : bool
            Whether to show progress bar

        Returns:
        --------
        dict
            Optimization results including optimal parameters and BIC
        """
        if loss_function is None:
            loss_function = self.neg_log_likelihood

        # Generate parameter grid
        alpha_bounds, theta_bounds = bounds
        alpha_values = np.linspace(alpha_bounds[0], alpha_bounds[1], n)
        theta_values = np.linspace(theta_bounds[0], theta_bounds[1], n)

        # Generate parameter combinations
        gen_experiments = self.generate_parameter_init_range(
            alpha_range=alpha_values,
            theta_range=theta_values,
            log_progress=log_progress
            )

        # Evaluate each parameter combination
        neg_log_likelihoods = []
        for _alpha, _theta in gen_experiments:
            _loss = loss_function((_alpha, _theta), stim_codes, actions, rewards)
            neg_log_likelihoods.append((_alpha, _theta, _loss))
        
        # Find optimal parameters
        alpha_optima, theta_optima, loss = min(neg_log_likelihoods, key=lambda x: x[2])

        # Compute BIC
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
        Optimize model parameters using scikit-learn's minimize function.

        Parameters:
        -----------
        init_guess : list
            Initial parameter guess
        args : tuple
            Additional arguments for loss function
        bounds : tuple
            Parameter bounds
        loss_function : callable
            Loss function to optimize
        single : bool
            Whether to return single result or full optimization object

        Returns:
        --------
        dict or tuple
            Optimization results
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
        stim_codes,
        actions,
        rewards,
        loss_function=None,
        alpha_init_range=np.linspace(0, 1, 5),
        theta_init_range=np.linspace(0.1, 10, 7),
        bounds=((0, 1), (0.1, 15)),
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
        theta_init_range : array
            Range of initial theta values
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
        gen_experiments = self.generate_parameter_init_range(alpha_range=alpha_init_range, theta_range=theta_init_range, log_progress=log_progress)

        # Try each initial condition
        for _alpha, _theta in gen_experiments:              
            result, res_nll, param_fits = self.optimize_scikit(
                loss_function=loss_function,
                init_guess=[_alpha, _theta],
                args=(stim_codes, actions, rewards),
                bounds=bounds)

            if result.fun < negLL:
                negLL = result.fun
                params_opt = result.x
                optimal_init_params = (_alpha, _theta)
      
        # Compute BIC
        BIC = self.compute_BIC(negLL, len(actions), 2)
        return {'negLL': negLL, 'param_opt': params_opt, 'BIC': BIC, 'optimal_init_params': optimal_init_params} 
    
    def crwcksa_exp(self, m=7):
        # Load data using the data_loader function
        jsondata, list_subjects, EDS_Easy, EDS_Hard = load_data()

        crwcksa_easy_results = {}
        crwcksa_hard_results = {}

        alpha_bounds = (0.01,1)
        theta_bounds = (0.01,5)
        alpha_init = np.linspace(0,1,m)
        theta_init = np.linspace(0.1,5,m)
        _bounds = (alpha_bounds, theta_bounds)
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
                theta_init_range=theta_init,
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

            crwcksa_easy_results[k]=new_res
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
                theta_init_range=theta_init,
                bounds=_bounds,
                log_progress=True
                )
            new_res = {
                'negLL': res['negLL'],
                'alpha_opt': res['param_opt'][0],
                'beta_opt': res['param_opt'][1],
                'BIC': res['BIC'],
                'optimal_init_params': res['optimal_init_params']
            }
            crwcksa_hard_results[k]=new_res
            counter +=1
            print(f"Progress:{(counter/30)*100}%")

        # Convert dictionaries to DataFrames
        df1 = pd.DataFrame.from_dict(crwcksa_easy_results, orient='index').reset_index().rename(columns={'index': 'subject'})
        df2 = pd.DataFrame.from_dict(crwcksa_hard_results, orient='index').reset_index().rename(columns={'index': 'subject'})

        # Add a column to distinguish between the two groups
        df1['group'] = 'Easy'
        df2['group'] = 'Hard'

        # Combine the DataFrames
        df_crwcksa = pd.concat([df1, df2])
        # Save to the correct path using PROJECT_ROOT
        output_path = os.path.join(PROJECT_ROOT, 'data', 'data_frames', 'df_crwcksa.csv')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_crwcksa.to_csv(output_path, index = False)
