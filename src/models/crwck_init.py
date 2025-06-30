"""
Contextual Rescorla-Wagner with Choice Kernel Model
-------------------------------------------------
This model combines two learning mechanisms:
1. Rescorla-Wagner learning: Updates action values based on reward prediction errors
2. Choice Kernel learning: Tracks choice history independently of rewards

The model maintains separate Q-values and choice kernels for each context (stimulus),
allowing it to learn context-specific action values and choice preferences.

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

from .cog_sci_learning_model_base import (MultiArmedBanditModels, add_diag_line, load_pre_data, load_data, get_actions_rewards)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ContextualRascorlaWagnerChoiceKernelInitModel(MultiArmedBanditModels):
    """
    A reinforcement learning model that combines Rescorla-Wagner learning with a choice kernel.
    
    The model learns in two ways:
    1. Through reward prediction errors (Rescorla-Wagner)
    2. Through choice history (Choice Kernel)
    
    Parameters:
    -----------
    alpha_rw : float
        Learning rate for Rescorla-Wagner updates (0-1)
    alpha_choice : float
        Learning rate for choice kernel updates (0-1)
    beta : float
        Inverse temperature parameter controlling exploration/exploitation
    """
    def __init__(self):
        """Initialize the model with default parameters."""
        pass

    def predict(self):
        """Predict actions given current model parameters."""
        pass

    def simulate(self, alpha_rw, alpha_choice, beta, N=100, Q_init=[0, 0], noise=True):
        """
        Simulate the Rescorla-Wagner with Choice Kernel model over N trials.

        Parameters:
        -----------
        alpha_rw : float
            Learning rate for Rescorla-Wagner updates
        alpha_choice : float
            Learning rate for choice kernel updates
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
        self : ContextualRascorlaWagnerChoiceKernelModel
            The model instance with simulation results stored in simulated_experiment
        """
        # Store simulation parameters
        self.simulated_params = {'alpha_rw': alpha_rw, 'alpha_choice': alpha_choice, 'beta': beta, 'N': N, 'Q_init': Q_init, 'noise': noise}

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
                # Softmax function with both Q and CK contributions
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

    def perform_sensitivity_analysis(
        self,
        alpha_rw_range=np.linspace(0, 1, 10),
        alpha_ck_range=np.linspace(0, 1, 10),
        beta_range=np.linspace(0, 10, 10),
        N=100,
        bounds=((0, 1), (0, 1), (1, 10)),
        log_progress=True
        ):
        """
        Perform sensitivity analysis to evaluate parameter stability.
        """
        results = {
            'alpha_rw (true)': [], 
            'alpha_ck (true)': [], 
            'beta (true)': [], 
            'N': [], 
            'alpha_rw (pred)': [], 
            'alpha_ck (pred)': [], 
            'beta (pred)': []
        }
        
        param_grid = self.generate_parameter_init_range(
            alpha_range=alpha_rw_range, 
            theta_range=beta_range, 
            log_progress=log_progress
        )
        
        for alpha_rw_true, alpha_ck_true, beta_true in param_grid:
            # Simulate data with current parameters
            self.simulate(
                alpha_rw=alpha_rw_true, 
                alpha_choice=alpha_ck_true, 
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
            alpha_rw_hat, alpha_ck_hat, beta_hat = param_fits
            
            # Store results
            results['N'].append(N)
            results['alpha_rw (true)'].append(alpha_rw_true)
            results['alpha_ck (true)'].append(alpha_ck_true)
            results['beta (true)'].append(beta_true)
            results['alpha_rw (pred)'].append(alpha_rw_hat)
            results['alpha_ck (pred)'].append(alpha_ck_hat)
            results['beta (pred)'].append(beta_hat)
        
        results = pd.DataFrame(results)

        # Create plots
        fig = make_subplots(rows=1, cols=3, subplot_titles=("Alpha RW Sensitivity", "Alpha CK Sensitivity", "Beta Sensitivity"))
        
        # Plot alpha_rw estimates
        fig.add_trace(
            go.Scatter(
                x=results['alpha_rw (true)'], 
                y=results['alpha_rw (pred)'], 
                mode='markers', 
                name='alpha_rw estimates'
            ), 
            row=1, 
            col=1
        )

        # Plot alpha_ck estimates
        fig.add_trace(
            go.Scatter(
                x=results['alpha_ck (true)'], 
                y=results['alpha_ck (pred)'], 
                mode='markers', 
                name='alpha_ck estimates'
            ), 
            row=1, 
            col=2
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
            col=3
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
                x=[0, 1], 
                y=[0, 1],   
                mode='lines', 
                line=dict(dash='dash'), 
                showlegend=False
            ), 
            row=1, 
            col=2
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
            col=3
        )

        # Update layout
        fig.update_layout(
            height=600, 
            width=1800, 
            title_text="Sensitivity Analysis: Parameter Stability Estimate", 
            template='none'
        )
        fig.update_xaxes(title_text='alpha_rw (true)', row=1, col=1)
        fig.update_yaxes(title_text='alpha_rw (pred)', row=1, col=1)
        fig.update_xaxes(title_text='alpha_ck (true)', row=1, col=2)
        fig.update_yaxes(title_text='alpha_ck (pred)', row=1, col=2)
        fig.update_xaxes(title_text='beta (true)', row=1, col=3)
        fig.update_yaxes(title_text='beta (pred)', row=1, col=3)
        
        # Save results to file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(current_dir, '..', 'data', 'data_frames', 'sensitivity_analysis')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'crwck_sensitivity_analysis.csv')
        results.to_csv(save_path, index=False)
        print("DONE!")
        
        return results, fig

    def plot_neg_log_likelihood(self):
        """Plot the negative log likelihood surface."""
        return super().plot_neg_log_likelihood()

    def neg_log_likelihood(self, parameters, stim_codes, actions, rewards, Q_init = [[0,0],[0,0]], CK_init = [[0,0],[0,0]]):
        """
        Compute the negative log likelihood of the model given parameters and data.

        Parameters:
        -----------
        parameters : tuple
            (alpha_rw, alpha_choice, beta) model parameters
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
        alpha_rw, alpha_choice, beta = parameters
        
        # Initialize Q-values and choice kernels
        Q = Q_init
        N = len(actions)
        CK = CK_init
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
                p0 = np.exp(beta*(Q[1][0]+CK[1][0]))/(np.exp(beta*(Q[1][0]+CK[1][0]))+np.exp(beta*(Q[1][1]+CK[1][1])))
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
            (alpha_rw, alpha_choice, theta) parameter combinations
        """
        alpha_iterable = tqdm(alpha_range) if log_progress else alpha_range

        for _alpha_rw in alpha_iterable:
            for _alpha_ck in alpha_range:
                for _theta in theta_range:
                    yield _alpha_rw,_alpha_ck, _theta

    def compute_BIC(self, LL, T, k_params=3):
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
        for _alpha_rw, _alpha_ck, _theta in gen_experiments:
            _loss = loss_function((_alpha_rw,_alpha_ck, _theta), stim_codes, actions, rewards)
            neg_log_likelihoods.append((_alpha_rw, _alpha_ck, _theta, _loss))
        
        # Find optimal parameters
        alpha_rw_optima, alpha_ck_optima, theta_optima, loss = min(neg_log_likelihoods, key=lambda x: x[3])

        # Compute BIC
        BIC = self.compute_BIC(loss, len(actions), 3)

        results = {
            'negLL': loss,
            'alpha_rw_pred': alpha_rw_optima,
            'alpha_ck_pred': alpha_ck_optima,
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
        Q_init = [[0,0],[0,0]],
        CK_init = [[0,0],[0,0]],
        bounds=((0, 1),(0,1), (0.1, 15)),
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
        for _alpha_rw, _alpha_ck, _theta in gen_experiments:              
            result, res_nll, param_fits = self.optimize_scikit(
                loss_function=loss_function,
                init_guess=[_alpha_rw, _alpha_ck, _theta],
                args=(stim_codes, actions, rewards, Q_init, CK_init),
                bounds=bounds)

            if result.fun < negLL:
                negLL = result.fun
                params_opt = result.x
                optimal_init_params = (_alpha_rw, _alpha_ck, _theta)
      
        # Compute BIC
        BIC = self.compute_BIC(negLL, len(actions), 3)
        return {'negLL': negLL, 'param_opt': params_opt, 'BIC': BIC, 'optimal_init_params': optimal_init_params}
         
    def crwck_init_exp(self, m=5):
        _,_,EDS_Easy, EDS_Hard = load_data()

        crwck_easy_results = {}
        crwck_hard_results = {}

        alpha_bounds = (0.01,1)
        theta_bounds = (0.01,5)
        alpha_init = np.linspace(0,1,m)
        theta_init = np.linspace(0.1,5,m)
        _bounds = (alpha_bounds,alpha_bounds, theta_bounds)
        counter = 0

        crwck_easy_df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'data_frames', 'pre_sessions', 'df_crwck_pre_init.csv'))
        crwck_hard_df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'data_frames', 'pre_sessions', 'df_crwck_pre_init.csv'))
        

        for k,v in EDS_Easy.items():

            _stim_codes = v[0]
            _resp_codes = v[1]
            _rele_mode = v[2]
            _actions, _rewards = get_actions_rewards(_resp_codes)
            _args = (_stim_codes, _actions, _rewards)
            

            Q_init = np.array(eval(crwck_easy_df[crwck_easy_df['subject'] == k]['Q_init'].values[0]))
            Q00, Q01, Q10, Q11 = Q_init
            CK_init = np.array(eval(crwck_easy_df[crwck_easy_df['subject'] == k]['CK_init'].values[0]))
            CK00, CK01, CK10, CK11 = CK_init

            if _rele_mode == 0:
                Q0 = (Q00 + Q10)/2
                Q1 = (Q01 + Q11)/2
                CK0 = (CK00 + CK10)/2
                CK1 = (CK01 + CK11)/2
                Q_init = [Q0, Q1]
                CK_init = [CK0, CK1]
            
            else: 
                Q0 = (Q00 + Q01)/2
                Q1 = (Q10 + Q11)/2
                CK0 = (CK00 + CK01)/2
                CK1 = (CK10 + CK11)/2
                Q_init = [Q0, Q1]
                CK_init = [CK0, CK1]

            res = self.optimize_scikit_model_over_init_parameters(
                stim_codes = _stim_codes,
                actions=_actions,
                rewards=_rewards,
                loss_function=None,
                alpha_init_range=alpha_init,
                theta_init_range=theta_init,
                Q_init = Q_init,
                CK_init = CK_init,
                bounds=_bounds,
                log_progress=True
                )
            
            #Extracting optimal parameters in to individual variables
            new_res = {
                'negLL': res['negLL'],
                'alpha_rw_opt': res['param_opt'][0],
                'alpha_ck_op': res['param_opt'][1],
                'beta_opt': res['param_opt'][2],
                'BIC': res['BIC'],
                'optimal_init_params': res['optimal_init_params']
            }

            crwck_easy_results[k]=new_res
            counter +=1
            print(f"Progress:{(counter/30)*100}%")

        for k,v in EDS_Hard.items():
            _stim_codes = v[0]
            _resp_codes = v[1]
            _rele_mode = v[2]
            _actions, _rewards = get_actions_rewards(_resp_codes)
            _args = (_stim_codes, _actions, _rewards)

            Q_init = np.array(eval(crwck_hard_df[crwck_hard_df['subject'] == k]['Q_init'].values[0]))
            Q00, Q01, Q10, Q11 = Q_init
            CK_init = np.array(eval(crwck_hard_df[crwck_hard_df['subject'] == k]['CK_init'].values[0]))
            CK00, CK01, CK10, CK11 = CK_init

            if _rele_mode == 0:
                Q0 = (Q00 + Q10)/2
                Q1 = (Q01 + Q11)/2
                CK0 = (CK00 + CK10)/2
                CK1 = (CK01 + CK11)/2
                Q_init = [Q0, Q1]
                CK_init = [CK0, CK1]
            
            else: 
                Q0 = (Q00 + Q01)/2
                Q1 = (Q10 + Q11)/2
                CK0 = (CK00 + CK01)/2
                CK1 = (CK10 + CK11)/2
                Q_init = [Q0, Q1]
                CK_init = [CK0, CK1]

            res = self.optimize_scikit_model_over_init_parameters(
                stim_codes = _stim_codes,
                actions=_actions,
                rewards=_rewards,
                loss_function=None,
                alpha_init_range=alpha_init,
                theta_init_range=theta_init,
                Q_init = Q_init,
                CK_init = CK_init,
                bounds=_bounds,
                log_progress=True
                )
            
            new_res = {
                'negLL': res['negLL'],
                'alpha_rw_opt': res['param_opt'][0],
                'alpha_ck_op': res['param_opt'][1],
                'beta_opt': res['param_opt'][2],
                'BIC': res['BIC'],
                'optimal_init_params': res['optimal_init_params']
            }
            crwck_hard_results[k]=new_res
            counter +=1
            print(f"Progress:{(counter/30)*100}%")

        # Convert dictionaries to DataFrames
        df1 = pd.DataFrame.from_dict(crwck_easy_results, orient='index').reset_index().rename(columns={'index': 'subject'})
        df2 = pd.DataFrame.from_dict(crwck_hard_results, orient='index').reset_index().rename(columns={'index': 'subject'})

        # Add a column to distinguish between the two groups
        df1['group'] = 'Easy'
        df2['group'] = 'Hard'

        # Combine the DataFrames
        df_crwck = pd.concat([df1, df2])
        output_path = os.path.join(PROJECT_ROOT, 'data', 'data_frames', 'df_crwck_init.csv')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_crwck.to_csv(output_path, index = False)

