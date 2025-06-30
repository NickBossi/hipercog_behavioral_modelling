"""
Contextual Rescorla-Wagner Model
-------------------------------
A reinforcement learning model that learns action values through reward prediction errors.
The model maintains separate Q-values for each context (stimulus), allowing it to learn
context-specific action values.

Source:
-------
- https://ccn.studentorg.berkeley.edu/pdfs/papers/WilsonCollins_modelFitting.pdf
- https://shawnrhoads.github.io/gu-psyc-347/module-03-01_Models-of-Learning.html

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

class ContextualRoscorlaWagnerModel(MultiArmedBanditModels):
    """
    Rescorla-Wagner Model implementation for contextual learning.

    The model learns action values through reward prediction errors, with separate
    Q-values maintained for each context (stimulus).

    Parameters:
    -----------
    alpha : float
        Learning rate (0-1)
    theta : float
        Inverse temperature parameter controlling exploration/exploitation
    """

    def __init__(self, alpha_range=(0,1), theta_range=(0.1, 10)):
        """
        Initialize the model with parameter ranges.

        Parameters:
        -----------
        alpha_range : tuple
            Range of alpha values (learning rate)
        theta_range : tuple
            Range of theta values (inverse temperature)
        """
        self.alpha_range = alpha_range
        self.theta_range = theta_range

    def simulate(self, alpha, theta, N=100, Q_init=[0, 0], noise=True):
        """
        Simulate the Rescorla-Wagner model over N trials.

        Parameters:
        -----------
        alpha : float
            Learning rate
        theta : float
            Inverse temperature parameter
        N : int
            Number of trials to simulate
        Q_init : list[float, float]
            Initial Q-values for each action
        noise : bool
            Whether to add noise to choices (True) or choose deterministically (False)

        Returns:
        --------
        self : ContextualRoscorlaWagnerModel
            The model instance with simulation results stored in simulated_experiment
        """
        # Store simulation parameters
        self.simulated_params = {'alpha': alpha, 'theta':theta, 'N': N, 'Q_init':Q_init, 'noise': noise}

        # Initialize arrays to store choices, rewards, and stimulus codes
        c = np.zeros((N), dtype=int)  # choices
        r = np.zeros((N), dtype=int)  # rewards
        s = np.zeros((N), dtype = int)  # stimulus codes

        # Initialize Q-values for both contexts
        Q_stored = np.zeros((2, N), dtype=float)
        Q0 = Q_init  # Q-values for context 0
        Q1 = Q_init  # Q-values for context 1

        # Run simulation for N trials
        for t in range(N):
            if np.random.rand() < 0.5:
                s[t] = 0  # Context 0
                # Store current Q-values
                Q_stored[:, t] = Q0
                
                # Compute choice probabilities using softmax
                p0 = np.exp(theta*Q0[0]) / (np.exp(theta*Q0[0]) + np.exp(theta*Q0[1]))
                p1 = 1 - p0

                # Make choice with or without noise
                if noise:
                    if np.random.random_sample(1) < p0:
                        c[t] = 0
                    else:
                        c[t] = 1
                else:
                    # Choose deterministically
                    c[t] = np.argmax([p0, p1])
                    
                # Generate reward (1 if choice matches context, 0 otherwise)
                if c[t] == s[t]:
                    r[t] = 1

                # Update Q-values using Rescorla-Wagner rule
                delta = r[t] - Q0[c[t]]  # Prediction error
                Q0[c[t]] = Q0[c[t]] + alpha * delta

            else:
                s[t] = 1  # Context 1
                # Store current Q-values
                Q_stored[:, t] = Q1
                
                # Compute choice probabilities using softmax
                p0 = np.exp(theta*Q1[0]) / (np.exp(theta*Q1[0]) + np.exp(theta*Q1[1]))
                p1 = 1 - p0

                # Make choice with or without noise
                if noise:
                    if np.random.random_sample(1) < p0:
                        c[t] = 0
                    else:
                        c[t] = 1
                else:
                    # Choose deterministically
                    c[t] = np.argmax([p0, p1])
                    
                # Generate reward
                if c[t] == s[t]:
                    r[t] = 1

                # Update Q-values using Rescorla-Wagner rule
                delta = r[t] - Q1[c[t]]  # Prediction error
                Q1[c[t]] = Q1[c[t]] + alpha * delta

        # Store simulation results
        self.simulated_experiment = {
            'action': c,          # Sequence of actions taken
            'reward': r,          # Sequence of rewards received
            'Q_stored': Q_stored, # Q-values over time for both contexts
            'stim_codes': s       # Sequence of contexts
        }
        return self

    def neg_log_likelihood(self, parameters, stim_codes, actions, rewards, Q_init=[0, 0], epsilon_clip=1e-10):
        """
        Compute the negative log likelihood of the model given parameters and data.

        Parameters:
        -----------
        parameters : tuple
            (alpha, theta) model parameters
        stim_codes : array
            Sequence of stimulus codes (contexts)
        actions : array
            Sequence of actions taken
        rewards : array
            Sequence of rewards received
        Q_init : list[float, float]
            Initial Q-values
        epsilon_clip : float
            Small value to prevent log(0) errors

        Returns:
        --------
        negLL : float
            Negative log likelihood of the data under the model
        """
        # Extract parameters
        alpha, theta = parameters
        _stim_codes, _actions, _rewards = stim_codes, actions, rewards

        # Initialize Q-values for both contexts
        Q0 = Q_init
        Q1 = Q_init
        T = len(_actions)
        choiceProb = np.zeros((T), dtype=float)

        # Compute likelihood for each trial
        for t in range(T):
            if _stim_codes[t] == 0:
                # Compute choice probabilities using softmax
                p0 = np.exp(theta*Q0[0]) / (np.exp(theta*Q0[0]) + np.exp(theta*Q0[1]))
                p = [p0, 1-p0]

                # Store choice probability for actual choice
                choiceProb[t] = p[_actions[t]]

                # Update Q-values using Rescorla-Wagner rule
                delta = _rewards[t] - Q0[_actions[t]]  # Prediction error
                Q0[_actions[t]] = Q0[_actions[t]] + alpha * delta

            if _stim_codes[t] == 1:
                # Compute choice probabilities using softmax
                p1 = np.exp(theta*Q1[0]) / (np.exp(theta*Q1[0]) + np.exp(theta*Q1[1]))
                p = [p1, 1-p1]

                # Store choice probability for actual choice
                choiceProb[t] = p[_actions[t]]

                # Update Q-values using Rescorla-Wagner rule
                delta = _rewards[t] - Q1[_actions[t]]  # Prediction error
                Q1[_actions[t]] = Q1[_actions[t]] + alpha * delta
        
        # Add clip for safety: ensure 0,1 values can log
        choiceProb = np.clip(choiceProb, epsilon_clip, 1 - epsilon_clip) 
        negLL = -np.sum(np.log(choiceProb))
        
        return negLL
    
    def generate_parameter_init_range(self, alpha_range, theta_range, log_progress=False):
        """
        Generate parameter combinations for grid search.

        Parameters:
        -----------
        alpha_range : array
            Range of alpha values to test
        theta_range : array
            Range of theta values to test
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

    def optimize_brute_force(self, stim_codes,actions, rewards, bounds=((0,1), (0.1, 10)), n = 100, loss_function=None, log_progress=True):
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

        if single:
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

    def predict(self, alpha, theta, reward_vector, noise=0):
        """
        Given parameters alpha and theta, predict a sequence of actions.

        Parameters:
        -----------
        alpha : float
            Learning rate
        theta : float
            Inverse temperature parameter
        reward_vector : array
            Sequence of rewards
        noise : float
            Amount of noise to add to choices

        Returns:
        --------
        list
            Sequence of predicted actions
        """
        raise Exception('Implementation not logically sound.')
        actions = []
        Q = [0.5, 0.5]  # Initialize Q-values
        for reward in reward_vector:
            # Compute choice probabilities
            p0 = np.exp(theta * Q[0]) / (np.exp(theta * Q[0]) + np.exp(theta * Q[1]))
            p1 = 1 - p0

            # Make choice with noise
            if noise is not None and noise > 0:
                if np.random.random_sample(1) < p0:
                    action = 0
                else:
                    action = 1
            else:
                # Choose deterministically
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

        Parameters:
        -----------
        alpha_range : array
            Range of alpha values to test
        theta_range : array
            Range of theta values to test
        N : int
            Number of trials per simulation
        bounds : tuple
            Parameter bounds for optimization
        log_progress : bool
            Whether to show progress bar

        Returns:
        --------
        tuple
            (results DataFrame, plot figure)
        """
        # Initialize results dictionary
        results = {'alpha (true)': [], 'theta (true)': [], 'N': [], 'alpha (pred)': [], 'theta (pred)': []}
        
        # Generate parameter combinations
        param_grid = self.generate_parameter_init_range(alpha_range=alpha_range, theta_range=theta_range, log_progress=log_progress)
        
        # Test each parameter combination
        for alpha_true, theta_true in param_grid:
            # Simulate data with true parameters
            self.simulate(alpha=alpha_true, theta=theta_true, N=N)

            # Get simulated data
            actions = self.simulated_experiment['action']
            rewards = self.simulated_experiment['reward']
            stim_codes = self.simulated_experiment['stim_codes']

            results_dict = self.optimize_scikit_model_over_init_parameters(
                stim_codes=stim_codes,
                actions=actions, 
                rewards=rewards, 
                bounds=bounds, 
                log_progress=False)
            
            res_nll = results_dict['negLL']
            param_fits = results_dict['param_opt']
            BIC = results_dict['BIC']
            optimal_init_params = results_dict['optimal_init_params']
            alpha_hat_scikit, beta_hat_scikit = param_fits

            alpha_hat_scikit, beta_hat_scikit = param_fits

            # Store results
            results['N'].append(N)
            results['alpha (true)'].append(alpha_true)
            results['theta (true)'].append(theta_true)
            results['alpha (pred)'].append(alpha_hat_scikit)
            results['theta (pred)'].append(beta_hat_scikit)
        
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

        # Plot theta estimates
        fig.add_trace(
            go.Scatter(
                x=results['theta (true)'], 
                y=results['theta (pred)'], 
                mode='markers', 
                name='theta estimates'
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
        fig.update_xaxes(title_text='theta (true)', row=1, col=2)
        fig.update_yaxes(title_text='theta (pred)', row=1, col=2)
        
        # Save results to file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(current_dir, '..', 'data', 'data_frames', 'sensitivity_analysis')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'crw_sensitivity_analysis.csv')
        results.to_csv(save_path, index=False)
        print("DONE!")
        
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
        Compare brute force and scikit optimization methods.

        Parameters:
        -----------
        alpha_range : array
            Range of alpha values to test
        theta_range : array
            Range of theta values to test
        fit_brute_force : bool
            Whether to use brute force optimization
        fit_scikit : bool
            Whether to use scikit optimization
        bounds : list
            Parameter bounds for optimization
        N : int
            Number of trials per simulation
        log_progress : bool
            Whether to show progress bar

        Returns:
        --------
        tuple
            (results DataFrame, plot figure)
        """
        # Generate parameter combinations
        gen_experiments = self.generate_parameter_init_range(
            alpha_range=alpha_range, 
            theta_range=theta_range, 
            log_progress=log_progress
        )

        # Initialize results dictionary
        results = {
            'alpha (true)': [], 
            'theta (true)': [], 
            'alpha (pred - brute force)': [], 
            'theta (pred - brute force)': [], 
            'alpha (pred - scikit-optim)': [], 
            'theta (pred - scikit-optim)': []
        }

        # Test each parameter combination
        for alpha_true, theta_true in gen_experiments:
            # Simulate data with true parameters
            self.simulate(alpha=alpha_true, theta=theta_true, N=N)
            actions = self.simulated_experiment['action']
            rewards = self.simulated_experiment['reward']

            # Brute force optimization
            if fit_brute_force:
                brute_force_results = self.optimize_brute_force(
                    bounds=bounds, 
                    actions=actions, 
                    rewards=rewards, 
                    log_progress=False
                )
                alpha_hat_brute_force = brute_force_results['alpha_pred']
                theta_hat_brute_force = brute_force_results['theta_pred']
                BIC_brute_force = brute_force_results['BIC']
            else:
                alpha_hat_brute_force, theta_hat_brute_force, BIC_brute_force = None, None, None
            
            # Scikit optimization
            if fit_scikit:
                res_nll, param_fits, BIC, _ = self.optimize_scikit_model_over_init_parameters(
                    actions=actions, 
                    rewards=rewards, 
                    bounds=bounds, 
                    log_progress=False
                )
                alpha_hat_scikit, beta_hat_scikit = param_fits
            else:
                alpha_hat_scikit, beta_hat_scikit = None, None
            
            # Store results
            results['alpha (true)'].append(alpha_true)
            results['theta (true)'].append(theta_true)
            results['alpha (pred - brute force)'].append(alpha_hat_brute_force)
            results['theta (pred - brute force)'].append(theta_hat_brute_force)
            results['alpha (pred - scikit-optim)'].append(alpha_hat_scikit)
            results['theta (pred - scikit-optim)'].append(beta_hat_scikit)

        results = pd.DataFrame(results)

        # Create plots
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Alpha Estimate", "Theta Estimate"))
        
        # Plot brute force results
        fig.add_trace(
            go.Scatter(
                x=results['alpha (true)'], 
                y=results['alpha (pred - brute force)'], 
                mode='markers', 
                name='alpha pred (brute force)'
            ), 
            row=1, 
            col=1
        )
        fig.add_trace(
            go.Scatter(
                x=results['theta (true)'], 
                y=results['theta (pred - brute force)'], 
                mode='markers', 
                name='theta pred (brute force)'
            ), 
            row=1, 
            col=2
        )
        
        # Plot scikit results
        fig.add_trace(
            go.Scatter(
                x=results['alpha (true)'], 
                y=results['alpha (pred - scikit-optim)'], 
                mode='markers', 
                name='alpha pred (scikit)'
            ), 
            row=1, 
            col=1
        )
        fig.add_trace(
            go.Scatter(
                x=results['theta (true)'], 
                y=results['theta (pred - scikit-optim)'], 
                mode='markers', 
                name='theta pred (scikit)'
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
                x=[0, 11], 
                y=[0, 11], 
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
            title_text="Compare Parameter Recovery: Brute Force vs Scikit", 
            template='none'
        )
        fig.update_xaxes(title_text='alpha (true)', row=1, col=1)
        fig.update_yaxes(title_text='alpha (pred)', row=1, col=1)
        fig.update_xaxes(title_text='theta (true)', row=1, col=2)
        fig.update_yaxes(title_text='theta (pred)', row=1, col=2)

        return results, fig

    def plot_neg_log_likelihood(self, _plt=None):
        """
        Plot the negative log likelihood surface.

        Parameters:
        -----------
        _plt : matplotlib.pyplot
            Optional matplotlib figure to plot on

        Returns:
        --------
        tuple
            (plt, negll, theta_pred, theta_range, alpha_pred, alpha_range)
        """
        # Get simulation parameters
        alpha_true, theta_true = self.simulated_params['alpha'], self.simulated_params['theta']
        actions, rewards = self.simulated_experiment['action'], self.simulated_experiment['reward']

        # Initialize arrays
        negll = []
        alpha_range = np.linspace(0, 1, 100)
        theta_range = np.linspace(0, 10, 100)

        # Generate parameter combinations
        gen_experiments = self.generate_parameter_init_range(
            alpha_range=alpha_range, 
            theta_range=theta_range, 
            log_progress=False
        )

        # Compute negative log likelihood for each parameter combination
        for _alpha, _theta in gen_experiments: 
            negll.append(self.neg_log_likelihood((_alpha, _theta), actions, rewards))
        
        # Find minimum
        min_index = np.argmin(negll)
        negll = np.array(negll).reshape((100, 100))
        alpha_idx, theta_idx = np.unravel_index(min_index, negll.shape)
        theta_idx, alpha_idx = np.unravel_index(min_index, negll.shape)

        # Get optimal parameters
        alpha_pred = alpha_range[alpha_idx]
        theta_pred = theta_range[theta_idx]

        # Print results
        print(f'alpha_idx:           {alpha_idx}')
        print(f'theta_idx:           {theta_idx}')
        print(f'min nll:             {np.min(negll)}')
        print()
        print(f"Minimum negll value: {negll[alpha_idx, theta_idx]}")
        print(f"Corresponding alpha: {alpha_pred}")
        print(f"Corresponding theta: {theta_pred}")

        # Create plot
        if _plt is None:
            _plt = plt.figure(figsize=(8, 6))

        # Plot contour
        plt.contourf(alpha_range, theta_range, negll, levels=50, cmap='viridis')
        plt.colorbar(label='Negative Log Likelihood')
        
        # Plot true and predicted parameters
        if alpha_true is not None and theta_true is not None:
            plt.scatter(
                alpha_true, 
                theta_true, 
                color='orange', 
                label=f'True (alpha:{round(alpha_true,2)}, theta:{round(theta_true,2)})', 
                edgecolors='black', 
                s=100
            )
        plt.scatter(
            alpha_pred, 
            theta_pred, 
            color='red', 
            label=f'Pred (alpha:{round(alpha_pred,2)}, theta:{round(theta_pred,2)})', 
            edgecolors='black', 
            s=100
        )
        
        # Update plot
        plt.xlabel('alpha')
        plt.ylabel('theta')
        plt.title(f'Negative Log Likelihood (alpha vs theta)')
        plt.legend()
        plt.tight_layout()

        return plt, negll, theta_pred, theta_range, alpha_pred, alpha_range
    
    def crw_exp(self, m = 10):
        # Load data using the data_loader function
        jsondata, list_subjects, EDS_Easy, EDS_Hard = load_data()

        cwr_easy_results = {}
        cwr_hard_results = {}

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
                stim_codes = _stim_codes,
                actions=_actions,
                rewards=_rewards,
                loss_function=None,
                alpha_init_range=np.linspace(0, 1, m),
                theta_init_range=np.linspace(.1, 10, m),
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

            cwr_easy_results[k]=new_res
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
                alpha_init_range=np.linspace(0, 1, m),
                theta_init_range=np.linspace(.1, 15, m),
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
            cwr_hard_results[k]=new_res
            counter +=1
            print(f"Progress:{(counter/30)*100}%")

        # Convert dictionaries to DataFrames
        df1 = pd.DataFrame.from_dict(cwr_easy_results, orient='index').reset_index().rename(columns={'index': 'subject'})
        df2 = pd.DataFrame.from_dict(cwr_hard_results, orient='index').reset_index().rename(columns={'index': 'subject'})

        # Add a column to distinguish between the two groups
        df1['group'] = 'Easy'
        df2['group'] = 'Hard'

        # Combine the DataFrames
        df_crw = pd.concat([df1, df2])
        
        # Save to the correct path using PROJECT_ROOT
        output_path = os.path.join(PROJECT_ROOT, 'data', 'data_frames', 'df_crw.csv')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_crw.to_csv(output_path, index = False)
