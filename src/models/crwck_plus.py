import plotly.graph_objects as go
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
from plotly.subplots import make_subplots

from .cog_sci_learning_model_base import (MultiArmedBanditModels, get_actions_rewards, load_data)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ContextualRascorlaWagnerChoiceKernelPlusModel(MultiArmedBanditModels):

    def __init__(self):
        pass

    def predict(self):
        pass

    def simulate(self, alpha_rw, alpha_choice, beta_rw, beta_ck, N=100, Q_init=[0, 0], noise=True):
        """
        Simulate the Rescorla-Wagner with Choice Kernel Plus model over N trials.

        Parameters:
        -----------
        alpha_rw : float
            Learning rate for Rescorla-Wagner updates
        alpha_choice : float
            Learning rate for choice kernel updates
        beta_rw : float
            Inverse temperature parameter for Rescorla-Wagner component
        beta_ck : float
            Inverse temperature parameter for Choice Kernel component
        N : int
            Number of trials to simulate
        Q_init : list[float, float]
            Initial Q-values for each action
        noise : bool
            Whether to add noise to choices (True) or choose deterministically (False)

        Returns:
        --------
        self : ContextualRascorlaWagnerChoiceKernelPlusModel
            The model instance with simulation results stored in simulated_experiment
        """
        # Store simulation parameters
        self.simulated_params = {
            'alpha_rw': alpha_rw, 
            'alpha_choice': alpha_choice, 
            'beta_rw': beta_rw,
            'beta_ck': beta_ck,
            'N': N, 
            'Q_init': Q_init, 
            'noise': noise
        }

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
                
                # Compute choice probabilities using both Q and CK values with separate betas
                # Softmax function with both Q and CK contributions
                p0 = np.exp(beta_rw*Q0[0] + beta_ck*CK0[0]) / (np.exp(beta_rw*Q0[0] + beta_ck*CK0[0]) + np.exp(beta_rw*Q0[1] + beta_ck*CK0[1]))
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
                
                # Compute choice probabilities using both Q and CK values with separate betas
                p0 = np.exp(beta_rw*Q1[0] + beta_ck*CK1[0]) / (np.exp(beta_rw*Q1[0] + beta_ck*CK1[0]) + np.exp(beta_rw*Q1[1] + beta_ck*CK1[1]))
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
        beta_rw_range=np.linspace(0, 10, 10),
        beta_ck_range=np.linspace(0, 10, 10),
        N=100,
        bounds=((0, 1), (0, 1), (1, 10), (1, 10)),
        log_progress=True
        ):
        """
        Perform sensitivity analysis to evaluate parameter stability.
        """
        results = {
            'alpha_rw (true)': [], 
            'alpha_ck (true)': [], 
            'beta_rw (true)': [], 
            'beta_ck (true)': [], 
            'N': [], 
            'alpha_rw (pred)': [], 
            'alpha_ck (pred)': [], 
            'beta_rw (pred)': [], 
            'beta_ck (pred)': []
        }
        
        param_grid = self.generate_parameter_init_range(
            alpha_range=alpha_rw_range, 
            theta_range=beta_rw_range, 
            log_progress=log_progress
        )
        
        for alpha_rw_true, alpha_ck_true, beta_rw_true, beta_ck_true in param_grid:
            # Simulate data with current parameters
            self.simulate(
                alpha_rw=alpha_rw_true, 
                alpha_choice=alpha_ck_true, 
                beta_rw=beta_rw_true,
                beta_ck=beta_ck_true,
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
            alpha_rw_hat, alpha_ck_hat, beta_rw_hat, beta_ck_hat = param_fits
            
            # Store results
            results['N'].append(N)
            results['alpha_rw (true)'].append(alpha_rw_true)
            results['alpha_ck (true)'].append(alpha_ck_true)
            results['beta_rw (true)'].append(beta_rw_true)
            results['beta_ck (true)'].append(beta_ck_true)
            results['alpha_rw (pred)'].append(alpha_rw_hat)
            results['alpha_ck (pred)'].append(alpha_ck_hat)
            results['beta_rw (pred)'].append(beta_rw_hat)
            results['beta_ck (pred)'].append(beta_ck_hat)
        
        results = pd.DataFrame(results)

        # Create plots
        fig = make_subplots(rows=2, cols=2, subplot_titles=(
            "Alpha RW Sensitivity", 
            "Alpha CK Sensitivity", 
            "Beta RW Sensitivity",
            "Beta CK Sensitivity"
        ))
        
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

        # Plot beta_rw estimates
        fig.add_trace(
            go.Scatter(
                x=results['beta_rw (true)'], 
                y=results['beta_rw (pred)'], 
                mode='markers', 
                name='beta_rw estimates'
            ), 
            row=2, 
            col=1
        )

        # Plot beta_ck estimates
        fig.add_trace(
            go.Scatter(
                x=results['beta_ck (true)'], 
                y=results['beta_ck (pred)'], 
                mode='markers', 
                name='beta_ck estimates'
            ), 
            row=2, 
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
            row=2, 
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
            row=2, 
            col=2
        )

        # Update layout
        fig.update_layout(
            height=1200, 
            width=1200, 
            title_text="Sensitivity Analysis: Parameter Stability Estimate", 
            template='none'
        )
        fig.update_xaxes(title_text='alpha_rw (true)', row=1, col=1)
        fig.update_yaxes(title_text='alpha_rw (pred)', row=1, col=1)
        fig.update_xaxes(title_text='alpha_ck (true)', row=1, col=2)
        fig.update_yaxes(title_text='alpha_ck (pred)', row=1, col=2)
        fig.update_xaxes(title_text='beta_rw (true)', row=2, col=1)
        fig.update_yaxes(title_text='beta_rw (pred)', row=2, col=1)
        fig.update_xaxes(title_text='beta_ck (true)', row=2, col=2)
        fig.update_yaxes(title_text='beta_ck (pred)', row=2, col=2)
        
        # Save results to file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(current_dir, '..', 'data', 'data_frames', 'sensitivity_analysis')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'crwck_plus_sensitivity_analysis.csv')
        results.to_csv(save_path, index=False)
        print("DONE!")
        
        return results, fig

    def probs_given_data(self, resp_codes, stim_codes, actions, rewards, params, Q_init = [0,0]):
        (alpha_rw, alpha_ck, beta_rw, beta_ck) = params
        N = len(actions)
        
        GO_probs = np.zeros((N), dtype = float)

        Q = [Q_init,Q_init]

        CK = [[0,0],[0,0]]

        #Iterating through time, calculating the probability of choosing the GO action under the model, and storing this in GO_probs
        for t in range(N):
            if stim_codes[t] == 0:
                p0 = np.exp(beta_rw*Q[0][0]+beta_ck*CK[0][0])/(np.exp(beta_rw*Q[0][0]+beta_ck*CK[0][0])+np.exp(beta_rw*Q[0][1]+beta_ck*CK[0][1]))

                #Updates Choice Kernel
                if actions[t] == 0:
                    CK[0][0] = CK[0][0] + alpha_ck * (1-CK[0][0])
                    CK[0][1] = CK[0][1] + alpha_ck * (0-CK[0][1])
                else:
                    CK[0][0] = CK[0][0] + alpha_ck * (0-CK[0][0])
                    CK[0][1] = CK[0][1] + alpha_ck * (1-CK[0][1])

                #Updates Q-values
                delta = rewards[t]-Q[0][actions[t]]
                
                Q[0][actions[t]] = Q[0][actions[t]] + alpha_rw*delta
                
                GO_probs[t] = 1-p0

            else:
                p0 = np.exp(beta_rw*Q[1][0]+beta_ck*CK[1][0])/(np.exp(beta_rw*Q[1][0]+beta_ck*CK[1][0])+np.exp(beta_rw*Q[1][1]+beta_ck*CK[1][1]))

                #Updates Choice Kernel
                if actions[t] == 0:
                    CK[1][0] = CK[1][0] + alpha_ck * (1-CK[1][0])
                    CK[1][1] = CK[1][1] + alpha_ck * (0-CK[1][1])
                else:
                    CK[1][0] = CK[1][0] + alpha_ck * (0-CK[1][0])
                    CK[1][1] = CK[1][1] + alpha_ck * (1-CK[1][1])

                #Updates Q-values
                delta = rewards[t]-Q[1][actions[t]]
                
                Q[1][actions[t]] = Q[1][actions[t]] + alpha_rw*delta

                GO_probs[t] = 1-p0

        return GO_probs

    def plot_neg_log_likelihood(self):
        return super().plot_neg_log_likelihood()

    def neg_log_likelihood(self, parameters, stim_codes, actions, rewards, Q_init = [0,0]):
        alpha_rw, alpha_choice, beta_rw, beta_ck = parameters
        Q = [Q_init,Q_init]
        N = len(actions)
        CK = [[0,0],[0,0]]
        log_likelihood = 0
        choice_probs = np.zeros((N), dtype = float)

        for i in range(N):

            if stim_codes[i] == 0:
                p0 = np.exp(beta_rw*Q[0][0]+beta_ck*CK[0][0])/(np.exp(beta_rw*Q[0][0]+beta_ck*CK[0][0])+np.exp(beta_rw*Q[0][1]+beta_ck*CK[0][1]))
                p = [p0, 1-p0]

                choice_probs[i] = p[actions[i]]

                #Updates Choice Kernel
                if actions[i] == 0:
                    CK[0][0] = CK[0][0] + alpha_choice * (1-CK[0][0])
                    CK[0][1] = CK[0][1] + alpha_choice * (0-CK[0][1])
                else:
                    CK[0][0] = CK[0][0] + alpha_choice * (0-CK[0][0])
                    CK[0][1] = CK[0][1] + alpha_choice * (1-CK[0][1])

                #Updates Q-values
                delta = rewards[i]-Q[0][actions[i]]
                
                Q[0][actions[i]] = Q[0][actions[i]] + alpha_rw*delta

            else:
                p0 = np.exp(beta_rw*Q[1][0]+beta_ck*CK[1][0])/(np.exp(beta_rw*Q[1][0]+beta_ck*CK[1][0])+np.exp(beta_rw*Q[1][1]+beta_ck*CK[1][1]))
                p = [p0, 1-p0]

                choice_probs[i] = p[actions[i]]

                #Updates Choice Kernel
                if actions[i] == 0:
                    CK[1][0] = CK[1][0] + alpha_choice * (1-CK[1][0])
                    CK[1][1] = CK[1][1] + alpha_choice * (0-CK[1][1])
                else:
                    CK[1][0] = CK[1][0] + alpha_choice * (0-CK[1][0])
                    CK[1][1] = CK[1][1] + alpha_choice * (1-CK[1][1])

                #Updates Q-values
                delta = rewards[i]-Q[1][actions[i]]
                
                Q[1][actions[i]] = Q[1][actions[i]] + alpha_rw*delta
        
        negLL = -np.sum(np.log(choice_probs))
        return negLL  
    
    def generate_parameter_init_range(self, alpha_range, theta_range, log_progress=False):
        """
        Generate params: (alpha, theta) pairs.
        """
        alpha_iterable = tqdm(alpha_range) if log_progress else alpha_range

        for _alpha_rw in alpha_iterable:
            for _alpha_ck in alpha_range:
                for _theta_rw in theta_range:
                    for _theta_ck in theta_range:
                        yield _alpha_rw,_alpha_ck, _theta_rw, _theta_ck

    def compute_BIC(self, LL, T, k_params=4):
        return super().compute_BIC(LL, T, k_params=k_params)
        # bic = k * np.log(N) + 2 * neg_log_likelihood
    
    def optimize_brute_force(self, stim_codes, actions, rewards, bounds=((0,1), (0.1, 10)), n = 10, loss_function=None, log_progress=True):
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
        for _alpha_rw, _alpha_ck, _theta_rw,_theta_ck in gen_experiments:
            _loss = loss_function((_alpha_rw,_alpha_ck, _theta_rw, _theta_ck), stim_codes, actions, rewards)
            neg_log_likelihoods.append((_alpha_rw, _alpha_ck, _theta_rw,_theta_ck, _loss))
        
        # Find the set with the minimum _loss
        alpha_rw_optima, alpha_ck_optima, theta_rw_optima,theta_ck_optima, loss = min(neg_log_likelihoods, key=lambda x: x[3])

        # compute BIC
        BIC = self.compute_BIC(loss, len(actions), 3)

        results = {
            'negLL': loss,
            'alpha_rw_pred': alpha_rw_optima,
            'alpha_ck_pred': alpha_ck_optima,
            'theta_rw_pred': theta_rw_optima,
            'theta_ck_pred': theta_ck_optima,
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
        stim_codes,
        actions,
        rewards,
        loss_function=None,
        alpha_init_range=np.linspace(0, 1, 5),
        theta_init_range=np.linspace(0.1, 10, 5),
        bounds=((0, 1),(0,1), (0.1, 5),(0.1,5)),
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
        for _alpha_rw, _alpha_ck, _theta_rw,_theta_ck in gen_experiments:              
            result, res_nll, param_fits = self.optimize_scikit(
                loss_function=loss_function,
                init_guess=[_alpha_rw, _alpha_ck, _theta_rw, _theta_ck],
                args=(stim_codes, actions, rewards),
                bounds=bounds)

            if result.fun < negLL:
                negLL = result.fun
                params_opt = result.x
                optimal_init_params = (_alpha_rw, _alpha_ck, _theta_rw, _theta_ck)
      
        # compute BIC

        BIC = self.compute_BIC(negLL, len(actions), 4)
        return {'negLL': negLL, 'param_opt': params_opt, 'BIC': BIC, 'optimal_init_params': optimal_init_params}
        
    def crwck_plus_exp(self, m=5):
        _,_,EDS_Easy, EDS_Hard = load_data()

        crwck_plus_easy_results = {}
        crwck_plus_hard_results = {}

        alpha_bounds = (0.01,1)
        theta_bounds = (0.01,5)
        alpha_init = np.linspace(0,1,m)
        theta_init = np.linspace(0.1,5,m)
        _bounds = (alpha_bounds,alpha_bounds, theta_bounds,theta_bounds)
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
                'alpha_rw_opt': res['param_opt'][0],
                'alpha_ck_op': res['param_opt'][1],
                'beta_rw_opt': res['param_opt'][2],
                'beta_ck_opt': res['param_opt'][3],
                'BIC': res['BIC'],
                'optimal_init_params': res['optimal_init_params']
            }

            crwck_plus_easy_results[k]=new_res
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
                'alpha_rw_opt': res['param_opt'][0],
                'alpha_ck_op': res['param_opt'][1],
                'beta_rw_opt': res['param_opt'][2],
                'beta_ck_opt': res['param_opt'][3],
                'BIC': res['BIC'],
                'optimal_init_params': res['optimal_init_params']
            }
            crwck_plus_hard_results[k]=new_res
            counter +=1
            print(f"Progress:{(counter/30)*100}%")

        # Convert dictionaries to DataFrames
        df1 = pd.DataFrame.from_dict(crwck_plus_easy_results, orient='index').reset_index().rename(columns={'index': 'subject'})
        df2 = pd.DataFrame.from_dict(crwck_plus_hard_results, orient='index').reset_index().rename(columns={'index': 'subject'})

        # Add a column to distinguish between the two groups
        df1['group'] = 'Easy'
        df2['group'] = 'Hard'

        # Combine the DataFrames
        df_crwck_plus = pd.concat([df1, df2])
        output_path = os.path.join(PROJECT_ROOT, 'data', 'data_frames', 'df_crwck_plus.csv')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_crwck_plus.to_csv(output_path, index = False)


