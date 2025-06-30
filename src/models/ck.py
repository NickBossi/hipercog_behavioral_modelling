import plotly.graph_objects as go
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

from plotly.subplots import make_subplots

# from src.rescorla_wagner_model import (RoscorlaWagner)
# from src.rescorla_wagner_model_plots import (RescorlaWagnerPlots)
# from src.rescorla_wagner_model_simulation import (RescorlaWagnerSimulate)
# from src.rescorla_wagner_model_diagnostics import (RoscorlaWagerModelDiagnostics)
from .cog_sci_learning_model_base import (MultiArmedBanditModels, add_diag_line, load_data, get_actions_rewards)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ChoiceKernelModel(MultiArmedBanditModels):

    def __init__(self):
        pass
        
    def simulate(self, alpha, beta, N=100, CK_init = [0,0], noise = True):
        self.simulated_params = {'alpha': alpha, 'beta':beta, 'N': N, 'CK_init':CK_init, 'noise': noise}
        
        c = np.zeros((N), dtype = int)
        CK_stored = np.zeros((2,N), dtype = float)
        CK = CK_init

        for t in range(N):
            CK_stored[:,t]= CK
            p0 = np.exp(beta*CK[0])/(np.exp(beta*CK[0])+np.exp(beta*CK[1]))
            p1 = 1-p0
            

            # sample choice K with probability p(k)
            if noise:
                if np.random.random_sample(1) < p0:
                    c[t] = 0
                else:
                    c[t] = 1
            else:
            # make choice without noise
                c[t] = np.argmax([p0, p1])
            
            #updated CK values:
            if c[t] == 0:
                CK[0] = CK[0] + alpha * (1-CK[0])
                CK[1] = CK[1] + alpha * (0-CK[1])
            else:
                CK[0] = CK[0] + alpha * (0-CK[0])
                CK[1] = CK[1] + alpha * (1-CK[1])
            
                
        self.simulated_experiment = {'actions':c, 'CK_stored': CK_stored}

    def predict(self, ):
        pass

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
        results = {'alpha (true)': [], 'beta (true)': [], 'N': [], 'alpha (pred)': [], 'beta (pred)': []}
        
        param_grid = self.generate_parameter_init_range(alpha_range=alpha_range, theta_range=beta_range, log_progress=log_progress)
        
        for alpha_true, beta_true in param_grid:
            # Simulate data with current parameters
            self.simulate(alpha=alpha_true, beta=beta_true, N=N)
        
            # Get simulated actions
            actions = self.simulated_experiment['actions']

            # Estimate parameters using multiple init conditions
            results_dict = self.optimize_scikit_model_over_init_parameters(
                actions=actions,
                bounds=bounds,
                log_progress=False
            )
            res_nll = results_dict['negLL']
            param_fits = results_dict['param_opt']
            BIC = results_dict['BIC']
            optimal_init_params = results_dict['optimal_init_params']
            alpha_hat_scikit, beta_hat_scikit = param_fits
            
            results['N'].append(N)
            results['alpha (true)'].append(alpha_true)
            results['beta (true)'].append(beta_true)
            results['alpha (pred)'].append(alpha_hat_scikit)
            results['beta (pred)'].append(beta_hat_scikit)
        
        results = pd.DataFrame(results)

        # Plot
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Alpha Sensitivity", "Beta Sensitivity"))
        fig.add_trace(go.Scatter(x=results['alpha (true)'], y=results['alpha (pred)'], mode='markers', name='alpha estimates'), row=1, col=1)
        fig.add_trace(go.Scatter(x=results['beta (true)'], y=results['beta (pred)'], mode='markers', name='beta estimates'), row=1, col=2)
        
        # Add diagonal lines for reference
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=[0, 12], y=[0, 12], mode='lines', line=dict(dash='dash'), showlegend=False), row=1, col=2)

        fig.update_layout(height=600, width=1200, title_text="Sensitivity Analysis: Parameter Stability Estimate", template='none')
        fig.update_xaxes(title_text='alpha (true)', row=1, col=1)
        fig.update_yaxes(title_text='alpha (pred)', row=1, col=1)
        fig.update_xaxes(title_text='beta (true)', row=1, col=2)
        fig.update_yaxes(title_text='beta (pred)', row=1, col=2)
        
        # Save results to file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(current_dir, '..', 'data', 'data_frames', 'sensitivity_analysis')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'ck_sensitivity_analysis.csv')
        results.to_csv(save_path, index=False)
        
        
        return results, fig

    def plot_neg_log_likelihood(self):
        return super().plot_neg_log_likelihood()

    def neg_log_likelihood(self, parameters, actions):
        alpha_choice, beta_choice = parameters
        N = len(actions)
        CK = [0,0]
        log_likelihood = 0
        choice_probs = np.zeros((N), dtype = float)

        for i in range(N):
            p0 = np.exp(beta_choice*CK[0])/(np.exp(beta_choice*CK[0])+np.exp(beta_choice*CK[1]))
            p = [p0, 1-p0]

            choice_probs[i] = p[actions[i]]

            if actions[i] == 0:
                CK[0] = CK[0] + alpha_choice * (1-CK[0])
                CK[1] = CK[1] + alpha_choice * (0-CK[1])
            else:
                CK[0] = CK[0] + alpha_choice * (0-CK[0])
                CK[1] = CK[1] + alpha_choice * (1-CK[1])
            
        negLL = -np.sum(np.log(choice_probs))
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
    
    def optimize_brute_force(self, actions, bounds=((0,1), (0.1, 10)), n = 10, loss_function=None, log_progress=True):
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
            _loss = loss_function((_alpha, _theta), actions)
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
                args=(actions),
                bounds=bounds)

            if result.fun < negLL:
                negLL = result.fun
                params_opt = result.x
                optimal_init_params = (_alpha, _theta)
      
        # compute BIC

        BIC = self.compute_BIC(negLL, len(actions), 2)
        return {'negLL': negLL, 'param_opt': params_opt, 'BIC': BIC, 'optimal_init_params': optimal_init_params}

    def ck_exp(self, m=1):
        # Load data using the data_loader function
        jsondata, list_subjects, EDS_Easy, EDS_Hard = load_data()

        ck_easy_results = {}
        ck_hard_results = {}

        alpha_bounds = (0.01,1)
        theta_bounds = (0.01,5)
        alpha_init_range = np.linspace(0,1,m)
        theta_init_range = np.linspace(.1, 5, m)
        _bounds = (alpha_bounds, theta_bounds)
        counter = 0

        for k,v in EDS_Easy.items():
            _stim_codes = v[0]
            _resp_codes = v[1]
            _actions, _rewards = get_actions_rewards(_resp_codes)
            _args = (_stim_codes, _actions, _rewards)
            res = self.optimize_scikit_model_over_init_parameters(
                actions=_actions,
                loss_function=None,
                alpha_init_range=alpha_init_range,
                theta_init_range=theta_init_range,
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
            ck_easy_results[k]=new_res
            counter +=1
            print(f"Progress:{(counter/30)*100}%")

        for k,v in EDS_Hard.items():
            _stim_codes = v[0]
            _resp_codes = v[1]
            _actions, _rewards = get_actions_rewards(_resp_codes)
            _args = (_stim_codes, _actions, _rewards)
            res = self.optimize_scikit_model_over_init_parameters(
                actions=_actions,
                loss_function=None,
                alpha_init_range=alpha_init_range,
                theta_init_range=theta_init_range,
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
            ck_hard_results[k]=new_res
            counter +=1
            print(f"Progress:{(counter/30)*100}%")

        # Convert dictionaries to DataFrames
        df1 = pd.DataFrame.from_dict(ck_easy_results, orient='index').reset_index().rename(columns={'index': 'subject'})
        df2 = pd.DataFrame.from_dict(ck_hard_results, orient='index').reset_index().rename(columns={'index': 'subject'})

        # Add a column to distinguish between the two groups
        df1['group'] = 'Easy'
        df2['group'] = 'Hard'

        # Combine the DataFrames
        df_ck = pd.concat([df1, df2])
        # Save to the correct path using PROJECT_ROOT
        output_path = os.path.join(PROJECT_ROOT, 'data', 'data_frames', 'df_ck.csv')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_ck.to_csv(output_path, index = False) 
