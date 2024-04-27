import os
import numpy as np
import argparse
#import yaml
import json
import pandas as pd
from tqdm import tqdm, trange
# from pathos import multiprocessing # speeds up large map functions
from sklearn import metrics
import scipy.stats as stats
from scipy.interpolate import interp1d
from scipy.integrate import simps
import pickle

from figures import make_tuning_plot_rmse, evidence_tuning_plots, save_plot, plot_spearman_r

def make_summary_df(df, summary_functions, summary_names):
    """ Convert the full_df object into a summary df.

    Args:
        df: full df of all experiments
        summary_functions: fns to be applied to each experiment run (e.g. cutoff rmse)
        summary_names: Names of outputs in df for the summary functions
    """
    # df = df.query("partition == 'test'")
    # Group by cutoff
    # subsetted = df.groupby(["dataset", "method_name", "trial_number", 
    #                         "task_name"])
    merge_list = []
    for name, fn in tqdm(zip(summary_names, summary_functions), total=len(summary_names)):
        res = fn(df)
        merge_list.append(res)

    summary_df = merge_list #pd.concat(merge_list, axis=1).reset_index()

    # column_names = ['rmse', 'mae', 'Predicted Probability', 'Expected Probability']

    return summary_df

### Make summary of extracted confs 

def ordered_regr_error(data, sort_factor, 
                       skip_factor=1, error_type = "mae"):
    """ Order df_subset by sort_factor and compute rmse or mae at each cutoff"""
    
    data = data.to_dict('records')
 
    sorted_data = sorted(data,
                         key=lambda pair: pair[sort_factor],        # sort_factor can be 'confidence'
                         reverse=True)
    cutoff,errors = [], []
    if error_type == "rmse":
        error_list = [set_['error']**2 for set_ in sorted_data]
    elif error_type == "mae":
        error_list = [np.abs(set_['error']) for set_ in sorted_data]
    else:
        raise NotImplementedError()

    total_error = np.sum(error_list)
    for i in tqdm(range(0, len(error_list), skip_factor)):
        cutoff.append(sorted_data[i][sort_factor])
        if error_type == "rmse": 
            errors.append(np.sqrt(total_error / len(error_list[i:])))
        elif error_type == "mae": 
            errors.append(total_error / len(error_list[i:]))
        else: 
            raise NotImplementedError()

        total_error -= np.sum(error_list[i :i+skip_factor])

    return np.array(errors)

def regr_calibration_fn(df_subset, num_partitions = 10):
    """ Create regression calibration curves in the observed bins.
    Full explanation and code taken from: 
    
    https://github.com/uncertainty-toolbox/uncertainty-toolbox/blob/89c42138d3028c8573a1a007ea8bef80ad2ed8e6/uncertainty_toolbox/metrics_calibration.py#L182

    """
    expected_p = np.arange(num_partitions+1)/num_partitions
    calibration_list = []
    # method = df_subset["method_name"].values[0]

    # df_subset = df_subset.query('partition == "test"')
    data = df_subset.to_dict('records')
    predictions = np.array([i['pred'] for i in data])
    confidence = np.array([i['conf'] for i in data])
    targets = np.array([i['target'] for i in data])
        
    # Taken from github link in docstring
    norm = stats.norm(loc=0, scale=1)
    gaussian_lower_bound = norm.ppf(0.5 - expected_p / 2.0)
    gaussian_upper_bound = norm.ppf(0.5 + expected_p / 2.0)
    residuals = predictions - targets
    normalized_residuals = (residuals.flatten() / confidence.flatten()).reshape(-1, 1)
    above_lower = normalized_residuals >= gaussian_lower_bound
    below_upper = normalized_residuals <= gaussian_upper_bound
    within_quantile = above_lower * below_upper
    obs_proportions = np.sum(within_quantile, axis=0).flatten() / len(residuals)

    return obs_proportions

def create_regression_summary(full_df, skip_factor = 1, num_partitions=40): 
    """ Given the regression full df as input, create a smaller summary by
    collecting data across trials

    Return:
        summary_df_names, summary_df

    """
    # Now make cutoff RMSE and MAE plots
    # Skip factor creates binning
    summary_names = ["rmse", "mae", "Predicted Probability", 
                     "Expected Probability"]
    cutoff_rmse = lambda x: ordered_regr_error(x, "conf", 
                                               skip_factor = skip_factor,
                                               error_type="rmse")
    cutoff_mae = lambda x: ordered_regr_error(x, "conf", 
                                              skip_factor = skip_factor, 
                                              error_type="mae")
    calibration_fn_ = lambda x : regr_calibration_fn(x, num_partitions = num_partitions)
    expected_p = lambda x : np.arange(num_partitions+1)/num_partitions
    summary_fns = [cutoff_rmse, cutoff_mae, calibration_fn_, expected_p]
    summary_df = make_summary_df(full_df, summary_fns, summary_names)
    # Delete this line 
    # summary_df.fillna(0, inplace=True)
    return summary_names, summary_df

def make_low_n_plots(full_df, summary_df, results_dir):
    """ make_low_n_plots"""
    
    y_points = 100
    low_n_rename = {"ensemble" : "ensemble",
                        "dropout": "dropout",
                        "evidence_new_reg_0.2" : "evidence"
                     }

    # Make calibration plot for evidence
    evidence_tuning_plots(summary_df, x_input="Expected Probability",
                          y_input = "Predicted Probability",
                          x_name="Expected Probability",
                          y_name= "Predicted Probability")
    save_plot(results_dir, f"low_n_evidence_tuning_plot")

    spearman_r_summary = plot_spearman_r(full_df, std=False)
    spearman_r_summary.to_csv(os.path.join(results_dir,
                                           f"spearman_r_low_n_summary_stats.csv"))
    save_plot(results_dir, f"spearman_r_low_n")

if __name__=="__main__":
    with open("/home/hpaat/my_exp/MTrans-evidential/output/evi_del/conf_0.pkl",'rb') as file:
        data = pickle.load(file)
    
    pred = data['pred'][:,0].reshape(-1,1)
    target = data['target'][:,0].reshape(-1,1)
    conf = data['conf'][:,0].reshape(-1,1)
    error = data['error'][:,0].reshape(-1,1)

    data_new = np.concatenate((pred, target, conf, error), axis=1)

    column_names = ['pred', 'target', 'conf', 'error']
    df = pd.DataFrame(data_new, columns=column_names)
    df['dataset'] = 'KITTI'
    summary_names, summary = create_regression_summary(df)
    
    summary = ['KITTI'] + [lst.tolist() for lst in summary]
    summary = pd.DataFrame([summary], columns=['dataset'] + summary_names)
    
    make_low_n_plots(df, summary, '/home/hpaat/my_exp/MTrans-evidential/small_experiments')

    
