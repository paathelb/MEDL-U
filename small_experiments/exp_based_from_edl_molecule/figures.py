import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import pandas as pd
import numpy as np
# import argparse
# from rdkit import Chem
# from rdkit.Chem import Draw
# from rdkit.Chem.Draw import rdMolDraw2D
import scipy.stats as stats 
# from scipy.interpolate import interp1d
# from scipy.integrate import simps
# from sklearn import metrics
# from tqdm import tqdm, trange
# from pathos import multiprocessing # speeds up large map functions

from numpy import nan
from ast import literal_eval

def make_tuning_plot_rmse(df, error_col_name="rmse",
                          error_title = "Top 10% RMSE",
                          cutoff = 0.10):

    """ Create the tuning plot for different lambda evidence parameters, but
    plot 10% RMSE instead of calibration. """

    df = df.copy()
    rmse_trial = df[0]
    method_df = []
    method = 'EDL'

    num_tested = len(rmse_trial)
    cutoff_index = int(cutoff * num_tested) - 1
    rmse_val = rmse_trial[-cutoff_index]
    to_append = {error_title: rmse_val,
                    "Regularizer Coeff, $\lambda$": method,
                    "method_name": method}
    method_df.append(to_append)
    method_df = pd.DataFrame(method_df)

    # Normalize by dataset
    # for dataset in datasets:
    #     # Make a divison vector of ones and change it to a different value only
    #     # for the correct dataset of interest to set max rmse to 1
    #     division_factor = np.ones(len(method_df))
    #     indices = (method_df["Data"] == dataset)

    #     # Normalize with respect to the ensemble so that this is 1
    #     max_val  = method_df[indices].query("method_name == 'ensemble'").mean()[error_title]

    #     # Take the maximum of the AVERAGE so it's normalized to 1
    #     division_factor[indices] = max_val
    #     method_df[error_title] = method_df[error_title] / division_factor

    method_df_evidence = method_df
    # method_df_ensemble = method_df[["ensemble" in str(i) for i in
    #                                 method_df['method_name']]].reset_index()

    data_colors = {
        'kitti' : sns.color_palette()[0]
    }

    min_x = 0.50 #np.min(method_df_evidence["Regularizer Coeff, $\lambda$"])
    max_x= 0.50 #np.max(method_df_evidence["Regularizer Coeff, $\lambda$"])

    sns.lineplot(x="Regularizer Coeff, $\lambda$", y=error_title,
                 hue="Data", alpha=0.8, data=method_df_evidence,
                 palette = data_colors)

    # color = data_colors['kitti']
    # area = subdf[error_title].mean()
    # std = subdf[error_title].std()
    # plt.hlines(area, min_x, max_x, linestyle="--", color=color, alpha=0.8)

    # # Add ensemble baseline
    # ensemble_line = plt.plot([], [], color='black', linestyle="--",
    #                              label="Ensemble")
    # # Now make ensemble plots
    # plt.legend(bbox_to_anchor=(1.1, 1.05))

def evidence_tuning_plots(df, x_input = "Mean Predicted Avg",
                          y_input = "Empirical Probability",
                          x_name="Mean Predicted",
                          y_name="Empirical Probability"):
    """ Plot the tuning plot at different evidence values """

    def lineplot(x, y, trials, methods, **kwargs):
        """method_lineplot.

        Args:
            y:
            methods:
            kwargs:
        """
        uniq_methods = set(methods.values)
        method_order = sorted(uniq_methods)

        method_new_names = [f"$\lambda={i:0.4f}$" for i in method_order]
        method_df = []
        for method_idx, (method, method_new_name) in enumerate(zip(method_order,
                                                                   method_new_names)):
            lines_y = y[methods == method]
            lines_x = x[methods == method]
            for index, (xx, yy,trial) in enumerate(zip(lines_x, lines_y, trials)):

                to_append = [{x_name  : x,
                              y_name: y,
                              "Method": method_new_name,
                              "Trial" : trial}
                    for i, (x,y) in enumerate(zip(xx,yy))]
                method_df.extend(to_append)
        method_df = pd.DataFrame(method_df)
        x = np.linspace(0,1,100)
        plt.plot(x, x, linestyle='--', color="black")
        sns.lineplot(x=x_name, y=y_name, hue="Method",
                     alpha=0.8,
                     hue_order=method_new_names, data=method_df,)
        # estimator=None, units = "Trial")

    df = df.copy()

    # Get the regularizer and reset coeff
    coeff = [0.50]        # [float(i.split("evidence_new_reg_")[1]) for i in df['method_name']]
    df["method_name"] = coeff
    df["Data"] = ['KITTI']  #convert_dataset_names(df["dataset"])
    df["Method"] = df["method_name"]
    df['trial_number'] = [0]

    g = sns.FacetGrid(df, col="Data",  height=6, sharex = False, sharey = False)
    g.map(lineplot, x_input,  y_input, "trial_number",
          methods=df["Method"]).add_legend()

def save_plot(outdir, outname):
    """ Save current plot"""
    plt.savefig(os.path.join(outdir, "png", outname+".png"), bbox_inches="tight")
    plt.savefig(os.path.join(outdir, "pdf", outname+".pdf"), bbox_inches="tight")
    plt.close()

def plot_spearman_r(full_df, std=True): 
    """ Plot spearman R summary stats """
    
    if std: 
        convert_to_std(full_df)
    # full_df["Data"] = convert_dataset_names(full_df["dataset"])
    
    grouped_df = full_df.groupby(["dataset"])
    spearman_r = grouped_df.apply(lambda x : stats.spearmanr(x['conf'].values,  np.abs(x['error'].values )).correlation)
    
    new_df = spearman_r.reset_index().rename({0: "Spearman Rho" },
                                             axis=1)

    method_order = ['evidence'] #[i for i in METHOD_ORDER 
    #                 if i in pd.unique(new_df['method_name'])]
    new_df['Method'] = 'evidence' #new_df['method_name']
    new_df['Dataset'] = new_df['dataset']

    plot_width = 2.6 * len(pd.unique(new_df['Dataset']))
    plt.figure(figsize=(plot_width, 5))

    sns.barplot(data=new_df , x="Dataset", y="Spearman Rho",
                hue="Method", hue_order = method_order)
    import pdb; pdb.set_trace()
    spearman_r_summary = new_df.groupby(["dataset", "Method"]).describe()['Spearman Rho'].reset_index()
    return spearman_r_summary 