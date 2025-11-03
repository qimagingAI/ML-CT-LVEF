# -*- coding: utf-8 -*-
"""
Created on Fri Oct  31 19:56:20 2025

@author: Piotr Slomka, Jianhang Zhou
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Assume df_results is available
"""
df_results = {
            'bias': [bias, bias_low, bias_high], 
            'MAE': [MAE, MAE_low, MAE_high], 
            'corr': [corr, corr_low, corr_high], 
            'ICC': [ICC, ICC_low, ICC_high], 
            'CoVa': [CoVa, CoVa_low, CoVa_high], 
            'AUC': [AUC, AUC_low, AUC_high], 
            'PPV': [PPV, PPV_low, PPV_high], 
            'NPV': [NPV, NPV_low, NPV_high]
            }
"""

def func_plot(df, test):
    df_plot = pd.DataFrame.from_dict(
        {
        'sites': [
            'Pooled', 'Site 1','Site 2','Site 3','Site 4',
            'Site 5','Site 6','Site 7','Site 8','Site 9',
            'Site 10','Site 11'
        ], 
        'test_value': df[test][0],
        'test_low':   df[test][1],
        'test_high':  df[test][2]
        }
        )
    # Error bars
    df_plot['error_low'] = df_plot['test_value'] - df_plot['test_low']
    df_plot['error_high'] = df_plot['test_high'] - df_plot['test_value']

    plt.style.use('default')
    plt.rcParams.update({
        "xtick.direction": "out",
        "ytick.direction": "out",
    })

    # Create figure
    fig, ax = plt.subplots(figsize=(5, 5))

    # Plot each site
    for i, row in df_plot.iterrows():
        ax.errorbar(
            row['test_value'], row['sites'], 
            xerr=[[row['error_low']], [row['error_high']]], 
            fmt='o', markersize=8, capsize=4
        )

    ax.invert_yaxis()
    ax.get_yaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()
    plt.show()