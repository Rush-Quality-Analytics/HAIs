import pandas as pd
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
from math import pi
import sys
import os
import scipy as sc
import warnings
from scipy.stats import binned_statistic
from numpy import log10, sqrt
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.patches as patches
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from scipy.stats.kde import gaussian_kde


pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 10)
warnings.filterwarnings('ignore')

mydir = os.path.expanduser("~/GitHub/HAIs/")

#########################################################################################
########################## IMPORT HAI DATA ##############################################


df = pd.read_pickle(mydir + "data/WinsorizedZscores.pkl")

quarters = ['_all_quarters', '_2020-04-22']

for q in quarters:

    if q == '_2020-04-22':
        df = df[df['file date'] == '2020-04-22']
        fdates = ['2020-04-22']

    else:
        fdates = ['2014-07-17', '2014-10-23', '2014-12-18', '2015-01-22', '2015-04-16', '2015-05-06', '2015-07-16', '2015-10-08', '2015-12-10', '2016-05-04', '2016-08-10', '2016-11-10', '2017-10-24', '2018-01-26', '2018-05-23', '2018-07-25', '2018-10-31', '2019-03-21', '2019-04-24', '2019-07-02', '2019-10-30', '2020-01-29', '2020-04-22']

    #########################################################################################
    ######################## DECLARE FIGURE OBJECT ##########################################
    #########################################################################################

    hais = ['CDIFF', 'CAUTI', 'CLABSI', 'MRSA']
    fig = plt.figure(figsize=(10, 10))
    rows, cols = 2, 2
    fs = 14
    sz = 20
    a = 0.8
    ec1 = '0.4'
    ec2 = '0.6'
    lw = 0.2
    
    ################################## SUBPLOT 1 ############################################
    metric = 'SIS'
    tdf = df[df['HAI'] == 'CAUTI']
    tdf['colors'] = tdf[metric + ', better than random'].replace({0: '0.9', 1:'k'})
    tdf['edge_colors'] = tdf[metric + ', better than random'].replace({0: ec1, 1:ec2})
    
    ax1 = plt.subplot2grid((rows, cols), (0, 0), colspan=1, rowspan=1)

    p75_min = 100
    p75_max = 0
    for fdate in fdates:
        ttdf = tdf[tdf['file date'] == fdate]
        p75 = np.percentile(ttdf['Winzorized z SIR'], 75)
        if p75 < p75_min:
            p75_min = float(p75)
        if p75 > p75_max:
            p75_max = float(p75)
    
    x = np.sqrt(tdf['Days'])
    y = tdf['Winzorized z SIR']
    plt.scatter(x, y, facecolors=tdf['colors'], edgecolors = tdf['edge_colors'], s=sz, lw=lw)
    
    weights = np.max(tdf['Winzorized z SIR']) + tdf['Winzorized z SIR']
    tdf1 = tdf[tdf['colors'] == 'k'].sample(n=10000, replace=True, weights = weights**10)
    tdf2 = tdf[tdf['colors'] == '0.9'].sample(n=10000, replace=True, weights = 1/(weights))
    tdf3 = pd.concat([tdf1, tdf2])
    tdf3 = tdf3.sample(frac=1)
    
    tdf1 = tdf1.sample(n=1)
    plt.scatter(np.sqrt(tdf1['Days']), tdf1['Winzorized z SIR'], facecolors=tdf1['colors'], edgecolors = tdf1['edge_colors'], s=sz+20, lw=lw+0.5,
        label = 'CAUTI cases < random (SIS)')
    tdf2 = tdf2.sample(n=1)
    plt.scatter(np.sqrt(tdf2['Days']), tdf2['Winzorized z SIR'], facecolors=tdf2['colors'], edgecolors = tdf2['edge_colors'], s=sz+20, lw=lw+0.5,
        label = 'CAUTI cases ≥ random (SIS)')
        
    x = np.sqrt(tdf3['Days'])
    y = tdf3['Winzorized z SIR']
    plt.scatter(x, y, facecolors=tdf3['colors'], edgecolors = tdf3['edge_colors'], s=sz, lw=lw)
    
    x = np.sqrt(tdf['Days'])
    if p75_min == p75_max:
        plt.hlines(p75_max, min(x), max(x), color='0.2')
    else:
        plt.fill_between([min(x), max(x)], p75_min, p75_max, color='0.2', linewidth=0, alpha=a)
    plt.ylabel('SIR, Winsorized z', fontsize=fs+2)
    plt.xlabel(r'$\sqrt{Urinary\ Catheter\ Days}$', fontsize=fs)
    plt.tick_params(axis='both', labelsize=fs-2)
    plt.legend(bbox_to_anchor=(-0.03, 1.05, 1.055, .1), loc=10, ncol=1, mode="expand",prop={'size':fs-2})
    
    
    ################################## SUBPLOT 2 ############################################
    metric = 'SISc'
    tdf = df[df['HAI'] == 'CAUTI']
    tdf['colors'] = tdf[metric + ', better than random'].replace({0: '0.9', 1:'k'})
    tdf['edge_colors'] = tdf[metric + ', better than random'].replace({0: ec1, 1:ec2})
    
    ax1 = plt.subplot2grid((rows, cols), (0, 1), colspan=1, rowspan=1)

    p75_min = 100
    p75_max = 0
    for fdate in fdates:
        ttdf = tdf[tdf['file date'] == fdate]
        p75 = np.percentile(ttdf['Winzorized z SIR'], 75)
        if p75 < p75_min:
            p75_min = float(p75)
        if p75 > p75_max:
            p75_max = float(p75)
    
    x = np.sqrt(tdf['Days'])
    y = tdf['Winzorized z SIR']
    plt.scatter(x, y, facecolors=tdf['colors'], edgecolors = tdf['edge_colors'], s=sz, lw=lw)
    
    weights = np.max(tdf['Winzorized z SIR']) + tdf['Winzorized z SIR']
    tdf1 = tdf[tdf['colors'] == 'k'].sample(n=10000, replace=True, weights = weights**10)
    tdf2 = tdf[tdf['colors'] == '0.9'].sample(n=10000, replace=True, weights = 1/(weights))
    tdf3 = pd.concat([tdf1, tdf2])
    tdf3 = tdf3.sample(frac=1)
    
    tdf1 = tdf1.sample(n=1)
    plt.scatter(np.sqrt(tdf1['Days']), tdf1['Winzorized z SIR'], facecolors=tdf1['colors'], edgecolors = tdf1['edge_colors'], s=sz+20, lw=lw+0.5,
        label = 'CAUTI cases < random (SIS'+'$_\mathbf{\mathrm{C}}$)')
    tdf2 = tdf2.sample(n=1)
    plt.scatter(np.sqrt(tdf2['Days']), tdf2['Winzorized z SIR'], facecolors=tdf2['colors'], edgecolors = tdf2['edge_colors'], s=sz+20, lw=lw+0.5,
        label = 'CAUTI cases ≥ random (SIS'+'$_\mathbf{\mathrm{C}}$)')
        
    x = np.sqrt(tdf3['Days'])
    y = tdf3['Winzorized z SIR']
    plt.scatter(x, y, facecolors=tdf3['colors'], edgecolors = tdf3['edge_colors'], s=sz, lw=lw)
    
    x = np.sqrt(tdf['Days'])
    if p75_min == p75_max:
        plt.hlines(p75_max, min(x), max(x), color='0.2')
    else:
        plt.fill_between([min(x), max(x)], p75_min, p75_max, color='0.2', linewidth=0, alpha=a)
    plt.ylabel('SIR, Winsorized z', fontsize=fs+2)
    plt.xlabel(r'$\sqrt{Urinary\ Catheter\ Days}$', fontsize=fs)
    plt.tick_params(axis='both', labelsize=fs-2)
    plt.legend(bbox_to_anchor=(-0.03, 1.05, 1.055, .1), loc=10, ncol=1, mode="expand",prop={'size':fs-2})
    
    
    #########################################################################################
    ################################ FINAL FORMATTING #######################################
    #########################################################################################

    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.savefig(mydir+'/figures/Fig4_' + q + '_' + metric +'_CAUTI_SIR.png', dpi=200, bbox_inches = "tight")
    plt.close()
