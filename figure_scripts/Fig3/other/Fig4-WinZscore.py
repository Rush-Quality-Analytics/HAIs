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

use_latest = False

if use_latest == True:
    df = df[df['file date'] == '2020-04-22']
    fdates = ['2020-04-22']

else:
    fdates = ['2014-07-17', '2014-10-23', '2014-12-18', '2015-01-22', '2015-04-16', '2015-05-06', '2015-07-16', '2015-10-08', '2015-12-10', '2016-05-04', '2016-08-10', '2016-11-10', '2017-10-24', '2018-01-26', '2018-05-23', '2018-07-25', '2018-10-31', '2019-03-21', '2019-04-24', '2019-07-02', '2019-10-30', '2020-01-29', '2020-04-22']


metrics = ['SIS', 'SISc']
for metric in metrics:

    #########################################################################################
    ##################### Get SIR delta colors ##############################################

    df['colors'] = df[metric + ', better than random'].replace({0: '0.9', 1:'k'})

    #########################################################################################
    ##################### DECLARE FIGURE 3A OBJECT ##########################################
    #########################################################################################

    hais = ['CDIFF', 'CAUTI', 'CLABSI', 'MRSA']
    fig = plt.figure(figsize=(10, 10))
    rows, cols = 4, 4
    fs = 8
    kernel = 0.05
    radius = 0.5

    #########################################################################################
    ################################ GENERATE FIGURE ########################################
    #########################################################################################

    kernel = 0.2
    sz = 10
    a = 0.8
    ec1 = '0.4'
    ec2 = '0.6'
    lw = 0.2
    ################################## SUBPLOT 1 ############################################
    tdf = df[df['HAI'] == 'CAUTI']
    ax1 = plt.subplot2grid((rows, cols), (0, 0), colspan=1, rowspan=1)

    p75_min = 100
    p75_max = 0
    for fdate in fdates:
        ttdf = tdf[tdf['file date'] == fdate]
        p75 = np.percentile(ttdf['Winzorized z ' + metric], 75)
        if p75 < p75_min:
            p75_min = float(p75)
        if p75 > p75_max:
            p75_max = float(p75)
            
    ttdf = tdf[tdf['colors'] == '0.9']
    x = np.sqrt(ttdf['Days'])
    y = ttdf['Winzorized z ' + metric]
    plt.scatter(x, y, facecolors=ttdf['colors'], edgecolors = ec1, s=sz, lw=lw)

    ttdf = tdf[tdf['colors'] == 'k']
    x = np.sqrt(ttdf['Days'])
    y = ttdf['Winzorized z ' + metric]
    plt.scatter(x, y, facecolors=ttdf['colors'], edgecolors = ec2, s=sz, lw=lw)
    x = np.sqrt(tdf['Days'])

    if p75_min == p75_max:
        plt.hlines(p75_max, min(x), max(x), color='0.2')
    else:
        plt.fill_between([min(x), max(x)], p75_min, p75_max, color='0.2', linewidth=0, alpha=a)
    plt.ylabel(metric + ', Winsorized z-score', fontsize=fs)
    plt.xlabel('CAUTI, ' + r'$\sqrt{Urinary\ Catheter\ Days}$', fontsize=fs)
    plt.text(-155, -0.1, 'CAUTI', fontsize=fs+3, fontweight='bold', rotation=90)
    plt.text(120, 3, metric, fontsize=fs+3, fontweight='bold')
    plt.tick_params(axis='both', labelsize=fs-2)

    ################################## SUBPLOT 2 ############################################
    ax2 = plt.subplot2grid((rows, cols), (0, 1), colspan=1, rowspan=1)

    p75_min = 100
    p75_max = 0
    for fdate in fdates:
        ttdf = tdf[tdf['file date'] == fdate]
        p75 = np.percentile(ttdf['Winzorized z SIR'], 75)
        if p75 < p75_min:
            p75_min = float(p75)
        if p75 > p75_max:
            p75_max = float(p75)
            
    ttdf = tdf[tdf['colors'] == '0.9']
    x = np.sqrt(ttdf['Days'])
    y = ttdf['Winzorized z SIR']
    plt.scatter(x, y, facecolors=ttdf['colors'], edgecolors = ec1, s=sz, lw=lw)

    ttdf = tdf[tdf['colors'] == 'k']
    x = np.sqrt(ttdf['Days'])
    y = ttdf['Winzorized z SIR']
    plt.scatter(x, y, facecolors=ttdf['colors'], edgecolors = ec2, s=sz, lw=lw)
    x = np.sqrt(tdf['Days'])
    if p75_min == p75_max:
        plt.hlines(p75_max, min(x), max(x), color='0.2')
    else:
        plt.fill_between([min(x), max(x)], p75_min, p75_max, color='0.2', linewidth=0, alpha=a)
    plt.ylabel('SIR, Winsorized z-score', fontsize=fs)
    plt.xlabel('CAUTI, ' + r'$\sqrt{Urinary\ Catheter\ Days}$', fontsize=fs)
    plt.text(120, 3, 'SIR', fontsize=fs+3, fontweight='bold')
    plt.tick_params(axis='both', labelsize=fs-2)


    ################################## SUBPLOT 3 ############################################
    tdf = df[df['HAI'] == 'CLABSI']
    ax3 = plt.subplot2grid((rows, cols), (1, 0), colspan=1, rowspan=1)

    p75_min = 100
    p75_max = 0
    for fdate in fdates:
        ttdf = tdf[tdf['file date'] == fdate]
        p75 = np.percentile(ttdf['Winzorized z ' + metric], 75)
        if p75 < p75_min:
            p75_min = float(p75)
        if p75 > p75_max:
            p75_max = float(p75)
            
    ttdf = tdf[tdf['colors'] == '0.9']
    x = np.sqrt(ttdf['Days'])
    y = ttdf['Winzorized z ' + metric]
    plt.scatter(x, y, facecolors=ttdf['colors'], edgecolors = ec1, s=sz, lw=lw)

    ttdf = tdf[tdf['colors'] == 'k']
    x = np.sqrt(ttdf['Days'])
    y = ttdf['Winzorized z ' + metric]
    plt.scatter(x, y, facecolors=ttdf['colors'], edgecolors = ec2, s=sz, lw=lw)
    x = np.sqrt(tdf['Days'])
    if p75_min == p75_max:
        plt.hlines(p75_max, min(x), max(x), color='0.2')
    else:
        plt.fill_between([min(x), max(x)], p75_min, p75_max, color='0.2', linewidth=0, alpha=a)
    plt.ylabel(metric + ', Winsorized z-score', fontsize=fs)
    plt.xlabel('CLABSI, ' + r'$\sqrt{Device\ Days}$', fontsize=fs)
    plt.text(-170, -0.1, 'CLABSI', fontsize=fs+3, fontweight='bold', rotation=90)
    plt.tick_params(axis='both', labelsize=fs-2)

    ################################## SUBPLOT 4 ############################################
    ax4 = plt.subplot2grid((rows, cols), (1, 1), colspan=1, rowspan=1)

    p75_min = 100
    p75_max = 0
    for fdate in fdates:
        ttdf = tdf[tdf['file date'] == fdate]
        p75 = np.percentile(ttdf['Winzorized z SIR'], 75)
        if p75 < p75_min:
            p75_min = float(p75)
        if p75 > p75_max:
            p75_max = float(p75)
            
    ttdf = tdf[tdf['colors'] == '0.9']
    x = np.sqrt(ttdf['Days'])
    y = ttdf['Winzorized z SIR']
    plt.scatter(x, y, facecolors=ttdf['colors'], edgecolors = ec1, s=sz, lw=lw)

    ttdf = tdf[tdf['colors'] == 'k']
    x = np.sqrt(ttdf['Days'])
    y = ttdf['Winzorized z SIR']
    plt.scatter(x, y, facecolors=ttdf['colors'], edgecolors = ec2, s=sz, lw=lw)
    x = np.sqrt(tdf['Days'])
    if p75_min == p75_max:
        plt.hlines(p75_max, min(x), max(x), color='0.2')
    else:
        plt.fill_between([min(x), max(x)], p75_min, p75_max, color='0.2', linewidth=0, alpha=a)
    plt.ylabel('SIR, Winsorized z-score', fontsize=fs)
    plt.xlabel('CLABSI, ' + r'$\sqrt{Device\ Days}$', fontsize=fs)
    plt.tick_params(axis='both', labelsize=fs-2)


    ################################## SUBPLOT 5 ############################################
    tdf = df[df['HAI'] == 'MRSA']
    ax5 = plt.subplot2grid((rows, cols), (2, 0), colspan=1, rowspan=1)

    p75_min = 100
    p75_max = 0
    for fdate in fdates:
        ttdf = tdf[tdf['file date'] == fdate]
        p75 = np.percentile(ttdf['Winzorized z ' + metric], 75)
        if p75 < p75_min:
            p75_min = float(p75)
        if p75 > p75_max:
            p75_max = float(p75)
            
    ttdf = tdf[tdf['colors'] == '0.9']
    x = np.sqrt(ttdf['Days'])
    y = ttdf['Winzorized z ' + metric]
    plt.scatter(x, y, facecolors=ttdf['colors'], edgecolors = ec1, s=sz, lw=lw)

    ttdf = tdf[tdf['colors'] == 'k']
    x = np.sqrt(ttdf['Days'])
    y = ttdf['Winzorized z ' + metric]
    plt.scatter(x, y, facecolors=ttdf['colors'], edgecolors = ec2, s=sz, lw=lw)
    x = np.sqrt(tdf['Days'])
    if p75_min == p75_max:
        plt.hlines(p75_max, min(x), max(x), color='0.2')
    else:
        plt.fill_between([min(x), max(x)], p75_min, p75_max, color='0.2', linewidth=0, alpha=a)
    plt.ylabel(metric + ', Winsorized z-score', fontsize=fs)
    plt.xlabel('MRSA, ' + r'$\sqrt{Patient\ days}$', fontsize=fs)
    plt.text(-330, -0.1, 'MRSA', fontsize=fs+3, fontweight='bold', rotation=90)
    plt.tick_params(axis='both', labelsize=fs-2)

    ################################## SUBPLOT 6 ############################################
    ax6 = plt.subplot2grid((rows, cols), (2, 1), colspan=1, rowspan=1)

    p75_min = 100
    p75_max = 0
    for fdate in fdates:
        ttdf = tdf[tdf['file date'] == fdate]
        p75 = np.percentile(ttdf['Winzorized z SIR'], 75)
        if p75 < p75_min:
            p75_min = float(p75)
        if p75 > p75_max:
            p75_max = float(p75)
            
    ttdf = tdf[tdf['colors'] == '0.9']
    x = np.sqrt(ttdf['Days'])
    y = ttdf['Winzorized z SIR']
    plt.scatter(x, y, facecolors=ttdf['colors'], edgecolors = ec1, s=sz, lw=lw)

    ttdf = tdf[tdf['colors'] == 'k']
    x = np.sqrt(ttdf['Days'])
    y = ttdf['Winzorized z SIR']
    plt.scatter(x, y, facecolors=ttdf['colors'], edgecolors = ec2, s=sz, lw=lw)
    x = np.sqrt(tdf['Days'])
    if p75_min == p75_max:
        plt.hlines(p75_max, min(x), max(x), color='0.2')
    else:
        plt.fill_between([min(x), max(x)], p75_min, p75_max, color='0.2', linewidth=0, alpha=a)
    plt.ylabel('SIR, Winsorized z-score', fontsize=fs)
    plt.xlabel('MRSA, ' + r'$\sqrt{Patient\ days}$', fontsize=fs)
    plt.tick_params(axis='both', labelsize=fs-2)


    ################################## SUBPLOT 7 ############################################
    tdf = df[df['HAI'] == 'CDIFF']
    ax7 = plt.subplot2grid((rows, cols), (3, 0), colspan=1, rowspan=1)

    p75_min = 100
    p75_max = 0
    for fdate in fdates:
        ttdf = tdf[tdf['file date'] == fdate]
        p75 = np.percentile(ttdf['Winzorized z ' + metric], 75)
        if p75 < p75_min:
            p75_min = float(p75)
        if p75 > p75_max:
            p75_max = float(p75)
            
    ttdf = tdf[tdf['colors'] == '0.9']
    x = np.sqrt(ttdf['Days'])
    y = ttdf['Winzorized z ' + metric]
    plt.scatter(x, y, facecolors=ttdf['colors'], edgecolors = ec1, s=sz, lw=lw)

    ttdf = tdf[tdf['colors'] == 'k']
    x = np.sqrt(ttdf['Days'])
    y = ttdf['Winzorized z ' + metric]
    plt.scatter(x, y, facecolors=ttdf['colors'], edgecolors = ec2, s=sz, lw=lw)
    x = np.sqrt(tdf['Days'])
    if p75_min == p75_max:
        plt.hlines(p75_max, min(x), max(x), color='0.2')
    else:
        plt.fill_between([min(x), max(x)], p75_min, p75_max, color='0.2', linewidth=0, alpha=a)
    plt.ylabel(metric + ', Winsorized z-score', fontsize=fs)
    plt.xlabel('CDIFF, ' + r'$\sqrt{Patient\ days}$', fontsize=fs)
    plt.text(-400, -0.1, 'CDIFF', fontsize=fs+3, fontweight='bold', rotation=90)
    plt.tick_params(axis='both', labelsize=fs-2)

    ################################## SUBPLOT 8 ############################################
    ax8 = plt.subplot2grid((rows, cols), (3, 1), colspan=1, rowspan=1)

    p75_min = 100
    p75_max = 0
    for fdate in fdates:
        ttdf = tdf[tdf['file date'] == fdate]
        p75 = np.percentile(ttdf['Winzorized z SIR'], 75)
        if p75 < p75_min:
            p75_min = float(p75)
        if p75 > p75_max:
            p75_max = float(p75)
            
    ttdf = tdf[tdf['colors'] == '0.9']
    x = np.sqrt(tdf['Days'])
    y = tdf['Winzorized z SIR']
    plt.scatter(x, y, facecolors=tdf['colors'], edgecolors = ec1, s=sz, lw=lw)

    ttdf = tdf[tdf['colors'] == 'k']
    x = np.sqrt(ttdf['Days'])
    y = ttdf['Winzorized z SIR']
    plt.scatter(x, y, facecolors=ttdf['colors'], edgecolors = ec2, s=sz, lw=lw)
    x = np.sqrt(tdf['Days'])
    if p75_min == p75_max:
        plt.hlines(p75_max, min(x), max(x), color='0.2')
    else:
        plt.fill_between([min(x), max(x)], p75_min, p75_max, color='0.2', linewidth=0, alpha=a)
    plt.ylabel('SIR, Winsorized z-score', fontsize=fs)
    plt.xlabel('CDIFF, ' + r'$\sqrt{Patient\ days}$', fontsize=fs)
    plt.tick_params(axis='both', labelsize=fs-2)

    #########################################################################################
    ################################ FINAL FORMATTING #######################################
    #########################################################################################

    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    if use_latest == True:
        plt.savefig(mydir+'/figures/Fig4_latest_quarter_' + metric +'.png', dpi=200, bbox_inches = "tight")
    else:
        plt.savefig(mydir+'/figures/Fig4_all_quarters_' + metric +'.png', dpi=200, bbox_inches = "tight")
    plt.close()
